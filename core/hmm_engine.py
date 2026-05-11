"""
Hidden Markov Model regime classifier.

Design contract
---------------
CAUSAL INFERENCE ONLY — this module never calls model.predict() (Viterbi)
during live detection.  Viterbi decodes the globally-optimal state sequence
using ALL observations, so label at time t changes when future bars are
added — that is look-ahead bias.  Instead we use the forward (filtering)
algorithm: P(q_t | o_{1:t}).  Adding a new bar at t+1 cannot change the
forward posteriors at t or earlier.

Regime labelling
----------------
Raw HMM states (0..K-1) are sorted by mean log-return so that label 0 is
always the most bearish state and label K-1 is the most bullish:

    K=3  →  bear | neutral | bull
    K=4  →  crash | bear | bull | euphoria
    K=5  →  crash | bear | neutral | bull | euphoria
    K=6  →  crash | deep_bear | bear | bull | euphoria | extreme_bull
    K=7  →  crash | deep_bear | bear | neutral | bull | euphoria | extreme_bull

Stability filter
----------------
A regime must persist for MIN_PERSISTENCE consecutive bars (default 3)
before it is reported.  If more than FLICKER_THRESHOLD transitions occur
in the last FLICKER_WINDOW bars (default: 4 in 20), the returned
RegimeState.is_uncertain is True and callers should reduce position sizing.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

logger = logging.getLogger(__name__)

_EPS: float = 1e-300  # guard against log(0)

# ---------------------------------------------------------------------------
# Regime name tables
# ---------------------------------------------------------------------------

_REGIME_NAMES: Dict[int, List[str]] = {
    3: ["bear", "neutral", "bull"],
    4: ["crash", "bear", "bull", "euphoria"],
    5: ["crash", "bear", "neutral", "bull", "euphoria"],
    6: ["crash", "deep_bear", "bear", "bull", "euphoria", "extreme_bull"],
    7: ["crash", "deep_bear", "bear", "neutral", "bull", "euphoria", "extreme_bull"],
}


def _make_generic_names(n: int) -> List[str]:
    """Fallback name list for n not in _REGIME_NAMES."""
    return [f"regime_{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    """Point-in-time regime assessment produced by detect_regime()."""

    regime: int           # sorted label (0 = most bearish, K-1 = most bullish)
    regime_name: str      # human-readable name from _REGIME_NAMES
    raw_regime: int       # internal HMM state index before sorting
    confidence: float     # P(raw_regime | o_{1:t}) from forward algorithm
    is_uncertain: bool    # True when flicker_count > flicker_threshold
    posteriors: np.ndarray  # full filtered posterior P(q_t=i | o_{1:t}), shape (K,)
    flicker_count: int    # regime transitions in the last flicker_window bars


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class HMMEngine:
    """
    Gaussian HMM wrapper for market-regime classification.

    All live inference uses the forward (filtering) algorithm to guarantee
    no look-ahead bias.  model.predict() (Viterbi) is only allowed inside
    fit() for the purpose of computing per-regime statistics on training data.
    """

    def __init__(
        self,
        n_regimes: int = 4,
        n_iter: int = 200,
        tol: float = 1e-4,
        covariance_type: str = "full",
        random_state: int = 42,
        auto_select: bool = False,
        n_regimes_min: int = 3,
        n_regimes_max: int = 7,
        min_persistence_bars: int = 3,
        flicker_window: int = 20,
        flicker_threshold: int = 4,
        return_feature_idx: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        n_regimes:
            Number of hidden states when auto_select=False.
        n_iter:
            Maximum EM iterations for GaussianHMM.
        tol:
            Log-likelihood convergence tolerance.
        covariance_type:
            'full', 'diag', or 'spherical'.
        random_state:
            Reproducibility seed.
        auto_select:
            If True, fit() grid-searches n_regimes in [n_regimes_min,
            n_regimes_max] using BIC and picks the best value.
        n_regimes_min / n_regimes_max:
            Search range for auto_select.
        min_persistence_bars:
            Consecutive bars a new regime must hold before it is accepted.
        flicker_window:
            Look-back window (bars) for flicker counting.
        flicker_threshold:
            Max allowed transitions in flicker_window before is_uncertain=True.
        return_feature_idx:
            Column index of the log-return feature in the scaled array.
            Used by _sort_regimes_by_return() to order states by mean return.
        """
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.auto_select = auto_select
        self.n_regimes_min = n_regimes_min
        self.n_regimes_max = n_regimes_max

        self._min_persistence = min_persistence_bars
        self._flicker_window = flicker_window
        self._flicker_threshold = flicker_threshold
        self._return_feature_idx = return_feature_idx

        self._model: Optional[GaussianHMM] = None
        self._is_fitted: bool = False

        # Built during _sort_regimes_by_return()
        self._sort_order: Optional[np.ndarray] = None   # sort_order[k] = raw state with k-th lowest mean return
        self._label_map: Dict[int, int] = {}             # raw_state → sorted position
        self._regime_names: List[str] = []
        self._regime_map: Dict[int, str] = {}            # sorted pos → name

        # Updated by detect_regime()
        self._last_state: Optional[RegimeState] = None
        self._prev_regime: Optional[int] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, features: Union[pd.DataFrame, np.ndarray]) -> "HMMEngine":
        """
        Fit the HMM on a scaled feature matrix.

        Parameters
        ----------
        features:
            Pre-scaled array or DataFrame of shape (n_bars, n_features).
            If auto_select=True, the optimal n_regimes is selected by BIC
            before fitting.

        Returns
        -------
        self
        """
        X = features.values if isinstance(features, pd.DataFrame) else np.asarray(features)

        if X.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {X.shape}")

        if self.auto_select:
            self.n_regimes = self.auto_select_n_regimes(
                X, self.n_regimes_min, self.n_regimes_max, criterion="bic"
            )
            logger.info("BIC-selected n_regimes = %d", self.n_regimes)

        _min_obs = self.n_regimes * 10
        if X.shape[0] < _min_obs:
            raise ValueError(
                f"Insufficient data: need at least {_min_obs} observations to fit "
                f"{self.n_regimes} regimes (got {X.shape[0]})."
            )

        model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        model.fit(X)
        self._model = model
        self._sort_regimes_by_return()
        self._is_fitted = True
        logger.info(
            "HMMEngine fitted: n_regimes=%d, converged=%s",
            self.n_regimes,
            getattr(model, "monitor_", None) and model.monitor_.converged,
        )
        return self

    def auto_select_n_regimes(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        n_min: int = 3,
        n_max: int = 7,
        criterion: str = "bic",
    ) -> int:
        """
        Grid-search n_regimes in [n_min, n_max] and return the best value.

        BIC = -2 * log_likelihood + k * log(n)

        where k = number of free model parameters and n = number of samples.
        Lower BIC is better (balances fit quality vs. model complexity).

        Parameters
        ----------
        features:
            Scaled feature matrix.
        n_min / n_max:
            Inclusive search range.
        criterion:
            'bic' or 'aic'.

        Returns
        -------
        Optimal number of regimes.
        """
        if criterion not in ("bic", "aic"):
            raise ValueError(f"Unknown criterion '{criterion}'. Use 'bic' or 'aic'.")

        X = features.values if isinstance(features, pd.DataFrame) else np.asarray(features)
        n_samples, n_features = X.shape
        best_score = np.inf
        best_n = n_min

        for n in range(n_min, n_max + 1):
            try:
                m = GaussianHMM(
                    n_components=n,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                )
                m.fit(X)
                log_lik = m.score(X) * n_samples  # score() returns per-sample avg

                k = self._count_free_params(n, n_features, self.covariance_type)
                score = (
                    -2 * log_lik + k * np.log(n_samples)
                    if criterion == "bic"
                    else -2 * log_lik + 2 * k
                )

                logger.debug("n_regimes=%d, %s=%.2f", n, criterion.upper(), score)

                if score < best_score:
                    best_score = score
                    best_n = n
            except Exception as exc:
                logger.warning("HMM fitting failed for n_regimes=%d: %s", n, exc)

        return best_n

    # ------------------------------------------------------------------
    # Live detection (forward algorithm — no look-ahead)
    # ------------------------------------------------------------------

    def detect_regime(
        self,
        features: Union[pd.DataFrame, np.ndarray],
    ) -> RegimeState:
        """
        Classify the current market regime using the forward algorithm.

        This method NEVER calls model.predict() (Viterbi).  The forward
        algorithm at step t uses only observations o_{1:t}, so no future
        information leaks into past regime labels.

        Parameters
        ----------
        features:
            Recent scaled feature history, shape (T, n_features).
            Pass enough history for the forward algorithm to stabilise
            (≥ 30 bars recommended).

        Returns
        -------
        RegimeState for the most recent bar.
        """
        self._require_fitted()
        X = features.values if isinstance(features, pd.DataFrame) else np.asarray(features)

        # Causal inference: P(q_t = i | o_{1:t}) for every t
        posteriors = self._forward_algorithm(X)  # (T, K)

        # Raw argmax at each bar
        raw_labels = np.argmax(posteriors, axis=1)

        # Map raw → sorted (bear…bull) labels
        sorted_labels = np.array([self._label_map[int(r)] for r in raw_labels])

        # Stability: count transitions on the RAW sorted labels (before smoothing).
        # The persistence filter gives a stable regime to act on, but the flicker
        # count on the underlying signal determines whether we flag UNCERTAIN.
        flicker_count = self._count_flickers(sorted_labels, self._flicker_window)
        is_uncertain = flicker_count > self._flicker_threshold

        # Causal persistence filter (only looks backward)
        filtered_labels = self._apply_persistence_filter(sorted_labels, self._min_persistence)

        # Current regime = last filtered label
        current_sorted = int(filtered_labels[-1])
        current_name = self._regime_names[current_sorted]
        # Raw state corresponding to the sorted label (for confidence lookup)
        current_raw = int(self._sort_order[current_sorted])
        current_confidence = float(posteriors[-1, current_raw])

        state = RegimeState(
            regime=current_sorted,
            regime_name=current_name,
            raw_regime=current_raw,
            confidence=current_confidence,
            is_uncertain=is_uncertain,
            posteriors=posteriors[-1].copy(),
            flicker_count=flicker_count,
        )

        if self._last_state is not None and self._last_state.regime != current_sorted:
            self.log_regime_change(self._last_state.regime, current_sorted)

        self._last_state = state
        return state

    def get_confidence(self) -> float:
        """
        Return the confidence of the most recently detected regime.

        Confidence = max(P(q_t = i | o_{1:t})) over all states i.
        Higher values indicate the model is more certain about the regime.

        Raises
        ------
        RuntimeError if detect_regime() has not been called yet.
        """
        if self._last_state is None:
            raise RuntimeError("No regime detected yet. Call detect_regime() first.")
        return float(np.max(self._last_state.posteriors))

    def log_regime_change(self, old_regime: int, new_regime: int, timestamp=None) -> None:
        """
        Log a regime transition at INFO level.

        Parameters
        ----------
        old_regime:
            Sorted label of the previous regime.
        new_regime:
            Sorted label of the new regime.
        timestamp:
            Optional datetime for the event; defaults to None (current time).
        """
        old_name = self._regime_map.get(old_regime, f"regime_{old_regime}")
        new_name = self._regime_map.get(new_regime, f"regime_{new_regime}")
        ts_str = f" at {timestamp}" if timestamp is not None else ""
        logger.info(
            "REGIME CHANGE%s: %s (label=%d) → %s (label=%d)",
            ts_str,
            old_name,
            old_regime,
            new_name,
            new_regime,
        )

    # ------------------------------------------------------------------
    # Batch inference (for backtest / walk-forward evaluation)
    # ------------------------------------------------------------------

    def predict(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Return the sorted regime label for every bar using the forward algorithm.

        This is the causal equivalent of model.predict() (Viterbi) for
        batch evaluation.  Labels at time t do not depend on t+1, t+2, …

        Returns
        -------
        Integer array of shape (n_bars,) with sorted regime labels.
        """
        self._require_fitted()
        X = features.values if isinstance(features, pd.DataFrame) else np.asarray(features)
        posteriors = self._forward_algorithm(X)
        raw_labels = np.argmax(posteriors, axis=1)
        return np.array([self._label_map[int(r)] for r in raw_labels])

    def predict_proba(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Return the filtered posterior P(q_t = i | o_{1:t}) for each bar.

        Columns are sorted by mean return (col 0 = most bearish state).

        Returns
        -------
        Array of shape (n_bars, n_regimes).
        """
        self._require_fitted()
        X = features.values if isinstance(features, pd.DataFrame) else np.asarray(features)
        raw_posteriors = self._forward_algorithm(X)
        # Re-order columns to match sorted regime labels
        return raw_posteriors[:, self._sort_order]

    def current_regime(
        self, features: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[int, np.ndarray]:
        """
        Classify the most recent bar using detect_regime().

        Returns
        -------
        (sorted_regime_label, posterior_vector) for the last bar.
        """
        state = self.detect_regime(features)
        return state.regime, state.posteriors

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialise the fitted engine to disk using pickle."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("HMMEngine saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HMMEngine":
        """Deserialise a previously saved HMMEngine from disk."""
        with open(Path(path), "rb") as fh:
            engine = pickle.load(fh)
        if not isinstance(engine, cls):
            raise TypeError(f"Loaded object is {type(engine)}, expected HMMEngine.")
        logger.info("HMMEngine loaded from %s (n_regimes=%d)", path, engine.n_regimes)
        return engine

    # ------------------------------------------------------------------
    # Internal: forward algorithm
    # ------------------------------------------------------------------

    def _forward_algorithm(self, X: np.ndarray) -> np.ndarray:
        """
        Run the causal forward (filtering) pass.

        Computes α_t(i) = P(o_1, …, o_t, q_t = i | λ) in log-space,
        then normalises each row to obtain the filtering distribution
        P(q_t = i | o_{1:t}).

        Because each α_t depends only on α_{t-1} and o_t, adding future
        observations never changes posteriors at earlier time steps —
        guaranteeing zero look-ahead bias.

        Parameters
        ----------
        X:
            Scaled feature array, shape (T, n_features).

        Returns
        -------
        posteriors: ndarray, shape (T, n_regimes).
            posteriors[t, i] = P(q_t = i | o_{1:t})
        """
        T = X.shape[0]
        K = self.n_regimes

        log_startprob = np.log(self._model.startprob_ + _EPS)        # (K,)
        log_transmat = np.log(self._model.transmat_ + _EPS)           # (K, K)
        log_emission = self._compute_log_emission(X)                  # (T, K)

        log_alpha = np.empty((T, K))
        log_alpha[0] = log_startprob + log_emission[0]

        for t in range(1, T):
            # Vectorised: log_alpha[t, j] = logsumexp_i(log_alpha[t-1,i] + log A[i,j])
            #             + log_emission[t, j]
            # log_alpha[t-1][:, None] + log_transmat  →  (K, K), axis-0 = i
            log_alpha[t] = (
                logsumexp(log_alpha[t - 1, :, np.newaxis] + log_transmat, axis=0)
                + log_emission[t]
            )

        # Normalise row-wise: P(q_t | o_{1:t}) = α_t / Σ_i α_t(i)
        log_row_sums = logsumexp(log_alpha, axis=1, keepdims=True)
        posteriors = np.exp(log_alpha - log_row_sums)
        return posteriors

    def _compute_log_emission(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log P(o_t | q_t = i) for all t and i.

        Delegates to hmmlearn's internal _compute_log_likelihood which
        correctly handles all covariance types and numerical edge cases.

        Returns
        -------
        Array of shape (T, n_regimes).
        """
        return self._model._compute_log_likelihood(X)

    # ------------------------------------------------------------------
    # Internal: regime labelling
    # ------------------------------------------------------------------

    def _sort_regimes_by_return(self) -> None:
        """
        Build the raw → sorted label mapping so that sorted label 0 is the
        most bearish state and sorted label K-1 is the most bullish.

        Uses model.means_[:, return_feature_idx] (expected log return per
        state) as the ordering criterion.  Mutates:
            self._sort_order   — sort_order[k] = raw state with k-th lowest mean return
            self._label_map    — raw_state → sorted position
            self._regime_names — ordered name list
            self._regime_map   — sorted position → name
        """
        mean_returns = self._model.means_[:, self._return_feature_idx]
        # Ascending sort: index 0 = lowest (most bearish)
        self._sort_order = np.argsort(mean_returns)

        self._label_map = {
            int(raw): int(pos) for pos, raw in enumerate(self._sort_order)
        }

        names = _REGIME_NAMES.get(self.n_regimes, _make_generic_names(self.n_regimes))
        self._regime_names = names
        self._regime_map = {pos: name for pos, name in enumerate(names)}

    # ------------------------------------------------------------------
    # Internal: stability filter
    # ------------------------------------------------------------------

    def _apply_persistence_filter(
        self,
        labels: np.ndarray,
        min_duration: int = 3,
    ) -> np.ndarray:
        """
        Suppress premature regime transitions.

        A regime change is only accepted once the new label has been the
        raw argmax for `min_duration` consecutive bars.  Until the
        persistence threshold is met, the previous (accepted) label is
        carried forward.  This filter is purely backward-looking (causal).

        Parameters
        ----------
        labels:
            Array of sorted regime labels, shape (T,).
        min_duration:
            Minimum run-length of a new label before it is accepted.

        Returns
        -------
        Filtered label array, same shape as labels.
        """
        T = len(labels)
        if T == 0:
            return labels.copy()

        filtered = labels.copy()

        for t in range(1, T):
            if t < min_duration - 1:
                # Not enough history to check persistence; keep raw label
                filtered[t] = labels[t]
                continue

            # Window of the last min_duration raw labels (inclusive of t)
            window_start = max(0, t - min_duration + 1)
            window = labels[window_start : t + 1]

            if np.all(window == window[0]):
                # Stable: accept the raw label
                filtered[t] = labels[t]
            else:
                # Not yet stable: hold the previously accepted regime
                filtered[t] = filtered[t - 1]

        return filtered

    def _count_flickers(self, labels: np.ndarray, window: int = 20) -> int:
        """
        Count regime transitions (consecutive label changes) in the last
        `window` bars of `labels`.

        A flicker count > flicker_threshold triggers is_uncertain=True.
        """
        recent = labels[-window:] if len(labels) >= window else labels
        if len(recent) < 2:
            return 0
        return int(np.sum(np.diff(recent) != 0))

    # ------------------------------------------------------------------
    # Internal: model selection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_free_params(n_states: int, n_features: int, cov_type: str) -> int:
        """
        Count free parameters in a GaussianHMM for BIC computation.

            initial distribution : n_states - 1
            transition matrix    : n_states * (n_states - 1)
            means                : n_states * n_features
            covariances          : depends on cov_type
        """
        init_params = n_states - 1
        trans_params = n_states * (n_states - 1)
        mean_params = n_states * n_features
        if cov_type == "full":
            cov_params = n_states * n_features * (n_features + 1) // 2
        elif cov_type == "diag":
            cov_params = n_states * n_features
        elif cov_type == "spherical":
            cov_params = n_states
        elif cov_type == "tied":
            cov_params = n_features * (n_features + 1) // 2
        else:
            cov_params = n_states * n_features * (n_features + 1) // 2
        return init_params + trans_params + mean_params + cov_params

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        """Raise RuntimeError if the model has not been fitted yet."""
        if not self._is_fitted:
            raise RuntimeError(
                "HMMEngine is not fitted. Call fit() before using inference methods."
            )

    @property
    def is_fitted(self) -> bool:
        """True after fit() has been called successfully."""
        return self._is_fitted

    @property
    def regime_names(self) -> List[str]:
        """Ordered list of regime names (index 0 = most bearish)."""
        return list(self._regime_names)

    def __repr__(self) -> str:
        return (
            f"HMMEngine(n_regimes={self.n_regimes}, "
            f"covariance_type='{self.covariance_type}', "
            f"fitted={self._is_fitted})"
        )
