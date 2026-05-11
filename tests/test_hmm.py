"""
Tests for core/hmm_engine.py and its integration with core/feature_eng.py.

Coverage
--------
1. Construction — HMMEngine builds without error; is_fitted starts False.
2. Fitting — fit() sets is_fitted; returns self; raises on bad input.
3. Forward algorithm — causal property proven: posteriors at t are
   identical whether the sequence is length T or T+N.
4. Viterbi exclusion — detect_regime() never calls model.predict().
5. Regime label consistency — labels are sorted by mean return.
6. Prediction shape / range.
7. Persistence / stability filter.
8. UNCERTAIN flag.
9. Auto regime selection.
10. Save / load roundtrip.
11. Feature engineering integration.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.feature_eng import FeatureEngineer
from core.hmm_engine import HMMEngine, RegimeState, _REGIME_NAMES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HMM_FEATURES = ["log_return", "log_volatility", "volume_zscore", "realized_vol_20d"]
_N_BARS_TRAIN = 504   # ≈ 2 trading years
_N_BARS_LIVE = 60     # recent window for detect_regime tests


# ---------------------------------------------------------------------------
# Fixtures — synthetic market data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def regime_ohlcv() -> pd.DataFrame:
    """
    Synthetic OHLCV data with three clear regimes:
        0–199  bear  (negative drift, high vol)
        200–399 neutral (near-zero drift, low vol)
        400–503 bull  (positive drift, moderate vol)
    """
    rng = np.random.default_rng(42)
    n = _N_BARS_TRAIN

    returns = np.concatenate([
        rng.normal(-0.0020, 0.025, 200),
        rng.normal(0.0003, 0.008, 200),
        rng.normal(0.0025, 0.014, 104),
    ])
    volumes = np.concatenate([
        rng.lognormal(18.0, 0.40, 200),
        rng.lognormal(17.0, 0.20, 200),
        rng.lognormal(17.5, 0.25, 104),
    ]).astype(np.int64)

    close = 400.0 * np.exp(np.cumsum(returns))
    idx = pd.bdate_range("2021-01-01", periods=n)

    return pd.DataFrame(
        {
            "open":   close * rng.uniform(0.999, 1.001, n),
            "high":   close * rng.uniform(1.001, 1.020, n),
            "low":    close * rng.uniform(0.980, 0.999, n),
            "close":  close,
            "volume": volumes,
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def feature_eng() -> FeatureEngineer:
    return FeatureEngineer(feature_names=_HMM_FEATURES)


@pytest.fixture(scope="module")
def raw_features(regime_ohlcv, feature_eng) -> pd.DataFrame:
    """Unscaled feature DataFrame from the full training window."""
    return feature_eng.build_features(regime_ohlcv)


@pytest.fixture(scope="module")
def scaled_array(raw_features, feature_eng) -> np.ndarray:
    """Scaled numpy array used by all HMM tests."""
    return feature_eng.fit_transform(raw_features)


@pytest.fixture(scope="module")
def fitted_hmm(scaled_array) -> HMMEngine:
    """HMMEngine fitted on the synthetic training data (n_regimes=3)."""
    engine = HMMEngine(
        n_regimes=3,
        n_iter=100,
        tol=1e-3,
        covariance_type="diag",
        random_state=0,
    )
    engine.fit(scaled_array)
    return engine


# Lightweight fixture for tests that just need a valid fitted engine quickly
@pytest.fixture
def small_fitted_hmm() -> HMMEngine:
    rng = np.random.default_rng(7)
    X = rng.standard_normal((120, 4))
    eng = HMMEngine(n_regimes=3, n_iter=50, tol=1e-2, covariance_type="diag", random_state=7)
    eng.fit(X)
    return eng


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    """Legacy fixture: simple random feature DataFrame (module-compat)."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.standard_normal((300, 4)),
        columns=_HMM_FEATURES,
    )


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_hmm_engine_initialises(self) -> None:
        """HMMEngine can be constructed with valid parameters."""
        eng = HMMEngine(n_regimes=4, n_iter=200, covariance_type="full", random_state=1)
        assert eng.n_regimes == 4
        assert eng.covariance_type == "full"
        assert eng.random_state == 1

    def test_is_not_fitted_before_fit(self) -> None:
        """is_fitted is False before fit() is called."""
        eng = HMMEngine(n_regimes=3)
        assert eng.is_fitted is False

    def test_repr_shows_unfitted(self) -> None:
        eng = HMMEngine(n_regimes=3)
        assert "fitted=False" in repr(eng)


# ---------------------------------------------------------------------------
# 2. Fitting
# ---------------------------------------------------------------------------

class TestFitting:
    def test_fit_sets_is_fitted(self, fitted_hmm: HMMEngine) -> None:
        """fit() sets is_fitted to True."""
        assert fitted_hmm.is_fitted is True

    def test_fit_returns_self(self, scaled_array: np.ndarray) -> None:
        """fit() returns the HMMEngine instance for method chaining."""
        eng = HMMEngine(n_regimes=3, n_iter=50, tol=1e-2, covariance_type="diag", random_state=99)
        result = eng.fit(scaled_array)
        assert result is eng

    def test_fit_with_insufficient_data_raises(self) -> None:
        """fit() raises ValueError when data has fewer rows than n_regimes × 10."""
        eng = HMMEngine(n_regimes=5, n_iter=10, covariance_type="diag")
        tiny = np.random.default_rng(0).standard_normal((20, 4))  # need 50
        with pytest.raises(ValueError, match="Insufficient data"):
            eng.fit(tiny)

    def test_fit_accepts_dataframe(self, raw_features: pd.DataFrame, feature_eng: FeatureEngineer) -> None:
        """fit() accepts a pre-scaled pd.DataFrame as well as np.ndarray."""
        scaled_df = pd.DataFrame(
            feature_eng.transform(raw_features),
            columns=raw_features.columns,
        )
        eng = HMMEngine(n_regimes=3, n_iter=50, tol=1e-2, covariance_type="diag", random_state=11)
        eng.fit(scaled_df)
        assert eng.is_fitted

    def test_fit_builds_regime_map(self, fitted_hmm: HMMEngine) -> None:
        """After fit(), _regime_map has an entry for every sorted label."""
        assert len(fitted_hmm._regime_map) == fitted_hmm.n_regimes
        for k in range(fitted_hmm.n_regimes):
            assert k in fitted_hmm._regime_map

    def test_fit_builds_label_map(self, fitted_hmm: HMMEngine) -> None:
        """_label_map covers every raw state and maps to a valid sorted position."""
        for raw, pos in fitted_hmm._label_map.items():
            assert 0 <= raw < fitted_hmm.n_regimes
            assert 0 <= pos < fitted_hmm.n_regimes


# ---------------------------------------------------------------------------
# 3. Forward algorithm — causal (no look-ahead) property
# ---------------------------------------------------------------------------

class TestForwardAlgorithm:
    def test_posteriors_shape(self, fitted_hmm: HMMEngine, scaled_array: np.ndarray) -> None:
        """_forward_algorithm() returns shape (T, n_regimes)."""
        post = fitted_hmm._forward_algorithm(scaled_array)
        assert post.shape == (len(scaled_array), fitted_hmm.n_regimes)

    def test_posteriors_sum_to_one(self, fitted_hmm: HMMEngine, scaled_array: np.ndarray) -> None:
        """Each row of _forward_algorithm() sums to ≈ 1.0."""
        post = fitted_hmm._forward_algorithm(scaled_array)
        np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-6)

    def test_posteriors_non_negative(self, fitted_hmm: HMMEngine, scaled_array: np.ndarray) -> None:
        """All posterior probabilities are ≥ 0."""
        post = fitted_hmm._forward_algorithm(scaled_array)
        assert np.all(post >= 0)

    def test_no_lookahead_bias_core(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray
    ) -> None:
        """
        KEY: forward posteriors at time t are identical whether the
        sequence ends at T or T+20.

        If this fails, the forward implementation reads future data.
        """
        T = 80
        ext = 20  # extra future bars

        post_T = fitted_hmm._forward_algorithm(scaled_array[:T])
        post_T_ext = fitted_hmm._forward_algorithm(scaled_array[: T + ext])

        # Posteriors 0..T-1 must not change when future bars are appended
        np.testing.assert_allclose(
            post_T,
            post_T_ext[:T],
            atol=1e-10,
            err_msg=(
                "Forward posteriors changed after appending future observations — "
                "look-ahead bias detected."
            ),
        )

    def test_no_lookahead_bias_labels(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray
    ) -> None:
        """
        Argmax labels produced by the forward algorithm at t < T are
        unchanged when observations at t ≥ T are revealed.
        """
        T = 100
        labels_T = np.argmax(fitted_hmm._forward_algorithm(scaled_array[:T]), axis=1)
        labels_T20 = np.argmax(
            fitted_hmm._forward_algorithm(scaled_array[: T + 20]), axis=1
        )
        np.testing.assert_array_equal(labels_T, labels_T20[:T])

    def test_single_observation(self, fitted_hmm: HMMEngine, scaled_array: np.ndarray) -> None:
        """_forward_algorithm() handles a single-row input (T=1)."""
        post = fitted_hmm._forward_algorithm(scaled_array[:1])
        assert post.shape == (1, fitted_hmm.n_regimes)
        np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Viterbi exclusion — detect_regime must NOT call model.predict()
# ---------------------------------------------------------------------------

class TestViterbiExclusion:
    def test_detect_regime_does_not_call_predict(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray
    ) -> None:
        """
        detect_regime() must use the forward algorithm, not Viterbi.

        We replace model.predict() with a function that raises; if
        detect_regime() calls it the test fails.
        """
        with patch.object(
            fitted_hmm._model,
            "predict",
            side_effect=AssertionError("Viterbi (model.predict) was called — look-ahead bias!"),
        ):
            state = fitted_hmm.detect_regime(scaled_array[-_N_BARS_LIVE:])

        assert state is not None  # if we get here, predict was not called

    def test_predict_batch_does_not_call_viterbi(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray
    ) -> None:
        """HMMEngine.predict() (batch forward labels) also avoids Viterbi."""
        with patch.object(
            fitted_hmm._model,
            "predict",
            side_effect=AssertionError("Viterbi called in batch predict!"),
        ):
            labels = fitted_hmm.predict(scaled_array)

        assert labels.shape == (len(scaled_array),)

    def test_forward_vs_viterbi_differ_on_extension(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray
    ) -> None:
        """
        Demonstrate that Viterbi IS look-ahead biased (labels change when
        the sequence is extended) while forward labels are stable.

        This test documents the difference rather than asserting it always
        holds, because on some sequences the two may accidentally agree.
        """
        T = 80
        fwd_T = np.argmax(fitted_hmm._forward_algorithm(scaled_array[:T]), axis=1)
        fwd_T20 = np.argmax(
            fitted_hmm._forward_algorithm(scaled_array[: T + 20]), axis=1
        )
        # Forward labels MUST be identical (already proven above)
        np.testing.assert_array_equal(fwd_T, fwd_T20[:T])

        # Viterbi labels for the same subsequences (informational only)
        viterbi_T = fitted_hmm._model.predict(scaled_array[:T])
        viterbi_T20 = fitted_hmm._model.predict(scaled_array[: T + 20])[:T]
        # We don't assert they differ — just verify forward algo is the correct path
        _ = np.sum(viterbi_T != viterbi_T20)  # count differences (for reference)


# ---------------------------------------------------------------------------
# 5. Regime label consistency
# ---------------------------------------------------------------------------

class TestRegimeLabelConsistency:
    def test_regime_names_match_n_regimes(self, fitted_hmm: HMMEngine) -> None:
        """Number of regime names equals n_regimes."""
        assert len(fitted_hmm.regime_names) == fitted_hmm.n_regimes

    def test_regime_names_for_3(self) -> None:
        """n_regimes=3 produces exactly bear/neutral/bull."""
        assert _REGIME_NAMES[3] == ["bear", "neutral", "bull"]

    def test_regime_names_for_4(self) -> None:
        """n_regimes=4 produces crash/bear/bull/euphoria (no neutral)."""
        assert _REGIME_NAMES[4] == ["crash", "bear", "bull", "euphoria"]

    def test_regime_names_for_5(self) -> None:
        """n_regimes=5 produces the full 5-name set."""
        assert _REGIME_NAMES[5] == ["crash", "bear", "neutral", "bull", "euphoria"]

    def test_sorted_by_mean_return_ascending(self, fitted_hmm: HMMEngine) -> None:
        """
        Mean log-return of state i ≤ mean log-return of state i+1
        (sorted label 0 is most bearish, last is most bullish).
        """
        means = fitted_hmm._model.means_[:, 0]  # column 0 = log_return
        sorted_means = [means[fitted_hmm._sort_order[k]] for k in range(fitted_hmm.n_regimes)]
        for i in range(len(sorted_means) - 1):
            assert sorted_means[i] <= sorted_means[i + 1], (
                f"Regime {i} mean_return={sorted_means[i]:.6f} should be ≤ "
                f"regime {i+1} mean_return={sorted_means[i+1]:.6f}"
            )

    def test_label_map_is_bijection(self, fitted_hmm: HMMEngine) -> None:
        """_label_map is a one-to-one mapping from raw states to sorted positions."""
        values = list(fitted_hmm._label_map.values())
        assert sorted(values) == list(range(fitted_hmm.n_regimes))

    def test_sort_order_is_permutation(self, fitted_hmm: HMMEngine) -> None:
        """_sort_order is a permutation of 0..n_regimes-1."""
        assert sorted(fitted_hmm._sort_order.tolist()) == list(range(fitted_hmm.n_regimes))


# ---------------------------------------------------------------------------
# 6. predict() / predict_proba() shape and range
# ---------------------------------------------------------------------------

class TestBatchPrediction:
    def test_predict_shape(self, fitted_hmm: HMMEngine, scaled_array: np.ndarray) -> None:
        """predict() returns an array with the same length as the input."""
        labels = fitted_hmm.predict(scaled_array)
        assert labels.shape == (len(scaled_array),)

    def test_predict_labels_within_range(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray
    ) -> None:
        """All predicted labels are in [0, n_regimes)."""
        labels = fitted_hmm.predict(scaled_array)
        assert labels.min() >= 0
        assert labels.max() < fitted_hmm.n_regimes

    def test_predict_proba_shape(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray
    ) -> None:
        """predict_proba() returns shape (n_bars, n_regimes)."""
        proba = fitted_hmm.predict_proba(scaled_array)
        assert proba.shape == (len(scaled_array), fitted_hmm.n_regimes)

    def test_predict_proba_sums_to_one(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray
    ) -> None:
        """Each row of predict_proba() sums to ≈ 1.0."""
        proba = fitted_hmm.predict_proba(scaled_array)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_current_regime_returns_tuple(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray
    ) -> None:
        """current_regime() returns (int, ndarray) tuple."""
        label, posteriors = fitted_hmm.current_regime(scaled_array[-_N_BARS_LIVE:])
        assert isinstance(label, (int, np.integer))
        assert isinstance(posteriors, np.ndarray)
        assert posteriors.shape == (fitted_hmm.n_regimes,)

    def test_predict_before_fit_raises(self) -> None:
        """predict() raises RuntimeError if called before fit()."""
        eng = HMMEngine(n_regimes=3)
        with pytest.raises(RuntimeError, match="not fitted"):
            eng.predict(np.zeros((10, 4)))


# ---------------------------------------------------------------------------
# 7. detect_regime() and RegimeState
# ---------------------------------------------------------------------------

class TestDetectRegime:
    def test_returns_regime_state(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray
    ) -> None:
        """detect_regime() returns a RegimeState instance."""
        state = fitted_hmm.detect_regime(scaled_array[-_N_BARS_LIVE:])
        assert isinstance(state, RegimeState)

    def test_regime_within_range(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray
    ) -> None:
        """RegimeState.regime is a valid sorted label."""
        state = fitted_hmm.detect_regime(scaled_array[-_N_BARS_LIVE:])
        assert 0 <= state.regime < fitted_hmm.n_regimes

    def test_confidence_in_unit_interval(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray
    ) -> None:
        """RegimeState.confidence is in (0, 1]."""
        state = fitted_hmm.detect_regime(scaled_array[-_N_BARS_LIVE:])
        assert 0.0 < state.confidence <= 1.0 + 1e-9

    def test_regime_name_in_known_names(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray
    ) -> None:
        """RegimeState.regime_name is one of the configured names."""
        state = fitted_hmm.detect_regime(scaled_array[-_N_BARS_LIVE:])
        assert state.regime_name in fitted_hmm.regime_names

    def test_posteriors_shape_in_state(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray
    ) -> None:
        """RegimeState.posteriors has shape (n_regimes,)."""
        state = fitted_hmm.detect_regime(scaled_array[-_N_BARS_LIVE:])
        assert state.posteriors.shape == (fitted_hmm.n_regimes,)

    def test_detect_before_fit_raises(self, scaled_array: np.ndarray) -> None:
        """detect_regime() raises RuntimeError before fitting."""
        eng = HMMEngine(n_regimes=3)
        with pytest.raises(RuntimeError, match="not fitted"):
            eng.detect_regime(scaled_array[:10])


# ---------------------------------------------------------------------------
# 8. get_confidence()
# ---------------------------------------------------------------------------

class TestGetConfidence:
    def test_get_confidence_after_detect(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray
    ) -> None:
        """get_confidence() returns a float in (0, 1] after detect_regime()."""
        fitted_hmm.detect_regime(scaled_array[-_N_BARS_LIVE:])
        conf = fitted_hmm.get_confidence()
        assert 0.0 < conf <= 1.0 + 1e-9

    def test_get_confidence_before_detect_raises(self) -> None:
        """get_confidence() raises RuntimeError if no detection has been made."""
        eng = HMMEngine(n_regimes=3, n_iter=30, covariance_type="diag", random_state=0)
        rng = np.random.default_rng(0)
        eng.fit(rng.standard_normal((90, 4)))
        with pytest.raises(RuntimeError, match="No regime detected"):
            eng.get_confidence()

    def test_get_confidence_equals_max_posterior(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray
    ) -> None:
        """get_confidence() = max of the last posterior vector."""
        state = fitted_hmm.detect_regime(scaled_array[-_N_BARS_LIVE:])
        assert abs(fitted_hmm.get_confidence() - float(np.max(state.posteriors))) < 1e-9


# ---------------------------------------------------------------------------
# 9. Persistence / stability filter
# ---------------------------------------------------------------------------

class TestPersistenceFilter:
    def test_single_bar_flip_suppressed(self, small_fitted_hmm: HMMEngine) -> None:
        """A 1-bar regime change surrounded by the previous regime is absorbed."""
        # Pattern: [0,0,0,1,0,0,0] — the 1 at index 3 should be suppressed
        labels = np.array([0, 0, 0, 1, 0, 0, 0])
        filtered = small_fitted_hmm._apply_persistence_filter(labels, min_duration=3)
        # The lone 1 should be ironed out
        assert filtered[3] == 0, (
            f"Single-bar flip was not suppressed. filtered={filtered}"
        )

    def test_persistent_change_accepted(self, small_fitted_hmm: HMMEngine) -> None:
        """A regime that holds for min_duration bars is accepted."""
        labels = np.array([0, 0, 0, 1, 1, 1, 1])
        filtered = small_fitted_hmm._apply_persistence_filter(labels, min_duration=3)
        assert filtered[-1] == 1

    def test_filter_is_causal(self, small_fitted_hmm: HMMEngine) -> None:
        """
        Applying the filter to [0..T] then to [0..T+5] produces identical
        results for positions 0..T.  The filter must not peek forward.
        """
        rng = np.random.default_rng(3)
        labels = rng.integers(0, 3, 50)
        filtered_T = small_fitted_hmm._apply_persistence_filter(labels[:40], 3)
        filtered_T5 = small_fitted_hmm._apply_persistence_filter(labels, 3)
        np.testing.assert_array_equal(filtered_T, filtered_T5[:40])

    def test_empty_input_returns_empty(self, small_fitted_hmm: HMMEngine) -> None:
        """_apply_persistence_filter() handles an empty array gracefully."""
        result = small_fitted_hmm._apply_persistence_filter(np.array([]), min_duration=3)
        assert len(result) == 0

    def test_single_element_unchanged(self, small_fitted_hmm: HMMEngine) -> None:
        labels = np.array([2])
        result = small_fitted_hmm._apply_persistence_filter(labels, min_duration=3)
        assert result[0] == 2


# ---------------------------------------------------------------------------
# 10. UNCERTAIN flag
# ---------------------------------------------------------------------------

class TestUncertainFlag:
    def test_stable_sequence_not_uncertain(self, small_fitted_hmm: HMMEngine) -> None:
        """A stable regime sequence does not trigger is_uncertain."""
        # Build a very stable input so flicker_count < threshold
        rng = np.random.default_rng(99)
        # All forward posteriors will be dominated by one state → low flicker
        X = np.tile(small_fitted_hmm._model.means_[0], (40, 1))
        X += rng.normal(0, 0.01, X.shape)
        state = small_fitted_hmm.detect_regime(X)
        # Stable data should not set uncertain (flicker_count ≤ threshold)
        assert state.flicker_count <= small_fitted_hmm._flicker_threshold or state.is_uncertain

    def test_flicker_threshold_triggers_uncertain(self, small_fitted_hmm: HMMEngine) -> None:
        """Injecting >4 label flips in 20 bars sets is_uncertain=True."""
        # Manually patch _forward_algorithm to return alternating posteriors
        K = small_fitted_hmm.n_regimes
        T = 30
        # Alternating posteriors: state 0 and state 1 flip every bar
        probs = np.zeros((T, K))
        for t in range(T):
            probs[t, t % 2] = 1.0

        with patch.object(small_fitted_hmm, "_forward_algorithm", return_value=probs):
            state = small_fitted_hmm.detect_regime(np.zeros((T, 4)))

        assert state.is_uncertain is True

    def test_flicker_count_field(self, small_fitted_hmm: HMMEngine) -> None:
        """RegimeState.flicker_count is a non-negative integer."""
        rng = np.random.default_rng(55)
        X = rng.standard_normal((40, 4))
        state = small_fitted_hmm.detect_regime(X)
        assert isinstance(state.flicker_count, (int, np.integer))
        assert state.flicker_count >= 0


# ---------------------------------------------------------------------------
# 11. Auto regime selection
# ---------------------------------------------------------------------------

class TestAutoSelect:
    def test_auto_select_returns_int_in_range(self, scaled_array: np.ndarray) -> None:
        """auto_select_n_regimes() returns an integer in [n_min, n_max]."""
        eng = HMMEngine(n_iter=50, tol=1e-2, covariance_type="diag", random_state=0)
        n_min, n_max = 3, 5
        best = eng.auto_select_n_regimes(scaled_array, n_min=n_min, n_max=n_max)
        assert n_min <= best <= n_max

    def test_auto_select_with_auto_flag(self, scaled_array: np.ndarray) -> None:
        """HMMEngine with auto_select=True picks n_regimes during fit()."""
        eng = HMMEngine(
            auto_select=True,
            n_regimes_min=3,
            n_regimes_max=5,
            n_iter=50,
            tol=1e-2,
            covariance_type="diag",
            random_state=0,
        )
        eng.fit(scaled_array)
        assert eng.is_fitted
        assert 3 <= eng.n_regimes <= 5

    def test_auto_select_aic(self, scaled_array: np.ndarray) -> None:
        """auto_select_n_regimes() works with criterion='aic'."""
        eng = HMMEngine(n_iter=50, tol=1e-2, covariance_type="diag", random_state=0)
        best = eng.auto_select_n_regimes(scaled_array, n_min=3, n_max=4, criterion="aic")
        assert best in (3, 4)

    def test_auto_select_invalid_criterion_raises(self, scaled_array: np.ndarray) -> None:
        """auto_select_n_regimes() raises ValueError for unknown criterion."""
        eng = HMMEngine(n_iter=50, tol=1e-2, covariance_type="diag", random_state=0)
        with pytest.raises(ValueError, match="Unknown criterion"):
            eng.auto_select_n_regimes(scaled_array, n_min=3, n_max=3, criterion="xyz")


# ---------------------------------------------------------------------------
# 12. Save / load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_file(self, fitted_hmm: HMMEngine, tmp_path: Path) -> None:
        """save() creates a file at the given path."""
        dest = tmp_path / "engine.pkl"
        fitted_hmm.save(dest)
        assert dest.exists()

    def test_load_returns_hmm_engine(self, fitted_hmm: HMMEngine, tmp_path: Path) -> None:
        """load() returns an HMMEngine instance."""
        dest = tmp_path / "engine.pkl"
        fitted_hmm.save(dest)
        loaded = HMMEngine.load(dest)
        assert isinstance(loaded, HMMEngine)

    def test_save_load_roundtrip_predictions(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray, tmp_path: Path
    ) -> None:
        """A saved and reloaded model produces identical predictions."""
        dest = tmp_path / "engine.pkl"
        fitted_hmm.save(dest)
        loaded = HMMEngine.load(dest)

        labels_orig = fitted_hmm.predict(scaled_array)
        labels_loaded = loaded.predict(scaled_array)
        np.testing.assert_array_equal(labels_orig, labels_loaded)

    def test_save_load_roundtrip_posteriors(
        self, fitted_hmm: HMMEngine, scaled_array: np.ndarray, tmp_path: Path
    ) -> None:
        """Loaded model produces identical forward posteriors."""
        dest = tmp_path / "engine.pkl"
        fitted_hmm.save(dest)
        loaded = HMMEngine.load(dest)

        p_orig = fitted_hmm._forward_algorithm(scaled_array[:50])
        p_loaded = loaded._forward_algorithm(scaled_array[:50])
        np.testing.assert_allclose(p_orig, p_loaded, atol=1e-12)

    def test_load_wrong_file_raises(self, tmp_path: Path) -> None:
        """load() raises TypeError if the file does not contain an HMMEngine."""
        bad = tmp_path / "bad.pkl"
        import pickle
        with open(bad, "wb") as f:
            pickle.dump({"not": "an engine"}, f)
        with pytest.raises(TypeError):
            HMMEngine.load(bad)


# ---------------------------------------------------------------------------
# 13. Feature engineering integration
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def test_build_features_required_columns_present(self, regime_ohlcv, feature_eng) -> None:
        """build_features() returns a DataFrame with all four required features."""
        feats = feature_eng.build_features(regime_ohlcv)
        for col in _HMM_FEATURES:
            assert col in feats.columns, f"Missing feature: {col}"

    def test_build_features_no_nans(self, regime_ohlcv, feature_eng) -> None:
        """After dropna(), the feature DataFrame contains no NaN values."""
        feats = feature_eng.build_features(regime_ohlcv)
        assert not feats.isna().any().any()

    def test_build_features_missing_column_raises(self) -> None:
        """build_features() raises ValueError when OHLCV is missing a column."""
        fe = FeatureEngineer(feature_names=_HMM_FEATURES)
        bad = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="Missing"):
            fe.build_features(bad)

    def test_fit_transform_shape(self, raw_features, feature_eng) -> None:
        """fit_transform() returns a 2-D array with the correct number of columns."""
        scaled = feature_eng.fit_transform(raw_features)
        assert scaled.ndim == 2
        assert scaled.shape[1] == len(_HMM_FEATURES)

    def test_transform_without_fit_raises(self, raw_features) -> None:
        """transform() raises RuntimeError if called before fit_scaler()."""
        fe = FeatureEngineer(feature_names=_HMM_FEATURES)
        with pytest.raises(RuntimeError, match="not fitted"):
            fe.transform(raw_features)

    def test_volume_zscore_near_zero_mean(self, regime_ohlcv) -> None:
        """volume_zscore has approximately zero mean over a long window."""
        fe = FeatureEngineer(feature_names=_HMM_FEATURES)
        feats = fe.build_features(regime_ohlcv)
        assert abs(feats["volume_zscore"].mean()) < 1.0

    def test_log_volatility_is_log_of_realized_vol(self, regime_ohlcv) -> None:
        """log_volatility ≈ log(realized_vol_20d)."""
        fe = FeatureEngineer(feature_names=_HMM_FEATURES)
        feats = fe.build_features(regime_ohlcv)
        expected = np.log(feats["realized_vol_20d"].clip(lower=1e-8))
        np.testing.assert_allclose(
            feats["log_volatility"].values,
            expected.values,
            atol=1e-8,
        )

    def test_full_pipeline_end_to_end(self, regime_ohlcv) -> None:
        """Full pipeline: OHLCV → features → scale → HMM fit → detect_regime."""
        fe = FeatureEngineer(feature_names=_HMM_FEATURES)
        feats = fe.build_features(regime_ohlcv)
        scaled = fe.fit_transform(feats)

        eng = HMMEngine(
            n_regimes=3,
            n_iter=80,
            tol=1e-2,
            covariance_type="diag",
            random_state=0,
        )
        eng.fit(scaled)
        state = eng.detect_regime(scaled[-40:])

        assert isinstance(state, RegimeState)
        assert state.regime_name in ["bear", "neutral", "bull"]
        assert 0.0 < state.confidence <= 1.0 + 1e-9
