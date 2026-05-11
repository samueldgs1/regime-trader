"""
Regime-conditional allocation strategies.

Primary API
-----------
StrategyOrchestrator — takes regime state fields and returns a SignalData
    object describing how much capital to deploy and at what leverage.

SignalData — the output contract: regime, allocation_pct, leverage,
    confidence, is_uncertain, notes, rebalance_required.

Regime defaults (customisable at construction time)
---------------------------------------------------
    crash     →  0 % invested, 0× leverage   — cash only
    deep_bear → 10 % invested, 1× leverage   — minimal exposure
    bear      → 20 % invested, 1× leverage   — defensive only
    neutral   → 50 % invested, 1× leverage   — balanced
    bull      → 95 % invested, 1.25× leverage — fully invested
    euphoria  → 70 % invested, 1× leverage   — take profits, tighten stops
    extreme_bull → 80 % invested, 1.25× leverage — high conviction

Uncertainty modifier
--------------------
If the HMM stability filter marks is_uncertain=True, allocation_pct is
multiplied by 0.50 and leverage is capped at 1.0.

Rebalancing gate
----------------
A trade is only warranted when the absolute difference between the
target allocation percentage and the current invested percentage exceeds
the rebalance_threshold (default 5 pp).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_REBALANCE_THRESHOLD: float = 0.05        # 5 percentage points
_UNCERTAINTY_ALLOCATION_SCALAR: float = 0.50       # halve allocation when uncertain
_UNCERTAINTY_MAX_LEVERAGE: float = 1.0             # no borrowing when uncertain


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeAllocationConfig:
    """Immutable per-regime allocation parameters."""
    allocation_pct: float   # fraction of NAV to invest (0.0–1.0)
    leverage: float         # portfolio leverage cap (1.0 = no leverage)
    notes: str              # human-readable strategy description


@dataclass
class SignalData:
    """
    Output of StrategyOrchestrator.get_signal().

    Fields
    ------
    regime:
        Sorted HMM label (0 = most bearish, K-1 = most bullish).
    regime_name:
        Human-readable regime name (e.g. 'bull', 'bear').
    allocation_pct:
        Fraction of NAV to deploy into equities after all modifiers.
        Remaining (1 − allocation_pct) stays in cash.
    leverage:
        Portfolio-level leverage cap after all modifiers.
    confidence:
        Posterior probability of the current regime from the forward algorithm.
    is_uncertain:
        True when the HMM stability filter detected excessive flickering.
    notes:
        Plain-text explanation of the signal and any applied modifiers.
    rebalance_required:
        True when the portfolio must be traded to reach the target.
        Populated by should_rebalance(); False by default at construction.
    """
    regime: int
    regime_name: str
    allocation_pct: float
    leverage: float
    confidence: float
    is_uncertain: bool
    notes: str
    rebalance_required: bool = False


@dataclass
class AllocationTarget:
    """Per-ticker weight map produced by compute_target_weights()."""
    regime_label: int
    regime_name: str
    weights: Dict[str, float]    # ticker → fraction of NAV
    leverage: float
    rebalance_threshold: float
    allocation_pct: float        # convenience: sum(weights.values())


# ---------------------------------------------------------------------------
# Default allocation table — covers all regime names from the HMM engine
# ---------------------------------------------------------------------------

_DEFAULT_REGIME_ALLOCATIONS: Dict[str, RegimeAllocationConfig] = {
    "crash": RegimeAllocationConfig(
        allocation_pct=0.00,
        leverage=0.00,
        notes="Crash regime — cash only, no equity exposure",
    ),
    "deep_bear": RegimeAllocationConfig(
        allocation_pct=0.10,
        leverage=1.00,
        notes="Deep bear — minimal exposure, defensive only",
    ),
    "bear": RegimeAllocationConfig(
        allocation_pct=0.20,
        leverage=1.00,
        notes="Bear regime — defensive allocation, reduced exposure",
    ),
    "neutral": RegimeAllocationConfig(
        allocation_pct=0.50,
        leverage=1.00,
        notes="Neutral regime — balanced equity/cash mix",
    ),
    "bull": RegimeAllocationConfig(
        allocation_pct=0.95,
        leverage=1.25,
        notes="Bull regime — fully invested with modest leverage",
    ),
    "euphoria": RegimeAllocationConfig(
        allocation_pct=0.70,
        leverage=1.00,
        notes="Euphoria regime — taking profits, tightening stops",
    ),
    "extreme_bull": RegimeAllocationConfig(
        allocation_pct=0.80,
        leverage=1.25,
        notes="Extreme bull — high conviction but volatility-aware",
    ),
}


# ---------------------------------------------------------------------------
# Primary class
# ---------------------------------------------------------------------------

class StrategyOrchestrator:
    """
    Translates a regime assessment into a concrete allocation signal.

    Designed to sit between the HMM engine and the order executor:

        HMMEngine.detect_regime()
            → RegimeState
            → StrategyOrchestrator.get_signal()
            → SignalData
            → OrderExecutor

    The orchestrator is stateless with respect to market data — it only
    reads the regime label and confidence values it receives.
    """

    def __init__(
        self,
        tickers: List[str],
        rebalance_threshold: float = _DEFAULT_REBALANCE_THRESHOLD,
        uncertainty_allocation_scalar: float = _UNCERTAINTY_ALLOCATION_SCALAR,
        custom_allocations: Optional[Dict[str, Dict]] = None,
    ) -> None:
        """
        Parameters
        ----------
        tickers:
            Universe of tradeable tickers.  Used by compute_target_weights()
            to spread the allocation_pct equally across the equity sleeve.
        rebalance_threshold:
            Minimum absolute difference in allocation percentage (e.g. 0.05
            = 5 pp) that triggers a rebalance recommendation.
        uncertainty_allocation_scalar:
            Multiplier applied to allocation_pct when is_uncertain=True.
            Default 0.50 halves the target exposure.
        custom_allocations:
            Optional dict of regime_name → dict with keys 'allocation_pct',
            'leverage', 'notes' that override the module defaults.  Partial
            overrides are supported — only the provided regimes are replaced.
        """
        if not tickers:
            raise ValueError("tickers must be a non-empty list.")
        if not 0.0 < rebalance_threshold < 1.0:
            raise ValueError(
                f"rebalance_threshold must be in (0, 1), got {rebalance_threshold}."
            )
        if not 0.0 < uncertainty_allocation_scalar <= 1.0:
            raise ValueError(
                "uncertainty_allocation_scalar must be in (0, 1], "
                f"got {uncertainty_allocation_scalar}."
            )

        self.tickers = list(tickers)
        self.rebalance_threshold = rebalance_threshold
        self.uncertainty_allocation_scalar = uncertainty_allocation_scalar

        # Merge module defaults with any caller overrides
        self._allocations: Dict[str, RegimeAllocationConfig] = {
            **_DEFAULT_REGIME_ALLOCATIONS
        }
        if custom_allocations:
            for name, cfg in custom_allocations.items():
                self._allocations[name] = RegimeAllocationConfig(
                    allocation_pct=float(cfg["allocation_pct"]),
                    leverage=float(cfg["leverage"]),
                    notes=str(cfg.get("notes", f"Custom config for {name}")),
                )

    # ------------------------------------------------------------------
    # Primary signal generation
    # ------------------------------------------------------------------

    def get_signal(
        self,
        regime: int,
        regime_name: str,
        confidence: float,
        is_uncertain: bool = False,
    ) -> SignalData:
        """
        Translate regime state fields into a SignalData allocation signal.

        Parameters
        ----------
        regime:
            Sorted HMM label (0 = most bearish).
        regime_name:
            Human-readable name from the HMM engine (e.g. 'bull', 'bear').
        confidence:
            Posterior probability of this regime from the forward algorithm.
        is_uncertain:
            True when the stability filter detected excessive regime flickering.

        Returns
        -------
        SignalData with allocation_pct, leverage, and notes populated.

        Raises
        ------
        KeyError if regime_name is not in the allocation table (neither
        default nor custom).
        """
        config = self._get_regime_config(regime_name)
        alloc_pct, leverage, notes = (
            config.allocation_pct,
            config.leverage,
            config.notes,
        )

        if is_uncertain:
            alloc_pct, leverage, notes = self._apply_uncertainty_modifier(
                alloc_pct, leverage, notes
            )

        signal = SignalData(
            regime=regime,
            regime_name=regime_name,
            allocation_pct=round(alloc_pct, 10),  # avoid float drift
            leverage=leverage,
            confidence=confidence,
            is_uncertain=is_uncertain,
            notes=notes,
            rebalance_required=False,  # populated by should_rebalance
        )

        logger.debug(
            "Signal: regime=%s alloc=%.1f%% leverage=%.2f× confident=%.1f%% uncertain=%s",
            regime_name,
            alloc_pct * 100,
            leverage,
            confidence * 100,
            is_uncertain,
        )
        return signal

    # ------------------------------------------------------------------
    # Rebalancing gate
    # ------------------------------------------------------------------

    def should_rebalance(
        self,
        target_alloc_pct: float,
        current_alloc_pct: float,
    ) -> bool:
        """
        Return True if the portfolio allocation has drifted past the threshold.

        Parameters
        ----------
        target_alloc_pct:
            Desired total invested fraction of NAV (from SignalData).
        current_alloc_pct:
            Actual current invested fraction of NAV.

        Returns
        -------
        True when |target − current| > rebalance_threshold.
        """
        return abs(target_alloc_pct - current_alloc_pct) > self.rebalance_threshold

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def compute_target_weights(
        self,
        signal: SignalData,
    ) -> Dict[str, float]:
        """
        Distribute signal.allocation_pct equally across self.tickers.

        The result is a dict of ticker → fraction-of-NAV weights that
        sums to signal.allocation_pct.  Remaining (1 − allocation_pct) is
        implicitly held as cash.

        Returns
        -------
        Dict[ticker, weight] where each weight is allocation_pct / n_tickers.
        """
        if signal.allocation_pct <= 0.0 or not self.tickers:
            return {t: 0.0 for t in self.tickers}

        per_ticker = signal.allocation_pct / len(self.tickers)
        return {t: per_ticker for t in self.tickers}

    # ------------------------------------------------------------------
    # Trade computation
    # ------------------------------------------------------------------

    def compute_rebalance_trades(
        self,
        signal: SignalData,
        current_positions: Dict[str, float],
        nav: float,
        prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute signed dollar amounts needed to reach the target weights.

        Parameters
        ----------
        signal:
            Current SignalData (output of get_signal()).
        current_positions:
            ticker → current market value (signed, $ ).
        nav:
            Current net asset value.
        prices:
            ticker → last price (reserved for future lot-size rounding).

        Returns
        -------
        ticker → signed dollar amount to trade (+ = buy, − = sell).
        Only tickers with a non-zero delta are included.
        """
        if nav <= 0:
            raise ValueError(f"NAV must be positive, got {nav}.")

        target_weights = self.compute_target_weights(signal)
        trades: Dict[str, float] = {}

        # Tickers in target universe
        for ticker, weight in target_weights.items():
            target_value = weight * nav
            current_value = current_positions.get(ticker, 0.0)
            delta = target_value - current_value
            if delta != 0.0:
                trades[ticker] = delta

        # Tickers held but NOT in the target universe → full liquidation
        for ticker, value in current_positions.items():
            if ticker not in target_weights and value != 0.0:
                trades[ticker] = -value

        return trades

    # ------------------------------------------------------------------
    # AllocationTarget helper (used by the rest of the system)
    # ------------------------------------------------------------------

    def build_allocation_target(
        self,
        signal: SignalData,
        rebalance_threshold: Optional[float] = None,
    ) -> AllocationTarget:
        """
        Convert a SignalData into the AllocationTarget format used
        by other modules (e.g. RiskManager, PositionTracker).
        """
        weights = self.compute_target_weights(signal)
        return AllocationTarget(
            regime_label=signal.regime,
            regime_name=signal.regime_name,
            weights=weights,
            leverage=signal.leverage,
            rebalance_threshold=rebalance_threshold or self.rebalance_threshold,
            allocation_pct=signal.allocation_pct,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_regime_config(self, regime_name: str) -> RegimeAllocationConfig:
        """
        Retrieve the allocation config for a regime name.

        Raises
        ------
        KeyError with an informative message if the name is unknown.
        """
        try:
            return self._allocations[regime_name]
        except KeyError:
            known = sorted(self._allocations.keys())
            raise KeyError(
                f"Unknown regime name '{regime_name}'. "
                f"Registered names: {known}. "
                "Pass custom_allocations to StrategyOrchestrator to add new regimes."
            ) from None

    def _apply_uncertainty_modifier(
        self,
        allocation_pct: float,
        leverage: float,
        notes: str,
    ) -> Tuple[float, float, str]:
        """
        Halve the allocation and cap leverage at 1× when uncertain.

        Returns
        -------
        (modified_allocation_pct, modified_leverage, updated_notes)
        """
        modified_alloc = allocation_pct * self.uncertainty_allocation_scalar
        modified_leverage = min(leverage, _UNCERTAINTY_MAX_LEVERAGE)
        uncertainty_note = (
            f" [UNCERTAIN: allocation reduced {self.uncertainty_allocation_scalar:.0%}"
            f" → {modified_alloc:.1%}, leverage capped at {modified_leverage:.2f}×]"
        )
        return modified_alloc, modified_leverage, notes + uncertainty_note

    @property
    def known_regimes(self) -> List[str]:
        """Sorted list of all regime names in the allocation table."""
        return sorted(self._allocations.keys())


# ---------------------------------------------------------------------------
# Legacy class — backward compatible with existing test stubs and main.py
# ---------------------------------------------------------------------------

class RegimeStrategyRouter:
    """
    Thin compatibility wrapper kept for the rest of the system.

    New code should use StrategyOrchestrator directly.
    """

    def __init__(
        self,
        regime_configs: Dict,
        tickers: List[str],
        rebalance_threshold: float = _DEFAULT_REBALANCE_THRESHOLD,
    ) -> None:
        # Build custom_allocations from the settings.REGIME_STRATEGIES format
        custom: Dict[str, Dict] = {}
        for _label, cfg in regime_configs.items():
            name = cfg.get("label", f"regime_{_label}")
            custom[name] = {
                "allocation_pct": cfg.get("equity_allocation", 0.5),
                "leverage": cfg.get("leverage_max", 1.0),
                "notes": f"From settings: {name}",
            }
        self._orchestrator = StrategyOrchestrator(
            tickers=tickers,
            rebalance_threshold=rebalance_threshold,
            custom_allocations=custom if custom else None,
        )
        self.tickers = tickers
        self.regime_configs = regime_configs

    def get_target_allocation(
        self,
        regime: int,
        prices=None,
    ) -> AllocationTarget:
        cfg = self.regime_configs.get(regime)
        if cfg is None:
            raise KeyError(f"Regime label {regime} not in regime_configs.")
        name = cfg.get("label", f"regime_{regime}")
        signal = self._orchestrator.get_signal(
            regime=regime,
            regime_name=name,
            confidence=1.0,
            is_uncertain=False,
        )
        return self._orchestrator.build_allocation_target(signal)

    def compute_rebalance_trades(
        self,
        target: AllocationTarget,
        current_positions: Dict[str, float],
        nav: float,
        prices: Dict[str, float],
    ) -> Dict[str, float]:
        signal = SignalData(
            regime=target.regime_label,
            regime_name=target.regime_name,
            allocation_pct=target.allocation_pct,
            leverage=target.leverage,
            confidence=1.0,
            is_uncertain=False,
            notes="",
        )
        return self._orchestrator.compute_rebalance_trades(
            signal, current_positions, nav, prices
        )

    def should_rebalance(
        self,
        target: AllocationTarget,
        current_weights: Dict[str, float],
    ) -> bool:
        current_alloc = sum(current_weights.values())
        return self._orchestrator.should_rebalance(
            target.allocation_pct, current_alloc
        )

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total == 0:
            return {k: 0.0 for k in weights}
        return {k: v / total for k, v in weights.items()}

    def _validate_leverage(self, weights: Dict[str, float], cap: float) -> None:
        gross = sum(abs(v) for v in weights.values())
        if gross > cap + 1e-9:
            raise ValueError(
                f"Gross exposure {gross:.4f} exceeds leverage cap {cap:.4f}."
            )
