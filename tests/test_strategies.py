"""
Tests for core/regime_strategies.py.

Coverage
--------
1.  Construction — valid params, bad params raise.
2.  get_signal() — one test per named regime, correct defaults.
3.  Uncertainty modifier — allocation halved, leverage capped at 1×.
4.  Unknown regime — KeyError with informative message.
5.  should_rebalance() — threshold logic (both sides of boundary).
6.  compute_target_weights() — equal split, zero when crash.
7.  compute_rebalance_trades() — flat→buy, overweight→sell, at-target→empty.
8.  Liquidation of unlisted tickers.
9.  build_allocation_target() round-trip.
10. Custom allocations override.
11. RegimeStrategyRouter backward-compatibility.
12. SignalData field validation.
"""

from __future__ import annotations

from typing import Dict

import pytest

from core.regime_strategies import (
    AllocationTarget,
    RegimeAllocationConfig,
    RegimeStrategyRouter,
    SignalData,
    StrategyOrchestrator,
    _DEFAULT_REGIME_ALLOCATIONS,
    _DEFAULT_REBALANCE_THRESHOLD,
    _UNCERTAINTY_ALLOCATION_SCALAR,
    _UNCERTAINTY_MAX_LEVERAGE,
)

# ---------------------------------------------------------------------------
# Constants used across tests
# ---------------------------------------------------------------------------

_TICKERS_1 = ["SPY"]
_TICKERS_3 = ["SPY", "QQQ", "TLT"]
_NAV = 100_000.0
_PRICE = {"SPY": 450.0, "QQQ": 380.0, "TLT": 95.0}

# All named regimes that must be addressable
_ALL_REGIMES = [
    "crash", "deep_bear", "bear", "neutral", "bull", "euphoria", "extreme_bull"
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def orc() -> StrategyOrchestrator:
    """Default single-ticker orchestrator."""
    return StrategyOrchestrator(tickers=_TICKERS_1)


@pytest.fixture
def orc3() -> StrategyOrchestrator:
    """Three-ticker orchestrator for weight-splitting tests."""
    return StrategyOrchestrator(tickers=_TICKERS_3)


@pytest.fixture
def flat_positions() -> Dict[str, float]:
    """All positions zeroed out."""
    return {t: 0.0 for t in _TICKERS_1}


@pytest.fixture
def flat_positions_3() -> Dict[str, float]:
    return {t: 0.0 for t in _TICKERS_3}


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_valid_construction(self) -> None:
        orc = StrategyOrchestrator(tickers=["SPY"], rebalance_threshold=0.05)
        assert orc.tickers == ["SPY"]
        assert orc.rebalance_threshold == 0.05

    def test_empty_tickers_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            StrategyOrchestrator(tickers=[])

    def test_bad_rebalance_threshold_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="rebalance_threshold"):
            StrategyOrchestrator(tickers=["SPY"], rebalance_threshold=0.0)

    def test_bad_rebalance_threshold_one_raises(self) -> None:
        with pytest.raises(ValueError, match="rebalance_threshold"):
            StrategyOrchestrator(tickers=["SPY"], rebalance_threshold=1.0)

    def test_bad_uncertainty_scalar_raises(self) -> None:
        with pytest.raises(ValueError, match="uncertainty_allocation_scalar"):
            StrategyOrchestrator(tickers=["SPY"], uncertainty_allocation_scalar=0.0)

    def test_known_regimes_contains_defaults(self, orc: StrategyOrchestrator) -> None:
        for name in _ALL_REGIMES:
            assert name in orc.known_regimes


# ---------------------------------------------------------------------------
# 2. get_signal() — per-regime correctness
# ---------------------------------------------------------------------------

class TestGetSignalPerRegime:
    """Each regime maps to the documented defaults."""

    def _sig(self, orc, name, uncertain=False):
        return orc.get_signal(regime=0, regime_name=name, confidence=0.9,
                              is_uncertain=uncertain)

    def test_crash_zero_allocation(self, orc: StrategyOrchestrator) -> None:
        sig = self._sig(orc, "crash")
        assert sig.allocation_pct == 0.0

    def test_crash_zero_leverage(self, orc: StrategyOrchestrator) -> None:
        sig = self._sig(orc, "crash")
        assert sig.leverage == 0.0

    def test_bear_allocation(self, orc: StrategyOrchestrator) -> None:
        sig = self._sig(orc, "bear")
        assert abs(sig.allocation_pct - 0.20) < 1e-9

    def test_bear_leverage_one(self, orc: StrategyOrchestrator) -> None:
        sig = self._sig(orc, "bear")
        assert sig.leverage == 1.0

    def test_neutral_allocation(self, orc: StrategyOrchestrator) -> None:
        sig = self._sig(orc, "neutral")
        assert abs(sig.allocation_pct - 0.50) < 1e-9

    def test_neutral_leverage_one(self, orc: StrategyOrchestrator) -> None:
        sig = self._sig(orc, "neutral")
        assert sig.leverage == 1.0

    def test_bull_allocation(self, orc: StrategyOrchestrator) -> None:
        sig = self._sig(orc, "bull")
        assert abs(sig.allocation_pct - 0.95) < 1e-9

    def test_bull_leverage_above_one(self, orc: StrategyOrchestrator) -> None:
        """Bull regime is the only default that uses leverage > 1."""
        sig = self._sig(orc, "bull")
        assert sig.leverage == 1.25

    def test_euphoria_allocation(self, orc: StrategyOrchestrator) -> None:
        sig = self._sig(orc, "euphoria")
        assert abs(sig.allocation_pct - 0.70) < 1e-9

    def test_euphoria_no_leverage(self, orc: StrategyOrchestrator) -> None:
        """Euphoria takes profits — no additional leverage."""
        sig = self._sig(orc, "euphoria")
        assert sig.leverage == 1.0

    def test_returns_signal_data_type(self, orc: StrategyOrchestrator) -> None:
        sig = self._sig(orc, "neutral")
        assert isinstance(sig, SignalData)

    def test_confidence_preserved(self, orc: StrategyOrchestrator) -> None:
        sig = orc.get_signal(regime=2, regime_name="neutral",
                             confidence=0.73, is_uncertain=False)
        assert abs(sig.confidence - 0.73) < 1e-9

    def test_regime_label_preserved(self, orc: StrategyOrchestrator) -> None:
        sig = orc.get_signal(regime=3, regime_name="bull",
                             confidence=0.85, is_uncertain=False)
        assert sig.regime == 3

    def test_regime_name_preserved(self, orc: StrategyOrchestrator) -> None:
        sig = self._sig(orc, "bear")
        assert sig.regime_name == "bear"

    def test_is_uncertain_false_by_default(self, orc: StrategyOrchestrator) -> None:
        sig = self._sig(orc, "bull")
        assert sig.is_uncertain is False

    def test_notes_non_empty(self, orc: StrategyOrchestrator) -> None:
        for name in _ALL_REGIMES:
            sig = self._sig(orc, name)
            assert len(sig.notes) > 0, f"Empty notes for regime '{name}'"

    def test_all_regimes_addressable(self, orc: StrategyOrchestrator) -> None:
        """Every named regime returns a SignalData without raising."""
        for name in _ALL_REGIMES:
            sig = self._sig(orc, name)
            assert isinstance(sig, SignalData)

    def test_allocation_pct_in_unit_interval(self, orc: StrategyOrchestrator) -> None:
        for name in _ALL_REGIMES:
            sig = self._sig(orc, name)
            assert 0.0 <= sig.allocation_pct <= 1.0, (
                f"allocation_pct={sig.allocation_pct} out of range for '{name}'"
            )

    def test_leverage_non_negative(self, orc: StrategyOrchestrator) -> None:
        for name in _ALL_REGIMES:
            sig = self._sig(orc, name)
            assert sig.leverage >= 0.0, (
                f"leverage={sig.leverage} < 0 for '{name}'"
            )


# ---------------------------------------------------------------------------
# 3. Uncertainty modifier
# ---------------------------------------------------------------------------

class TestUncertaintyModifier:
    def _sig_pair(self, orc, name):
        certain = orc.get_signal(regime=1, regime_name=name, confidence=0.6,
                                 is_uncertain=False)
        uncertain = orc.get_signal(regime=1, regime_name=name, confidence=0.6,
                                   is_uncertain=True)
        return certain, uncertain

    def test_allocation_halved_bull(self, orc: StrategyOrchestrator) -> None:
        certain, uncertain = self._sig_pair(orc, "bull")
        expected = certain.allocation_pct * _UNCERTAINTY_ALLOCATION_SCALAR
        assert abs(uncertain.allocation_pct - expected) < 1e-9

    def test_allocation_halved_neutral(self, orc: StrategyOrchestrator) -> None:
        certain, uncertain = self._sig_pair(orc, "neutral")
        expected = certain.allocation_pct * _UNCERTAINTY_ALLOCATION_SCALAR
        assert abs(uncertain.allocation_pct - expected) < 1e-9

    def test_allocation_halved_bear(self, orc: StrategyOrchestrator) -> None:
        certain, uncertain = self._sig_pair(orc, "bear")
        expected = certain.allocation_pct * _UNCERTAINTY_ALLOCATION_SCALAR
        assert abs(uncertain.allocation_pct - expected) < 1e-9

    def test_leverage_capped_at_one_for_bull(self, orc: StrategyOrchestrator) -> None:
        """Bull default leverage=1.25 must be capped at 1.0 when uncertain."""
        _, uncertain = self._sig_pair(orc, "bull")
        assert uncertain.leverage <= _UNCERTAINTY_MAX_LEVERAGE

    def test_leverage_unchanged_when_already_at_one(
        self, orc: StrategyOrchestrator
    ) -> None:
        """Bear leverage is already 1.0 — cap has no effect."""
        certain, uncertain = self._sig_pair(orc, "bear")
        assert uncertain.leverage == certain.leverage == 1.0

    def test_uncertainty_note_appended(self, orc: StrategyOrchestrator) -> None:
        _, uncertain = self._sig_pair(orc, "neutral")
        assert "UNCERTAIN" in uncertain.notes

    def test_is_uncertain_flag_true(self, orc: StrategyOrchestrator) -> None:
        _, uncertain = self._sig_pair(orc, "neutral")
        assert uncertain.is_uncertain is True

    def test_crash_uncertain_still_zero(self, orc: StrategyOrchestrator) -> None:
        """50% of 0% is still 0% — crash allocation stays zero."""
        _, uncertain = self._sig_pair(orc, "crash")
        assert uncertain.allocation_pct == 0.0

    def test_custom_scalar_applied(self) -> None:
        orc = StrategyOrchestrator(tickers=["SPY"],
                                   uncertainty_allocation_scalar=0.25)
        certain = orc.get_signal(regime=2, regime_name="bull",
                                 confidence=0.8, is_uncertain=False)
        uncertain = orc.get_signal(regime=2, regime_name="bull",
                                   confidence=0.8, is_uncertain=True)
        assert abs(uncertain.allocation_pct - certain.allocation_pct * 0.25) < 1e-9


# ---------------------------------------------------------------------------
# 4. Unknown regime
# ---------------------------------------------------------------------------

class TestUnknownRegime:
    def test_unknown_name_raises_key_error(self, orc: StrategyOrchestrator) -> None:
        with pytest.raises(KeyError):
            orc.get_signal(regime=0, regime_name="moon_shot", confidence=0.9)

    def test_error_message_mentions_name(self, orc: StrategyOrchestrator) -> None:
        with pytest.raises(KeyError, match="moon_shot"):
            orc.get_signal(regime=0, regime_name="moon_shot", confidence=0.9)

    def test_error_message_lists_known(self, orc: StrategyOrchestrator) -> None:
        with pytest.raises(KeyError, match="bull"):
            orc.get_signal(regime=0, regime_name="moon_shot", confidence=0.9)


# ---------------------------------------------------------------------------
# 5. should_rebalance()
# ---------------------------------------------------------------------------

class TestShouldRebalance:
    def test_exact_match_no_rebalance(self, orc: StrategyOrchestrator) -> None:
        assert orc.should_rebalance(0.50, 0.50) is False

    def test_within_threshold_no_rebalance(self, orc: StrategyOrchestrator) -> None:
        """Diff = 0.04 < 0.05 threshold → False."""
        assert orc.should_rebalance(0.50, 0.54) is False

    def test_just_below_threshold_no_rebalance(
        self, orc: StrategyOrchestrator
    ) -> None:
        """Diff = 0.049 pp < 0.05 threshold → no rebalance."""
        assert orc.should_rebalance(0.50, 0.549) is False

    def test_above_threshold_rebalance(self, orc: StrategyOrchestrator) -> None:
        """Diff = 0.06 > 0.05 → True."""
        assert orc.should_rebalance(0.50, 0.56) is True

    def test_below_target_triggers_rebalance(self, orc: StrategyOrchestrator) -> None:
        """Allocation fell below target by more than threshold."""
        assert orc.should_rebalance(0.95, 0.80) is True

    def test_crash_target_zero_current_invested(
        self, orc: StrategyOrchestrator
    ) -> None:
        """Going from 95 % invested to crash (0 %) triggers rebalance."""
        assert orc.should_rebalance(0.0, 0.95) is True

    def test_custom_threshold_respected(self) -> None:
        orc = StrategyOrchestrator(tickers=["SPY"], rebalance_threshold=0.10)
        assert orc.should_rebalance(0.50, 0.55) is False   # 5 pp < 10 pp
        assert orc.should_rebalance(0.50, 0.61) is True    # 11 pp > 10 pp


# ---------------------------------------------------------------------------
# 6. compute_target_weights()
# ---------------------------------------------------------------------------

class TestComputeTargetWeights:
    def _sig(self, orc, name):
        return orc.get_signal(regime=0, regime_name=name, confidence=0.9)

    def test_crash_all_zero_weights(self, orc: StrategyOrchestrator) -> None:
        sig = self._sig(orc, "crash")
        weights = orc.compute_target_weights(sig)
        assert all(v == 0.0 for v in weights.values())

    def test_single_ticker_weight_equals_alloc(
        self, orc: StrategyOrchestrator
    ) -> None:
        sig = self._sig(orc, "neutral")  # 50 %
        weights = orc.compute_target_weights(sig)
        assert abs(weights["SPY"] - 0.50) < 1e-9

    def test_three_tickers_equal_split(self, orc3: StrategyOrchestrator) -> None:
        sig = orc3.get_signal(regime=2, regime_name="neutral",
                              confidence=0.8)
        weights = orc3.compute_target_weights(sig)
        expected = 0.50 / 3
        for ticker in _TICKERS_3:
            assert abs(weights[ticker] - expected) < 1e-9

    def test_weights_sum_to_allocation_pct(
        self, orc3: StrategyOrchestrator
    ) -> None:
        for name in _ALL_REGIMES:
            sig = orc3.get_signal(regime=0, regime_name=name, confidence=0.9)
            weights = orc3.compute_target_weights(sig)
            assert abs(sum(weights.values()) - sig.allocation_pct) < 1e-9, (
                f"Weights for '{name}' sum to {sum(weights.values()):.4f}, "
                f"expected {sig.allocation_pct:.4f}"
            )

    def test_all_tickers_present_in_output(
        self, orc3: StrategyOrchestrator
    ) -> None:
        sig = orc3.get_signal(regime=1, regime_name="bull", confidence=0.9)
        weights = orc3.compute_target_weights(sig)
        for t in _TICKERS_3:
            assert t in weights


# ---------------------------------------------------------------------------
# 7. compute_rebalance_trades()
# ---------------------------------------------------------------------------

class TestComputeRebalanceTrades:
    def test_flat_to_full_allocation_produces_buy(
        self, orc: StrategyOrchestrator, flat_positions: Dict
    ) -> None:
        """Starting from zero positions, all trades should be positive (buys)."""
        sig = orc.get_signal(regime=2, regime_name="bull", confidence=0.9)
        trades = orc.compute_rebalance_trades(sig, flat_positions, _NAV)
        assert len(trades) == 1
        assert trades["SPY"] > 0

    def test_no_trades_when_already_at_target(
        self, orc: StrategyOrchestrator
    ) -> None:
        """When current positions exactly match target, trades dict is empty."""
        sig = orc.get_signal(regime=2, regime_name="neutral", confidence=0.9)
        target_value = sig.allocation_pct * _NAV    # 50 % × 100 000 = 50 000
        at_target = {"SPY": target_value}
        trades = orc.compute_rebalance_trades(sig, at_target, _NAV)
        assert trades == {} or all(abs(v) < 1e-6 for v in trades.values())

    def test_overweight_produces_sell(self, orc: StrategyOrchestrator) -> None:
        """A position above target weight generates a negative (sell) trade."""
        sig = orc.get_signal(regime=1, regime_name="neutral", confidence=0.9)
        # currently 80 % invested in SPY, target is 50 %
        current = {"SPY": 0.80 * _NAV}
        trades = orc.compute_rebalance_trades(sig, current, _NAV)
        assert "SPY" in trades
        assert trades["SPY"] < 0

    def test_underweight_produces_buy(self, orc: StrategyOrchestrator) -> None:
        sig = orc.get_signal(regime=1, regime_name="bull", confidence=0.9)
        current = {"SPY": 0.30 * _NAV}   # 30 % < 95 % target
        trades = orc.compute_rebalance_trades(sig, current, _NAV)
        assert trades["SPY"] > 0

    def test_trade_signs_match_direction(
        self, orc3: StrategyOrchestrator
    ) -> None:
        """Buy → positive, sell → negative, as labelled."""
        sig = orc3.get_signal(regime=2, regime_name="neutral", confidence=0.8)
        # SPY underweight, QQQ overweight, TLT flat
        per_t = sig.allocation_pct / 3
        current = {
            "SPY": 0.0,               # needs buy
            "QQQ": per_t * 2 * _NAV,  # overweight → sell
            "TLT": per_t * _NAV,      # exactly at target → no trade
        }
        trades = orc3.compute_rebalance_trades(sig, current, _NAV)
        assert trades.get("SPY", 0) > 0
        assert trades.get("QQQ", 0) < 0
        assert abs(trades.get("TLT", 0.0)) < 1e-6

    def test_crash_regime_liquidates_all(
        self, orc: StrategyOrchestrator
    ) -> None:
        """Crash target is 0 % — existing positions are fully sold."""
        sig = orc.get_signal(regime=0, regime_name="crash", confidence=0.99)
        current = {"SPY": 80_000.0}
        trades = orc.compute_rebalance_trades(sig, current, _NAV)
        assert "SPY" in trades
        assert trades["SPY"] < 0

    def test_trade_magnitude_correct(self, orc: StrategyOrchestrator) -> None:
        """Trade amount = target_value − current_value."""
        sig = orc.get_signal(regime=1, regime_name="neutral", confidence=0.9)
        target_value = sig.allocation_pct * _NAV  # 50 %
        current_value = 30_000.0
        trades = orc.compute_rebalance_trades(
            sig, {"SPY": current_value}, _NAV
        )
        expected_delta = target_value - current_value
        assert abs(trades["SPY"] - expected_delta) < 1e-6

    def test_zero_nav_raises(self, orc: StrategyOrchestrator) -> None:
        sig = orc.get_signal(regime=1, regime_name="neutral", confidence=0.9)
        with pytest.raises(ValueError, match="NAV"):
            orc.compute_rebalance_trades(sig, {}, nav=0.0)

    def test_negative_nav_raises(self, orc: StrategyOrchestrator) -> None:
        sig = orc.get_signal(regime=1, regime_name="neutral", confidence=0.9)
        with pytest.raises(ValueError, match="NAV"):
            orc.compute_rebalance_trades(sig, {}, nav=-1.0)


# ---------------------------------------------------------------------------
# 8. Liquidation of tickers not in target universe
# ---------------------------------------------------------------------------

class TestUniverseLiquidation:
    def test_position_outside_universe_is_liquidated(
        self, orc: StrategyOrchestrator
    ) -> None:
        """
        A ticker held in current_positions but absent from self.tickers must
        receive a full-sell trade so the portfolio stays within the universe.
        """
        sig = orc.get_signal(regime=2, regime_name="bull", confidence=0.9)
        current = {"SPY": 30_000.0, "UNLISTED": 5_000.0}
        trades = orc.compute_rebalance_trades(sig, current, _NAV)
        assert "UNLISTED" in trades
        assert trades["UNLISTED"] == -5_000.0

    def test_zero_value_outside_universe_not_traded(
        self, orc: StrategyOrchestrator
    ) -> None:
        """A flat (zero) position outside the universe should not generate a trade."""
        sig = orc.get_signal(regime=2, regime_name="bull", confidence=0.9)
        current = {"SPY": 0.0, "UNLISTED": 0.0}
        trades = orc.compute_rebalance_trades(sig, current, _NAV)
        assert "UNLISTED" not in trades


# ---------------------------------------------------------------------------
# 9. build_allocation_target() round-trip
# ---------------------------------------------------------------------------

class TestBuildAllocationTarget:
    def test_returns_allocation_target(self, orc: StrategyOrchestrator) -> None:
        sig = orc.get_signal(regime=2, regime_name="bull", confidence=0.9)
        target = orc.build_allocation_target(sig)
        assert isinstance(target, AllocationTarget)

    def test_allocation_pct_matches_signal(self, orc: StrategyOrchestrator) -> None:
        sig = orc.get_signal(regime=2, regime_name="neutral", confidence=0.8)
        target = orc.build_allocation_target(sig)
        assert abs(target.allocation_pct - sig.allocation_pct) < 1e-9

    def test_leverage_matches_signal(self, orc: StrategyOrchestrator) -> None:
        sig = orc.get_signal(regime=3, regime_name="bull", confidence=0.9)
        target = orc.build_allocation_target(sig)
        assert target.leverage == sig.leverage

    def test_weights_sum_to_allocation_pct(self, orc3: StrategyOrchestrator) -> None:
        sig = orc3.get_signal(regime=2, regime_name="neutral", confidence=0.8)
        target = orc3.build_allocation_target(sig)
        assert abs(sum(target.weights.values()) - sig.allocation_pct) < 1e-9


# ---------------------------------------------------------------------------
# 10. Custom allocation override
# ---------------------------------------------------------------------------

class TestCustomAllocations:
    def test_custom_override_replaces_default(self) -> None:
        orc = StrategyOrchestrator(
            tickers=["SPY"],
            custom_allocations={
                "bull": {"allocation_pct": 0.80, "leverage": 1.0,
                         "notes": "Conservative bull"}
            },
        )
        sig = orc.get_signal(regime=3, regime_name="bull", confidence=0.9)
        assert abs(sig.allocation_pct - 0.80) < 1e-9
        assert sig.leverage == 1.0

    def test_custom_new_regime_addressable(self) -> None:
        orc = StrategyOrchestrator(
            tickers=["SPY"],
            custom_allocations={
                "moon_shot": {"allocation_pct": 1.0, "leverage": 2.0,
                              "notes": "YOLO"}
            },
        )
        sig = orc.get_signal(regime=99, regime_name="moon_shot", confidence=1.0)
        assert abs(sig.allocation_pct - 1.0) < 1e-9

    def test_custom_override_does_not_affect_other_regimes(self) -> None:
        orc = StrategyOrchestrator(
            tickers=["SPY"],
            custom_allocations={
                "bull": {"allocation_pct": 0.80, "leverage": 1.0, "notes": "x"}
            },
        )
        sig = orc.get_signal(regime=1, regime_name="neutral", confidence=0.9)
        assert abs(sig.allocation_pct - 0.50) < 1e-9  # default unchanged

    def test_none_custom_allocations_uses_defaults(self) -> None:
        orc = StrategyOrchestrator(tickers=["SPY"], custom_allocations=None)
        sig = orc.get_signal(regime=2, regime_name="bear", confidence=0.9)
        assert abs(sig.allocation_pct - 0.20) < 1e-9


# ---------------------------------------------------------------------------
# 11. RegimeStrategyRouter backward-compatibility
# ---------------------------------------------------------------------------

class TestRegimeStrategyRouter:
    """Verify the legacy wrapper honours the original scaffold interface."""

    @pytest.fixture
    def router_settings(self):
        from config.settings import REGIME_STRATEGIES, UNIVERSE
        return RegimeStrategyRouter(
            regime_configs=REGIME_STRATEGIES,
            tickers=UNIVERSE.tickers,
        )

    def test_get_target_allocation_returns_target(
        self, router_settings
    ) -> None:
        target = router_settings.get_target_allocation(regime=0)
        assert isinstance(target, AllocationTarget)

    def test_all_regime_labels_addressable(self, router_settings) -> None:
        from config.settings import REGIME_STRATEGIES
        for label in REGIME_STRATEGIES:
            target = router_settings.get_target_allocation(regime=label)
            assert isinstance(target, AllocationTarget)

    def test_unknown_label_raises(self, router_settings) -> None:
        with pytest.raises(KeyError):
            router_settings.get_target_allocation(regime=999)

    def test_weights_sum_lte_one(self, router_settings) -> None:
        from config.settings import REGIME_STRATEGIES
        for label in REGIME_STRATEGIES:
            target = router_settings.get_target_allocation(regime=label)
            assert sum(target.weights.values()) <= 1.0 + 1e-9

    def test_flat_to_full_produces_buy(self, router_settings) -> None:
        from config.settings import REGIME_STRATEGIES
        # Use the regime with highest equity allocation
        label = max(
            REGIME_STRATEGIES,
            key=lambda k: REGIME_STRATEGIES[k]["equity_allocation"],
        )
        target = router_settings.get_target_allocation(regime=label)
        trades = router_settings.compute_rebalance_trades(
            target=target,
            current_positions={"SPY": 0.0},
            nav=_NAV,
            prices=_PRICE,
        )
        if target.allocation_pct > 0:
            assert any(v > 0 for v in trades.values())

    def test_should_rebalance_false_within_threshold(
        self, router_settings
    ) -> None:
        target = router_settings.get_target_allocation(regime=0)
        # Current weights exactly match allocation_pct
        current = {"SPY": target.allocation_pct}
        assert router_settings.should_rebalance(target, current) is False

    def test_should_rebalance_true_above_threshold(
        self, router_settings
    ) -> None:
        target = router_settings.get_target_allocation(regime=0)
        # Massive drift
        current = {"SPY": target.allocation_pct + 0.50}
        assert router_settings.should_rebalance(target, current) is True

    def test_normalize_weights_sums_to_one(self, router_settings) -> None:
        weights = {"SPY": 3.0, "QQQ": 1.0, "TLT": 1.0}
        normalized = router_settings._normalize_weights(weights)
        assert abs(sum(normalized.values()) - 1.0) < 1e-9

    def test_normalize_weights_zero_total(self, router_settings) -> None:
        weights = {"SPY": 0.0}
        normalized = router_settings._normalize_weights(weights)
        assert normalized["SPY"] == 0.0

    def test_validate_leverage_passes_under_cap(self, router_settings) -> None:
        router_settings._validate_leverage({"SPY": 0.5}, cap=1.0)  # no raise

    def test_validate_leverage_raises_over_cap(self, router_settings) -> None:
        with pytest.raises(ValueError, match="leverage cap"):
            router_settings._validate_leverage({"SPY": 1.5}, cap=1.0)


# ---------------------------------------------------------------------------
# 12. SignalData field types
# ---------------------------------------------------------------------------

class TestSignalDataFields:
    def test_rebalance_required_defaults_false(
        self, orc: StrategyOrchestrator
    ) -> None:
        sig = orc.get_signal(regime=1, regime_name="neutral", confidence=0.8)
        assert sig.rebalance_required is False

    def test_can_set_rebalance_required(self, orc: StrategyOrchestrator) -> None:
        sig = orc.get_signal(regime=1, regime_name="neutral", confidence=0.8)
        sig.rebalance_required = True
        assert sig.rebalance_required is True

    def test_all_required_fields_present(self, orc: StrategyOrchestrator) -> None:
        sig = orc.get_signal(regime=2, regime_name="bull", confidence=0.9)
        assert hasattr(sig, "regime")
        assert hasattr(sig, "regime_name")
        assert hasattr(sig, "allocation_pct")
        assert hasattr(sig, "leverage")
        assert hasattr(sig, "confidence")
        assert hasattr(sig, "notes")
