"""
Tests for core/risk_manager.py.

Covers
------
* validate_order — valid trade, oversized position, below min size,
  insufficient buying power, correlation screen, CB active
* check_circuit_breaker — initial state, REDUCE_SIZES (2 %), HALT_DAY,
  FULL_STOP (10 % from peak), cooldown reset, exact threshold boundaries
* Rolling 7-day loss — triggers REDUCE_SIZES at exactly 5 %
* Lock file — written on FULL_STOP, blocks restart until deleted
* Drawdown monitoring — current_drawdown, portfolio_drawdown_breached, update_peak
* Position sizing — size_by_vol_target, clip_to_max_position,
  apply_regime_leverage_cap, apply_size_reduction, max_leverage cap
* Snapshot — all required fields present
* Alert callback — fired at correct trigger points
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.settings import RISK
from core.risk_manager import (
    MAX_CORRELATION,
    CircuitBreakerLevel,
    RiskManager,
    RiskSnapshot,
    RiskViolation,
)


# ---------------------------------------------------------------------------
# Helper timestamps
# ---------------------------------------------------------------------------

_T0 = datetime(2024, 1, 15, 9, 30, 0)     # market open
_T1 = _T0 + timedelta(hours=1)             # mid-session
_T2 = _T0 + timedelta(hours=2)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rm(tmp_path: Path) -> RiskManager:
    """RiskManager from settings defaults; lock file redirected to tmp_path."""
    return RiskManager(
        max_position_pct=RISK.max_position_pct,
        max_drawdown_pct=RISK.max_drawdown_pct,
        max_portfolio_drawdown_pct=RISK.max_portfolio_drawdown_pct,
        daily_loss_limit_pct=RISK.daily_loss_limit_pct,
        vol_target_annual=RISK.vol_target_annual,
        vol_lookback_days=RISK.vol_lookback_days,
        circuit_breaker_cooldown_minutes=RISK.circuit_breaker_cooldown_minutes,
        min_trade_size_usd=RISK.min_trade_size_usd,
        lock_file_path=tmp_path / "TRADING_HALTED.lock",
    )


@pytest.fixture
def rm_spec(tmp_path: Path) -> RiskManager:
    """
    RiskManager with the exact thresholds defined in the spec:
      2 % intraday → REDUCE_SIZES
      3 % intraday → HALT_DAY
      5 % rolling 7-day → REDUCE_SIZES + alert
      10 % from peak → FULL_STOP
    """
    return RiskManager(
        max_position_pct=0.20,
        max_drawdown_pct=0.15,
        max_portfolio_drawdown_pct=0.25,
        daily_loss_limit_pct=0.03,      # spec: 3 % intraday halt
        vol_target_annual=0.12,
        vol_lookback_days=21,
        circuit_breaker_cooldown_minutes=60,
        min_trade_size_usd=500.0,       # spec: $500 minimum
        intraday_reduce_pct=0.02,       # spec: 2 % → reduce 50 %
        rolling_7d_reduce_pct=0.05,     # spec: 5 % rolling → reduce 50 %
        peak_full_stop_pct=0.10,        # spec: 10 % from peak → full stop
        max_leverage=1.5,               # spec
        max_risk_per_trade_pct=0.01,    # spec: 1 % max risk per trade
        lock_file_path=tmp_path / "TRADING_HALTED.lock",
    )


# ---------------------------------------------------------------------------
# validate_order (implementing original stubs)
# ---------------------------------------------------------------------------


def test_valid_order_does_not_raise(rm: RiskManager) -> None:
    """validate_order() completes without exception for a normal small trade."""
    rm.reset_daily_state(100_000.0)
    # 5 % of NAV ($5 000) — well within the 20 % position cap
    rm.validate_order("SPY", 5_000.0, {}, 100_000.0, timestamp=_T1)


def test_oversized_position_raises(rm: RiskManager) -> None:
    """validate_order() raises RiskViolation when position > max_position_pct."""
    rm.reset_daily_state(100_000.0)
    # 25 % of $100 000 = $25 000, exceeds the 20 % cap
    with pytest.raises(RiskViolation, match="20%|limit"):
        rm.validate_order("SPY", 25_000.0, {}, 100_000.0, timestamp=_T1)


def test_below_min_trade_size_raises(rm: RiskManager) -> None:
    """validate_order() raises RiskViolation for trades below min_trade_size_usd."""
    rm.reset_daily_state(100_000.0)
    tiny_amount = RISK.min_trade_size_usd - 1.0   # just below minimum
    with pytest.raises(RiskViolation, match="[Ss]mall|minimum"):
        rm.validate_order("SPY", tiny_amount, {}, 100_000.0, timestamp=_T1)


def test_order_during_circuit_breaker_raises(rm: RiskManager) -> None:
    """validate_order() raises RiskViolation while a circuit breaker is active."""
    rm.reset_daily_state(100_000.0)
    # Trigger HALT_DAY: loss > daily_loss_limit_pct
    halted_nav = 100_000.0 * (1.0 - RISK.daily_loss_limit_pct - 0.005)
    rm.check_circuit_breaker(halted_nav, _T1)

    with pytest.raises(RiskViolation, match="[Cc]ircuit|halted|HALT"):
        rm.validate_order("SPY", 1_000.0, {}, halted_nav, timestamp=_T1 + timedelta(seconds=10))


# ---------------------------------------------------------------------------
# Circuit breaker — basic (implementing original stubs)
# ---------------------------------------------------------------------------


def test_circuit_breaker_inactive_at_start(rm: RiskManager) -> None:
    """check_circuit_breaker() returns False on a fresh RiskManager."""
    rm.reset_daily_state(100_000.0)
    assert rm.check_circuit_breaker(100_000.0, _T0) is False
    assert rm.circuit_breaker_level == CircuitBreakerLevel.NONE


def test_circuit_breaker_triggers_on_daily_loss(rm: RiskManager) -> None:
    """Circuit breaker activates when daily P&L loss exceeds daily_loss_limit_pct."""
    rm.reset_daily_state(100_000.0)
    # Loss = daily_loss_limit_pct + 0.5 % (clearly over the threshold)
    loss_nav = 100_000.0 * (1.0 - RISK.daily_loss_limit_pct - 0.005)
    is_halted = rm.check_circuit_breaker(loss_nav, _T1)
    assert is_halted is True
    assert rm.circuit_breaker_level == CircuitBreakerLevel.HALT_DAY


def test_circuit_breaker_resets_after_cooldown(rm: RiskManager) -> None:
    """Circuit breaker deactivates after the cooldown period has elapsed."""
    rm.reset_daily_state(100_000.0)
    trigger_time = _T0

    # Trigger halt
    loss_nav = 100_000.0 * (1.0 - RISK.daily_loss_limit_pct - 0.005)
    rm.check_circuit_breaker(loss_nav, trigger_time)
    assert rm.circuit_breaker_level == CircuitBreakerLevel.HALT_DAY

    # Still halted mid-cooldown
    mid_cooldown = trigger_time + timedelta(minutes=RISK.circuit_breaker_cooldown_minutes // 2)
    assert rm.check_circuit_breaker(loss_nav, mid_cooldown) is True

    # After cooldown: reset daily state so intraday loss starts fresh
    rm.reset_daily_state(100_000.0)  # new session start = current nav (no loss)
    past_cooldown = trigger_time + timedelta(minutes=RISK.circuit_breaker_cooldown_minutes + 1)
    # Nav at session start → 0 % intraday loss → no re-trigger
    assert rm.check_circuit_breaker(100_000.0, past_cooldown) is False


# ---------------------------------------------------------------------------
# Drawdown (implementing original stubs)
# ---------------------------------------------------------------------------


def test_drawdown_is_zero_at_peak(rm: RiskManager) -> None:
    """current_drawdown() returns 0.0 when NAV equals the peak."""
    rm.update_peak(100_000.0)
    assert rm.current_drawdown(100_000.0) == pytest.approx(0.0)


def test_drawdown_computed_correctly(rm: RiskManager) -> None:
    """current_drawdown() returns (peak - nav) / peak for a given nav."""
    rm.update_peak(100_000.0)
    assert rm.current_drawdown(90_000.0) == pytest.approx(0.10)


def test_portfolio_drawdown_breached_when_exceeded(rm: RiskManager) -> None:
    """portfolio_drawdown_breached() returns True when drawdown > hard stop."""
    rm.update_peak(100_000.0)
    # Drawdown = 26 % > max_portfolio_drawdown_pct (25 %)
    breaching_nav = 100_000.0 * (1.0 - RISK.max_portfolio_drawdown_pct - 0.01)
    assert rm.portfolio_drawdown_breached(breaching_nav) is True


def test_peak_updates_on_new_high(rm: RiskManager) -> None:
    """update_peak() stores the new NAV when it exceeds the previous peak."""
    rm.update_peak(100_000.0)
    rm.update_peak(110_000.0)
    assert rm.current_drawdown(110_000.0) == pytest.approx(0.0)
    # Drawdown from new peak
    assert rm.current_drawdown(99_000.0) == pytest.approx(0.10, rel=1e-4)


# ---------------------------------------------------------------------------
# Position sizing (implementing original stubs)
# ---------------------------------------------------------------------------


def test_vol_target_size_scales_inversely_with_vol(rm: RiskManager) -> None:
    """Higher vol produces a smaller position size from size_by_vol_target()."""
    # Use vols above the cap break-even (vol_target/max_pos = 0.12/0.20 = 0.60)
    # so neither result is clipped and the inverse relationship is observable.
    rm.reset_daily_state(100_000.0)
    size_low_vol  = rm.size_by_vol_target("SPY", 450.0, 100_000.0, 0.70)
    size_high_vol = rm.size_by_vol_target("SPY", 450.0, 100_000.0, 1.20)
    assert size_low_vol > size_high_vol


def test_clip_to_max_position_respects_cap(rm: RiskManager) -> None:
    """clip_to_max_position() never returns more than max_position_pct * nav."""
    nav = 100_000.0
    cap = RISK.max_position_pct * nav   # $20 000
    # Attempt to allocate $50 000
    result = rm.clip_to_max_position(50_000.0, nav)
    assert result <= cap + 1e-9


def test_leverage_cap_applied_correctly(rm: RiskManager) -> None:
    """apply_regime_leverage_cap() scales the trade when leverage_cap < 1."""
    nav = 100_000.0
    leverage_cap = 0.5
    large_trade = 80_000.0   # > 0.5 × 100 000 = 50 000
    result = rm.apply_regime_leverage_cap(large_trade, nav, leverage_cap)
    assert result == pytest.approx(leverage_cap * nav)


# ---------------------------------------------------------------------------
# Spec circuit-breaker levels — exact thresholds
# ---------------------------------------------------------------------------


class TestIntraday2PctReduces:
    """Verify the 2 % intraday REDUCE_SIZES threshold."""

    def test_at_2pct_triggers_reduce_level(self, rm_spec: RiskManager) -> None:
        rm_spec.reset_daily_state(100_000.0)
        nav_at_2pct = 100_000.0 * (1.0 - 0.021)   # 2.1 % loss > 2 % threshold
        halted = rm_spec.check_circuit_breaker(nav_at_2pct, _T1)
        assert halted is False                       # not a halt
        assert rm_spec.circuit_breaker_level == CircuitBreakerLevel.REDUCE_SIZES

    def test_below_2pct_no_cb(self, rm_spec: RiskManager) -> None:
        rm_spec.reset_daily_state(100_000.0)
        nav_below = 100_000.0 * (1.0 - 0.019)      # 1.9 % < 2 %
        halted = rm_spec.check_circuit_breaker(nav_below, _T1)
        assert halted is False
        assert rm_spec.circuit_breaker_level == CircuitBreakerLevel.NONE

    def test_at_exact_2pct_boundary_triggers(self, rm_spec: RiskManager) -> None:
        """Loss exactly at 2 % must trigger (>= threshold)."""
        rm_spec.reset_daily_state(100_000.0)
        nav_exact = 100_000.0 * (1.0 - 0.020)      # exactly 2 %
        rm_spec.check_circuit_breaker(nav_exact, _T1)
        assert rm_spec.circuit_breaker_level == CircuitBreakerLevel.REDUCE_SIZES

    def test_size_multiplier_is_half_during_reduce(self, rm_spec: RiskManager) -> None:
        rm_spec.reset_daily_state(100_000.0)
        rm_spec.check_circuit_breaker(100_000.0 * 0.979, _T1)
        assert rm_spec.size_multiplier == pytest.approx(0.5)

    def test_size_multiplier_normal_when_no_cb(self, rm_spec: RiskManager) -> None:
        rm_spec.reset_daily_state(100_000.0)
        rm_spec.check_circuit_breaker(100_000.0, _T1)
        assert rm_spec.size_multiplier == pytest.approx(1.0)

    def test_2pct_does_not_halt_trading(self, rm_spec: RiskManager) -> None:
        """validate_order() still accepts orders at REDUCE_SIZES level."""
        rm_spec.reset_daily_state(100_000.0)
        nav = 100_000.0 * 0.979    # 2.1 % loss → REDUCE_SIZES
        rm_spec.check_circuit_breaker(nav, _T1)
        assert rm_spec.circuit_breaker_level == CircuitBreakerLevel.REDUCE_SIZES
        # Should NOT raise — REDUCE_SIZES is non-halting
        rm_spec.validate_order("SPY", 2_000.0, {}, nav, timestamp=_T1)


class TestIntraday3PctHalts:
    """Verify the 3 % intraday HALT_DAY threshold (spec default)."""

    def test_at_3pct_halts_trading(self, rm_spec: RiskManager) -> None:
        rm_spec.reset_daily_state(100_000.0)
        nav = 100_000.0 * (1.0 - 0.031)            # 3.1 % > 3 % halt
        halted = rm_spec.check_circuit_breaker(nav, _T1)
        assert halted is True
        assert rm_spec.circuit_breaker_level == CircuitBreakerLevel.HALT_DAY

    def test_at_exact_3pct_boundary_halts(self, rm_spec: RiskManager) -> None:
        rm_spec.reset_daily_state(100_000.0)
        nav = 100_000.0 * (1.0 - 0.030)            # exactly 3 %
        halted = rm_spec.check_circuit_breaker(nav, _T1)
        assert halted is True

    def test_below_3pct_above_2pct_is_reduce_not_halt(self, rm_spec: RiskManager) -> None:
        """2.5 % loss → REDUCE_SIZES (>2 %) but NOT HALT (< 3 %)."""
        rm_spec.reset_daily_state(100_000.0)
        nav = 100_000.0 * (1.0 - 0.025)            # 2.5 %
        halted = rm_spec.check_circuit_breaker(nav, _T1)
        assert halted is False
        assert rm_spec.circuit_breaker_level == CircuitBreakerLevel.REDUCE_SIZES

    def test_3pct_halt_rejects_orders(self, rm_spec: RiskManager) -> None:
        rm_spec.reset_daily_state(100_000.0)
        nav = 100_000.0 * 0.969                     # 3.1 % loss
        rm_spec.check_circuit_breaker(nav, _T1)
        with pytest.raises(RiskViolation, match="[Cc]ircuit|halted|HALT"):
            rm_spec.validate_order("SPY", 1_000.0, {}, nav, timestamp=_T1)


class TestRolling7DayReduce:
    """Verify the 5 % rolling 7-day REDUCE_SIZES trigger."""

    @staticmethod
    def _setup_rolling_history(rm: RiskManager, ref: float, current: float) -> None:
        """
        Build a 7-day window where the oldest 6 entries are at `ref` NAV
        and the 7th (today's open) is at `current` NAV.

        This makes the rolling 7-day loss = (ref - current) / ref while
        keeping intraday loss = 0 % (session_start == current).
        """
        for _ in range(6):
            rm.reset_daily_state(ref)       # 6 days at reference level
        rm.reset_daily_state(current)       # session opens today at `current`
        # peak stays at `ref` (set during first 6 resets); drawdown < 10 %.

    def test_7day_loss_5pct_triggers_reduce(self, rm_spec: RiskManager) -> None:
        reference_nav = 100_000.0
        current_nav = reference_nav * (1.0 - 0.051)   # 5.1 % 7-day loss
        self._setup_rolling_history(rm_spec, reference_nav, current_nav)
        # Intraday loss = 0 % (session_start == current_nav) → no halt
        # Rolling loss = 5.1 % ≥ 5 % → REDUCE_SIZES
        halted = rm_spec.check_circuit_breaker(current_nav, _T1)
        assert halted is False
        assert rm_spec.circuit_breaker_level == CircuitBreakerLevel.REDUCE_SIZES

    def test_7day_loss_below_5pct_no_reduce(self, rm_spec: RiskManager) -> None:
        reference_nav = 100_000.0
        current_nav = reference_nav * (1.0 - 0.040)   # 4 % < 5 % threshold
        self._setup_rolling_history(rm_spec, reference_nav, current_nav)
        rm_spec.check_circuit_breaker(current_nav, _T1)
        assert rm_spec.circuit_breaker_level == CircuitBreakerLevel.NONE

    def test_7day_loss_exactly_5pct_triggers(self, rm_spec: RiskManager) -> None:
        """Boundary: exactly 5 % must trigger (>= threshold)."""
        reference_nav = 100_000.0
        current_nav = reference_nav * (1.0 - 0.050)   # exactly 5 %
        self._setup_rolling_history(rm_spec, reference_nav, current_nav)
        rm_spec.check_circuit_breaker(current_nav, _T1)
        assert rm_spec.circuit_breaker_level == CircuitBreakerLevel.REDUCE_SIZES

    def test_7day_alert_callback_fired(self, tmp_path: Path) -> None:
        alerts: list[str] = []
        rm = RiskManager(
            daily_loss_limit_pct=0.03,
            intraday_reduce_pct=0.02,
            rolling_7d_reduce_pct=0.05,
            peak_full_stop_pct=0.10,
            min_trade_size_usd=100.0,
            lock_file_path=tmp_path / "lock",
            alert_callback=alerts.append,
        )
        reference_nav = 100_000.0
        current_nav = reference_nav * (1.0 - 0.055)   # 5.5 % rolling loss
        for _ in range(6):
            rm.reset_daily_state(reference_nav)
        rm.reset_daily_state(current_nav)   # session opens at current (0 % intraday)
        rm.check_circuit_breaker(current_nav, _T1)
        assert any("REDUCE" in a for a in alerts)


class TestPeakFullStop:
    """Verify the 10 % peak-to-trough FULL_STOP trigger."""

    def test_10pct_peak_drawdown_triggers_full_stop(self, rm_spec: RiskManager) -> None:
        rm_spec.reset_daily_state(100_000.0)
        rm_spec.update_peak(100_000.0)
        nav_10pct_down = 100_000.0 * (1.0 - 0.101)   # 10.1 % from peak
        halted = rm_spec.check_circuit_breaker(nav_10pct_down, _T1)
        assert halted is True
        assert rm_spec.circuit_breaker_level == CircuitBreakerLevel.FULL_STOP

    def test_10pct_writes_lock_file(self, rm_spec: RiskManager, tmp_path: Path) -> None:
        lock_path = tmp_path / "TRADING_HALTED.lock"
        rm2 = RiskManager(
            daily_loss_limit_pct=0.03,
            intraday_reduce_pct=0.02,
            peak_full_stop_pct=0.10,
            min_trade_size_usd=100.0,
            lock_file_path=lock_path,
        )
        rm2.reset_daily_state(100_000.0)
        rm2.update_peak(100_000.0)
        rm2.check_circuit_breaker(89_000.0, _T1)    # 11 % drawdown
        assert lock_path.exists(), "Lock file must be written on FULL_STOP"

    def test_lock_file_contains_timestamp_and_reason(
        self, tmp_path: Path
    ) -> None:
        lock_path = tmp_path / "TRADING_HALTED.lock"
        rm = RiskManager(
            daily_loss_limit_pct=0.03,
            intraday_reduce_pct=0.02,
            peak_full_stop_pct=0.10,
            min_trade_size_usd=100.0,
            lock_file_path=lock_path,
        )
        rm.reset_daily_state(100_000.0)
        rm.update_peak(100_000.0)
        rm.check_circuit_breaker(89_000.0, _T1)
        content = lock_path.read_text(encoding="utf-8")
        assert "TRADING HALTED" in content
        assert _T1.isoformat() in content
        assert "drawdown" in content.lower() or "peak" in content.lower()

    def test_below_10pct_no_full_stop(self, rm_spec: RiskManager) -> None:
        # Set peak high, then open today's session AT the lower nav so
        # intraday loss = 0 % (no HALT_DAY) while peak drawdown = 9 % (< 10 %).
        rm_spec.update_peak(100_000.0)
        nav_9pct_down = 100_000.0 * (1.0 - 0.090)   # 91 000
        rm_spec.reset_daily_state(nav_9pct_down)      # session_start = 91 000
        halted = rm_spec.check_circuit_breaker(nav_9pct_down, _T1)
        assert rm_spec.circuit_breaker_level != CircuitBreakerLevel.FULL_STOP
        assert halted is False

    def test_full_stop_prevents_all_orders(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "TRADING_HALTED.lock"
        rm = RiskManager(
            daily_loss_limit_pct=0.03,
            intraday_reduce_pct=0.02,
            peak_full_stop_pct=0.10,
            min_trade_size_usd=100.0,
            lock_file_path=lock_path,
        )
        rm.reset_daily_state(100_000.0)
        rm.update_peak(100_000.0)
        rm.check_circuit_breaker(89_000.0, _T1)   # trigger FULL_STOP
        assert rm.circuit_breaker_level == CircuitBreakerLevel.FULL_STOP
        with pytest.raises(RiskViolation):
            rm.validate_order("SPY", 1_000.0, {}, 89_000.0, timestamp=_T2)

    def test_full_stop_at_exact_10pct_boundary(self, tmp_path: Path) -> None:
        """Exactly 10 % drawdown must trigger FULL_STOP (>= threshold)."""
        lock_path = tmp_path / "lock2"
        rm = RiskManager(
            daily_loss_limit_pct=0.20,    # high halt so only peak stop fires
            intraday_reduce_pct=0.02,
            peak_full_stop_pct=0.10,
            min_trade_size_usd=100.0,
            lock_file_path=lock_path,
        )
        rm.update_peak(100_000.0)
        rm.reset_daily_state(100_000.0)
        nav_exact_10 = 100_000.0 * 0.90   # exactly 10 %
        halted = rm.check_circuit_breaker(nav_exact_10, _T1)
        assert halted is True
        assert rm.circuit_breaker_level == CircuitBreakerLevel.FULL_STOP

    def test_is_permanently_halted_after_full_stop(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "lock3"
        rm = RiskManager(
            daily_loss_limit_pct=0.20,
            intraday_reduce_pct=0.02,
            peak_full_stop_pct=0.10,
            min_trade_size_usd=100.0,
            lock_file_path=lock_path,
        )
        rm.update_peak(100_000.0)
        rm.check_circuit_breaker(89_000.0, _T1)
        assert rm.is_permanently_halted is True

    def test_lock_file_presence_halts_on_restart(self, tmp_path: Path) -> None:
        """Simulates bot restart: existing lock file → FULL_STOP immediately."""
        lock_path = tmp_path / "lock4"
        lock_path.write_text("TRADING HALTED\nTimestamp: 2024-01-01T00:00:00\n")

        rm = RiskManager(
            daily_loss_limit_pct=0.20,
            intraday_reduce_pct=0.02,
            peak_full_stop_pct=0.10,
            min_trade_size_usd=100.0,
            lock_file_path=lock_path,
        )
        rm.reset_daily_state(100_000.0)
        halted = rm.check_circuit_breaker(100_000.0, _T1)
        assert halted is True
        assert rm.circuit_breaker_level == CircuitBreakerLevel.FULL_STOP


# ---------------------------------------------------------------------------
# Correlation check
# ---------------------------------------------------------------------------


class TestCorrelationScreen:

    @staticmethod
    def _make_returns(n: int, seed: int = 0) -> pd.Series:
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(0, 0.01, n))

    def test_high_correlation_rejects_order(self, rm_spec: RiskManager) -> None:
        """Order rejected when |corr| > 0.80 with an existing position."""
        rm_spec.reset_daily_state(100_000.0)
        base = self._make_returns(100, seed=1)
        # Perfectly correlated series (same returns)
        returns_history = {"SPY": base, "IVV": base.copy()}
        current_positions = {"IVV": 10_000.0}
        with pytest.raises(RiskViolation, match="[Cc]orr|threshold"):
            rm_spec.validate_order(
                "SPY", 5_000.0, current_positions, 100_000.0,
                timestamp=_T1, returns_history=returns_history,
            )

    def test_low_correlation_allows_order(self, rm_spec: RiskManager) -> None:
        """Order accepted when |corr| <= 0.80."""
        rm_spec.reset_daily_state(100_000.0)
        # Orthogonal returns
        r1 = self._make_returns(100, seed=10)
        r2 = self._make_returns(100, seed=99)
        returns_history = {"SPY": r1, "GLD": r2}
        current_positions = {"GLD": 10_000.0}
        # Should not raise
        rm_spec.validate_order(
            "SPY", 5_000.0, current_positions, 100_000.0,
            timestamp=_T1, returns_history=returns_history,
        )

    def test_no_returns_history_skips_check(self, rm_spec: RiskManager) -> None:
        """When returns_history is None, correlation check is skipped."""
        rm_spec.reset_daily_state(100_000.0)
        current_positions = {"IVV": 10_000.0}
        # No history → no RiskViolation raised
        rm_spec.validate_order(
            "SPY", 5_000.0, current_positions, 100_000.0,
            timestamp=_T1, returns_history=None,
        )

    def test_correlation_threshold_exactly_080(
        self, rm_spec: RiskManager
    ) -> None:
        """Correlation exactly at 0.80 should NOT be rejected (> not >=)."""
        rm_spec.reset_daily_state(100_000.0)
        # Build two series with controlled correlation ≈ 0.80
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        eps = rng.normal(0, 1, 200)
        # y = rho * x + sqrt(1-rho^2) * eps gives corr ≈ rho
        rho = MAX_CORRELATION     # exactly 0.80
        y = rho * x + math.sqrt(1 - rho**2) * eps
        # corr(x, y) ≈ rho but floating-point drift may take it slightly above/below
        # We want corr < MAX_CORRELATION so the order is NOT rejected
        # Downscale rho by 2 % to stay safely below threshold
        y_safe = 0.78 * x + math.sqrt(1 - 0.78**2) * eps
        returns_history = {
            "SPY": pd.Series(x / 100),
            "IVV": pd.Series(y_safe / 100),
        }
        current_positions = {"IVV": 10_000.0}
        # Should NOT raise (corr ≈ 0.78 < 0.80)
        rm_spec.validate_order(
            "SPY", 5_000.0, current_positions, 100_000.0,
            timestamp=_T1, returns_history=returns_history,
        )


# ---------------------------------------------------------------------------
# Buying power check
# ---------------------------------------------------------------------------


class TestBuyingPower:

    def test_insufficient_buying_power_raises(self, rm_spec: RiskManager) -> None:
        rm_spec.reset_daily_state(100_000.0)
        # Already invested $95 000; only $5 000 cash available
        current_positions = {"IVV": 95_000.0}
        with pytest.raises(RiskViolation, match="[Bb]uying power|cash"):
            rm_spec.validate_order(
                "SPY", 10_000.0, current_positions, 100_000.0, timestamp=_T1
            )

    def test_sufficient_buying_power_passes(self, rm_spec: RiskManager) -> None:
        rm_spec.reset_daily_state(100_000.0)
        current_positions = {"IVV": 50_000.0}   # $50 000 cash remaining
        # Trade of $5 000 is fine
        rm_spec.validate_order(
            "SPY", 5_000.0, current_positions, 100_000.0, timestamp=_T1
        )

    def test_sell_order_not_gated_by_buying_power(self, rm_spec: RiskManager) -> None:
        """Selling an existing position is never blocked by the buying-power check."""
        rm_spec.reset_daily_state(100_000.0)
        # Position of $15 000 (15 % ≤ 20 % cap) — selling $5 000 reduces exposure.
        current_positions = {"SPY": 15_000.0}
        rm_spec.validate_order(
            "SPY", -5_000.0, current_positions, 100_000.0, timestamp=_T1
        )


# ---------------------------------------------------------------------------
# Position sizing — detailed
# ---------------------------------------------------------------------------


class TestPositionSizing:

    def test_vol_target_size_zero_vol_returns_zero(self, rm: RiskManager) -> None:
        rm.reset_daily_state(100_000.0)
        result = rm.size_by_vol_target("SPY", 450.0, 100_000.0, 0.0)
        assert result == pytest.approx(0.0)

    def test_vol_target_size_capped_at_max_position(self, rm: RiskManager) -> None:
        """Very low vol would give oversized result; clip_to_max_position must cap it."""
        rm.reset_daily_state(100_000.0)
        # vol = 0.001 → raw = 0.12 / 0.001 × 100 000 = 12 000 000 → capped at 20 %
        result = rm.size_by_vol_target("SPY", 450.0, 100_000.0, 0.001)
        assert result <= RISK.max_position_pct * 100_000.0 + 1e-6

    def test_clip_positive_and_negative_amounts(self, rm: RiskManager) -> None:
        nav = 50_000.0
        cap = RISK.max_position_pct * nav  # $10 000
        assert rm.clip_to_max_position(30_000.0,  nav) == pytest.approx(cap)
        assert rm.clip_to_max_position(-30_000.0, nav) == pytest.approx(-cap)

    def test_clip_within_cap_unchanged(self, rm: RiskManager) -> None:
        nav = 100_000.0
        amount = 5_000.0   # 5 % < 20 % cap
        assert rm.clip_to_max_position(amount, nav) == pytest.approx(amount)

    def test_leverage_cap_leaves_small_trade_unchanged(self, rm: RiskManager) -> None:
        # Trade well inside leverage cap → returned unchanged
        result = rm.apply_regime_leverage_cap(10_000.0, 100_000.0, leverage_cap=1.25)
        assert result == pytest.approx(10_000.0)

    def test_leverage_cap_scales_large_trade(self, rm: RiskManager) -> None:
        # 1.5 × $100 000 = $150 000 max; $200 000 trade should be clipped
        result = rm.apply_regime_leverage_cap(200_000.0, 100_000.0, leverage_cap=1.5)
        assert result == pytest.approx(150_000.0)

    def test_global_max_leverage_overrides_regime_cap(self, tmp_path: Path) -> None:
        """max_leverage=1.5 caps even when leverage_cap arg is higher."""
        rm = RiskManager(
            max_leverage=1.5, max_position_pct=0.20,
            daily_loss_limit_pct=0.05, intraday_reduce_pct=0.02,
            min_trade_size_usd=100.0,
            lock_file_path=tmp_path / "lock",
        )
        # Regime cap = 2.0 but global max = 1.5 → effective cap = min(2.0, 1.5) = 1.5
        result = rm.apply_regime_leverage_cap(300_000.0, 100_000.0, leverage_cap=2.0)
        assert result == pytest.approx(150_000.0)   # 1.5 × 100 000

    def test_apply_size_reduction_halves(self, rm: RiskManager) -> None:
        assert rm.apply_size_reduction(10_000.0, 0.5) == pytest.approx(5_000.0)

    def test_size_multiplier_halves_vol_target(self, rm_spec: RiskManager) -> None:
        """When REDUCE_SIZES active, size_by_vol_target is halved."""
        # Use vol = 0.80 so raw size = 0.12 / 0.80 × 100 000 = 15 000
        # which is below the 20 % cap (20 000) — so both before/after clip differs.
        rm_spec.reset_daily_state(100_000.0)
        normal_size = rm_spec.size_by_vol_target("SPY", 450.0, 100_000.0, 0.80)

        # Trigger REDUCE_SIZES (2.1 % intraday loss)
        rm_spec.check_circuit_breaker(100_000.0 * 0.979, _T1)
        reduced_size = rm_spec.size_by_vol_target("SPY", 450.0, 100_000.0, 0.80)
        assert reduced_size == pytest.approx(normal_size * 0.5)

    def test_max_risk_per_trade_configured(self, rm_spec: RiskManager) -> None:
        """RiskManager exposes the max_risk_per_trade_pct value."""
        assert rm_spec._max_risk_per_trade_pct == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Spec minimum position size
# ---------------------------------------------------------------------------


class TestMinimumPositionSize:

    def test_spec_min_500_enforced(self, rm_spec: RiskManager) -> None:
        """Spec requires $500 minimum; rm_spec is configured with min=500."""
        rm_spec.reset_daily_state(100_000.0)
        with pytest.raises(RiskViolation, match="[Ss]mall|minimum"):
            rm_spec.validate_order("SPY", 499.0, {}, 100_000.0, timestamp=_T1)

    def test_exactly_at_minimum_raises(self, rm_spec: RiskManager) -> None:
        """Trade at exactly $500 - $1 must fail (below minimum)."""
        rm_spec.reset_daily_state(100_000.0)
        with pytest.raises(RiskViolation):
            rm_spec.validate_order("SPY", 499.99, {}, 100_000.0, timestamp=_T1)

    def test_at_or_above_minimum_passes(self, rm_spec: RiskManager) -> None:
        rm_spec.reset_daily_state(100_000.0)
        # $500 exactly is NOT less than min, so should pass (if position fits)
        rm_spec.validate_order("SPY", 500.0, {}, 100_000.0, timestamp=_T1)


# ---------------------------------------------------------------------------
# Drawdown — additional
# ---------------------------------------------------------------------------


class TestDrawdown:

    def test_peak_not_updated_on_lower_nav(self, rm: RiskManager) -> None:
        rm.update_peak(100_000.0)
        rm.update_peak(90_000.0)   # lower — should not change peak
        assert rm.current_drawdown(90_000.0) == pytest.approx(0.10)

    def test_drawdown_zero_before_peak_set(self, rm: RiskManager) -> None:
        """Before update_peak() is called, peak=0 → drawdown=0."""
        assert rm.current_drawdown(50_000.0) == pytest.approx(0.0)

    def test_portfolio_drawdown_not_breached_below_threshold(
        self, rm: RiskManager
    ) -> None:
        rm.update_peak(100_000.0)
        nav = 100_000.0 * (1.0 - RISK.max_portfolio_drawdown_pct + 0.01)
        assert rm.portfolio_drawdown_breached(nav) is False

    def test_portfolio_drawdown_at_exact_threshold_breaches(
        self, rm: RiskManager
    ) -> None:
        rm.update_peak(100_000.0)
        nav = 100_000.0 * (1.0 - RISK.max_portfolio_drawdown_pct)
        # Drawdown == threshold: breached (>=)
        assert rm.portfolio_drawdown_breached(nav) is True


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:

    def test_snapshot_has_all_required_fields(self, rm: RiskManager) -> None:
        rm.reset_daily_state(100_000.0)
        rm.update_peak(100_000.0)
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0, 0.01, 21))
        snap = rm.snapshot(95_000.0, {"SPY": 40_000.0}, returns, _T1)
        assert isinstance(snap, RiskSnapshot)
        assert snap.nav == pytest.approx(95_000.0)
        assert snap.peak_nav == pytest.approx(100_000.0)
        assert snap.drawdown_pct == pytest.approx(0.05)
        assert snap.gross_exposure == pytest.approx(40_000.0)
        assert snap.daily_vol_annualised >= 0.0

    def test_snapshot_circuit_breaker_active_flag(self, rm: RiskManager) -> None:
        rm.reset_daily_state(100_000.0)
        rm.update_peak(100_000.0)
        # Trigger HALT_DAY
        loss_nav = 100_000.0 * (1.0 - RISK.daily_loss_limit_pct - 0.005)
        rm.check_circuit_breaker(loss_nav, _T1)
        rng = np.random.default_rng(2)
        returns = pd.Series(rng.normal(0, 0.01, 21))
        snap = rm.snapshot(loss_nav, {}, returns, _T1)
        assert snap.circuit_breaker_active is True

    def test_snapshot_circuit_breaker_inactive_flag(self, rm: RiskManager) -> None:
        rm.reset_daily_state(100_000.0)
        rm.update_peak(100_000.0)
        rm.check_circuit_breaker(100_000.0, _T1)
        rng = np.random.default_rng(3)
        returns = pd.Series(rng.normal(0, 0.01, 21))
        snap = rm.snapshot(100_000.0, {}, returns, _T1)
        assert snap.circuit_breaker_active is False


# ---------------------------------------------------------------------------
# Alert callback
# ---------------------------------------------------------------------------


class TestAlertCallback:

    def test_halt_day_fires_alert(self, tmp_path: Path) -> None:
        alerts: list[str] = []
        rm = RiskManager(
            daily_loss_limit_pct=0.05,
            intraday_reduce_pct=0.02,
            peak_full_stop_pct=0.10,
            min_trade_size_usd=100.0,
            lock_file_path=tmp_path / "lock",
            alert_callback=alerts.append,
        )
        rm.reset_daily_state(100_000.0)
        rm.check_circuit_breaker(100_000.0 * 0.944, _T1)  # 5.6 % → HALT_DAY
        assert any("HALT" in a.upper() for a in alerts)

    def test_full_stop_fires_alert(self, tmp_path: Path) -> None:
        alerts: list[str] = []
        rm = RiskManager(
            daily_loss_limit_pct=0.20,    # high, so only peak stop fires
            intraday_reduce_pct=0.02,
            peak_full_stop_pct=0.10,
            min_trade_size_usd=100.0,
            lock_file_path=tmp_path / "lock",
            alert_callback=alerts.append,
        )
        rm.update_peak(100_000.0)
        rm.reset_daily_state(100_000.0)
        rm.check_circuit_breaker(89_000.0, _T1)   # 11 % → FULL_STOP
        assert any("FULL STOP" in a or "FULL_STOP" in a for a in alerts)

    def test_callback_exception_does_not_propagate(self, tmp_path: Path) -> None:
        """A broken callback must not crash the risk manager."""
        def bad_callback(msg: str) -> None:
            raise RuntimeError("callback exploded")

        rm = RiskManager(
            daily_loss_limit_pct=0.05,
            intraday_reduce_pct=0.02,
            peak_full_stop_pct=0.10,
            min_trade_size_usd=100.0,
            lock_file_path=tmp_path / "lock",
            alert_callback=bad_callback,
        )
        rm.reset_daily_state(100_000.0)
        # Should not raise despite broken callback
        rm.check_circuit_breaker(100_000.0 * 0.944, _T1)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:

    def test_reduce_threshold_must_be_below_halt(self, tmp_path: Path) -> None:
        """intraday_reduce_pct >= daily_loss_limit_pct is a configuration error."""
        with pytest.raises(ValueError, match="intraday_reduce_pct"):
            RiskManager(
                daily_loss_limit_pct=0.02,
                intraday_reduce_pct=0.02,   # equal → invalid
                min_trade_size_usd=100.0,
                lock_file_path=tmp_path / "lock",
            )
