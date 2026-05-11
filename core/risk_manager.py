"""
Risk management: circuit breakers, position sizing, order validation,
and drawdown limits.

Architecture
------------
RiskManager has ABSOLUTE VETO POWER over all other systems.
Every proposed order must pass validate_order() before reaching
the OrderExecutor.  Any rule violation raises RiskViolation.

Circuit breaker hierarchy (triggers independently of HMM)
----------------------------------------------------------
Level 1 — REDUCE_SIZES (non-halting):
    • Intraday loss >= intraday_reduce_pct (default 2 %)
    • All position sizes multiplied by 0.50 for the rest of the session.
    • Rolling 7-day loss >= rolling_7d_reduce_pct (default 5 %)
    • Same 0.50 size scalar applied + alert fired.

Level 2 — HALT_DAY:
    • Intraday loss >= daily_loss_limit_pct (default 5 %, matches RISK config).
    • All positions closed, no new orders accepted.
    • Trading resumes after circuit_breaker_cooldown_minutes.

Level 3 — FULL_STOP (permanent until manual reset):
    • Peak-to-trough drawdown >= peak_full_stop_pct (default 10 %).
    • All positions closed, lock file written to logs/TRADING_HALTED.lock
      containing timestamp and reason.  Bot cannot restart until the file
      is manually deleted by the operator.

Position sizing rules
---------------------
• Max risk per trade:  max_risk_per_trade_pct × NAV  (default 1 %)
• Max single position: max_position_pct × NAV         (default 20 %)
• Max total leverage:  max_leverage                    (default 1.5×)
• Minimum trade size:  min_trade_size_usd              (default $100)

Order validation (pre-trade)
-----------------------------
1. Circuit breaker not active at HALT_DAY or FULL_STOP level.
2. Trade size >= min_trade_size_usd.
3. Resulting position <= max_position_pct × NAV.
4. Buying power (cash) is sufficient.
5. Correlation with any existing open position <= MAX_CORRELATION (0.80).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_LOCK_FILE = Path("logs") / "TRADING_HALTED.lock"

MAX_CORRELATION: float = 0.80          # reject order if |corr| > this
_ANNUALISE: float = math.sqrt(252)


# ---------------------------------------------------------------------------
# Circuit breaker levels
# ---------------------------------------------------------------------------

class CircuitBreakerLevel(IntEnum):
    """Ordered severity levels — higher value = more restrictive."""
    NONE         = 0   # normal operation
    REDUCE_SIZES = 1   # size ×0.50, trading still allowed
    HALT_DAY     = 2   # no new orders until cooldown expires
    FULL_STOP    = 3   # permanent halt; lock file written


# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------

class RiskViolation(Exception):
    """Raised when a proposed trade or portfolio state violates a rule."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RiskSnapshot:
    """Point-in-time portfolio risk metrics for dashboarding / logging."""
    timestamp: datetime
    nav: float
    peak_nav: float
    daily_pnl: float
    drawdown_pct: float
    portfolio_drawdown_pct: float
    gross_exposure: float
    net_exposure: float
    largest_position_pct: float
    daily_vol_annualised: float
    circuit_breaker_active: bool
    circuit_breaker_level: CircuitBreakerLevel = CircuitBreakerLevel.NONE


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Validates orders and monitors portfolio-level risk limits.

    The single source of truth for all risk decisions.  Stateful — must
    be kept alive for the duration of a trading session.

    Usage
    -----
    At session open:
        rm.reset_daily_state(opening_nav)

    Before every valuation update:
        rm.update_peak(nav)

    Before every order:
        rm.validate_order(ticker, dollar_amount, positions, nav, timestamp)

    For live vol-aware sizing:
        size = rm.size_by_vol_target(ticker, price, nav, realised_vol_annual)
        size = rm.clip_to_max_position(size, nav)
    """

    def __init__(
        self,
        # ---- existing params (kept for backward compatibility) ----
        max_position_pct: float = 0.20,
        max_drawdown_pct: float = 0.15,
        max_portfolio_drawdown_pct: float = 0.25,
        daily_loss_limit_pct: float = 0.05,    # maps to HALT_DAY threshold
        vol_target_annual: float = 0.12,
        vol_lookback_days: int = 21,
        circuit_breaker_cooldown_minutes: int = 60,
        min_trade_size_usd: float = 100.0,
        # ---- new circuit-breaker thresholds (per spec) ----
        intraday_reduce_pct: float = 0.02,     # 2 %  → REDUCE_SIZES
        rolling_7d_reduce_pct: float = 0.05,   # 5 %  rolling → REDUCE_SIZES + alert
        peak_full_stop_pct: float = 0.10,      # 10 % from peak → FULL_STOP
        # ---- new sizing params ----
        max_leverage: float = 1.5,
        max_risk_per_trade_pct: float = 0.01,
        # ---- infrastructure ----
        lock_file_path: Optional[Path] = None,
        alert_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        # Validate key relationships
        if intraday_reduce_pct >= daily_loss_limit_pct:
            raise ValueError(
                f"intraday_reduce_pct ({intraday_reduce_pct}) must be less than "
                f"daily_loss_limit_pct ({daily_loss_limit_pct})."
            )

        # Store configuration
        self._max_position_pct           = max_position_pct
        self._max_drawdown_pct           = max_drawdown_pct
        self._max_portfolio_drawdown_pct = max_portfolio_drawdown_pct
        self._daily_loss_limit_pct       = daily_loss_limit_pct   # HALT_DAY
        self._vol_target_annual          = vol_target_annual
        self._vol_lookback_days          = vol_lookback_days
        self._cooldown_minutes           = circuit_breaker_cooldown_minutes
        self._min_trade_size_usd         = min_trade_size_usd
        self._intraday_reduce_pct        = intraday_reduce_pct
        self._rolling_7d_reduce_pct      = rolling_7d_reduce_pct
        self._peak_full_stop_pct         = peak_full_stop_pct
        self._max_leverage               = max_leverage
        self._max_risk_per_trade_pct     = max_risk_per_trade_pct
        self._lock_file_path             = lock_file_path or _DEFAULT_LOCK_FILE
        self._alert_callback             = alert_callback

        # Mutable state
        self._peak_nav:                    float                         = 0.0
        self._session_start_nav:           float                         = 0.0
        self._daily_nav_history:           List[float]                   = []
        self._circuit_breaker_level:       CircuitBreakerLevel           = CircuitBreakerLevel.NONE
        self._circuit_breaker_triggered_at: Optional[datetime]           = None
        self._permanently_halted:          bool                          = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def circuit_breaker_level(self) -> CircuitBreakerLevel:
        """Current circuit breaker level (last evaluated)."""
        return self._circuit_breaker_level

    @property
    def size_multiplier(self) -> float:
        """
        Factor to multiply all trade sizes by.

        Returns 0.5 when REDUCE_SIZES is active, 1.0 otherwise.
        HALT_DAY / FULL_STOP: trading is not allowed at all (validate_order raises).
        """
        return 0.5 if self._circuit_breaker_level == CircuitBreakerLevel.REDUCE_SIZES else 1.0

    @property
    def is_permanently_halted(self) -> bool:
        """True if a FULL_STOP has been triggered (lock file exists)."""
        return self._permanently_halted or self._lock_file_path.exists()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_daily_state(self, opening_nav: float) -> None:
        """
        Call at market open to reset intraday accumulators.

        Also records the opening NAV in the rolling 7-day history.
        """
        self._session_start_nav = opening_nav

        # Rolling history: keep at most 7 entries
        self._daily_nav_history.append(opening_nav)
        if len(self._daily_nav_history) > 7:
            self._daily_nav_history.pop(0)

        # Clear HALT_DAY level if cooldown has been more than implicitly served
        # (the explicit cooldown check happens in check_circuit_breaker)

        self.update_peak(opening_nav)
        logger.debug(
            "Daily state reset: session_start=%.2f peak=%.2f",
            opening_nav, self._peak_nav,
        )

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def check_circuit_breaker(self, nav: float, timestamp: datetime) -> bool:
        """
        Evaluate all circuit breaker conditions and update internal state.

        Returns
        -------
        True  → trading is halted (HALT_DAY or FULL_STOP level).
        False → trading is permitted (possibly with reduced sizes).

        Side effects
        ------------
        • Sets _circuit_breaker_level.
        • Writes the lock file and sets _permanently_halted on FULL_STOP.
        • Clears HALT_DAY state when cooldown expires.
        """
        # 1. Permanent halt takes precedence
        if self.is_permanently_halted:
            self._circuit_breaker_level = CircuitBreakerLevel.FULL_STOP
            return True

        # 2. Peak drawdown → FULL_STOP
        if self._peak_nav > 0:
            peak_dd = (self._peak_nav - nav) / self._peak_nav
            if peak_dd >= self._peak_full_stop_pct:
                reason = (
                    f"Peak-to-trough drawdown {peak_dd:.2%} >= "
                    f"{self._peak_full_stop_pct:.0%} from peak ${self._peak_nav:,.2f}"
                )
                self._trigger_full_stop(nav, timestamp, reason)
                return True

        # 3. Cooldown from a previous HALT_DAY
        if (
            self._circuit_breaker_level == CircuitBreakerLevel.HALT_DAY
            and self._circuit_breaker_triggered_at is not None
        ):
            elapsed_min = (
                (timestamp - self._circuit_breaker_triggered_at).total_seconds() / 60.0
            )
            if elapsed_min < self._cooldown_minutes:
                return True   # still in cooldown, keep halting
            else:
                # Cooldown elapsed: clear halt state, then re-evaluate
                logger.info(
                    "Circuit breaker cooldown elapsed (%.0f min); resuming evaluation.",
                    elapsed_min,
                )
                self._circuit_breaker_level       = CircuitBreakerLevel.NONE
                self._circuit_breaker_triggered_at = None

        # 4. Intraday halt — HALT_DAY
        if self._session_start_nav > 0:
            intraday_loss = (self._session_start_nav - nav) / self._session_start_nav
            if intraday_loss >= self._daily_loss_limit_pct:
                self._circuit_breaker_level       = CircuitBreakerLevel.HALT_DAY
                self._circuit_breaker_triggered_at = timestamp
                logger.warning(
                    "Circuit breaker HALT_DAY: intraday loss %.2f%% >= %.0f%%",
                    intraday_loss * 100, self._daily_loss_limit_pct * 100,
                )
                self._fire_alert(
                    f"HALT_DAY: intraday loss {intraday_loss:.2%} at {timestamp}"
                )
                return True

        # 5. REDUCE_SIZES checks (non-halting)
        should_reduce = False

        # Rolling 7-day loss
        rolling_loss = self._compute_rolling_7d_loss(nav)
        if rolling_loss >= self._rolling_7d_reduce_pct:
            should_reduce = True
            logger.warning(
                "CB REDUCE_SIZES: rolling 7-day loss %.2f%% >= %.0f%%",
                rolling_loss * 100, self._rolling_7d_reduce_pct * 100,
            )
            self._fire_alert(
                f"REDUCE_SIZES: rolling 7-day loss {rolling_loss:.2%} at {timestamp}"
            )

        # Intraday 2 % reduce
        if self._session_start_nav > 0:
            intraday_loss = (self._session_start_nav - nav) / self._session_start_nav
            if intraday_loss >= self._intraday_reduce_pct:
                should_reduce = True
                logger.warning(
                    "CB REDUCE_SIZES: intraday loss %.2f%% >= %.0f%%",
                    intraday_loss * 100, self._intraday_reduce_pct * 100,
                )

        if should_reduce:
            self._circuit_breaker_level = CircuitBreakerLevel.REDUCE_SIZES
        elif self._circuit_breaker_level == CircuitBreakerLevel.REDUCE_SIZES:
            # Loss recovered below reduce threshold — clear
            self._circuit_breaker_level = CircuitBreakerLevel.NONE

        return False   # not halted

    # ------------------------------------------------------------------
    # Order validation
    # ------------------------------------------------------------------

    def validate_order(
        self,
        ticker: str,
        dollar_amount: float,
        current_positions: Dict[str, float],
        nav: float,
        timestamp: Optional[datetime] = None,
        returns_history: Optional[Dict[str, "pd.Series"]] = None,
    ) -> None:
        """
        Raise RiskViolation if the proposed order breaks any rule.

        Parameters
        ----------
        ticker:
            Symbol being traded.
        dollar_amount:
            Signed dollar notional (+ buy, − sell).
        current_positions:
            Current market values by ticker (signed dollars).
        nav:
            Current net asset value.
        timestamp:
            Evaluation time; defaults to datetime.now().
        returns_history:
            Optional dict of ticker → pd.Series of historical returns,
            used for correlation screening.

        Raises
        ------
        RiskViolation on any rule violation.
        """
        ts = timestamp or datetime.now()

        # 1. Circuit breaker — absolute veto for HALT_DAY / FULL_STOP
        if self.check_circuit_breaker(nav, ts):
            raise RiskViolation(
                "Circuit breaker active — trading halted. "
                f"Level: {self._circuit_breaker_level.name}"
            )

        # 2. Minimum trade size
        if abs(dollar_amount) < self._min_trade_size_usd:
            raise RiskViolation(
                f"Trade too small: ${abs(dollar_amount):.2f} < "
                f"minimum ${self._min_trade_size_usd:.2f}."
            )

        # 3. Max position size
        # Only block when the trade INCREASES absolute exposure beyond the cap.
        # Sells that reduce an existing (even oversized) position are always allowed.
        if nav > 0:
            existing = current_positions.get(ticker, 0.0)
            new_position = existing + dollar_amount
            if (
                abs(new_position) > abs(existing)                   # exposure growing
                and abs(new_position) / nav > self._max_position_pct
            ):
                raise RiskViolation(
                    f"Position in {ticker} would reach "
                    f"{abs(new_position) / nav:.1%} of NAV, "
                    f"exceeding {self._max_position_pct:.0%} limit."
                )

        # 4. Buying power (for buy orders)
        if dollar_amount > 0 and nav > 0:
            total_invested = sum(abs(v) for v in current_positions.values())
            available_cash = max(0.0, nav - total_invested)
            if dollar_amount > available_cash:
                raise RiskViolation(
                    f"Insufficient buying power: need ${dollar_amount:,.2f}, "
                    f"have ${available_cash:,.2f} cash."
                )

        # 5. Correlation screen
        if returns_history and len(current_positions) > 0:
            self._check_correlation(ticker, current_positions, returns_history)

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def size_by_vol_target(
        self,
        ticker: str,
        price: float,
        nav: float,
        realised_vol_annual: float,
    ) -> float:
        """
        Return the dollar amount that makes this position contribute
        vol_target_annual to portfolio volatility (simplified single-asset).

        dollar_size = (vol_target / asset_vol) × NAV × size_multiplier

        The result is clipped to max_position_pct × NAV.
        """
        if realised_vol_annual <= 0.0 or nav <= 0.0:
            return 0.0
        raw = (self._vol_target_annual / realised_vol_annual) * nav
        raw *= self.size_multiplier   # halve if REDUCE_SIZES active
        return self.clip_to_max_position(raw, nav)

    def clip_to_max_position(
        self,
        dollar_amount: float,
        nav: float,
    ) -> float:
        """Clip dollar_amount so |position| ≤ max_position_pct × NAV."""
        if nav <= 0.0:
            return 0.0
        cap = self._max_position_pct * nav
        sign = 1.0 if dollar_amount >= 0.0 else -1.0
        return sign * min(abs(dollar_amount), cap)

    def apply_regime_leverage_cap(
        self,
        dollar_amount: float,
        nav: float,
        leverage_cap: float,
    ) -> float:
        """
        Scale down the trade if |dollar_amount| > leverage_cap × NAV.

        Also enforces the global max_leverage ceiling.
        """
        if nav <= 0.0 or leverage_cap <= 0.0:
            return 0.0
        effective_cap = min(leverage_cap, self._max_leverage)
        max_exposure = effective_cap * nav
        if abs(dollar_amount) > max_exposure:
            sign = 1.0 if dollar_amount >= 0.0 else -1.0
            return sign * max_exposure
        return dollar_amount

    def apply_size_reduction(
        self,
        dollar_amount: float,
        factor: float = 0.5,
    ) -> float:
        """Scale dollar_amount by factor (used manually or by size_multiplier)."""
        return dollar_amount * factor

    # ------------------------------------------------------------------
    # Drawdown monitoring
    # ------------------------------------------------------------------

    def update_peak(self, nav: float) -> None:
        """Update peak NAV; call after each valuation."""
        if nav > self._peak_nav:
            self._peak_nav = nav

    def current_drawdown(self, nav: float) -> float:
        """Return (peak − nav) / peak as a positive fraction.  Zero if at peak."""
        if self._peak_nav <= 0.0:
            return 0.0
        return max(0.0, (self._peak_nav - nav) / self._peak_nav)

    def portfolio_drawdown_breached(self, nav: float) -> bool:
        """Return True when drawdown exceeds the max_portfolio_drawdown_pct hard stop."""
        return self.current_drawdown(nav) >= self._max_portfolio_drawdown_pct

    def record_daily_nav(self, nav: float) -> None:
        """
        Record end-of-day NAV for the rolling 7-day loss computation.

        Call once per trading day after market close.
        """
        self._daily_nav_history.append(nav)
        if len(self._daily_nav_history) > 7:
            self._daily_nav_history.pop(0)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(
        self,
        nav: float,
        positions: Dict[str, float],
        returns: "pd.Series",
        timestamp: datetime,
    ) -> RiskSnapshot:
        """Build and return a RiskSnapshot for dashboarding / logging."""
        values = list(positions.values())
        gross_exp = sum(abs(v) for v in values)
        net_exp   = sum(values)
        largest_pct = (
            max(abs(v) / nav for v in values) if (nav > 0 and values) else 0.0
        )
        vol = (
            float(returns.std() * _ANNUALISE)
            if (not returns.empty and len(returns) > 1)
            else 0.0
        )
        daily_pnl = nav - self._session_start_nav if self._session_start_nav > 0 else 0.0

        return RiskSnapshot(
            timestamp=timestamp,
            nav=nav,
            peak_nav=self._peak_nav,
            daily_pnl=daily_pnl,
            drawdown_pct=self.current_drawdown(nav),
            portfolio_drawdown_pct=self.current_drawdown(nav),
            gross_exposure=gross_exp,
            net_exposure=net_exp,
            largest_position_pct=largest_pct,
            daily_vol_annualised=vol,
            circuit_breaker_active=(
                self._circuit_breaker_level >= CircuitBreakerLevel.HALT_DAY
            ),
            circuit_breaker_level=self._circuit_breaker_level,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_rolling_7d_loss(self, current_nav: float) -> float:
        """
        Compute the loss fraction over the rolling 7-day window.

        Uses the oldest entry in _daily_nav_history as the reference.
        Returns 0.0 when history is empty or reference NAV is zero.
        """
        if not self._daily_nav_history:
            return 0.0
        ref_nav = self._daily_nav_history[0]
        if ref_nav <= 0.0:
            return 0.0
        return max(0.0, (ref_nav - current_nav) / ref_nav)

    def _trigger_full_stop(
        self,
        nav: float,
        timestamp: datetime,
        reason: str,
    ) -> None:
        """Permanently halt the bot and write the lock file."""
        self._permanently_halted   = True
        self._circuit_breaker_level = CircuitBreakerLevel.FULL_STOP

        lock_path = self._lock_file_path
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            content = (
                "TRADING HALTED\n"
                "==============\n"
                f"Timestamp : {timestamp.isoformat()}\n"
                f"Reason    : {reason}\n"
                f"NAV       : ${nav:,.2f}\n"
                "\n"
                "Delete this file to allow the bot to restart.\n"
            )
            lock_path.write_text(content, encoding="utf-8")
            logger.critical(
                "FULL STOP triggered. Lock file written to %s. Reason: %s",
                lock_path, reason,
            )
        except OSError as exc:
            logger.error("Failed to write lock file %s: %s", lock_path, exc)

        self._fire_alert(f"FULL STOP: {reason}")

    def _check_correlation(
        self,
        ticker: str,
        current_positions: Dict[str, float],
        returns_history: Dict[str, "pd.Series"],
    ) -> None:
        """Raise RiskViolation if new ticker is >MAX_CORRELATION with any open pos."""
        if ticker not in returns_history:
            return
        new_ret = returns_history[ticker].dropna()
        for other, pos_val in current_positions.items():
            if other == ticker or pos_val == 0.0:
                continue
            if other not in returns_history:
                continue
            other_ret = returns_history[other].dropna()
            aligned = pd.concat([new_ret, other_ret], axis=1).dropna()
            if len(aligned) < 5:
                continue   # not enough data to compute correlation
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if math.isnan(corr):
                continue
            if abs(corr) > MAX_CORRELATION:
                raise RiskViolation(
                    f"Order rejected: |corr({ticker}, {other})| = {corr:.3f} "
                    f"exceeds {MAX_CORRELATION:.0%} threshold."
                )

    def _fire_alert(self, message: str) -> None:
        """Call the alert callback if one was registered."""
        if self._alert_callback is not None:
            try:
                self._alert_callback(message)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Alert callback raised: %s", exc)
