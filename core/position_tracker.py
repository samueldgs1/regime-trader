"""
Open position tracker and portfolio exposure calculator.

Maintains a live view of holdings, unrealised P&L, and sector/gross
exposure by polling Alpaca or reconciling fill events.

Sync strategy
-------------
* Call sync_from_broker() once at session open (or after connect()).
* For live trading, call start_background_sync() to keep state fresh
  every 60 seconds in a daemon thread.
* After every confirmed fill, call apply_fill() to update immediately
  without waiting for the next background poll.

Thread safety
-------------
All state mutations use a threading.Lock.  Reads (get_position, snapshot,
etc.) acquire the same lock so callers always see a consistent view.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Snapshot of a single open position."""
    ticker:              str
    qty:                 float          # signed (positive = long, negative = short)
    avg_entry_price:     float
    current_price:       float
    market_value:        float          # qty × current_price
    unrealised_pnl:      float          # market_value − cost_basis
    unrealised_pnl_pct:  float          # unrealised_pnl / cost_basis
    cost_basis:          float          # qty × avg_entry_price
    side:                str            # "long" | "short"
    opened_at:           Optional[datetime]


@dataclass
class PortfolioSnapshot:
    """Aggregate portfolio state at a point in time."""
    timestamp:        datetime
    cash:             float
    nav:              float             # cash + sum(market_values)
    positions:        Dict[str, Position]
    gross_exposure:   float
    net_exposure:     float
    long_exposure:    float
    short_exposure:   float
    unrealised_pnl:   float
    weights:          Dict[str, float]  # ticker → market_value / nav


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PositionTracker:
    """
    Tracks current holdings and derives portfolio-level statistics.

    Designed to be the single source of truth for position data within
    the bot.  Sync with Alpaca on startup and after every fill.
    """

    def __init__(self) -> None:
        self._positions:    Dict[str, Position] = {}
        self._cash:         float               = 0.0
        self._last_sync:    Optional[datetime]  = None
        self._pnl_history:  List[Dict]          = []   # realised P&L records
        self._lock:         threading.Lock      = threading.Lock()
        self._syncing:      bool                = False
        self._sync_thread:  Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Background sync
    # ------------------------------------------------------------------

    def start_background_sync(
        self,
        alpaca_client,
        interval_s: int = 60,
    ) -> None:
        """
        Start a daemon thread that calls sync_from_broker() every
        interval_s seconds (default 60).

        Safe to call multiple times; a second call stops the old thread
        and starts a fresh one.
        """
        self.stop_background_sync()   # stop any existing loop

        self._syncing = True

        def _loop() -> None:
            logger.info(
                "Position sync thread started (interval=%ds)", interval_s
            )
            while self._syncing:
                try:
                    self.sync_from_broker(alpaca_client)
                except Exception as exc:
                    logger.error("Background position sync failed: %s", exc)
                time.sleep(interval_s)

        self._sync_thread = threading.Thread(
            target=_loop, daemon=True, name="position-sync"
        )
        self._sync_thread.start()

    def stop_background_sync(self) -> None:
        """Signal the background thread to stop.  Returns immediately."""
        self._syncing = False

    # ------------------------------------------------------------------
    # Sync with broker
    # ------------------------------------------------------------------

    def sync_from_broker(self, alpaca_client) -> None:
        """
        Pull current positions and cash balance from Alpaca and update
        internal state.

        Parameters
        ----------
        alpaca_client:
            A connected ``alpaca.trading.client.TradingClient`` instance
            (or a MagicMock for testing).
        """
        raw_positions = alpaca_client.get_all_positions()
        account       = alpaca_client.get_account()

        new_positions: Dict[str, Position] = {}
        for p in raw_positions:
            ticker  = str(p.symbol)
            qty     = float(p.qty)
            entry   = float(p.avg_entry_price)
            cur     = float(p.current_price)
            mv      = float(p.market_value)
            cb      = float(p.cost_basis)
            upnl    = float(p.unrealized_pl)
            upnl_pc = float(p.unrealized_plpc)
            side_v  = (
                p.side.value if hasattr(p.side, "value") else str(p.side)
            )

            new_positions[ticker] = Position(
                ticker=ticker,
                qty=qty,
                avg_entry_price=entry,
                current_price=cur,
                market_value=mv,
                unrealised_pnl=upnl,
                unrealised_pnl_pct=upnl_pc,
                cost_basis=cb,
                side=side_v,
                opened_at=self._positions.get(ticker, Position(
                    ticker, 0, 0, 0, 0, 0, 0, 0, "long", None
                )).opened_at,
            )

        try:
            cash = float(account.cash)
        except (TypeError, AttributeError, ValueError):
            cash = self._cash

        with self._lock:
            self._positions = new_positions
            self._cash      = cash
            self._last_sync = datetime.now(tz=timezone.utc)

        logger.debug(
            "Positions synced: %d open  cash=$%.2f  last_sync=%s",
            len(new_positions), cash, self._last_sync,
        )

    def apply_fill(
        self,
        ticker:       str,
        filled_qty:   float,
        filled_price: float,
        side:         str,
        timestamp:    datetime,
    ) -> None:
        """
        Update position state immediately after an order fill.

        Parameters
        ----------
        ticker:
            Symbol filled.
        filled_qty:
            Unsigned share quantity filled.
        filled_price:
            Average fill price.
        side:
            'buy' or 'sell'.
        timestamp:
            Fill timestamp from the broker.
        """
        with self._lock:
            if side == "buy":
                if ticker in self._positions:
                    existing = self._positions[ticker]
                    new_avg = self._update_average_cost(
                        existing, filled_qty, filled_price
                    )
                    new_qty   = existing.qty + filled_qty
                    new_cb    = new_qty * new_avg
                    new_mv    = new_qty * existing.current_price
                    new_upnl  = new_mv - new_cb
                    new_uppc  = (new_upnl / new_cb) if new_cb != 0.0 else 0.0
                    self._positions[ticker] = Position(
                        ticker=ticker,
                        qty=new_qty,
                        avg_entry_price=new_avg,
                        current_price=existing.current_price,
                        market_value=new_mv,
                        unrealised_pnl=new_upnl,
                        unrealised_pnl_pct=new_uppc,
                        cost_basis=new_cb,
                        side="long" if new_qty > 0 else "short",
                        opened_at=existing.opened_at,
                    )
                else:
                    self._positions[ticker] = self._new_position(
                        ticker, filled_qty, filled_price, timestamp
                    )

            elif side == "sell":
                if ticker in self._positions:
                    existing  = self._positions[ticker]
                    new_qty   = existing.qty - filled_qty
                    # Record realised P&L
                    realised  = (filled_price - existing.avg_entry_price) * filled_qty
                    self._pnl_history.append({
                        "timestamp": timestamp,
                        "ticker":    ticker,
                        "pnl":       realised,
                    })
                    if abs(new_qty) < 1e-9:
                        del self._positions[ticker]
                    else:
                        new_cb   = new_qty * existing.avg_entry_price
                        new_mv   = new_qty * existing.current_price
                        new_upnl = new_mv - new_cb
                        new_uppc = (new_upnl / new_cb) if new_cb != 0.0 else 0.0
                        self._positions[ticker] = Position(
                            ticker=ticker,
                            qty=new_qty,
                            avg_entry_price=existing.avg_entry_price,
                            current_price=existing.current_price,
                            market_value=new_mv,
                            unrealised_pnl=new_upnl,
                            unrealised_pnl_pct=new_uppc,
                            cost_basis=new_cb,
                            side="long" if new_qty > 0 else "short",
                            opened_at=existing.opened_at,
                        )

    # ------------------------------------------------------------------
    # Price updates
    # ------------------------------------------------------------------

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Refresh current_price, market_value, and unrealised_pnl for all
        positions in prices.

        Parameters
        ----------
        prices:
            Map of ticker → latest price.
        """
        with self._lock:
            for ticker, price in prices.items():
                if ticker not in self._positions or price <= 0:
                    continue
                pos      = self._positions[ticker]
                new_mv   = pos.qty * price
                new_upnl = new_mv - pos.cost_basis
                new_uppc = (new_upnl / pos.cost_basis) if pos.cost_basis != 0.0 else 0.0
                self._positions[ticker] = Position(
                    ticker=ticker,
                    qty=pos.qty,
                    avg_entry_price=pos.avg_entry_price,
                    current_price=price,
                    market_value=new_mv,
                    unrealised_pnl=new_upnl,
                    unrealised_pnl_pct=new_uppc,
                    cost_basis=pos.cost_basis,
                    side=pos.side,
                    opened_at=pos.opened_at,
                )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_position(self, ticker: str) -> Optional[Position]:
        """Return the Position for ticker, or None if flat."""
        with self._lock:
            return self._positions.get(ticker)

    def all_positions(self) -> Dict[str, Position]:
        """Return a shallow copy of the internal positions dict."""
        with self._lock:
            return dict(self._positions)

    def snapshot(self, cash: Optional[float] = None) -> PortfolioSnapshot:
        """Build a PortfolioSnapshot from current state."""
        with self._lock:
            cash_val = cash if cash is not None else self._cash
            positions_copy = dict(self._positions)

        values        = [p.market_value for p in positions_copy.values()]
        long_exp      = sum(v for v in values if v > 0)
        short_exp     = sum(v for v in values if v < 0)
        gross_exp     = sum(abs(v) for v in values)
        net_exp       = sum(values)
        total_upnl    = sum(p.unrealised_pnl for p in positions_copy.values())
        nav           = cash_val + sum(values)

        weights: Dict[str, float] = {}
        if nav > 0:
            weights = {
                t: p.market_value / nav
                for t, p in positions_copy.items()
            }

        return PortfolioSnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            cash=cash_val,
            nav=nav,
            positions=positions_copy,
            gross_exposure=gross_exp,
            net_exposure=net_exp,
            long_exposure=long_exp,
            short_exposure=short_exp,
            unrealised_pnl=total_upnl,
            weights=weights,
        )

    def current_weights(self, nav: float) -> Dict[str, float]:
        """Return ticker → (market_value / nav) for all open positions."""
        if nav <= 0:
            return {}
        with self._lock:
            return {
                t: p.market_value / nav
                for t, p in self._positions.items()
            }

    def is_flat(self, ticker: str) -> bool:
        """Return True if there is no open position in ticker."""
        with self._lock:
            pos = self._positions.get(ticker)
            return pos is None or abs(pos.qty) < 1e-9

    def net_exposure(self) -> float:
        """Sum of signed market values across all open positions."""
        with self._lock:
            return sum(p.market_value for p in self._positions.values())

    def gross_exposure(self) -> float:
        """Sum of absolute market values across all open positions."""
        with self._lock:
            return sum(abs(p.market_value) for p in self._positions.values())

    def position_values(self) -> Dict[str, float]:
        """Return ticker → market_value for all open positions."""
        with self._lock:
            return {t: p.market_value for t, p in self._positions.items()}

    def unrealised_pnl_by_ticker(self) -> Dict[str, float]:
        """Return ticker → unrealised_pnl for all open positions."""
        with self._lock:
            return {t: p.unrealised_pnl for t, p in self._positions.items()}

    # ------------------------------------------------------------------
    # P&L history
    # ------------------------------------------------------------------

    def pnl_series(self) -> pd.Series:
        """
        Return a time-indexed Series of daily realised P&L.
        Populated as positions are closed via apply_fill().
        """
        if not self._pnl_history:
            return pd.Series(dtype=float)

        df = pd.DataFrame(self._pnl_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        return df["pnl"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_position(
        self,
        ticker:    str,
        qty:       float,
        price:     float,
        timestamp: datetime,
    ) -> Position:
        """Create a Position dataclass for a fresh entry."""
        cost_basis = qty * price
        return Position(
            ticker=ticker,
            qty=qty,
            avg_entry_price=price,
            current_price=price,
            market_value=cost_basis,
            unrealised_pnl=0.0,
            unrealised_pnl_pct=0.0,
            cost_basis=cost_basis,
            side="long" if qty > 0 else "short",
            opened_at=timestamp,
        )

    def _update_average_cost(
        self,
        existing:       Position,
        additional_qty: float,
        fill_price:     float,
    ) -> float:
        """
        Return the new weighted-average entry price after adding to a
        position.

        new_avg = (old_cost_basis + additional_qty × fill_price)
                  / (old_qty + additional_qty)
        """
        total_qty  = existing.qty + additional_qty
        if abs(total_qty) < 1e-12:
            return fill_price
        total_cost = existing.cost_basis + additional_qty * fill_price
        return total_cost / total_qty
