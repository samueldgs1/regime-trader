"""
Alpaca order execution layer.

Architecture
------------
OrderExecutor wraps the alpaca-py TradingClient to place, modify, and
cancel orders.  All methods are synchronous; the caller is responsible
for running them off the main thread if needed.

Credential policy
-----------------
Production code MUST use the ``from_env()`` class method, which loads
ALPACA_API_KEY, ALPACA_SECRET_KEY, and ALPACA_BASE_URL exclusively from
a .env file.  Raw API keys must never appear in source code or be passed
as arguments in application logic.

The constructor accepts api_key / secret_key only to allow unit tests
to inject a MagicMock client without touching the filesystem.

Safety checks
-------------
* paper=True + live URL → ValueError at construction time.
* All orders retry on HTTP 429 / 503 with exponential back-off.
* Consecutive failures ≥ _REPEATED_FAILURE_THRESHOLD fire an alert.
* cancel_all_orders() is the emergency stop used by the circuit breaker.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry / safety constants
# ---------------------------------------------------------------------------

_MAX_RETRIES:    int   = 3
_RETRY_BACKOFF_S       = (1.0, 2.0, 4.0)          # exponential back-off
_RETRYABLE_CODES       = frozenset({"429", "503"}) # HTTP status codes
_REPEATED_FAILURE_THRESHOLD: int = 3

_LIVE_DOMAINS  = ("api.alpaca.markets",)
_PAPER_MARKER  = "paper-api"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    BUY  = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET     = "market"
    LIMIT      = "limit"
    STOP       = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class OrderStatus(str, Enum):
    PENDING          = "pending_new"
    NEW              = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED           = "filled"
    CANCELLED        = "canceled"
    REJECTED         = "rejected"
    EXPIRED          = "expired"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enum_val(v) -> str:
    """Return the string value of v whether it is an Enum or a plain str."""
    return v.value if isinstance(v, Enum) else str(v)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """Lightweight representation of an Alpaca order."""
    id:                str
    client_order_id:   str
    ticker:            str
    side:              OrderSide
    order_type:        OrderType
    qty:               Optional[float]
    notional:          Optional[float]     # dollar-denominated order size
    limit_price:       Optional[float]
    stop_price:        Optional[float]
    time_in_force:     TimeInForce
    status:            OrderStatus
    filled_qty:        float
    filled_avg_price:  Optional[float]
    submitted_at:      datetime
    filled_at:         Optional[datetime]


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class AlpacaAPIError(Exception):
    """Raised when the Alpaca API returns a non-retryable error."""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OrderExecutor:
    """
    Places and manages orders through the Alpaca brokerage API.

    Production entrypoint
    ---------------------
    Always construct via ``OrderExecutor.from_env()`` which reads credentials
    exclusively from a .env file.

    Testing
    -------
    The constructor accepts credentials directly and exposes ``_client`` so
    tests can inject ``MagicMock()``.

    Usage
    -----
    ::

        executor = OrderExecutor.from_env()
        executor.connect()
        if executor.is_market_open():
            order = executor.market_order("SPY", 5_000, OrderSide.BUY)
    """

    def __init__(
        self,
        api_key:        str,
        secret_key:     str,
        base_url:       str,
        paper:          bool = True,
        alert_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        api_key / secret_key:
            Alpaca credentials.  In production use from_env() instead.
        base_url:
            REST endpoint URL.
        paper:
            Safety flag — if True the executor refuses to use a live URL.
        alert_callback:
            Optional callable fired on repeated order failures.
        """
        _is_live_url = (
            any(d in base_url for d in _LIVE_DOMAINS)
            and _PAPER_MARKER not in base_url
        )
        if paper and _is_live_url:
            raise ValueError(
                f"paper=True but base_url '{base_url}' points to a live endpoint. "
                "Use the paper trading URL (https://paper-api.alpaca.markets) "
                "or set paper=False explicitly for live trading."
            )

        self._api_key    = api_key
        self._secret_key = secret_key
        self._base_url   = base_url
        self._paper      = paper
        self._alert_cb   = alert_callback
        self._client     = None       # set by connect() or replaced by tests
        self._fail_count = 0          # consecutive submission failures

    # ------------------------------------------------------------------
    # Production factory — loads credentials from .env
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        alert_callback: Optional[Callable[[str], None]] = None,
    ) -> "OrderExecutor":
        """
        Create an OrderExecutor from environment variables.

        Required in .env
        ----------------
        ALPACA_API_KEY, ALPACA_SECRET_KEY

        Optional in .env
        ----------------
        ALPACA_BASE_URL  (default: https://paper-api.alpaca.markets)
        PAPER            (default: true)

        Raises
        ------
        EnvironmentError if required credentials are missing.
        """
        from dotenv import load_dotenv
        load_dotenv()

        api_key    = os.getenv("ALPACA_API_KEY",    "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        base_url   = os.getenv("ALPACA_BASE_URL",   "https://paper-api.alpaca.markets")
        paper      = os.getenv("PAPER", "true").lower() == "true"

        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env.  "
                "Copy .env.example to .env and fill in your Alpaca credentials."
            )

        return cls(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url,
            paper=paper,
            alert_callback=alert_callback,
        )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Initialise the Alpaca TradingClient and verify connectivity.

        Sets self._client and logs the account status + buying power.
        Call once after constructing the executor.
        """
        from alpaca.trading.client import TradingClient

        self._client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self._paper,
        )
        account = self._client.get_account()
        logger.info(
            "Connected to Alpaca.  status=%s  buying_power=$%.2f  equity=$%.2f",
            account.status,
            float(account.buying_power),
            float(account.equity),
        )

    # ------------------------------------------------------------------
    # Market status / account
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        """Return True if the US equity market is currently open."""
        clock = self._client.get_clock()
        return bool(clock.is_open)

    def get_clock(self) -> Dict:
        """Return market clock (is_open, next_open, next_close, timestamp)."""
        clock = self._client.get_clock()
        return {
            "is_open":    bool(clock.is_open),
            "next_open":  clock.next_open,
            "next_close": clock.next_close,
            "timestamp":  clock.timestamp,
        }

    def get_account(self) -> Dict:
        """Return a dict summary of the Alpaca account."""
        a = self._client.get_account()
        return {
            "id":                 str(a.id),
            "status":             str(a.status),
            "currency":           a.currency,
            "buying_power":       float(a.buying_power),
            "equity":             float(a.equity),
            "cash":               float(a.cash),
            "portfolio_value":    float(a.portfolio_value),
            "pattern_day_trader": a.pattern_day_trader,
            "trading_blocked":    a.trading_blocked,
        }

    def get_positions(self) -> List[Dict]:
        """Return current open positions as a list of dicts."""
        raw_positions = self._client.get_all_positions()
        result: List[Dict] = []
        for p in raw_positions:
            side_val = (
                p.side.value if hasattr(p.side, "value") else str(p.side)
            )
            result.append({
                "symbol":          p.symbol,
                "qty":             float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "market_value":    float(p.market_value),
                "unrealized_pl":   float(p.unrealized_pl),
                "current_price":   float(p.current_price),
                "side":            side_val,
            })
        return result

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def market_order(
        self,
        ticker:        str,
        notional:      float,
        side:          OrderSide,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> Order:
        """
        Place a notional market order (dollar amount, not shares).

        Parameters
        ----------
        ticker:
            Equity symbol.
        notional:
            Unsigned dollar amount.
        side:
            OrderSide.BUY or OrderSide.SELL.
        time_in_force:
            Default DAY.
        """
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums    import OrderSide as ASide, TimeInForce as ATIF

        req = MarketOrderRequest(
            symbol=ticker,
            notional=round(notional, 2),
            side=ASide(_enum_val(side)),
            time_in_force=ATIF(_enum_val(time_in_force)),
        )
        raw = self._submit_with_retry(req)
        return self._parse_order(raw)

    def limit_order(
        self,
        ticker:        str,
        qty:           float,
        side:          OrderSide,
        limit_price:   float,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> Order:
        """Place a share-quantity limit order."""
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums    import OrderSide as ASide, TimeInForce as ATIF

        req = LimitOrderRequest(
            symbol=ticker,
            qty=qty,
            side=ASide(_enum_val(side)),
            limit_price=round(limit_price, 2),
            time_in_force=ATIF(time_in_force.value),
        )
        raw = self._submit_with_retry(req)
        return self._parse_order(raw)

    def stop_order(
        self,
        ticker:        str,
        qty:           float,
        side:          OrderSide,
        stop_price:    float,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> Order:
        """Place a stop order."""
        from alpaca.trading.requests import StopOrderRequest
        from alpaca.trading.enums    import OrderSide as ASide, TimeInForce as ATIF

        req = StopOrderRequest(
            symbol=ticker,
            qty=qty,
            side=ASide(_enum_val(side)),
            stop_price=round(stop_price, 2),
            time_in_force=ATIF(time_in_force.value),
        )
        raw = self._submit_with_retry(req)
        return self._parse_order(raw)

    def bracket_order(
        self,
        ticker:            str,
        qty:               float,
        side:              OrderSide,
        take_profit_price: float,
        stop_loss_price:   float,
    ) -> Order:
        """
        Place an OCO bracket order (entry + take-profit + stop-loss) as one
        submission.
        """
        from alpaca.trading.requests import (
            MarketOrderRequest,
            TakeProfitRequest,
            StopLossRequest,
        )
        from alpaca.trading.enums import (
            OrderSide   as ASide,
            TimeInForce as ATIF,
            OrderClass,
        )

        req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=ASide(_enum_val(side)),
            time_in_force=ATIF.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
            stop_loss=StopLossRequest(stop_price=round(stop_loss_price, 2)),
        )
        raw = self._submit_with_retry(req)
        return self._parse_order(raw)

    def submit_order(
        self,
        ticker:        str,
        dollar_amount: float,
        side:          OrderSide,
    ) -> Order:
        """
        Convenience wrapper: notional market order.

        Used by the backtester / strategy layer which works in dollar amounts.
        """
        return self.market_order(ticker, abs(dollar_amount), side)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a specific order by ID.

        Returns
        -------
        True if cancelled; False if already in a terminal state or the
        API call failed.
        """
        try:
            self._client.cancel_order_by_id(order_id)
            logger.info("Cancelled order %s", order_id)
            return True
        except Exception as exc:
            logger.warning("Failed to cancel order %s: %s", order_id, exc)
            return False

    def cancel_all_orders(self) -> List[str]:
        """
        Cancel every open order.  Used by the circuit breaker emergency stop.

        Returns
        -------
        List of cancelled order IDs.
        """
        cancelled: List[str] = []
        for order in self.list_open_orders():
            if self.cancel_order(order.id):
                cancelled.append(order.id)
        return cancelled

    def modify_stop(self, order_id: str, new_stop_price: float) -> Order:
        """Modify the stop price of an existing stop or stop-limit order."""
        from alpaca.trading.requests import ReplaceOrderRequest

        req = ReplaceOrderRequest(stop_price=round(new_stop_price, 2))
        raw = self._client.replace_order_by_id(order_id, req)
        return self._parse_order(raw)

    def replace_order(
        self,
        order_id:    str,
        qty:         Optional[float] = None,
        limit_price: Optional[float] = None,
    ) -> Order:
        """Modify qty or limit_price of an open, not-yet-filled order."""
        from alpaca.trading.requests import ReplaceOrderRequest

        req = ReplaceOrderRequest(
            qty=qty,
            limit_price=round(limit_price, 2) if limit_price is not None else None,
        )
        raw = self._client.replace_order_by_id(order_id, req)
        return self._parse_order(raw)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_order(self, order_id: str) -> Order:
        """Fetch the latest state of a single order."""
        raw = self._client.get_order_by_id(order_id)
        return self._parse_order(raw)

    def list_open_orders(self) -> List[Order]:
        """Return all currently open (non-terminal) orders."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums    import QueryOrderStatus

        req  = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        raws = self._client.get_orders(filter=req)
        return [self._parse_order(r) for r in (raws or [])]

    def list_recent_orders(self, limit: int = 50) -> List[Order]:
        """Return the most recent orders regardless of status."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums    import QueryOrderStatus

        req  = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
        raws = self._client.get_orders(filter=req)
        return [self._parse_order(r) for r in (raws or [])]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _notional_to_qty(self, ticker: str, notional: float, price: float) -> float:
        """
        Convert a dollar notional to share quantity.

        Parameters
        ----------
        ticker:
            Unused — reserved for lot-size / min-increment logic.
        notional:
            Dollar amount to convert.
        price:
            Current ask/last price in dollars.

        Returns
        -------
        Positive float shares.

        Raises
        ------
        ValueError if price ≤ 0.
        """
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price!r}")
        return notional / price

    def _parse_order(self, raw) -> Order:
        """
        Map a raw API response (dict or Alpaca Pydantic model) to Order.

        Accepts
        -------
        * dict  — raw Alpaca REST response
        * Alpaca ``alpaca.trading.models.Order`` Pydantic object
        * MagicMock with attribute access (for tests that set attributes
          explicitly — see test helpers in tests/test_orders.py)
        """
        if isinstance(raw, dict):
            def _get(key: str, default=None):
                return raw.get(key, default)
        else:
            def _get(key: str, default=None):
                return getattr(raw, key, default)

        def _float(v) -> Optional[float]:
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        def _dt(v) -> Optional[datetime]:
            if v is None:
                return None
            if isinstance(v, datetime):
                return v
            try:
                return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                return None

        def _enum(cls, val, default):
            if val is None:
                return default
            try:
                return cls(str(val))
            except ValueError:
                return default

        # Alpaca REST dict uses "type"; Pydantic model exposes "order_type"
        order_type_raw = _get("type") or _get("order_type")

        return Order(
            id               = str(_get("id", "")),
            client_order_id  = str(_get("client_order_id", "")),
            ticker           = str(_get("symbol", "")),
            side             = _enum(OrderSide,    _get("side"),          OrderSide.BUY),
            order_type       = _enum(OrderType,    order_type_raw,        OrderType.MARKET),
            qty              = _float(_get("qty")),
            notional         = _float(_get("notional")),
            limit_price      = _float(_get("limit_price")),
            stop_price       = _float(_get("stop_price")),
            time_in_force    = _enum(TimeInForce,  _get("time_in_force"), TimeInForce.DAY),
            status           = _enum(OrderStatus,  _get("status"),        OrderStatus.NEW),
            filled_qty       = float(_get("filled_qty") or 0.0),
            filled_avg_price = _float(_get("filled_avg_price")),
            submitted_at     = _dt(_get("submitted_at")) or datetime.now(tz=timezone.utc),
            filled_at        = _dt(_get("filled_at")),
        )

    def _submit_with_retry(self, request) -> object:
        """
        Submit an order with exponential back-off on HTTP 429 and 503.

        Logs every error.  Fires the alert callback after
        _REPEATED_FAILURE_THRESHOLD consecutive failures.

        Raises
        ------
        The original exception if non-retryable or all retries exhausted.
        """
        last_exc: Optional[Exception] = None

        for attempt, backoff in enumerate(_RETRY_BACKOFF_S):
            try:
                result = self._client.submit_order(order_data=request)
                self._fail_count = 0          # success — reset counter
                return result
            except Exception as exc:
                exc_str  = str(exc)
                retryable = any(code in exc_str for code in _RETRYABLE_CODES)
                logger.error(
                    "Order submission error (attempt %d/%d, retryable=%s): %s",
                    attempt + 1, _MAX_RETRIES, retryable, exc,
                )
                self._fail_count += 1

                if self._fail_count >= _REPEATED_FAILURE_THRESHOLD:
                    msg = (
                        f"Repeated order failure #{self._fail_count}: {exc}.  "
                        "Manual inspection required."
                    )
                    logger.critical(msg)
                    if self._alert_cb:
                        try:
                            self._alert_cb(msg)
                        except Exception:
                            pass

                if retryable and attempt < len(_RETRY_BACKOFF_S) - 1:
                    time.sleep(backoff)
                    last_exc = exc
                    continue

                raise  # non-retryable or final attempt

        raise RuntimeError(
            f"Order failed after {_MAX_RETRIES} retries"
        ) from last_exc
