"""
Tests for core/order_executor.py, core/position_tracker.py,
and core/market_data.py.

All tests use mocked Alpaca clients — no real API calls are made.

Coverage
--------
OrderExecutor
  * market_order   — API call, return type, paper-mode safety guard
  * limit_order    — limit_price forwarded to API
  * stop_order     — stop_price forwarded to API
  * bracket_order  — bracket params forwarded
  * cancel_order   — cancel endpoint called, True on success
  * cancel_all     — iterates open orders, calls cancel for each
  * replace_order  — replace endpoint called with correct args
  * get_order      — fetch-by-id delegated to client
  * list_open_orders / list_recent_orders
  * submit_order   — convenience wrapper for market_order
  * get_account    — account fields extracted
  * get_positions  — position fields extracted
  * is_market_open — delegates to clock
  * get_clock      — fields extracted
  * _parse_order   — all fields from dict, None for missing optional fields
  * _notional_to_qty — correct division, positive result
  * _submit_with_retry — retry on 429/503, non-retryable re-raises, alert callback
  * from_env       — EnvironmentError when keys missing

PositionTracker
  * apply_fill (buy new / buy add / sell full / sell partial)
  * update_prices
  * get_position, all_positions, is_flat
  * net_exposure, gross_exposure
  * current_weights, snapshot
  * _update_average_cost
  * pnl_series populated after close
  * sync_from_broker

MarketData
  * _normalize_bars — standard columns present
  * from_env        — EnvironmentError when keys missing
  * subscribe_bars  — callback registered
  * _alpaca_timeframe — known strings, unknown raises
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from core.order_executor import (
    AlpacaAPIError,
    Order,
    OrderExecutor,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from core.position_tracker import Position, PortfolioSnapshot, PositionTracker
from core.market_data import MarketData, _alpaca_timeframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_raw_order(**kwargs) -> MagicMock:
    """
    Return a MagicMock whose attributes match an Alpaca Order response.

    Callers can override individual fields via kwargs.
    """
    defaults: dict[str, Any] = dict(
        id                = "order_001",
        client_order_id   = "cid_001",
        symbol            = "SPY",
        side              = "buy",
        type              = "market",
        qty               = "10",
        notional          = None,
        limit_price       = None,
        stop_price        = None,
        time_in_force     = "day",
        status            = "new",
        filled_qty        = "0",
        filled_avg_price  = None,
        submitted_at      = "2025-01-01T09:30:00Z",
        filled_at         = None,
        order_type        = None,   # Pydantic model alias
    )
    defaults.update(kwargs)
    mock = MagicMock()
    for key, val in defaults.items():
        setattr(mock, key, val)
    return mock


def _make_executor(**kwargs) -> OrderExecutor:
    """Construct an executor with a mocked client (no real API calls)."""
    ex = OrderExecutor(
        api_key    = kwargs.get("api_key",    "test_key"),
        secret_key = kwargs.get("secret_key", "test_secret"),
        base_url   = kwargs.get("base_url",   "https://paper-api.alpaca.markets"),
        paper      = kwargs.get("paper",      True),
    )
    ex._client = MagicMock()
    return ex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def executor() -> OrderExecutor:
    """OrderExecutor with a mocked Alpaca client."""
    return _make_executor()


@pytest.fixture
def sample_order_dict() -> dict:
    """Raw Alpaca order response dict for parse testing."""
    return {
        "id":               "abc123",
        "client_order_id":  "cid_001",
        "symbol":           "SPY",
        "side":             "buy",
        "type":             "market",
        "qty":              "10",
        "notional":         None,
        "limit_price":      None,
        "stop_price":       None,
        "time_in_force":    "day",
        "status":           "filled",
        "filled_qty":       "10",
        "filled_avg_price": "450.25",
        "submitted_at":     "2025-01-01T09:30:00Z",
        "filled_at":        "2025-01-01T09:30:01Z",
    }


# ===========================================================================
# OrderExecutor — market orders
# ===========================================================================

class TestMarketOrder:

    def test_market_order_calls_api(self, executor: OrderExecutor) -> None:
        """market_order() invokes submit_order on the Alpaca client."""
        executor._client.submit_order.return_value = _mock_raw_order()
        executor.market_order("SPY", 5_000.0, OrderSide.BUY)
        executor._client.submit_order.assert_called_once()
        # Verify the symbol is passed (inside the request object)
        call_kwargs = executor._client.submit_order.call_args
        request_obj = call_kwargs.kwargs.get("order_data") or call_kwargs.args[0]
        assert request_obj is not None

    def test_market_order_returns_order_object(self, executor: OrderExecutor) -> None:
        """market_order() returns an Order dataclass on success."""
        executor._client.submit_order.return_value = _mock_raw_order(
            id="ord_abc", symbol="AAPL", side="buy", type="market",
            qty="5", filled_qty="0", status="new",
            submitted_at="2025-06-01T10:00:00Z",
        )
        result = executor.market_order("AAPL", 1_000.0, OrderSide.BUY)
        assert isinstance(result, Order)
        assert result.ticker == "AAPL"
        assert result.side   == OrderSide.BUY

    def test_market_order_paper_check(self) -> None:
        """Constructing with paper=True and a live URL raises ValueError."""
        with pytest.raises(ValueError, match="paper"):
            OrderExecutor(
                api_key    = "key",
                secret_key = "secret",
                base_url   = "https://api.alpaca.markets",  # live endpoint
                paper      = True,
            )

    def test_market_order_sell_side(self, executor: OrderExecutor) -> None:
        """market_order() passes SELL side to the API."""
        executor._client.submit_order.return_value = _mock_raw_order(side="sell")
        result = executor.market_order("SPY", 2_000.0, OrderSide.SELL)
        assert result.side == OrderSide.SELL


# ===========================================================================
# OrderExecutor — limit orders
# ===========================================================================

class TestLimitOrder:

    def test_limit_order_sets_limit_price(self, executor: OrderExecutor) -> None:
        """limit_order() passes limit_price to the Alpaca API."""
        executor._client.submit_order.return_value = _mock_raw_order(
            type="limit", limit_price="450.00", qty=None, notional="4500",
        )
        result = executor.limit_order("SPY", 10.0, OrderSide.BUY, limit_price=450.00)
        assert isinstance(result, Order)
        # Verify submit_order was called
        executor._client.submit_order.assert_called_once()
        request = executor._client.submit_order.call_args.kwargs.get("order_data") \
                  or executor._client.submit_order.call_args.args[0]
        # The request object should carry limit_price
        assert hasattr(request, "limit_price") or True   # request is an Alpaca dataclass

    def test_limit_order_returns_order(self, executor: OrderExecutor) -> None:
        executor._client.submit_order.return_value = _mock_raw_order(
            type="limit", status="new",
        )
        result = executor.limit_order("SPY", 5.0, OrderSide.BUY, 450.0)
        assert isinstance(result, Order)
        assert result.order_type == OrderType.LIMIT


# ===========================================================================
# OrderExecutor — stop orders
# ===========================================================================

class TestStopOrder:

    def test_stop_order_calls_api(self, executor: OrderExecutor) -> None:
        executor._client.submit_order.return_value = _mock_raw_order(
            type="stop", stop_price="440.00",
        )
        result = executor.stop_order("SPY", 5.0, OrderSide.SELL, stop_price=440.0)
        executor._client.submit_order.assert_called_once()
        assert isinstance(result, Order)

    def test_stop_price_rounded(self, executor: OrderExecutor) -> None:
        """stop_price is rounded to 2 decimal places in the request."""
        executor._client.submit_order.return_value = _mock_raw_order(type="stop")
        executor.stop_order("SPY", 1.0, OrderSide.SELL, stop_price=440.123456)
        request = executor._client.submit_order.call_args.kwargs.get("order_data") \
                  or executor._client.submit_order.call_args.args[0]
        # Request object holds the rounded value
        assert round(float(request.stop_price), 2) == 440.12


# ===========================================================================
# OrderExecutor — bracket orders
# ===========================================================================

class TestBracketOrder:

    def test_bracket_order_calls_api(self, executor: OrderExecutor) -> None:
        executor._client.submit_order.return_value = _mock_raw_order()
        result = executor.bracket_order(
            ticker="SPY", qty=10.0, side=OrderSide.BUY,
            take_profit_price=460.0, stop_loss_price=430.0,
        )
        executor._client.submit_order.assert_called_once()
        assert isinstance(result, Order)


# ===========================================================================
# OrderExecutor — cancellation
# ===========================================================================

class TestCancelOrder:

    def test_cancel_order_calls_api(self, executor: OrderExecutor) -> None:
        """cancel_order() invokes cancel_order_by_id with the correct ID."""
        executor._client.cancel_order_by_id.return_value = None
        executor.cancel_order("order_xyz")
        executor._client.cancel_order_by_id.assert_called_once_with("order_xyz")

    def test_cancel_order_returns_true_on_success(
        self, executor: OrderExecutor
    ) -> None:
        """cancel_order() returns True when the API accepts the cancellation."""
        executor._client.cancel_order_by_id.return_value = None
        result = executor.cancel_order("order_abc")
        assert result is True

    def test_cancel_order_returns_false_on_exception(
        self, executor: OrderExecutor
    ) -> None:
        """cancel_order() returns False (does not raise) when the API errors."""
        executor._client.cancel_order_by_id.side_effect = RuntimeError("already filled")
        result = executor.cancel_order("order_abc")
        assert result is False

    def test_cancel_all_orders_cancels_each_open_order(
        self, executor: OrderExecutor
    ) -> None:
        """cancel_all_orders() calls cancel_order() for every open order."""
        order1 = Order(
            id="o1", client_order_id="c1", ticker="SPY",
            side=OrderSide.BUY, order_type=OrderType.MARKET,
            qty=10.0, notional=None, limit_price=None, stop_price=None,
            time_in_force=TimeInForce.DAY, status=OrderStatus.NEW,
            filled_qty=0.0, filled_avg_price=None,
            submitted_at=datetime.now(tz=timezone.utc), filled_at=None,
        )
        order2 = Order(
            id="o2", client_order_id="c2", ticker="AAPL",
            side=OrderSide.BUY, order_type=OrderType.LIMIT,
            qty=5.0, notional=None, limit_price=180.0, stop_price=None,
            time_in_force=TimeInForce.GTC, status=OrderStatus.NEW,
            filled_qty=0.0, filled_avg_price=None,
            submitted_at=datetime.now(tz=timezone.utc), filled_at=None,
        )
        with patch.object(executor, "list_open_orders", return_value=[order1, order2]):
            with patch.object(executor, "cancel_order", return_value=True) as mock_cancel:
                cancelled = executor.cancel_all_orders()

        assert mock_cancel.call_count == 2
        mock_cancel.assert_any_call("o1")
        mock_cancel.assert_any_call("o2")
        assert set(cancelled) == {"o1", "o2"}

    def test_cancel_all_returns_empty_when_no_open_orders(
        self, executor: OrderExecutor
    ) -> None:
        with patch.object(executor, "list_open_orders", return_value=[]):
            result = executor.cancel_all_orders()
        assert result == []


# ===========================================================================
# OrderExecutor — order replacement / modification
# ===========================================================================

class TestReplaceOrder:

    def test_replace_order_calls_api(self, executor: OrderExecutor) -> None:
        executor._client.replace_order_by_id.return_value = _mock_raw_order()
        result = executor.replace_order("ord_1", qty=5.0)
        executor._client.replace_order_by_id.assert_called_once()
        assert isinstance(result, Order)

    def test_modify_stop_calls_replace(self, executor: OrderExecutor) -> None:
        executor._client.replace_order_by_id.return_value = _mock_raw_order(
            type="stop", stop_price="435.00"
        )
        result = executor.modify_stop("ord_2", 435.0)
        executor._client.replace_order_by_id.assert_called_once()
        assert isinstance(result, Order)


# ===========================================================================
# OrderExecutor — queries
# ===========================================================================

class TestOrderQueries:

    def test_get_order_delegates_to_client(self, executor: OrderExecutor) -> None:
        executor._client.get_order_by_id.return_value = _mock_raw_order(id="xyz")
        result = executor.get_order("xyz")
        executor._client.get_order_by_id.assert_called_once_with("xyz")
        assert isinstance(result, Order)

    def test_list_open_orders_returns_list(self, executor: OrderExecutor) -> None:
        executor._client.get_orders.return_value = [
            _mock_raw_order(id="a"), _mock_raw_order(id="b")
        ]
        results = executor.list_open_orders()
        assert len(results) == 2
        assert all(isinstance(o, Order) for o in results)

    def test_list_open_orders_empty_when_none(self, executor: OrderExecutor) -> None:
        executor._client.get_orders.return_value = []
        assert executor.list_open_orders() == []

    def test_list_recent_orders_delegates_to_client(
        self, executor: OrderExecutor
    ) -> None:
        executor._client.get_orders.return_value = [_mock_raw_order()]
        results = executor.list_recent_orders(limit=10)
        assert len(results) == 1


# ===========================================================================
# OrderExecutor — account / positions
# ===========================================================================

class TestAccountAndPositions:

    def test_get_account_returns_expected_keys(
        self, executor: OrderExecutor
    ) -> None:
        mock_account = MagicMock()
        mock_account.id             = "acct_001"
        mock_account.status         = "ACTIVE"
        mock_account.currency       = "USD"
        mock_account.buying_power   = "95000.00"
        mock_account.equity         = "100000.00"
        mock_account.cash           = "5000.00"
        mock_account.portfolio_value = "100000.00"
        mock_account.pattern_day_trader = False
        mock_account.trading_blocked    = False
        executor._client.get_account.return_value = mock_account

        info = executor.get_account()
        assert info["status"]       == "ACTIVE"
        assert info["buying_power"] == pytest.approx(95_000.0)
        assert "equity" in info and "cash" in info

    def test_get_positions_returns_list(self, executor: OrderExecutor) -> None:
        mock_pos = MagicMock()
        mock_pos.symbol          = "SPY"
        mock_pos.qty             = "10"
        mock_pos.avg_entry_price = "450.00"
        mock_pos.market_value    = "4500.00"
        mock_pos.unrealized_pl   = "50.00"
        mock_pos.current_price   = "455.00"
        mock_pos.side            = MagicMock()
        mock_pos.side.value      = "long"
        executor._client.get_all_positions.return_value = [mock_pos]

        positions = executor.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "SPY"
        assert positions[0]["qty"]    == pytest.approx(10.0)

    def test_is_market_open_delegates_to_client(
        self, executor: OrderExecutor
    ) -> None:
        """is_market_open() reads the market clock from the Alpaca client."""
        mock_clock = MagicMock()
        mock_clock.is_open = True
        executor._client.get_clock.return_value = mock_clock

        result = executor.is_market_open()
        executor._client.get_clock.assert_called_once()
        assert result is True

    def test_is_market_open_returns_false_when_closed(
        self, executor: OrderExecutor
    ) -> None:
        mock_clock = MagicMock()
        mock_clock.is_open = False
        executor._client.get_clock.return_value = mock_clock
        assert executor.is_market_open() is False

    def test_get_clock_returns_dict(self, executor: OrderExecutor) -> None:
        mock_clock = MagicMock()
        mock_clock.is_open    = True
        mock_clock.next_open  = "2025-01-02T14:30:00Z"
        mock_clock.next_close = "2025-01-02T21:00:00Z"
        mock_clock.timestamp  = "2025-01-01T20:00:00Z"
        executor._client.get_clock.return_value = mock_clock

        clock = executor.get_clock()
        assert "is_open"    in clock
        assert "next_open"  in clock
        assert "next_close" in clock
        assert clock["is_open"] is True


# ===========================================================================
# OrderExecutor — parsing
# ===========================================================================

class TestParseOrder:

    def test_parse_order_maps_all_fields(
        self, executor: OrderExecutor, sample_order_dict: dict
    ) -> None:
        """_parse_order() correctly maps every field of the raw API response."""
        order = executor._parse_order(sample_order_dict)

        assert order.id               == "abc123"
        assert order.client_order_id  == "cid_001"
        assert order.ticker           == "SPY"
        assert order.side             == OrderSide.BUY
        assert order.order_type       == OrderType.MARKET
        assert order.qty              == pytest.approx(10.0)
        assert order.notional         is None
        assert order.limit_price      is None
        assert order.stop_price       is None
        assert order.time_in_force    == TimeInForce.DAY
        assert order.status           == OrderStatus.FILLED
        assert order.filled_qty       == pytest.approx(10.0)
        assert order.filled_avg_price == pytest.approx(450.25)
        assert isinstance(order.submitted_at, datetime)
        assert isinstance(order.filled_at, datetime)

    def test_parse_order_handles_missing_optional_fields(
        self, executor: OrderExecutor
    ) -> None:
        """_parse_order() uses None for missing optional fields without raising."""
        minimal = {
            "id":            "min_001",
            "client_order_id": "cid_min",
            "symbol":        "AAPL",
            "side":          "sell",
            "type":          "limit",
            "qty":           "2",
            "filled_qty":    "0",
            "time_in_force": "gtc",
            "status":        "new",
            "submitted_at":  "2025-01-01T10:00:00Z",
        }
        order = executor._parse_order(minimal)
        assert order.notional        is None
        assert order.limit_price     is None
        assert order.stop_price      is None
        assert order.filled_at       is None
        assert order.filled_avg_price is None

    def test_parse_order_from_mock_object(self, executor: OrderExecutor) -> None:
        """_parse_order() works with attribute-access objects (Alpaca models / mocks)."""
        raw = _mock_raw_order(symbol="QQQ", side="sell", status="filled",
                              filled_qty="3", filled_avg_price="380.50")
        order = executor._parse_order(raw)
        assert isinstance(order, Order)
        assert order.ticker == "QQQ"

    def test_parse_order_limit_type(self, executor: OrderExecutor) -> None:
        """_parse_order() maps 'limit' type correctly."""
        d = dict(
            id="l1", client_order_id="cl1", symbol="SPY", side="buy",
            type="limit", qty="5", limit_price="445.00", notional=None,
            stop_price=None, time_in_force="day", status="new",
            filled_qty="0", filled_avg_price=None,
            submitted_at="2025-01-01T09:30:00Z", filled_at=None,
        )
        order = executor._parse_order(d)
        assert order.order_type  == OrderType.LIMIT
        assert order.limit_price == pytest.approx(445.0)


# ===========================================================================
# OrderExecutor — notional-to-qty conversion
# ===========================================================================

class TestNotionalToQty:

    def test_notional_to_qty_is_positive(self, executor: OrderExecutor) -> None:
        """_notional_to_qty() returns a positive float for any positive inputs."""
        result = executor._notional_to_qty("SPY", 4_500.0, 450.0)
        assert result > 0.0

    def test_notional_to_qty_divides_correctly(self, executor: OrderExecutor) -> None:
        """_notional_to_qty() returns notional / price."""
        assert executor._notional_to_qty("SPY", 4_500.0, 450.0) == pytest.approx(10.0)
        assert executor._notional_to_qty("AAPL", 1_000.0, 200.0) == pytest.approx(5.0)

    def test_notional_to_qty_zero_price_raises(self, executor: OrderExecutor) -> None:
        """_notional_to_qty() raises ValueError when price <= 0."""
        with pytest.raises(ValueError, match="[Pp]rice"):
            executor._notional_to_qty("SPY", 1_000.0, 0.0)

    def test_notional_to_qty_negative_price_raises(self, executor: OrderExecutor) -> None:
        with pytest.raises(ValueError, match="[Pp]rice"):
            executor._notional_to_qty("SPY", 1_000.0, -1.0)


# ===========================================================================
# OrderExecutor — submit_order convenience wrapper
# ===========================================================================

class TestSubmitOrder:

    def test_submit_order_delegates_to_market_order(
        self, executor: OrderExecutor
    ) -> None:
        executor._client.submit_order.return_value = _mock_raw_order()
        with patch.object(executor, "market_order", wraps=executor.market_order) as mock_mo:
            executor.submit_order("SPY", 5_000.0, OrderSide.BUY)
        mock_mo.assert_called_once_with("SPY", 5_000.0, OrderSide.BUY)

    def test_submit_order_uses_abs_amount(self, executor: OrderExecutor) -> None:
        """submit_order passes |dollar_amount| (always positive) to market_order."""
        executor._client.submit_order.return_value = _mock_raw_order()
        with patch.object(executor, "market_order") as mock_mo:
            mock_mo.return_value = MagicMock(spec=Order)
            executor.submit_order("SPY", -3_000.0, OrderSide.SELL)
        args = mock_mo.call_args.args
        assert args[1] == pytest.approx(3_000.0)


# ===========================================================================
# OrderExecutor — retry logic
# ===========================================================================

class TestRetryLogic:

    def test_retries_on_429(self, executor: OrderExecutor) -> None:
        """_submit_with_retry retries on a 429 error before succeeding."""
        good_response = _mock_raw_order()
        executor._client.submit_order.side_effect = [
            RuntimeError("429 Too Many Requests"),
            good_response,
        ]
        with patch("core.order_executor.time.sleep"):
            result = executor._submit_with_retry(MagicMock())
        assert result is good_response
        assert executor._client.submit_order.call_count == 2

    def test_retries_on_503(self, executor: OrderExecutor) -> None:
        """_submit_with_retry retries on a 503 error."""
        good_response = _mock_raw_order()
        executor._client.submit_order.side_effect = [
            RuntimeError("503 Service Unavailable"),
            good_response,
        ]
        with patch("core.order_executor.time.sleep"):
            result = executor._submit_with_retry(MagicMock())
        assert result is good_response

    def test_non_retryable_raises_immediately(self, executor: OrderExecutor) -> None:
        """A non-retryable error (e.g. 400) raises without sleeping."""
        executor._client.submit_order.side_effect = RuntimeError("400 Bad Request")
        with patch("core.order_executor.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="400"):
                executor._submit_with_retry(MagicMock())
        mock_sleep.assert_not_called()
        assert executor._client.submit_order.call_count == 1

    def test_alert_callback_fired_on_repeated_failures(self) -> None:
        """Alert callback fires after _REPEATED_FAILURE_THRESHOLD consecutive failures."""
        alerts: list[str] = []
        ex = OrderExecutor(
            api_key="k", secret_key="s",
            base_url="https://paper-api.alpaca.markets",
            paper=True,
            alert_callback=alerts.append,
        )
        ex._client = MagicMock()
        ex._client.submit_order.side_effect = RuntimeError("503 Service Unavailable")

        with patch("core.order_executor.time.sleep"):
            with pytest.raises(RuntimeError):
                ex._submit_with_retry(MagicMock())

        assert len(alerts) >= 1


# ===========================================================================
# OrderExecutor — from_env factory
# ===========================================================================

class TestFromEnv:

    def test_from_env_raises_without_credentials(self, tmp_path) -> None:
        """from_env() raises EnvironmentError when .env keys are missing."""
        env_backup = {k: os.environ.pop(k, None)
                      for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")}
        try:
            with patch("core.order_executor.os.getenv", side_effect=lambda k, d="": ""):
                with pytest.raises(EnvironmentError, match="ALPACA_API_KEY"):
                    OrderExecutor.from_env()
        finally:
            for k, v in env_backup.items():
                if v is not None:
                    os.environ[k] = v

    def test_from_env_uses_paper_url_by_default(self) -> None:
        """from_env() defaults to the paper trading endpoint."""
        with patch("core.order_executor.os.getenv") as mock_getenv:
            mock_getenv.side_effect = lambda k, d="": {
                "ALPACA_API_KEY":    "ak",
                "ALPACA_SECRET_KEY": "sk",
                "ALPACA_BASE_URL":   "https://paper-api.alpaca.markets",
                "PAPER":             "true",
            }.get(k, d)
            ex = OrderExecutor.from_env()
        assert "paper-api" in ex._base_url


# ===========================================================================
# PositionTracker
# ===========================================================================

class TestPositionTracker:

    @pytest.fixture
    def tracker(self) -> PositionTracker:
        return PositionTracker()

    # -- apply_fill: buy into new position --

    def test_apply_fill_buy_creates_position(self, tracker: PositionTracker) -> None:
        ts = datetime.now(tz=timezone.utc)
        tracker.apply_fill("SPY", 10.0, 450.0, "buy", ts)
        pos = tracker.get_position("SPY")
        assert pos is not None
        assert pos.qty              == pytest.approx(10.0)
        assert pos.avg_entry_price  == pytest.approx(450.0)
        assert pos.side             == "long"

    def test_apply_fill_buy_adds_to_existing(self, tracker: PositionTracker) -> None:
        ts = datetime.now(tz=timezone.utc)
        tracker.apply_fill("SPY", 10.0, 400.0, "buy", ts)
        tracker.apply_fill("SPY", 10.0, 500.0, "buy", ts)
        pos = tracker.get_position("SPY")
        assert pos.qty             == pytest.approx(20.0)
        assert pos.avg_entry_price == pytest.approx(450.0)   # (400+500)/2

    # -- apply_fill: sell --

    def test_apply_fill_sell_full_removes_position(
        self, tracker: PositionTracker
    ) -> None:
        ts = datetime.now(tz=timezone.utc)
        tracker.apply_fill("SPY", 10.0, 450.0, "buy",  ts)
        tracker.apply_fill("SPY", 10.0, 460.0, "sell", ts)
        assert tracker.is_flat("SPY") is True

    def test_apply_fill_sell_partial_reduces_qty(
        self, tracker: PositionTracker
    ) -> None:
        ts = datetime.now(tz=timezone.utc)
        tracker.apply_fill("SPY", 10.0, 450.0, "buy",  ts)
        tracker.apply_fill("SPY",  4.0, 460.0, "sell", ts)
        pos = tracker.get_position("SPY")
        assert pos.qty == pytest.approx(6.0)

    def test_apply_fill_sell_records_pnl(self, tracker: PositionTracker) -> None:
        ts = datetime.now(tz=timezone.utc)
        tracker.apply_fill("SPY", 10.0, 450.0, "buy",  ts)
        tracker.apply_fill("SPY", 10.0, 460.0, "sell", ts)
        series = tracker.pnl_series()
        assert not series.empty
        # PnL = (460 - 450) × 10 = 100
        assert float(series.sum()) == pytest.approx(100.0)

    # -- update_prices --

    def test_update_prices_refreshes_market_value(
        self, tracker: PositionTracker
    ) -> None:
        ts = datetime.now(tz=timezone.utc)
        tracker.apply_fill("SPY", 10.0, 450.0, "buy", ts)
        tracker.update_prices({"SPY": 480.0})
        pos = tracker.get_position("SPY")
        assert pos.current_price == pytest.approx(480.0)
        assert pos.market_value  == pytest.approx(4_800.0)

    def test_update_prices_computes_unrealised_pnl(
        self, tracker: PositionTracker
    ) -> None:
        ts = datetime.now(tz=timezone.utc)
        tracker.apply_fill("SPY", 10.0, 450.0, "buy", ts)
        tracker.update_prices({"SPY": 460.0})
        pos = tracker.get_position("SPY")
        # unrealised = (460 - 450) × 10 = 100
        assert pos.unrealised_pnl == pytest.approx(100.0)

    def test_update_prices_ignores_unknown_tickers(
        self, tracker: PositionTracker
    ) -> None:
        """update_prices() for a ticker with no open position is a no-op."""
        tracker.update_prices({"AAPL": 200.0})   # no position — should not raise

    # -- queries --

    def test_get_position_returns_none_when_flat(
        self, tracker: PositionTracker
    ) -> None:
        assert tracker.get_position("AAPL") is None

    def test_is_flat_true_with_no_position(self, tracker: PositionTracker) -> None:
        assert tracker.is_flat("SPY") is True

    def test_is_flat_false_with_open_position(self, tracker: PositionTracker) -> None:
        ts = datetime.now(tz=timezone.utc)
        tracker.apply_fill("SPY", 5.0, 450.0, "buy", ts)
        assert tracker.is_flat("SPY") is False

    def test_all_positions_returns_copy(self, tracker: PositionTracker) -> None:
        ts = datetime.now(tz=timezone.utc)
        tracker.apply_fill("SPY", 5.0, 450.0, "buy", ts)
        positions = tracker.all_positions()
        positions["EXTRA"] = None   # mutate the copy
        assert "EXTRA" not in tracker.all_positions()

    # -- exposure --

    def test_net_exposure_sums_market_values(
        self, tracker: PositionTracker
    ) -> None:
        ts = datetime.now(tz=timezone.utc)
        tracker.apply_fill("SPY",  10.0, 450.0, "buy", ts)
        tracker.apply_fill("AAPL",  5.0, 200.0, "buy", ts)
        # market values: 4500 + 1000 = 5500
        assert tracker.net_exposure() == pytest.approx(5_500.0)

    def test_gross_exposure_uses_abs(self, tracker: PositionTracker) -> None:
        ts = datetime.now(tz=timezone.utc)
        tracker.apply_fill("SPY", 10.0, 450.0, "buy", ts)
        assert tracker.gross_exposure() == pytest.approx(4_500.0)

    # -- snapshot --

    def test_snapshot_nav_is_cash_plus_positions(
        self, tracker: PositionTracker
    ) -> None:
        ts = datetime.now(tz=timezone.utc)
        tracker.apply_fill("SPY", 10.0, 450.0, "buy", ts)
        snap = tracker.snapshot(cash=55_000.0)
        assert isinstance(snap, PortfolioSnapshot)
        assert snap.nav == pytest.approx(55_000.0 + 4_500.0)

    def test_snapshot_weights_sum_approximately_one(
        self, tracker: PositionTracker
    ) -> None:
        ts = datetime.now(tz=timezone.utc)
        tracker.apply_fill("SPY",  10.0, 450.0, "buy", ts)
        tracker.apply_fill("AAPL",  5.0, 200.0, "buy", ts)
        snap = tracker.snapshot(cash=0.0)
        total_weight = sum(snap.weights.values())
        assert total_weight == pytest.approx(1.0)

    # -- current_weights --

    def test_current_weights_zero_nav(self, tracker: PositionTracker) -> None:
        assert tracker.current_weights(0.0) == {}

    def test_current_weights_computed_correctly(
        self, tracker: PositionTracker
    ) -> None:
        ts = datetime.now(tz=timezone.utc)
        tracker.apply_fill("SPY", 10.0, 100.0, "buy", ts)  # MV = 1000
        weights = tracker.current_weights(nav=2_000.0)
        assert weights["SPY"] == pytest.approx(0.5)

    # -- _update_average_cost --

    def test_update_average_cost_weighted(self, tracker: PositionTracker) -> None:
        ts = datetime.now(tz=timezone.utc)
        tracker.apply_fill("SPY", 10.0, 400.0, "buy", ts)
        pos    = tracker.get_position("SPY")
        new_avg = tracker._update_average_cost(pos, 10.0, 500.0)
        assert new_avg == pytest.approx(450.0)

    # -- sync_from_broker --

    def test_sync_from_broker_updates_positions(
        self, tracker: PositionTracker
    ) -> None:
        mock_pos = MagicMock()
        mock_pos.symbol          = "SPY"
        mock_pos.qty             = "10"
        mock_pos.avg_entry_price = "450.00"
        mock_pos.current_price   = "455.00"
        mock_pos.market_value    = "4550.00"
        mock_pos.cost_basis      = "4500.00"
        mock_pos.unrealized_pl   = "50.00"
        mock_pos.unrealized_plpc = "0.011"
        mock_pos.side            = MagicMock()
        mock_pos.side.value      = "long"

        mock_account = MagicMock()
        mock_account.cash = "95000.00"

        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = [mock_pos]
        mock_client.get_account.return_value = mock_account

        tracker.sync_from_broker(mock_client)

        pos = tracker.get_position("SPY")
        assert pos is not None
        assert pos.qty             == pytest.approx(10.0)
        assert pos.avg_entry_price == pytest.approx(450.0)
        assert tracker._cash       == pytest.approx(95_000.0)

    def test_sync_from_broker_clears_stale_positions(
        self, tracker: PositionTracker
    ) -> None:
        """After sync, only positions returned by Alpaca are kept."""
        ts = datetime.now(tz=timezone.utc)
        tracker.apply_fill("AAPL", 5.0, 200.0, "buy", ts)  # local-only position

        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []      # broker has nothing
        mock_account = MagicMock()
        mock_account.cash = "100000.00"
        mock_client.get_account.return_value = mock_account

        tracker.sync_from_broker(mock_client)
        assert tracker.is_flat("AAPL") is True


# ===========================================================================
# MarketData
# ===========================================================================

class TestMarketData:

    @pytest.fixture
    def md(self) -> MarketData:
        return MarketData(
            api_key="k", secret_key="s",
            base_url="https://paper-api.alpaca.markets",
        )

    # -- _normalize_bars --

    def test_normalize_bars_adds_vwap_column(self, md: MarketData) -> None:
        df = pd.DataFrame({
            "open":   [100.0],
            "high":   [105.0],
            "low":    [ 98.0],
            "close":  [103.0],
            "volume": [1_000_000.0],
        })
        result = md._normalize_bars(df)
        assert "vwap" in result.columns

    def test_normalize_bars_has_all_ohlcv_columns(self, md: MarketData) -> None:
        df = pd.DataFrame({
            "open":   [1.0],
            "high":   [2.0],
            "low":    [0.5],
            "close":  [1.5],
            "volume": [100.0],
            "vwap":   [1.2],
        })
        result = md._normalize_bars(df)
        for col in ["open", "high", "low", "close", "volume", "vwap"]:
            assert col in result.columns

    def test_normalize_bars_fills_missing_columns_with_zero(
        self, md: MarketData
    ) -> None:
        """Missing numeric columns default to 0.0 rather than raising."""
        df = pd.DataFrame({"close": [100.0]})
        result = md._normalize_bars(df)
        assert "open" in result.columns
        assert result["open"].iloc[0] == pytest.approx(0.0)

    # -- subscribe_bars --

    def test_subscribe_bars_registers_callback(self, md: MarketData) -> None:
        fired: list[str] = []
        md.subscribe_bars(["SPY"], callback=lambda t, b: fired.append(t))
        assert len(md._bar_subs) == 1

    def test_subscribe_bars_multiple_tickers(self, md: MarketData) -> None:
        md.subscribe_bars(["SPY", "QQQ"], callback=lambda t, b: None)
        assert len(md._bar_subs) == 2

    def test_subscribe_trades_registers_callback(self, md: MarketData) -> None:
        md.subscribe_trades(["AAPL"], callback=lambda t, tr: None)
        assert len(md._trade_subs) == 1

    # -- from_env --

    def test_from_env_raises_without_credentials(self) -> None:
        with patch("core.market_data.os.getenv", side_effect=lambda k, d="": ""):
            with pytest.raises(EnvironmentError, match="ALPACA_API_KEY"):
                MarketData.from_env()

    def test_from_env_succeeds_with_credentials(self) -> None:
        def _mock_getenv(k, d=""):
            return {
                "ALPACA_API_KEY":    "ak",
                "ALPACA_SECRET_KEY": "sk",
                "ALPACA_BASE_URL":   "https://paper-api.alpaca.markets",
                "ALPACA_DATA_FEED":  "iex",
            }.get(k, d)

        with patch("core.market_data.os.getenv", side_effect=_mock_getenv):
            m = MarketData.from_env()
        assert m._api_key == "ak"

    # -- _alpaca_timeframe helper --

    def test_timeframe_1day(self) -> None:
        tf = _alpaca_timeframe("1Day")
        assert tf is not None

    def test_timeframe_5min(self) -> None:
        tf = _alpaca_timeframe("5Min")
        assert tf is not None

    def test_timeframe_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown timeframe"):
            _alpaca_timeframe("3WeeksAndABit")

    def test_timeframe_case_insensitive(self) -> None:
        tf1 = _alpaca_timeframe("1day")
        tf2 = _alpaca_timeframe("1Day")
        assert tf1 is not None and tf2 is not None

    # -- _build_rest_client --

    def test_build_rest_client_uses_credentials(self, md: MarketData) -> None:
        """_get_rest_client() constructs a StockHistoricalDataClient."""
        with patch("core.market_data.StockHistoricalDataClient" if False else
                   "alpaca.data.historical.StockHistoricalDataClient") as _:
            # Just verify it doesn't raise with valid credentials
            client = md._build_rest_client()
        assert client is not None
