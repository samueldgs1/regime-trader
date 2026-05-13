"""
Market data feeds: historical bars and real-time bar streaming.

Provides a unified interface for both Alpaca REST (historical / backtest)
and WebSocket (live trading) data sources.

Streaming resilience
--------------------
* Primary feed: alpaca-py StockDataStream (WebSocket).
* Fallback: REST polling every ``poll_interval_s`` seconds if the
  WebSocket disconnects or fails to start.
* The fallback thread monitors a "stream healthy" flag; when the
  primary reconnects, polling stops automatically.

Retry policy
------------
REST calls retry on HTTP 429 / 503 with exponential back-off.
All API errors are logged; repeated failures fire the alert callback.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RETRY_BACKOFF_S  = (1.0, 2.0, 4.0)
_RETRYABLE_CODES  = frozenset({"429", "503"})
_POLL_INTERVAL_S  = 30           # REST fallback polling interval (seconds)
_STREAM_BAR_MIN   = "5Min"       # default stream bar timeframe
_MAX_RETRIES: int = 3


# ---------------------------------------------------------------------------
# Timeframe helpers
# ---------------------------------------------------------------------------

def _alpaca_timeframe(timeframe_str: str):
    """
    Convert a human-readable timeframe string to an alpaca-py TimeFrame.

    Recognised strings (case-insensitive)
    --------------------------------------
    '1Min' / '1Minute', '5Min', '15Min', '30Min',
    '1Hour', '1Day', '1Week', '1Month'
    """
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tfl = timeframe_str.lower().replace(" ", "")
    _map: Dict[str, object] = {
        "1min":    TimeFrame(1,   TimeFrameUnit.Minute),
        "1minute": TimeFrame(1,   TimeFrameUnit.Minute),
        "5min":    TimeFrame(5,   TimeFrameUnit.Minute),
        "15min":   TimeFrame(15,  TimeFrameUnit.Minute),
        "30min":   TimeFrame(30,  TimeFrameUnit.Minute),
        "1hour":   TimeFrame(1,   TimeFrameUnit.Hour),
        "1day":    TimeFrame(1,   TimeFrameUnit.Day),
        "1week":   TimeFrame(1,   TimeFrameUnit.Week),
        "1month":  TimeFrame(1,   TimeFrameUnit.Month),
    }
    if tfl not in _map:
        raise ValueError(
            f"Unknown timeframe '{timeframe_str}'. "
            f"Supported: {sorted(_map.keys())}"
        )
    return _map[tfl]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MarketData:
    """
    Fetches and streams market data from Alpaca.

    Historical mode  — REST polling, used for backtest and HMM training.
    Live mode        — WebSocket stream, used during market hours with
                       automatic REST fallback on disconnect.

    Production entrypoint
    ---------------------
    Use ``MarketData.from_env()`` to load credentials from .env.

    Usage (live)
    ------------
    ::

        md = MarketData.from_env()
        md.subscribe_bars(["SPY", "QQQ"], callback=on_bar)
        md.start_stream()          # blocks; run in a thread

    Usage (historical)
    ------------------
    ::

        bars = md.get_bars(["SPY"], start=date(2023,1,1), end=date(2024,1,1))
    """

    def __init__(
        self,
        api_key:        str,
        secret_key:     str,
        base_url:       str = "https://paper-api.alpaca.markets",
        data_feed:      str = "iex",
        alert_callback: Optional[Callable[[str], None]] = None,
        poll_interval_s: int = _POLL_INTERVAL_S,
    ) -> None:
        """
        Parameters
        ----------
        api_key / secret_key:
            Alpaca credentials.  In production use from_env().
        base_url:
            Alpaca trading REST endpoint (used only for account checks).
        data_feed:
            'iex' (free) or 'sip' (consolidated, paid).
        alert_callback:
            Optional callable fired on repeated API failures.
        poll_interval_s:
            Seconds between REST polls when the WebSocket is down.
        """
        self._api_key        = api_key
        self._secret_key     = secret_key
        self._base_url       = base_url
        self._data_feed      = data_feed
        self._alert_cb       = alert_callback
        self._poll_interval  = poll_interval_s
        self._is_crypto      = data_feed == "crypto"

        self._rest_client    = None     # Historical data client (lazy init)
        self._stream_client  = None     # Streaming client (lazy init)

        # Subscription state
        self._bar_subs:    List[Tuple[str, Callable]] = []  # (ticker, cb)
        self._trade_subs:  List[Tuple[str, Callable]] = []

        # Streaming / fallback state
        self._stream_healthy:   bool                        = False
        self._fallback_thread:  Optional[threading.Thread] = None
        self._fallback_running: bool                        = False
        self._stream_thread:    Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Production factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        alert_callback: Optional[Callable[[str], None]] = None,
    ) -> "MarketData":
        """Load credentials from .env and construct a MarketData client."""
        from dotenv import load_dotenv
        load_dotenv()

        api_key    = os.getenv("ALPACA_API_KEY",    "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        base_url   = os.getenv("ALPACA_BASE_URL",   "https://paper-api.alpaca.markets")
        data_feed  = os.getenv("ALPACA_DATA_FEED",  "iex")

        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env."
            )

        return cls(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url,
            data_feed=data_feed,
            alert_callback=alert_callback,
        )

    # ------------------------------------------------------------------
    # Historical data (REST)
    # ------------------------------------------------------------------

    def get_bars(
        self,
        tickers:    List[str],
        start:      date,
        end:        date,
        timeframe:  str = "1Day",
        adjustment: str = "split",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars for one or more tickers.

        Parameters
        ----------
        tickers:
            List of equity symbols.
        start / end:
            Inclusive date range.
        timeframe:
            '1Min', '5Min', '15Min', '30Min', '1Hour', '1Day', etc.
        adjustment:
            Price adjustment: 'raw', 'split', or 'all'.

        Returns
        -------
        DataFrame indexed by (symbol, timestamp) with columns
        ['open', 'high', 'low', 'close', 'volume', 'vwap'].
        """
        client = self._get_rest_client()
        tf     = _alpaca_timeframe(timeframe)

        if self._is_crypto:
            from alpaca.data.requests import CryptoBarsRequest
            req = CryptoBarsRequest(
                symbol_or_symbols=tickers,
                start=datetime.combine(start, datetime.min.time()),
                end=datetime.combine(end,   datetime.max.time()),
                timeframe=tf,
            )
        else:
            from alpaca.data.requests import StockBarsRequest
            req = StockBarsRequest(
                symbol_or_symbols=tickers,
                start=datetime.combine(start, datetime.min.time()),
                end=datetime.combine(end,   datetime.max.time()),
                timeframe=tf,
                adjustment=adjustment,
                feed=self._data_feed,
            )

        for attempt, backoff in enumerate(_RETRY_BACKOFF_S):
            try:
                if self._is_crypto:
                    response = client.get_crypto_bars(req)
                else:
                    response = client.get_stock_bars(req)
                df = response.df
                if df.empty:
                    return pd.DataFrame(
                        columns=["open", "high", "low", "close", "volume", "vwap"]
                    )
                df = df.rename(columns=str.lower)
                return self._normalize_bars(df)
            except Exception as exc:
                retryable = any(c in str(exc) for c in _RETRYABLE_CODES)
                logger.error(
                    "get_bars error (attempt %d/%d, retryable=%s): %s",
                    attempt + 1, _MAX_RETRIES, retryable, exc,
                )
                if retryable and attempt < len(_RETRY_BACKOFF_S) - 1:
                    time.sleep(backoff)
                    continue
                raise

    def get_latest_bars(
        self,
        tickers:   List[str],
        timeframe: str = "1Day",
        limit:     int = 500,
    ) -> pd.DataFrame:
        """Fetch the most recent ``limit`` bars up to now."""
        client = self._get_rest_client()
        tf     = _alpaca_timeframe(timeframe)
        end    = datetime.now(tz=timezone.utc)
        start  = end - timedelta(days=limit * 2)   # generous window

        if self._is_crypto:
            from alpaca.data.requests import CryptoBarsRequest
            req = CryptoBarsRequest(
                symbol_or_symbols=tickers,
                start=start,
                end=end,
                timeframe=tf,
                limit=limit,
            )
            response = client.get_crypto_bars(req)
        else:
            from alpaca.data.requests import StockBarsRequest
            req = StockBarsRequest(
                symbol_or_symbols=tickers,
                start=start,
                end=end,
                timeframe=tf,
                limit=limit,
            )
            response = client.get_stock_bars(req)

        df = response.df
        if df.empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "vwap"]
            )
        return self._normalize_bars(df.rename(columns=str.lower))

    def get_quotes(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        Fetch the latest NBBO quote for each ticker.

        Returns
        -------
        Map of ticker → {'bid': float, 'ask': float, 'mid': float}.
        """
        from alpaca.data.requests import StockLatestQuoteRequest

        client   = self._get_rest_client()
        req      = StockLatestQuoteRequest(symbol_or_symbols=tickers)
        response = client.get_stock_latest_quote(req)

        result: Dict[str, Dict] = {}
        for ticker, quote in response.items():
            bid = float(quote.bid_price or 0.0)
            ask = float(quote.ask_price or 0.0)
            result[ticker] = {
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2.0 if (bid and ask) else 0.0,
            }
        return result

    def get_latest_trade(self, ticker: str) -> Dict:
        """Return the most recent trade for a single ticker."""
        from alpaca.data.requests import StockLatestTradeRequest

        client   = self._get_rest_client()
        req      = StockLatestTradeRequest(symbol_or_symbols=[ticker])
        response = client.get_stock_latest_trade(req)
        trade    = response[ticker]
        return {
            "price":     float(trade.price),
            "size":      float(trade.size),
            "timestamp": trade.timestamp,
        }

    def get_trading_calendar(self, start: date, end: date) -> pd.DataFrame:
        """
        Return the NYSE trading calendar for the date range.

        Returns DataFrame with columns ['date', 'open', 'close'].
        """
        from alpaca.trading.client   import TradingClient
        from alpaca.trading.requests import GetCalendarRequest

        trading_client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=True,
        )
        req      = GetCalendarRequest(start=str(start), end=str(end))
        calendar = trading_client.get_calendar(req)

        rows = []
        for day in calendar:
            rows.append({
                "date":  day.date,
                "open":  day.open,
                "close": day.close,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Real-time streaming (WebSocket)
    # ------------------------------------------------------------------

    def subscribe_bars(
        self,
        tickers:  List[str],
        callback: Callable[[str, Dict], None],
    ) -> None:
        """
        Register a callback for 1-minute bar updates.

        Parameters
        ----------
        tickers:
            Symbols to subscribe.
        callback:
            Called with (ticker, bar_dict) on each bar arrival.
            bar_dict contains: open, high, low, close, volume, timestamp.
        """
        for ticker in tickers:
            self._bar_subs.append((ticker, callback))
        logger.info(
            "Subscribed bar callbacks for %s (%d total subs)",
            tickers, len(self._bar_subs),
        )

    def subscribe_trades(
        self,
        tickers:  List[str],
        callback: Callable[[str, Dict], None],
    ) -> None:
        """Register a callback for real-time trade prints."""
        for ticker in tickers:
            self._trade_subs.append((ticker, callback))
        logger.info(
            "Subscribed trade callbacks for %s (%d total subs)",
            tickers, len(self._trade_subs),
        )

    def start_stream(self) -> None:
        """
        Connect the WebSocket stream and begin delivering bars.

        Blocks until stop_stream() is called or the stream errors out.
        Launches a REST polling fallback in a daemon thread; the fallback
        stops automatically once the WebSocket reconnects.

        Intended to be run in a dedicated thread:
        ::

            t = threading.Thread(target=md.start_stream, daemon=True)
            t.start()
        """
        stream = self._build_stream_client()

        # Register handlers for all subscribed tickers
        bar_tickers   = list({t for t, _ in self._bar_subs})
        trade_tickers = list({t for t, _ in self._trade_subs})

        if bar_tickers:
            async def _on_bar(bar) -> None:
                self._stream_healthy = True
                payload = {
                    "open":      float(bar.open),
                    "high":      float(bar.high),
                    "low":       float(bar.low),
                    "close":     float(bar.close),
                    "volume":    float(bar.volume),
                    "timestamp": bar.timestamp,
                }
                for ticker, cb in self._bar_subs:
                    if ticker == bar.symbol:
                        try:
                            cb(bar.symbol, payload)
                        except Exception as exc:
                            logger.warning(
                                "Bar callback error for %s: %s", bar.symbol, exc
                            )

            stream.subscribe_bars(_on_bar, *bar_tickers)

        if trade_tickers:
            async def _on_trade(trade) -> None:
                self._stream_healthy = True
                payload = {
                    "price":     float(trade.price),
                    "size":      float(trade.size),
                    "timestamp": trade.timestamp,
                }
                for ticker, cb in self._trade_subs:
                    if ticker == trade.symbol:
                        try:
                            cb(trade.symbol, payload)
                        except Exception as exc:
                            logger.warning(
                                "Trade callback error for %s: %s", trade.symbol, exc
                            )

            stream.subscribe_trades(_on_trade, *trade_tickers)

        # Start REST fallback in the background
        self._start_rest_fallback()

        self._stream_client = stream
        logger.info("Starting WebSocket stream…")
        try:
            self._stream_healthy = True
            stream.run()
        except Exception as exc:
            logger.error("WebSocket stream crashed: %s", exc)
            self._stream_healthy = False
            if self._alert_cb:
                try:
                    self._alert_cb(f"WebSocket stream crashed: {exc}")
                except Exception:
                    pass

    def stop_stream(self) -> None:
        """Gracefully disconnect the WebSocket stream and stop fallback polling."""
        self._fallback_running = False
        if self._stream_client is not None:
            try:
                self._stream_client.stop()
            except Exception as exc:
                logger.warning("Error stopping stream: %s", exc)
        logger.info("Stream stopped.")

    # ------------------------------------------------------------------
    # REST fallback polling
    # ------------------------------------------------------------------

    def _start_rest_fallback(self) -> None:
        """
        Start a daemon thread that polls via REST when the WebSocket is down.

        The thread checks self._stream_healthy and skips polling while the
        stream is active.
        """
        self._fallback_running = True

        bar_tickers = list({t for t, _ in self._bar_subs})
        if not bar_tickers:
            return

        def _poll_loop() -> None:
            logger.info(
                "REST fallback thread started (poll_interval=%ds)",
                self._poll_interval,
            )
            while self._fallback_running:
                if not self._stream_healthy:
                    logger.warning(
                        "WebSocket unhealthy — REST-polling bars for %s",
                        bar_tickers,
                    )
                    try:
                        client = self._get_rest_client()
                        if self._is_crypto:
                            from alpaca.data.requests import CryptoLatestBarRequest
                            req  = CryptoLatestBarRequest(symbol_or_symbols=bar_tickers)
                            bars = client.get_crypto_latest_bar(req)
                        else:
                            from alpaca.data.requests import StockLatestBarRequest
                            req  = StockLatestBarRequest(symbol_or_symbols=bar_tickers)
                            bars = client.get_stock_latest_bar(req)
                        for ticker, bar in bars.items():
                            payload = {
                                "open":      float(bar.open),
                                "high":      float(bar.high),
                                "low":       float(bar.low),
                                "close":     float(bar.close),
                                "volume":    float(bar.volume),
                                "timestamp": bar.timestamp,
                            }
                            for sub_ticker, cb in self._bar_subs:
                                if sub_ticker == ticker:
                                    try:
                                        cb(ticker, payload)
                                    except Exception as exc:
                                        logger.warning(
                                            "Fallback bar callback error: %s", exc
                                        )
                    except Exception as exc:
                        logger.error("REST fallback poll failed: %s", exc)
                time.sleep(self._poll_interval)

        self._fallback_thread = threading.Thread(
            target=_poll_loop, daemon=True, name="market-data-fallback"
        )
        self._fallback_thread.start()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the DataFrame has standard OHLCV columns and a clean index.

        Handles both single-ticker (DatetimeIndex) and multi-ticker
        (MultiIndex with (symbol, timestamp)) DataFrames returned by
        alpaca-py.
        """
        standard_cols = ["open", "high", "low", "close", "volume"]
        for col in standard_cols:
            if col not in df.columns:
                df[col] = 0.0

        # alpaca-py returns vwap as a column on intraday bars
        if "vwap" not in df.columns:
            df["vwap"] = float("nan")

        return df[standard_cols + ["vwap"]]

    def _handle_rate_limit(self, retry_after_s: float) -> None:
        """Back off when the Alpaca API rate limit is hit."""
        logger.warning(
            "Rate limited — sleeping %.1f s before retry", retry_after_s
        )
        time.sleep(retry_after_s)

    def _get_rest_client(self):
        """Return (and lazily create) the StockHistoricalDataClient."""
        if self._rest_client is None:
            self._rest_client = self._build_rest_client()
        return self._rest_client

    def _build_rest_client(self):
        """Instantiate and return the appropriate historical data client."""
        if self._is_crypto:
            from alpaca.data.historical import CryptoHistoricalDataClient
            return CryptoHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )
        from alpaca.data.historical import StockHistoricalDataClient
        return StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )

    def _build_stream_client(self):
        """Instantiate and return the appropriate streaming client."""
        if self._is_crypto:
            from alpaca.data.live import CryptoDataStream
            return CryptoDataStream(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )
        from alpaca.data.live import StockDataStream
        from alpaca.data.enums import DataFeed
        feed_enum = DataFeed(self._data_feed) if isinstance(self._data_feed, str) else self._data_feed
        return StockDataStream(
            api_key=self._api_key,
            secret_key=self._secret_key,
            feed=feed_enum,
        )
