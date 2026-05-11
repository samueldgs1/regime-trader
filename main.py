"""
regime_trader — entry point.

Orchestrates the full live-trading loop:
  1. Load configuration and credentials.
  2. Connect to Alpaca and initialise components.
  3. Download history and fit the HMM.
  4. Enter the market-hours event loop.
  5. On each bar: update features → classify regime →
     compute target allocation → validate risk → execute orders.
  6. Monitor, alert, and log throughout.

Usage:
    python main.py [--backtest] [--tickers SPY QQQ]
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_HALT_LOCK_PATH  = Path("logs/TRADING_HALTED.lock")
_STATE_FILE      = Path("logs/current_state.json")
_TRADES_FILE     = Path("logs/trade_log.jsonl")
_PRICE_FILE      = Path("logs/price_history.json")
_BAR_BUFFER_SIZE = 600   # rolling bars kept per ticker for live feature computation


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Flags
    -----
    --backtest  Run a walk-forward backtest instead of live trading.
    --tickers   Space-separated list of ticker symbols to trade.
    --config    Path to an optional JSON/YAML config override file.
    --log-level Logging verbosity (DEBUG, INFO, WARNING, ERROR).
    """
    parser = argparse.ArgumentParser(
        prog="regime_trader",
        description="HMM-based regime-aware automated trading bot.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        default=False,
        help="Run walk-forward backtest instead of live trading.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        metavar="TICKER",
        help="Space-separated list of ticker symbols to trade (e.g. SPY QQQ).",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to an optional JSON config override file.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(level: str = "INFO") -> None:
    """Configure root logger with console and rotating file handlers."""
    from logging.handlers import RotatingFileHandler

    from config.settings import MONITORING

    log_level = getattr(logging, level.upper(), logging.INFO)
    log_dir = Path(MONITORING.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(log_level)

    if not root.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root.addHandler(ch)

        fh = RotatingFileHandler(
            log_dir / "regime_trader.log",
            maxBytes=MONITORING.log_rotation_mb * 1024 * 1024,
            backupCount=5,
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)


# ---------------------------------------------------------------------------
# Component wiring
# ---------------------------------------------------------------------------

def build_components(args: argparse.Namespace) -> dict:
    """
    Instantiate and wire together all bot components.

    Returns
    -------
    Dict with keys: tickers, market_data, feature_eng, hmm_engine,
    strategy_router, risk_manager, order_executor,
    position_tracker, alert_dispatcher.
    """
    from config.settings import HMM, MONITORING, RISK, UNIVERSE
    from core.alerts import AlertDispatcher
    from core.feature_eng import FeatureEngineer
    from core.hmm_engine import HMMEngine
    from core.market_data import MarketData
    from core.order_executor import OrderExecutor
    from core.position_tracker import PositionTracker
    from core.regime_strategies import StrategyOrchestrator
    from core.risk_manager import RiskManager

    tickers = args.tickers or UNIVERSE.tickers

    # --- Alert dispatcher (no Alpaca dependency) ---
    try:
        from config import credentials as creds
        email_enabled = MONITORING.email_enabled and bool(creds.SMTP_HOST)
        webhook_enabled = MONITORING.webhook_enabled and bool(creds.WEBHOOK_URL)
        alert_dispatcher = AlertDispatcher(
            email_enabled=email_enabled,
            webhook_enabled=webhook_enabled,
            smtp_host=creds.SMTP_HOST,
            smtp_port=creds.SMTP_PORT,
            smtp_user=creds.SMTP_USER,
            smtp_password=creds.SMTP_PASSWORD,
            email_from=creds.ALERT_EMAIL_FROM,
            email_to=creds.ALERT_EMAIL_TO,
            webhook_url=creds.WEBHOOK_URL,
        )
    except EnvironmentError as exc:
        logger.critical("Credential load failed: %s", exc)
        raise SystemExit(1) from exc

    # --- Order executor ---
    def _alert_cb(msg: str) -> None:
        from core.alerts import Alert, AlertSeverity
        alert_dispatcher.custom(Alert(
            severity=AlertSeverity.WARNING,
            title="Repeated Order Failures",
            body=msg,
            timestamp=datetime.now(tz=timezone.utc),
            metadata={},
        ))

    try:
        order_executor = OrderExecutor.from_env(alert_callback=_alert_cb)
    except EnvironmentError as exc:
        logger.critical("OrderExecutor init failed: %s", exc)
        raise SystemExit(1) from exc

    # --- Position tracker ---
    position_tracker = PositionTracker()

    # --- Market data ---
    market_data = MarketData.from_env()

    # --- Feature engineer ---
    feature_eng = FeatureEngineer(feature_names=HMM.features)

    # --- HMM engine ---
    hmm_engine = HMMEngine(
        n_regimes=HMM.n_regimes_default,
        n_iter=HMM.n_iter,
        tol=HMM.tol,
        covariance_type=HMM.covariance_type,
        random_state=HMM.random_state,
        auto_select=HMM.auto_select_regimes,
        n_regimes_min=HMM.n_regimes_min,
        n_regimes_max=HMM.n_regimes_max,
    )

    # --- Strategy router ---
    strategy_router = StrategyOrchestrator(tickers=tickers)

    # --- Risk manager ---
    risk_manager = RiskManager(
        max_position_pct=RISK.max_position_pct,
        daily_loss_limit_pct=RISK.daily_loss_limit_pct,
        min_trade_size_usd=RISK.min_trade_size_usd,
        lock_file_path=_HALT_LOCK_PATH,
    )

    return {
        "tickers": tickers,
        "market_data": market_data,
        "feature_eng": feature_eng,
        "hmm_engine": hmm_engine,
        "strategy_router": strategy_router,
        "risk_manager": risk_manager,
        "order_executor": order_executor,
        "position_tracker": position_tracker,
        "alert_dispatcher": alert_dispatcher,
        # Rolling bar buffers keyed by ticker (populated in on_bar)
        "_bar_buffers": collections.defaultdict(list),
    }


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialise(components: dict, tickers: List[str]) -> None:
    """
    Download history, fit the HMM, sync positions from the broker,
    and verify connectivity before entering the main loop.
    """
    from config.settings import HMM

    hmm_engine = components["hmm_engine"]
    feature_eng = components["feature_eng"]
    market_data = components["market_data"]
    order_executor = components["order_executor"]
    position_tracker = components["position_tracker"]

    # 1. Check for halt lock
    if _HALT_LOCK_PATH.exists():
        content = _HALT_LOCK_PATH.read_text().strip()
        logger.critical(
            "TRADING_HALTED.lock exists — aborting startup.\n%s\n"
            "Delete %s manually to re-enable trading.",
            content,
            _HALT_LOCK_PATH,
        )
        raise SystemExit(1)

    # 2. Verify Alpaca connectivity
    logger.info("Verifying Alpaca connectivity…")
    try:
        order_executor.connect()
        account = order_executor.get_account()
        logger.info(
            "Alpaca connected — status=%s  buying_power=%s",
            account.get("status", "unknown"),
            account.get("buying_power", "unknown"),
        )
    except Exception as exc:
        logger.critical("Cannot connect to Alpaca: %s", exc)
        raise SystemExit(1) from exc

    # 3. Download ~2 years of daily bars for HMM training
    logger.info("Downloading %d days of history for %s…", HMM.training_window_days, tickers)
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=HMM.training_window_days * 2)
        bars_df = market_data.get_bars(
            tickers=tickers,
            start=start_date,
            end=end_date,
            timeframe="1Day",
        )
    except Exception as exc:
        logger.critical("Failed to download historical data: %s", exc)
        raise SystemExit(1) from exc

    # 4. Fit the HMM on primary ticker's history
    logger.info("Training HMM…")
    try:
        primary_bars = _extract_ticker_bars(bars_df, tickers[0])
        if primary_bars.empty:
            raise ValueError(f"No historical data returned for {tickers[0]}")

        features_df = feature_eng.build_features(primary_bars).dropna()
        if len(features_df) < 50:
            raise ValueError(
                f"Only {len(features_df)} feature rows — need at least 50 to train HMM"
            )

        feature_eng.fit_scaler(features_df)
        scaled = feature_eng.transform(features_df)
        hmm_engine.fit(scaled)
        logger.info(
            "HMM trained — n_regimes=%d  converged=%s",
            hmm_engine.n_regimes,
            hmm_engine.is_fitted,
        )

        # Pre-fill bar buffers with the most recent _BAR_BUFFER_SIZE bars
        for ticker in tickers:
            ticker_bars = _extract_ticker_bars(bars_df, ticker)
            if not ticker_bars.empty:
                recent = ticker_bars.tail(_BAR_BUFFER_SIZE)
                components["_bar_buffers"][ticker] = recent.to_dict("records")

    except Exception as exc:
        logger.critical("HMM training failed: %s", exc)
        raise SystemExit(1) from exc

    # 5. Sync positions from broker
    logger.info("Syncing positions from broker…")
    try:
        position_tracker.sync_from_broker(order_executor._client)
        snap = position_tracker.snapshot()
        logger.info(
            "Positions synced — %d open  NAV=$%.2f  cash=$%.2f",
            len(snap.positions),
            snap.nav,
            snap.cash,
        )
    except Exception as exc:
        logger.warning("Initial position sync failed (will retry in background): %s", exc)

    # 6. Start background position sync (daemon thread)
    position_tracker.start_background_sync(order_executor._client)

    logger.info("Initialisation complete — ready to trade %s", tickers)


def _extract_ticker_bars(bars_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Extract single-ticker OHLCV rows from a potentially multi-indexed DataFrame.

    alpaca-py returns a MultiIndex (symbol, timestamp) for multi-ticker requests.
    """
    if bars_df.empty:
        return pd.DataFrame()
    if isinstance(bars_df.index, pd.MultiIndex):
        try:
            return bars_df.xs(ticker, level="symbol")
        except KeyError:
            return pd.DataFrame()
    return bars_df


# ---------------------------------------------------------------------------
# Event loop
# ---------------------------------------------------------------------------

def run_live(components: dict, tickers: List[str]) -> None:
    """
    Enter the live trading loop.

    Steps per bar
    -------------
    1. Subscribe to streaming 5-min bars.
    2. On each bar: update rolling buffer, build features, detect regime,
       compute allocation, validate risk, execute rebalance trades.
    3. Periodically retrain the HMM (settings.HMM.retrain_interval_days).
    4. On shutdown signal: save final snapshot and stop all feeds.
    """
    from config.settings import HMM

    market_data = components["market_data"]

    logger.info("Entering live trading loop for %s…", tickers)

    last_retrain_day: Optional[date] = None
    _shutdown_event = threading.Event()
    components["_shutdown_event"] = _shutdown_event

    def _handle_signal(sig, frame):
        logger.info("Received signal %s — initiating graceful shutdown…", sig)
        _shutdown_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    def _bar_callback(ticker: str, bar: dict) -> None:
        try:
            on_bar(ticker, bar, components)
        except Exception as exc:
            logger.error("on_bar error for %s: %s", ticker, exc, exc_info=True)

    market_data.subscribe_bars(tickers, _bar_callback)

    stream_thread = threading.Thread(
        target=market_data.start_stream,
        daemon=True,
        name="market-data-stream",
    )
    stream_thread.start()

    logger.info("Market data stream started — waiting for bars…")

    try:
        while not _shutdown_event.is_set():
            today = date.today()
            if last_retrain_day is None or (today - last_retrain_day).days >= HMM.retrain_interval_days:
                _maybe_retrain(components, tickers)
                last_retrain_day = today
            _shutdown_event.wait(timeout=60)
    finally:
        shutdown(components)


def on_bar(ticker: str, bar: dict, components: dict) -> None:
    """
    Process a single incoming bar event.

    Called by the streaming callback registered in run_live().
    """
    from config.settings import MONITORING, RISK

    feature_eng = components["feature_eng"]
    hmm_engine = components["hmm_engine"]
    strategy_router = components["strategy_router"]
    risk_manager = components["risk_manager"]
    order_executor = components["order_executor"]
    position_tracker = components["position_tracker"]
    alert_dispatcher = components["alert_dispatcher"]
    tickers = components["tickers"]
    bar_buffers: Dict[str, list] = components["_bar_buffers"]

    ts = datetime.now(tz=timezone.utc)
    close_price = float(bar.get("close", 0))
    if close_price <= 0:
        return

    logger.debug("on_bar %s @ %s  close=%.4f", ticker, ts.isoformat(), close_price)

    # 1. Update rolling bar buffer
    bar_buffers[ticker].append(bar)
    if len(bar_buffers[ticker]) > _BAR_BUFFER_SIZE:
        bar_buffers[ticker] = bar_buffers[ticker][-_BAR_BUFFER_SIZE:]

    # 2. Update position prices
    position_tracker.update_prices({ticker: close_price})

    # 3. Get portfolio snapshot
    snap = position_tracker.snapshot()
    nav = snap.nav

    # 4. Check circuit breakers
    try:
        halted = risk_manager.check_circuit_breaker(nav=nav, timestamp=ts)
        if halted:
            logger.warning("Circuit breaker active (HALT_DAY or FULL_STOP) — skipping bar for %s", ticker)
            return
    except Exception as exc:
        logger.error("Circuit breaker check failed: %s", exc)
        return

    # 5. Build features from rolling bar buffer (need ≥50 bars)
    buf = bar_buffers[ticker]
    if len(buf) < 50:
        logger.debug("Bar buffer too small (%d) for %s — skipping regime detection", len(buf), ticker)
        return

    try:
        ohlcv_df = pd.DataFrame(buf)
        ohlcv_df.columns = [c.lower() for c in ohlcv_df.columns]
        for col in ("open", "high", "low", "close", "volume"):
            if col not in ohlcv_df.columns:
                ohlcv_df[col] = 0.0
        if "timestamp" in ohlcv_df.columns:
            ohlcv_df = ohlcv_df.set_index("timestamp")
        features_df = feature_eng.build_features(ohlcv_df).dropna()
        if features_df.empty:
            return
        scaled = feature_eng.transform(features_df)
    except Exception as exc:
        logger.warning("Feature computation failed for %s: %s", ticker, exc)
        return

    # 6. Detect regime
    try:
        regime_state = hmm_engine.detect_regime(scaled)
    except Exception as exc:
        logger.warning("Regime detection failed for %s: %s", ticker, exc)
        return

    # 7. Get allocation signal
    try:
        signal = strategy_router.get_signal(
            regime=regime_state.regime,
            regime_name=regime_state.regime_name,
            confidence=regime_state.confidence,
            is_uncertain=regime_state.is_uncertain,
        )
    except Exception as exc:
        logger.warning("Strategy signal failed: %s", exc)
        return

    logger.info(
        "Bar %s regime=%s(%d) conf=%.2f alloc=%.0f%%  uncertain=%s",
        ticker, signal.regime_name, signal.regime,
        signal.confidence, signal.allocation_pct * 100,
        signal.is_uncertain,
    )

    # 7b. Write dashboard state files (best-effort, never blocks trading)
    _write_dashboard_state(regime_state, risk_manager, nav)
    _write_price_history(ticker, bar_buffers, regime_state)

    # 8. Drawdown alert
    if nav > 0 and snap.unrealised_pnl / nav < -MONITORING.alert_drawdown_pct:
        alert_dispatcher.drawdown_warning(
            drawdown_pct=abs(snap.unrealised_pnl / nav),
            threshold_pct=MONITORING.alert_drawdown_pct,
            timestamp=ts,
        )

    # 9. Check if rebalance is needed
    current_invested_pct = snap.gross_exposure / nav if nav > 0 else 0.0
    if not strategy_router.should_rebalance(
        target_alloc_pct=signal.allocation_pct,
        current_alloc_pct=current_invested_pct,
    ):
        logger.debug("No rebalance needed (regime=%s)", signal.regime_name)
        return

    # 10. Compute target weights and execute rebalance
    try:
        target = strategy_router.compute_target_weights(signal=signal)
    except Exception as exc:
        logger.warning("Target weight computation failed: %s", exc)
        return

    for t, target_weight in target.items():
        current_pos = snap.positions.get(t)
        current_weight = (current_pos.market_value / nav) if current_pos and nav > 0 else 0.0
        delta_weight = target_weight - current_weight
        notional = abs(delta_weight) * nav

        if notional < RISK.min_trade_size_usd:
            continue

        side = "buy" if delta_weight > 0 else "sell"

        try:
            signed_amount = notional if side == "buy" else -notional
            position_values = {sym: pos.market_value for sym, pos in snap.positions.items()}
            risk_manager.validate_order(
                ticker=t,
                dollar_amount=signed_amount,
                current_positions=position_values,
                nav=nav,
                timestamp=ts,
            )
        except Exception as risk_exc:
            logger.warning("Risk rejected %s %s $%.0f: %s", side, t, notional, risk_exc)
            continue

        try:
            order_executor.market_order(ticker=t, notional=notional, side=side)
            logger.info("Order submitted: %s %s $%.2f", side, t, notional)

            fill_price = close_price if t == ticker else notional / max(notional, 1.0)
            fill_qty = notional / fill_price if fill_price > 0 else 0.0
            if fill_qty > 0:
                position_tracker.apply_fill(
                    ticker=t,
                    filled_qty=fill_qty,
                    filled_price=fill_price,
                    side=side,
                    timestamp=ts,
                )
                _append_trade_log(
                    ticker=t, side=side, notional=notional,
                    fill_price=fill_price, fill_qty=fill_qty,
                    regime_state=regime_state,
                    allocation_pct=signal.allocation_pct,
                    timestamp=ts,
                )
        except Exception as exc:
            logger.error("Order failed for %s: %s", t, exc)
            alert_dispatcher.order_failure(
                ticker=t, side=side, notional=notional, error=str(exc), timestamp=ts,
            )


def _maybe_retrain(components: dict, tickers: List[str]) -> None:
    """Re-fit the HMM on the most recent daily bars."""
    from config.settings import HMM

    hmm_engine = components["hmm_engine"]
    feature_eng = components["feature_eng"]
    market_data = components["market_data"]

    logger.info("Retraining HMM…")
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=HMM.training_window_days * 2)
        bars_df = market_data.get_bars(
            tickers=tickers,
            start=start_date,
            end=end_date,
            timeframe="1Day",
        )
        primary_bars = _extract_ticker_bars(bars_df, tickers[0])
        if not primary_bars.empty:
            features_df = feature_eng.build_features(primary_bars).dropna()
            feature_eng.fit_scaler(features_df)
            scaled = feature_eng.transform(features_df)
            hmm_engine.fit(scaled)
            logger.info("HMM retrain complete — n_regimes=%d", hmm_engine.n_regimes)
    except Exception as exc:
        logger.warning("HMM retrain failed: %s", exc)


# ---------------------------------------------------------------------------
# Backtest mode
# ---------------------------------------------------------------------------

def run_backtest(components: dict, tickers: List[str]) -> None:
    """
    Execute the walk-forward backtest and print a performance report.
    """
    from config.settings import BACKTEST, HMM
    from core.backtester import Backtester

    market_data = components["market_data"]
    feature_eng = components["feature_eng"]
    hmm_engine = components["hmm_engine"]
    strategy_router = components["strategy_router"]
    risk_manager = components["risk_manager"]

    logger.info("Running walk-forward backtest for %s…", tickers)

    try:
        needed_days = BACKTEST.in_sample_days + BACKTEST.out_of_sample_days + 60
        end_date = date.today()
        start_date = end_date - timedelta(days=needed_days * 2)
        bars_df = market_data.get_bars(
            tickers=tickers,
            start=start_date,
            end=end_date,
            timeframe="1Day",
        )
        primary_bars = _extract_ticker_bars(bars_df, tickers[0])
        if primary_bars.empty:
            logger.error("No historical data returned for backtest")
            return

        features_df = feature_eng.build_features(primary_bars).dropna()

        backtester = Backtester(
            features=features_df,
            price_series=primary_bars["close"].reindex(features_df.index),
            hmm_engine=hmm_engine,
            strategy_router=strategy_router,
            risk_manager=risk_manager,
            in_sample_days=BACKTEST.in_sample_days,
            out_of_sample_days=BACKTEST.out_of_sample_days,
            initial_capital=BACKTEST.initial_capital,
            commission_pct=BACKTEST.commission_pct,
            walk_forward_anchored=BACKTEST.walk_forward_anchored,
        )

        result = backtester.run()
        print(result.format_report())

    except Exception as exc:
        logger.error("Backtest failed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def shutdown(components: dict) -> None:
    """Gracefully stop all components and save a final state snapshot."""
    logger.info("Shutting down regime_trader…")

    position_tracker = components.get("position_tracker")
    market_data = components.get("market_data")
    alert_dispatcher = components.get("alert_dispatcher")

    if position_tracker:
        try:
            position_tracker.stop_background_sync()
        except Exception as exc:
            logger.warning("Error stopping position sync: %s", exc)

    if market_data:
        try:
            market_data.stop_stream()
        except Exception as exc:
            logger.warning("Error stopping market data stream: %s", exc)

    if position_tracker:
        try:
            snap = position_tracker.snapshot()
            _save_snapshot(snap)
            logger.info(
                "Final snapshot saved — NAV=$%.2f  positions=%d",
                snap.nav,
                len(snap.positions),
            )
            if alert_dispatcher:
                alert_dispatcher.daily_summary(
                    date=snap.timestamp,
                    pnl=snap.unrealised_pnl,
                    pnl_pct=snap.unrealised_pnl / snap.nav if snap.nav > 0 else 0.0,
                    regime=-1,
                    nav=snap.nav,
                )
        except Exception as exc:
            logger.warning("Failed to save final snapshot: %s", exc)

    logger.info("Shutdown complete.")


def _write_dashboard_state(
    regime_state: "RegimeState",
    risk_manager: "RiskManager",
    nav: float,
) -> None:
    """Write current bot state to logs/current_state.json for the dashboard."""
    try:
        _LOGS_DIR = Path("logs")
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        posteriors = regime_state.posteriors.tolist() if hasattr(regime_state.posteriors, "tolist") else list(regime_state.posteriors)
        state = {
            "timestamp":              datetime.now(tz=timezone.utc).isoformat(),
            "regime": {
                "label":       regime_state.regime,
                "name":        regime_state.regime_name,
                "confidence":  float(regime_state.confidence),
                "posteriors":  posteriors,
                "is_uncertain": regime_state.is_uncertain,
                "n_regimes":   len(posteriors),
            },
            "circuit_breaker_level": int(risk_manager.circuit_breaker_level),
            "size_multiplier":       float(risk_manager.size_multiplier),
            "session_start_nav":     float(risk_manager._session_start_nav),
            "peak_nav":              float(risk_manager._peak_nav),
        }
        _STATE_FILE.write_text(json.dumps(state))
    except Exception as exc:
        logger.debug("Dashboard state write failed: %s", exc)


def _append_trade_log(
    ticker: str,
    side: str,
    notional: float,
    fill_price: float,
    fill_qty: float,
    regime_state: "RegimeState",
    allocation_pct: float,
    timestamp: datetime,
) -> None:
    """Append a trade record to logs/trade_log.jsonl."""
    try:
        Path("logs").mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp":       timestamp.isoformat(),
            "ticker":          ticker,
            "side":            side,
            "notional":        round(notional, 2),
            "entry_price":     round(fill_price, 4),
            "qty":             round(fill_qty, 6),
            "allocation_pct":  round(allocation_pct, 4),
            "regime_at_entry": regime_state.regime_name,
            "regime_label":    regime_state.regime,
            "confidence":      round(float(regime_state.confidence), 4),
            "stop":            round(fill_price * 0.98, 4),
        }
        with open(_TRADES_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.debug("Trade log append failed: %s", exc)


def _write_price_history(
    ticker: str,
    bar_buffers: dict,
    regime_state: "RegimeState",
) -> None:
    """Write rolling price+regime history to logs/price_history.json."""
    try:
        buf = bar_buffers.get(ticker, [])
        if not buf:
            return
        # Only write the most recent 390 bars (one trading day at 1-min)
        recent = buf[-390:]
        bars_out = []
        n = len(recent)
        for i, b in enumerate(recent):
            ts = b.get("timestamp", "")
            if hasattr(ts, "isoformat"):
                ts = ts.isoformat()
            bars_out.append({
                "timestamp":   str(ts),
                "open":        float(b.get("open", 0)),
                "high":        float(b.get("high", 0)),
                "low":         float(b.get("low", 0)),
                "close":       float(b.get("close", 0)),
                "volume":      float(b.get("volume", 0)),
                # Assign current regime to the latest bar; older bars unknown
                "regime_name": regime_state.regime_name if i == n - 1 else b.get("regime_name", "unknown"),
                "confidence":  float(regime_state.confidence) if i == n - 1 else float(b.get("confidence", 0.5)),
            })
        Path("logs").mkdir(parents=True, exist_ok=True)
        _PRICE_FILE.write_text(json.dumps({"ticker": ticker, "bars": bars_out}))
    except Exception as exc:
        logger.debug("Price history write failed: %s", exc)


def _save_snapshot(snap: "PortfolioSnapshot") -> None:
    """Persist a PortfolioSnapshot to JSON in the logs directory."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    ts_str = snap.timestamp.strftime("%Y%m%dT%H%M%SZ")
    path = log_dir / f"snapshot_{ts_str}.json"

    data = {
        "timestamp": snap.timestamp.isoformat(),
        "cash": snap.cash,
        "nav": snap.nav,
        "gross_exposure": snap.gross_exposure,
        "net_exposure": snap.net_exposure,
        "unrealised_pnl": snap.unrealised_pnl,
        "weights": snap.weights,
        "positions": {
            ticker: {
                "qty": pos.qty,
                "avg_entry_price": pos.avg_entry_price,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "unrealised_pnl": pos.unrealised_pnl,
                "side": pos.side,
            }
            for ticker, pos in snap.positions.items()
        },
    }
    path.write_text(json.dumps(data, indent=2))
    logger.debug("Snapshot saved to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse args, wire components, and dispatch to live or backtest mode."""
    args = parse_args()

    log_level = args.log_level
    if log_level is None:
        from config.settings import MONITORING
        log_level = MONITORING.log_level

    _setup_logging(log_level)
    logger.info("regime_trader starting up…")

    components = build_components(args)
    tickers = args.tickers or components["tickers"]

    if args.backtest:
        run_backtest(components, tickers)
    else:
        initialise(components, tickers)
        run_live(components, tickers)


if __name__ == "__main__":
    main()
