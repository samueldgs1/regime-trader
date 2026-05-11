"""
Tests for main.py orchestrator and core/alerts.py AlertDispatcher.

Uses extensive mocking so no real Alpaca credentials are required.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path regardless of invocation
# ---------------------------------------------------------------------------
import os
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    """Minimal OHLCV DataFrame with n rows."""
    import numpy as np
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = 400 + rng.normal(0, 1, n).cumsum()
    df = pd.DataFrame(
        {
            "open":   close * 0.999,
            "high":   close * 1.002,
            "low":    close * 0.998,
            "close":  close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )
    return df


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ===========================================================================
# AlertDispatcher tests
# ===========================================================================

class TestAlertDispatcherInit:
    def _dispatcher(self, email=False, webhook=False):
        from core.alerts import AlertDispatcher
        return AlertDispatcher(
            email_enabled=email,
            webhook_enabled=webhook,
            smtp_host="smtp.example.com",
            smtp_port=587,
            smtp_user="user@example.com",
            smtp_password="secret",
            email_from="bot@example.com",
            email_to="ops@example.com",
            webhook_url="https://hooks.example.com/abc",
        )

    def test_init_stores_config(self):
        d = self._dispatcher(email=True, webhook=True)
        assert d._email_enabled is True
        assert d._webhook_enabled is True
        assert d._smtp_host == "smtp.example.com"
        assert d._smtp_port == 587
        assert d._webhook_url == "https://hooks.example.com/abc"

    def test_init_disabled_by_default(self):
        d = self._dispatcher()
        assert d._email_enabled is False
        assert d._webhook_enabled is False


class TestBuildAlert:
    def _dispatcher(self):
        from core.alerts import AlertDispatcher
        return AlertDispatcher(
            email_enabled=False, webhook_enabled=False,
            smtp_host="", smtp_port=587, smtp_user="",
            smtp_password="", email_from="", email_to="", webhook_url="",
        )

    def test_build_alert_fields(self):
        from core.alerts import AlertSeverity
        d = self._dispatcher()
        alert = d._build_alert(
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            body="Something happened",
            metadata={"key": "val"},
        )
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "Test Alert"
        assert alert.body == "Something happened"
        assert alert.metadata == {"key": "val"}
        assert isinstance(alert.timestamp, datetime)

    def test_build_alert_default_metadata(self):
        from core.alerts import AlertSeverity
        d = self._dispatcher()
        alert = d._build_alert(AlertSeverity.INFO, "T", "B")
        assert alert.metadata == {}


class TestAlertLog:
    def _dispatcher(self):
        from core.alerts import AlertDispatcher
        return AlertDispatcher(
            email_enabled=False, webhook_enabled=False,
            smtp_host="", smtp_port=587, smtp_user="",
            smtp_password="", email_from="", email_to="", webhook_url="",
        )

    def test_log_critical(self):
        from core.alerts import Alert, AlertSeverity
        d = self._dispatcher()
        alert = Alert(
            severity=AlertSeverity.CRITICAL,
            title="Halt",
            body="Full stop triggered",
            timestamp=_now(),
            metadata={},
        )
        with patch("core.alerts.logger") as mock_log:
            d._log(alert)
            mock_log.critical.assert_called_once()

    def test_log_warning(self):
        from core.alerts import Alert, AlertSeverity
        d = self._dispatcher()
        alert = Alert(
            severity=AlertSeverity.WARNING,
            title="Drawdown",
            body="10%",
            timestamp=_now(),
            metadata={},
        )
        with patch("core.alerts.logger") as mock_log:
            d._log(alert)
            mock_log.warning.assert_called_once()

    def test_log_info(self):
        from core.alerts import Alert, AlertSeverity
        d = self._dispatcher()
        alert = Alert(
            severity=AlertSeverity.INFO,
            title="Summary",
            body="Daily",
            timestamp=_now(),
            metadata={},
        )
        with patch("core.alerts.logger") as mock_log:
            d._log(alert)
            mock_log.info.assert_called_once()


class TestRegimeChangeAlert:
    def _dispatcher(self):
        from core.alerts import AlertDispatcher
        return AlertDispatcher(
            email_enabled=False, webhook_enabled=False,
            smtp_host="", smtp_port=587, smtp_user="",
            smtp_password="", email_from="", email_to="", webhook_url="",
        )

    def test_regime_change_builds_info_alert(self):
        from core.alerts import AlertSeverity
        d = self._dispatcher()
        with patch.object(d, "custom") as mock_custom:
            d.regime_change(old_regime=1, new_regime=2, timestamp=_now())
            mock_custom.assert_called_once()
            alert = mock_custom.call_args[0][0]
            assert alert.severity == AlertSeverity.INFO
            assert "1" in alert.title or "2" in alert.title
            assert alert.metadata["old_regime"] == 1
            assert alert.metadata["new_regime"] == 2

    def test_regime_change_dispatches_to_log(self):
        d = self._dispatcher()
        with patch("core.alerts.logger") as mock_log:
            d.regime_change(old_regime=0, new_regime=3, timestamp=_now())
            assert mock_log.info.called or mock_log.critical.called or mock_log.warning.called


class TestCircuitBreakerAlert:
    def _dispatcher(self):
        from core.alerts import AlertDispatcher
        return AlertDispatcher(
            email_enabled=False, webhook_enabled=False,
            smtp_host="", smtp_port=587, smtp_user="",
            smtp_password="", email_from="", email_to="", webhook_url="",
        )

    def test_circuit_breaker_triggered_is_critical(self):
        from core.alerts import AlertSeverity
        d = self._dispatcher()
        with patch.object(d, "custom") as mock_custom:
            d.circuit_breaker_triggered(
                reason="10% drawdown",
                nav=90_000.0,
                drawdown_pct=0.10,
                timestamp=_now(),
            )
            alert = mock_custom.call_args[0][0]
            assert alert.severity == AlertSeverity.CRITICAL
            assert "90,000" in alert.body or "90000" in alert.body
            assert alert.metadata["drawdown_pct"] == 0.10


class TestOrderFailureAlert:
    def _dispatcher(self):
        from core.alerts import AlertDispatcher
        return AlertDispatcher(
            email_enabled=False, webhook_enabled=False,
            smtp_host="", smtp_port=587, smtp_user="",
            smtp_password="", email_from="", email_to="", webhook_url="",
        )

    def test_order_failure_is_warning(self):
        from core.alerts import AlertSeverity
        d = self._dispatcher()
        with patch.object(d, "custom") as mock_custom:
            d.order_failure(
                ticker="SPY",
                side="buy",
                notional=5000.0,
                error="Connection timeout",
                timestamp=_now(),
            )
            alert = mock_custom.call_args[0][0]
            assert alert.severity == AlertSeverity.WARNING
            assert "SPY" in alert.title
            assert alert.metadata["ticker"] == "SPY"
            assert alert.metadata["notional"] == 5000.0


class TestDrawdownWarningAlert:
    def _dispatcher(self):
        from core.alerts import AlertDispatcher
        return AlertDispatcher(
            email_enabled=False, webhook_enabled=False,
            smtp_host="", smtp_port=587, smtp_user="",
            smtp_password="", email_from="", email_to="", webhook_url="",
        )

    def test_drawdown_warning_is_warning(self):
        from core.alerts import AlertSeverity
        d = self._dispatcher()
        with patch.object(d, "custom") as mock_custom:
            d.drawdown_warning(drawdown_pct=0.12, threshold_pct=0.10, timestamp=_now())
            alert = mock_custom.call_args[0][0]
            assert alert.severity == AlertSeverity.WARNING
            assert alert.metadata["drawdown_pct"] == 0.12


class TestVolSpikeAlert:
    def _dispatcher(self):
        from core.alerts import AlertDispatcher
        return AlertDispatcher(
            email_enabled=False, webhook_enabled=False,
            smtp_host="", smtp_port=587, smtp_user="",
            smtp_password="", email_from="", email_to="", webhook_url="",
        )

    def test_vol_spike_is_warning(self):
        from core.alerts import AlertSeverity
        d = self._dispatcher()
        with patch.object(d, "custom") as mock_custom:
            d.vol_spike(current_vol=0.40, rolling_avg_vol=0.15, spike_factor=2.67, timestamp=_now())
            alert = mock_custom.call_args[0][0]
            assert alert.severity == AlertSeverity.WARNING
            assert alert.metadata["spike_factor"] == 2.67


class TestDailySummaryAlert:
    def _dispatcher(self):
        from core.alerts import AlertDispatcher
        return AlertDispatcher(
            email_enabled=False, webhook_enabled=False,
            smtp_host="", smtp_port=587, smtp_user="",
            smtp_password="", email_from="", email_to="", webhook_url="",
        )

    def test_daily_summary_is_info(self):
        from core.alerts import AlertSeverity
        d = self._dispatcher()
        with patch.object(d, "custom") as mock_custom:
            d.daily_summary(
                date=_now(), pnl=1200.0, pnl_pct=0.012, regime=2, nav=102_000.0,
            )
            alert = mock_custom.call_args[0][0]
            assert alert.severity == AlertSeverity.INFO
            assert "102,000" in alert.body or "102000" in alert.body

    def test_daily_summary_negative_pnl(self):
        from core.alerts import AlertSeverity
        d = self._dispatcher()
        with patch.object(d, "custom") as mock_custom:
            d.daily_summary(
                date=_now(), pnl=-500.0, pnl_pct=-0.005, regime=1, nav=99_500.0,
            )
            alert = mock_custom.call_args[0][0]
            assert alert.severity == AlertSeverity.INFO


class TestCustomAlert:
    def _dispatcher(self, email=False, webhook=False):
        from core.alerts import AlertDispatcher
        return AlertDispatcher(
            email_enabled=email, webhook_enabled=webhook,
            smtp_host="smtp.ex.com", smtp_port=587, smtp_user="u",
            smtp_password="p", email_from="f@e.com", email_to="t@e.com",
            webhook_url="https://hook.example.com",
        )

    def test_custom_calls_log_always(self):
        from core.alerts import Alert, AlertSeverity
        d = self._dispatcher()
        alert = Alert(severity=AlertSeverity.INFO, title="T", body="B", timestamp=_now(), metadata={})
        with patch.object(d, "_log") as mock_log:
            d.custom(alert)
            mock_log.assert_called_once_with(alert)

    def test_custom_calls_email_when_enabled(self):
        from core.alerts import Alert, AlertSeverity
        d = self._dispatcher(email=True)
        alert = Alert(severity=AlertSeverity.INFO, title="T", body="B", timestamp=_now(), metadata={})
        with patch.object(d, "_log"), patch.object(d, "_send_email") as mock_email:
            d.custom(alert)
            mock_email.assert_called_once_with(alert)

    def test_custom_skips_email_when_disabled(self):
        from core.alerts import Alert, AlertSeverity
        d = self._dispatcher(email=False)
        alert = Alert(severity=AlertSeverity.INFO, title="T", body="B", timestamp=_now(), metadata={})
        with patch.object(d, "_log"), patch.object(d, "_send_email") as mock_email:
            d.custom(alert)
            mock_email.assert_not_called()

    def test_custom_calls_webhook_when_enabled(self):
        from core.alerts import Alert, AlertSeverity
        d = self._dispatcher(webhook=True)
        alert = Alert(severity=AlertSeverity.INFO, title="T", body="B", timestamp=_now(), metadata={})
        with patch.object(d, "_log"), patch.object(d, "_send_webhook") as mock_wh:
            d.custom(alert)
            mock_wh.assert_called_once_with(alert)


class TestFormatHelpers:
    def _dispatcher(self):
        from core.alerts import AlertDispatcher
        return AlertDispatcher(
            email_enabled=False, webhook_enabled=False,
            smtp_host="", smtp_port=587, smtp_user="",
            smtp_password="", email_from="", email_to="", webhook_url="",
        )

    def test_format_email_body_contains_severity(self):
        from core.alerts import Alert, AlertSeverity
        d = self._dispatcher()
        alert = Alert(
            severity=AlertSeverity.CRITICAL,
            title="Test",
            body="Body text here.",
            timestamp=_now(),
            metadata={"k": "v"},
        )
        body = d._format_email_body(alert)
        assert "CRITICAL" in body
        assert "Body text here." in body
        assert "k" in body

    def test_format_webhook_payload_is_json_serializable(self):
        from core.alerts import Alert, AlertSeverity
        d = self._dispatcher()
        alert = Alert(
            severity=AlertSeverity.WARNING,
            title="Webhook test",
            body="payload",
            timestamp=_now(),
            metadata={"num": 42},
        )
        payload = d._format_webhook_payload(alert)
        # Must be JSON-serializable
        serialized = json.dumps(payload)
        parsed = json.loads(serialized)
        assert parsed["severity"] == "WARNING"
        assert parsed["title"] == "Webhook test"
        assert parsed["metadata"]["num"] == 42

    def test_format_webhook_payload_keys(self):
        from core.alerts import Alert, AlertSeverity
        d = self._dispatcher()
        alert = Alert(severity=AlertSeverity.INFO, title="T", body="B", timestamp=_now(), metadata={})
        payload = d._format_webhook_payload(alert)
        for key in ("severity", "title", "body", "timestamp", "metadata"):
            assert key in payload


class TestSendEmailErrors:
    def test_smtp_error_is_logged_not_raised(self):
        from core.alerts import Alert, AlertSeverity, AlertDispatcher
        d = AlertDispatcher(
            email_enabled=True, webhook_enabled=False,
            smtp_host="bad-host.invalid", smtp_port=587, smtp_user="u",
            smtp_password="p", email_from="f@e.com", email_to="t@e.com",
            webhook_url="",
        )
        alert = Alert(severity=AlertSeverity.INFO, title="T", body="B", timestamp=_now(), metadata={})
        import smtplib
        with patch("smtplib.SMTP", side_effect=OSError("connection refused")):
            with patch("core.alerts.logger") as mock_log:
                d._send_email(alert)  # must not raise
                mock_log.error.assert_called_once()


class TestSendWebhookErrors:
    def test_webhook_error_is_logged_not_raised(self):
        from core.alerts import Alert, AlertSeverity, AlertDispatcher
        d = AlertDispatcher(
            email_enabled=False, webhook_enabled=True,
            smtp_host="", smtp_port=587, smtp_user="",
            smtp_password="", email_from="", email_to="",
            webhook_url="https://hooks.example.com/test",
        )
        alert = Alert(severity=AlertSeverity.CRITICAL, title="T", body="B", timestamp=_now(), metadata={})
        import urllib.request
        with patch("urllib.request.urlopen", side_effect=OSError("network error")):
            with patch("core.alerts.logger") as mock_log:
                d._send_webhook(alert)  # must not raise
                mock_log.error.assert_called_once()


# ===========================================================================
# parse_args tests
# ===========================================================================

class TestParseArgs:
    def test_defaults(self):
        from main import parse_args
        args = parse_args([])
        assert args.backtest is False
        assert args.tickers is None
        assert args.config is None
        assert args.log_level is None

    def test_backtest_flag(self):
        from main import parse_args
        args = parse_args(["--backtest"])
        assert args.backtest is True

    def test_tickers(self):
        from main import parse_args
        args = parse_args(["--tickers", "SPY", "QQQ", "IWM"])
        assert args.tickers == ["SPY", "QQQ", "IWM"]

    def test_config_path(self):
        from main import parse_args
        args = parse_args(["--config", "/tmp/myconfig.json"])
        assert args.config == "/tmp/myconfig.json"

    def test_log_level(self):
        from main import parse_args
        args = parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_combined_args(self):
        from main import parse_args
        args = parse_args(["--backtest", "--tickers", "SPY", "--log-level", "WARNING"])
        assert args.backtest is True
        assert args.tickers == ["SPY"]
        assert args.log_level == "WARNING"


# ===========================================================================
# _setup_logging tests
# ===========================================================================

class TestSetupLogging:
    def test_setup_logging_does_not_raise(self, tmp_path):
        from main import _setup_logging
        from config.settings import MONITORING
        original = MONITORING.log_dir
        MONITORING.log_dir = str(tmp_path / "logs")
        try:
            _setup_logging("INFO")
        finally:
            MONITORING.log_dir = original
            # Clean up handlers to avoid interference with other tests
            import logging
            root = logging.getLogger()
            root.handlers.clear()

    def test_setup_logging_respects_level(self, tmp_path):
        import logging
        from main import _setup_logging
        from config.settings import MONITORING
        original = MONITORING.log_dir
        MONITORING.log_dir = str(tmp_path / "logs")
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        root.handlers.clear()
        try:
            _setup_logging("DEBUG")
            assert root.level == logging.DEBUG
        finally:
            root.handlers.clear()
            root.handlers.extend(original_handlers)
            MONITORING.log_dir = original


# ===========================================================================
# _save_snapshot tests
# ===========================================================================

class TestSaveSnapshot:
    def test_save_snapshot_creates_file(self, tmp_path):
        from main import _save_snapshot
        from core.position_tracker import PortfolioSnapshot

        snap = PortfolioSnapshot(
            timestamp=_now(),
            cash=10_000.0,
            nav=110_000.0,
            positions={},
            gross_exposure=100_000.0,
            net_exposure=100_000.0,
            long_exposure=100_000.0,
            short_exposure=0.0,
            unrealised_pnl=5_000.0,
            weights={"SPY": 0.9},
        )

        with patch("main.Path", lambda s: tmp_path if s == "logs" else Path(s)):
            # Call with patched log dir
            import main as m
            original = m.Path
            m.Path = lambda s: tmp_path if s == "logs" else original(s)
            try:
                _save_snapshot(snap)
            finally:
                m.Path = original

    def test_save_snapshot_valid_json(self, tmp_path):
        from main import _save_snapshot
        from core.position_tracker import PortfolioSnapshot

        snap = PortfolioSnapshot(
            timestamp=_now(),
            cash=5_000.0,
            nav=50_000.0,
            positions={},
            gross_exposure=45_000.0,
            net_exposure=45_000.0,
            long_exposure=45_000.0,
            short_exposure=0.0,
            unrealised_pnl=500.0,
            weights={},
        )

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        with patch("main.Path") as mock_path:
            mock_path.return_value = log_dir
            # Redirect Path("logs") to tmp_path/logs
            import main as m
            original_path = m.Path

            def patched_path(s):
                if s == "logs":
                    return log_dir
                return original_path(s)

            m.Path = patched_path
            try:
                _save_snapshot(snap)
                files = list(log_dir.iterdir())
                assert len(files) >= 1
                data = json.loads(files[0].read_text())
                assert "nav" in data
                assert data["cash"] == 5_000.0
            finally:
                m.Path = original_path


# ===========================================================================
# _extract_ticker_bars tests
# ===========================================================================

class TestExtractTickerBars:
    def test_single_index_returns_same_df(self):
        from main import _extract_ticker_bars
        df = _make_ohlcv(50)
        result = _extract_ticker_bars(df, "SPY")
        assert len(result) == 50

    def test_multi_index_extracts_ticker(self):
        from main import _extract_ticker_bars
        df = _make_ohlcv(30)
        # Create multi-index with (symbol, timestamp)
        multi = pd.concat({"SPY": df, "QQQ": df}, names=["symbol"])
        result = _extract_ticker_bars(multi, "SPY")
        assert len(result) == 30

    def test_multi_index_missing_ticker_returns_empty(self):
        from main import _extract_ticker_bars
        df = _make_ohlcv(20)
        multi = pd.concat({"QQQ": df}, names=["symbol"])
        result = _extract_ticker_bars(multi, "SPY")
        assert result.empty

    def test_empty_df_returns_empty(self):
        from main import _extract_ticker_bars
        result = _extract_ticker_bars(pd.DataFrame(), "SPY")
        assert result.empty


# ===========================================================================
# build_components tests (heavily mocked)
# ===========================================================================

class TestBuildComponents:
    def _mock_args(self, tickers=None, backtest=False):
        ns = SimpleNamespace(tickers=tickers, backtest=backtest, config=None, log_level=None)
        return ns

    def _patch_all(self):
        """Return a dict of patches to apply."""
        return {
            "core.order_executor.OrderExecutor.from_env": MagicMock(return_value=MagicMock()),
            "core.market_data.MarketData.from_env": MagicMock(return_value=MagicMock()),
        }

    @patch.dict("os.environ", {
        "ALPACA_API_KEY": "test_key",
        "ALPACA_SECRET_KEY": "test_secret",
    })
    def test_build_components_returns_required_keys(self):
        from main import build_components
        args = self._mock_args(tickers=["SPY"])

        mock_executor = MagicMock()
        mock_md = MagicMock()

        with patch("core.order_executor.OrderExecutor.from_env", return_value=mock_executor), \
             patch("core.market_data.MarketData.from_env", return_value=mock_md):
            components = build_components(args)

        required = {
            "tickers", "market_data", "feature_eng", "hmm_engine",
            "strategy_router", "risk_manager", "order_executor",
            "position_tracker", "alert_dispatcher", "_bar_buffers",
        }
        for key in required:
            assert key in components, f"Missing key: {key}"

    @patch.dict("os.environ", {
        "ALPACA_API_KEY": "test_key",
        "ALPACA_SECRET_KEY": "test_secret",
    })
    def test_build_components_uses_args_tickers(self):
        from main import build_components
        args = self._mock_args(tickers=["QQQ", "IWM"])

        with patch("core.order_executor.OrderExecutor.from_env", return_value=MagicMock()), \
             patch("core.market_data.MarketData.from_env", return_value=MagicMock()):
            components = build_components(args)

        assert components["tickers"] == ["QQQ", "IWM"]

    @patch.dict("os.environ", {
        "ALPACA_API_KEY": "test_key",
        "ALPACA_SECRET_KEY": "test_secret",
    })
    def test_build_components_falls_back_to_settings_tickers(self):
        from main import build_components
        from config.settings import UNIVERSE
        args = self._mock_args(tickers=None)

        with patch("core.order_executor.OrderExecutor.from_env", return_value=MagicMock()), \
             patch("core.market_data.MarketData.from_env", return_value=MagicMock()):
            components = build_components(args)

        assert components["tickers"] == UNIVERSE.tickers


# ===========================================================================
# initialise tests
# ===========================================================================

class TestInitialise:
    def _make_components(self, tmp_path):
        from core.position_tracker import PositionTracker, PortfolioSnapshot
        from core.alerts import AlertDispatcher
        from core.hmm_engine import HMMEngine
        from core.feature_eng import FeatureEngineer
        from config.settings import HMM

        mock_executor = MagicMock()
        mock_executor.get_account.return_value = SimpleNamespace(status="ACTIVE", buying_power="100000")

        mock_md = MagicMock()
        ohlcv = _make_ohlcv(600)
        mock_md.get_bars.return_value = ohlcv  # single-ticker df

        mock_pt = MagicMock(spec=PositionTracker)
        mock_pt.snapshot.return_value = PortfolioSnapshot(
            timestamp=_now(), cash=100_000.0, nav=100_000.0, positions={},
            gross_exposure=0.0, net_exposure=0.0, long_exposure=0.0,
            short_exposure=0.0, unrealised_pnl=0.0, weights={},
        )

        mock_hmm = MagicMock()
        mock_hmm.n_regimes = 4
        mock_hmm.is_fitted = True

        mock_feat = MagicMock()
        ohlcv_features = _make_ohlcv(600).rename(columns=lambda c: c)
        mock_feat.build_features.return_value = ohlcv_features.drop(columns=["volume"])
        mock_feat.transform.return_value = ohlcv_features.drop(columns=["volume"]).values

        mock_dispatcher = MagicMock(spec=AlertDispatcher)

        return {
            "tickers": ["SPY"],
            "market_data": mock_md,
            "feature_eng": mock_feat,
            "hmm_engine": mock_hmm,
            "strategy_router": MagicMock(),
            "risk_manager": MagicMock(),
            "order_executor": mock_executor,
            "position_tracker": mock_pt,
            "alert_dispatcher": mock_dispatcher,
            "_bar_buffers": {},
        }

    def test_aborts_when_halt_lock_exists(self, tmp_path):
        from main import initialise, _HALT_LOCK_PATH
        import main as m

        lock = tmp_path / "TRADING_HALTED.lock"
        lock.write_text("halted for test")

        components = self._make_components(tmp_path)
        original_lock = m._HALT_LOCK_PATH
        m._HALT_LOCK_PATH = lock
        try:
            with pytest.raises(SystemExit):
                initialise(components, ["SPY"])
        finally:
            m._HALT_LOCK_PATH = original_lock

    def test_aborts_when_alpaca_unreachable(self, tmp_path):
        from main import initialise
        components = self._make_components(tmp_path)
        components["order_executor"].get_account.side_effect = ConnectionError("unreachable")

        with pytest.raises(SystemExit):
            initialise(components, ["SPY"])

    def test_successful_init_calls_hmm_fit(self, tmp_path):
        from main import initialise
        components = self._make_components(tmp_path)

        initialise(components, ["SPY"])
        components["hmm_engine"].fit.assert_called_once()

    def test_successful_init_starts_background_sync(self, tmp_path):
        from main import initialise
        components = self._make_components(tmp_path)

        initialise(components, ["SPY"])
        components["position_tracker"].start_background_sync.assert_called_once()

    def test_aborts_when_no_historical_data(self, tmp_path):
        from main import initialise
        components = self._make_components(tmp_path)
        components["market_data"].get_bars.return_value = pd.DataFrame()

        with pytest.raises(SystemExit):
            initialise(components, ["SPY"])


# ===========================================================================
# shutdown tests
# ===========================================================================

class TestShutdown:
    def _make_components(self):
        from core.position_tracker import PortfolioSnapshot
        mock_pt = MagicMock()
        mock_pt.snapshot.return_value = PortfolioSnapshot(
            timestamp=_now(), cash=100_000.0, nav=100_000.0, positions={},
            gross_exposure=0.0, net_exposure=0.0, long_exposure=0.0,
            short_exposure=0.0, unrealised_pnl=0.0, weights={},
        )
        return {
            "position_tracker": mock_pt,
            "market_data": MagicMock(),
            "alert_dispatcher": MagicMock(),
        }

    def test_shutdown_stops_position_sync(self):
        from main import shutdown
        components = self._make_components()
        with patch("main._save_snapshot"):
            shutdown(components)
        components["position_tracker"].stop_background_sync.assert_called_once()

    def test_shutdown_stops_stream(self):
        from main import shutdown
        components = self._make_components()
        with patch("main._save_snapshot"):
            shutdown(components)
        components["market_data"].stop_stream.assert_called_once()

    def test_shutdown_sends_daily_summary(self):
        from main import shutdown
        components = self._make_components()
        with patch("main._save_snapshot"):
            shutdown(components)
        components["alert_dispatcher"].daily_summary.assert_called_once()

    def test_shutdown_tolerates_stop_errors(self):
        from main import shutdown
        components = self._make_components()
        components["position_tracker"].stop_background_sync.side_effect = RuntimeError("boom")
        components["market_data"].stop_stream.side_effect = RuntimeError("boom")
        with patch("main._save_snapshot"):
            shutdown(components)  # must not raise


# ===========================================================================
# on_bar tests
# ===========================================================================

class TestOnBar:
    def _make_components(self):
        from core.position_tracker import PortfolioSnapshot, Position
        from core.alerts import AlertDispatcher
        from core.risk_manager import CircuitBreakerLevel

        snap = PortfolioSnapshot(
            timestamp=_now(), cash=80_000.0, nav=100_000.0,
            positions={"SPY": Position(
                ticker="SPY", qty=50.0, avg_entry_price=400.0,
                current_price=400.0, market_value=20_000.0,
                unrealised_pnl=0.0, unrealised_pnl_pct=0.0,
                cost_basis=20_000.0, side="long", opened_at=_now(),
            )},
            gross_exposure=20_000.0, net_exposure=20_000.0,
            long_exposure=20_000.0, short_exposure=0.0,
            unrealised_pnl=0.0, weights={"SPY": 0.2},
        )

        mock_pt = MagicMock()
        mock_pt.snapshot.return_value = snap

        mock_risk = MagicMock()
        mock_risk.check_circuit_breaker.return_value = False  # not halted

        mock_hmm = MagicMock()
        from core.hmm_engine import RegimeState
        import numpy as np
        mock_hmm.detect_regime.return_value = RegimeState(
            regime=2, regime_name="bull", raw_regime=1,
            confidence=0.85, is_uncertain=False,
            posteriors=np.array([0.05, 0.10, 0.85]),
            flicker_count=0,
        )

        mock_feat = MagicMock()
        mock_feat.build_features.return_value = _make_ohlcv(100).drop(columns=["volume"])
        mock_feat.transform.return_value = _make_ohlcv(100).drop(columns=["volume"]).values

        mock_strategy = MagicMock()
        from core.regime_strategies import SignalData
        mock_strategy.get_signal.return_value = SignalData(
            regime=2, regime_name="bull", allocation_pct=0.95,
            leverage=1.0, confidence=0.85, is_uncertain=False,
            notes="Bull", rebalance_required=False,
        )
        mock_strategy.should_rebalance.return_value = True
        mock_strategy.compute_target_weights.return_value = {"SPY": 0.95}

        mock_executor = MagicMock()
        mock_dispatcher = MagicMock(spec=AlertDispatcher)

        # Pre-fill bar buffer with 100 bars
        buf = _make_ohlcv(100).reset_index().to_dict("records")

        return {
            "tickers": ["SPY"],
            "feature_eng": mock_feat,
            "hmm_engine": mock_hmm,
            "strategy_router": mock_strategy,
            "risk_manager": mock_risk,
            "order_executor": mock_executor,
            "position_tracker": mock_pt,
            "alert_dispatcher": mock_dispatcher,
            "_bar_buffers": {"SPY": buf},
        }

    def test_updates_position_prices(self):
        from main import on_bar
        components = self._make_components()
        bar = {"close": 410.0, "open": 408.0, "high": 412.0, "low": 407.0, "volume": 1_000_000}
        on_bar("SPY", bar, components)
        components["position_tracker"].update_prices.assert_called_with({"SPY": 410.0})

    def test_skips_on_halt(self):
        from main import on_bar
        components = self._make_components()
        components["risk_manager"].check_circuit_breaker.return_value = True  # halted
        bar = {"close": 410.0, "open": 408.0, "high": 412.0, "low": 407.0, "volume": 1_000_000}
        on_bar("SPY", bar, components)
        # HMM should not be called when halted
        components["hmm_engine"].detect_regime.assert_not_called()

    def test_skips_when_zero_close(self):
        from main import on_bar
        components = self._make_components()
        bar = {"close": 0.0, "open": 0.0, "high": 0.0, "low": 0.0, "volume": 0}
        on_bar("SPY", bar, components)
        components["position_tracker"].update_prices.assert_not_called()

    def test_submits_order_on_rebalance(self):
        from main import on_bar
        components = self._make_components()
        bar = {"close": 410.0, "open": 408.0, "high": 412.0, "low": 407.0, "volume": 1_000_000}
        on_bar("SPY", bar, components)
        components["order_executor"].market_order.assert_called_once()

    def test_no_order_when_no_rebalance_needed(self):
        from main import on_bar
        components = self._make_components()
        components["strategy_router"].should_rebalance.return_value = False
        bar = {"close": 410.0, "open": 408.0, "high": 412.0, "low": 407.0, "volume": 1_000_000}
        on_bar("SPY", bar, components)
        components["order_executor"].market_order.assert_not_called()

    def test_alert_on_order_failure(self):
        from main import on_bar
        components = self._make_components()
        components["order_executor"].market_order.side_effect = RuntimeError("timeout")
        bar = {"close": 410.0, "open": 408.0, "high": 412.0, "low": 407.0, "volume": 1_000_000}
        on_bar("SPY", bar, components)
        components["alert_dispatcher"].order_failure.assert_called_once()

    def test_skips_when_buffer_too_small(self):
        from main import on_bar
        components = self._make_components()
        components["_bar_buffers"]["SPY"] = []  # empty buffer
        bar = {"close": 410.0, "open": 408.0, "high": 412.0, "low": 407.0, "volume": 1_000_000}
        on_bar("SPY", bar, components)
        components["hmm_engine"].detect_regime.assert_not_called()

    def test_bar_appended_to_buffer(self):
        from main import on_bar
        components = self._make_components()
        initial_len = len(components["_bar_buffers"]["SPY"])
        bar = {"close": 410.0, "open": 408.0, "high": 412.0, "low": 407.0, "volume": 1_000_000}
        on_bar("SPY", bar, components)
        assert len(components["_bar_buffers"]["SPY"]) == initial_len + 1

    def test_drawdown_alert_fired(self):
        from main import on_bar
        from core.position_tracker import PortfolioSnapshot
        components = self._make_components()
        # Force 15% drawdown
        snap_dd = PortfolioSnapshot(
            timestamp=_now(), cash=80_000.0, nav=100_000.0, positions={},
            gross_exposure=20_000.0, net_exposure=20_000.0, long_exposure=20_000.0,
            short_exposure=0.0, unrealised_pnl=-15_000.0, weights={},
        )
        components["position_tracker"].snapshot.return_value = snap_dd
        bar = {"close": 410.0, "open": 408.0, "high": 412.0, "low": 407.0, "volume": 1_000_000}
        on_bar("SPY", bar, components)
        components["alert_dispatcher"].drawdown_warning.assert_called_once()


# ===========================================================================
# run_backtest smoke test
# ===========================================================================

class TestRunBacktest:
    def test_run_backtest_calls_get_bars(self):
        from main import run_backtest

        mock_md = MagicMock()
        ohlcv = _make_ohlcv(800)
        mock_md.get_bars.return_value = ohlcv

        mock_feat = MagicMock()
        feat_df = ohlcv.drop(columns=["volume"])
        mock_feat.build_features.return_value = feat_df

        mock_backtester_cls = MagicMock()
        mock_result = MagicMock()
        mock_result.format_report.return_value = "Report"
        mock_backtester_cls.return_value.run.return_value = mock_result

        components = {
            "market_data": mock_md,
            "feature_eng": mock_feat,
            "hmm_engine": MagicMock(),
            "strategy_router": MagicMock(),
            "risk_manager": MagicMock(),
        }

        with patch("core.backtester.Backtester", mock_backtester_cls), patch("builtins.print"):
            run_backtest(components, ["SPY"])

        mock_md.get_bars.assert_called_once()
        mock_backtester_cls.return_value.run.assert_called_once()

    def test_run_backtest_handles_empty_data(self):
        from main import run_backtest

        mock_md = MagicMock()
        mock_md.get_bars.return_value = pd.DataFrame()

        components = {
            "market_data": mock_md,
            "feature_eng": MagicMock(),
            "hmm_engine": MagicMock(),
            "strategy_router": MagicMock(),
            "risk_manager": MagicMock(),
        }

        run_backtest(components, ["SPY"])  # should not raise


# ===========================================================================
# run_live smoke tests (very minimal — avoids blocking)
# ===========================================================================

class TestRunLive:
    def _make_components(self):
        mock_md = MagicMock()
        mock_md.stop_stream = MagicMock()
        mock_md.subscribe_bars = MagicMock()

        from core.position_tracker import PortfolioSnapshot
        mock_pt = MagicMock()
        mock_pt.snapshot.return_value = PortfolioSnapshot(
            timestamp=_now(), cash=100_000.0, nav=100_000.0, positions={},
            gross_exposure=0.0, net_exposure=0.0, long_exposure=0.0,
            short_exposure=0.0, unrealised_pnl=0.0, weights={},
        )

        return {
            "tickers": ["SPY"],
            "market_data": mock_md,
            "feature_eng": MagicMock(),
            "hmm_engine": MagicMock(),
            "strategy_router": MagicMock(),
            "risk_manager": MagicMock(),
            "order_executor": MagicMock(),
            "position_tracker": mock_pt,
            "alert_dispatcher": MagicMock(),
            "_bar_buffers": {},
        }

    def test_run_live_subscribes_bars(self):
        import threading
        from main import run_live

        components = self._make_components()

        # start_stream will set the internal _shutdown_event stored in components
        def _fake_stream():
            # By the time this runs, run_live has already set components["_shutdown_event"]
            ev = components.get("_shutdown_event")
            if ev is not None:
                ev.set()

        components["market_data"].start_stream.side_effect = _fake_stream

        with patch("main.shutdown"), patch("main._maybe_retrain"):
            run_live(components, ["SPY"])

        components["market_data"].subscribe_bars.assert_called_once()
