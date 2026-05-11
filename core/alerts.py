"""
Alert dispatcher for critical trading events.

Supports email (SMTP) and generic webhook (Slack, Discord, etc.)
notification channels.  All methods are no-ops when the respective
channel is disabled in settings.
"""

from __future__ import annotations

import json
import logging
import smtplib
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.text import MIMEText
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    """Structured alert payload."""
    severity: AlertSeverity
    title: str
    body: str
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


class AlertDispatcher:
    """
    Routes structured alerts to configured notification channels.

    Channels
    --------
    Email   — SMTP, configured via credentials.py.
    Webhook — HTTP POST JSON, configured via credentials.py (WEBHOOK_URL).
    Log     — always active; writes to the application logger.
    """

    def __init__(
        self,
        email_enabled: bool,
        webhook_enabled: bool,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str,
        email_from: str,
        email_to: str,
        webhook_url: str,
    ) -> None:
        self._email_enabled = email_enabled
        self._webhook_enabled = webhook_enabled
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._smtp_user = smtp_user
        self._smtp_password = smtp_password
        self._email_from = email_from
        self._email_to = email_to
        self._webhook_url = webhook_url

    # ------------------------------------------------------------------
    # High-level alert methods
    # ------------------------------------------------------------------

    def regime_change(self, old_regime: int, new_regime: int, timestamp: datetime) -> None:
        """Dispatch an alert when the HMM transitions to a new regime."""
        alert = self._build_alert(
            severity=AlertSeverity.INFO,
            title=f"Regime Change: {old_regime} → {new_regime}",
            body=(
                f"The HMM classifier transitioned from regime {old_regime} "
                f"to regime {new_regime} at {timestamp.isoformat()}."
            ),
            metadata={"old_regime": old_regime, "new_regime": new_regime},
        )
        self.custom(alert)

    def circuit_breaker_triggered(
        self,
        reason: str,
        nav: float,
        drawdown_pct: float,
        timestamp: datetime,
    ) -> None:
        """Dispatch a CRITICAL alert when a circuit breaker fires."""
        alert = self._build_alert(
            severity=AlertSeverity.CRITICAL,
            title="Circuit Breaker Triggered",
            body=(
                f"Circuit breaker activated at {timestamp.isoformat()}.\n"
                f"Reason: {reason}\n"
                f"NAV: ${nav:,.2f}\n"
                f"Drawdown: {drawdown_pct:.2%}"
            ),
            metadata={"reason": reason, "nav": nav, "drawdown_pct": drawdown_pct},
        )
        self.custom(alert)

    def order_failure(
        self,
        ticker: str,
        side: str,
        notional: float,
        error: str,
        timestamp: datetime,
    ) -> None:
        """Dispatch a WARNING when an order submission fails."""
        alert = self._build_alert(
            severity=AlertSeverity.WARNING,
            title=f"Order Failure: {side.upper()} {ticker}",
            body=(
                f"Order failed at {timestamp.isoformat()}.\n"
                f"Ticker: {ticker}  Side: {side}  Notional: ${notional:,.2f}\n"
                f"Error: {error}"
            ),
            metadata={"ticker": ticker, "side": side, "notional": notional, "error": error},
        )
        self.custom(alert)

    def drawdown_warning(
        self,
        drawdown_pct: float,
        threshold_pct: float,
        timestamp: datetime,
    ) -> None:
        """Dispatch a WARNING when drawdown exceeds the alert threshold."""
        alert = self._build_alert(
            severity=AlertSeverity.WARNING,
            title="Drawdown Warning",
            body=(
                f"Portfolio drawdown of {drawdown_pct:.2%} has exceeded the "
                f"alert threshold of {threshold_pct:.2%} at {timestamp.isoformat()}."
            ),
            metadata={"drawdown_pct": drawdown_pct, "threshold_pct": threshold_pct},
        )
        self.custom(alert)

    def vol_spike(
        self,
        current_vol: float,
        rolling_avg_vol: float,
        spike_factor: float,
        timestamp: datetime,
    ) -> None:
        """Dispatch a WARNING when realised vol spikes above the threshold."""
        alert = self._build_alert(
            severity=AlertSeverity.WARNING,
            title="Volatility Spike Detected",
            body=(
                f"Realised volatility of {current_vol:.4f} is {spike_factor:.1f}× the "
                f"rolling average of {rolling_avg_vol:.4f} at {timestamp.isoformat()}."
            ),
            metadata={
                "current_vol": current_vol,
                "rolling_avg_vol": rolling_avg_vol,
                "spike_factor": spike_factor,
            },
        )
        self.custom(alert)

    def daily_summary(
        self,
        date: datetime,
        pnl: float,
        pnl_pct: float,
        regime: int,
        nav: float,
    ) -> None:
        """Send an end-of-day INFO summary."""
        pnl_sign = "+" if pnl >= 0 else ""
        alert = self._build_alert(
            severity=AlertSeverity.INFO,
            title=f"Daily Summary — {date.strftime('%Y-%m-%d')}",
            body=(
                f"Date:   {date.strftime('%Y-%m-%d')}\n"
                f"P&L:    {pnl_sign}${pnl:,.2f}  ({pnl_sign}{pnl_pct:.2%})\n"
                f"NAV:    ${nav:,.2f}\n"
                f"Regime: {regime}"
            ),
            metadata={"pnl": pnl, "pnl_pct": pnl_pct, "regime": regime, "nav": nav},
        )
        self.custom(alert)

    def custom(self, alert: Alert) -> None:
        """Dispatch an arbitrary pre-built Alert to all active channels."""
        self._log(alert)
        if self._email_enabled:
            self._send_email(alert)
        if self._webhook_enabled:
            self._send_webhook(alert)

    # ------------------------------------------------------------------
    # Channel implementations
    # ------------------------------------------------------------------

    def _send_email(self, alert: Alert) -> None:
        """Send alert via SMTP."""
        subject = f"[{alert.severity.value}] {alert.title}"
        body = self._format_email_body(alert)

        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = self._email_from
        msg["To"] = self._email_to

        try:
            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self._smtp_user, self._smtp_password)
                server.sendmail(self._email_from, [self._email_to], msg.as_string())
            logger.debug("Email alert sent: %s", alert.title)
        except Exception as exc:
            logger.error("Failed to send email alert '%s': %s", alert.title, exc)

    def _send_webhook(self, alert: Alert) -> None:
        """POST alert JSON to the configured webhook URL."""
        payload = self._format_webhook_payload(alert)
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            self._webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.getcode()
            if status not in (200, 201, 202, 204):
                logger.warning(
                    "Webhook returned unexpected status %d for '%s'", status, alert.title
                )
            else:
                logger.debug("Webhook alert sent: %s", alert.title)
        except Exception as exc:
            logger.error("Failed to send webhook alert '%s': %s", alert.title, exc)

    def _log(self, alert: Alert) -> None:
        """Write alert to the Python logger at the appropriate level."""
        message = "[%s] %s — %s", alert.severity.value, alert.title, alert.body
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(*message)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(*message)
        else:
            logger.info(*message)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_alert(
        self,
        severity: AlertSeverity,
        title: str,
        body: str,
        metadata: Optional[Dict] = None,
    ) -> Alert:
        """Construct a timestamped Alert dataclass."""
        return Alert(
            severity=severity,
            title=title,
            body=body,
            timestamp=datetime.now(tz=timezone.utc),
            metadata=metadata or {},
        )

    def _format_email_body(self, alert: Alert) -> str:
        """Render alert as plain-text email body."""
        lines = [
            f"Severity:  {alert.severity.value}",
            f"Timestamp: {alert.timestamp.isoformat()}",
            f"Title:     {alert.title}",
            "",
            alert.body,
        ]
        if alert.metadata:
            lines += ["", "--- Metadata ---"]
            for key, val in alert.metadata.items():
                lines.append(f"  {key}: {val}")
        return "\n".join(lines)

    def _format_webhook_payload(self, alert: Alert) -> Dict:
        """Render alert as a JSON-serialisable dict for the webhook POST."""
        return {
            "severity": alert.severity.value,
            "title": alert.title,
            "body": alert.body,
            "timestamp": alert.timestamp.isoformat(),
            "metadata": alert.metadata,
        }
