"""
Credential loader for regime_trader.

Reads secrets exclusively from environment variables (set via a .env
file in development). Nothing is hardcoded here.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """Return the env-var value or raise if missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Copy .env.example to .env and fill in your credentials."
        )
    return value


def _optional(key: str, default: str = "") -> str:
    """Return the env-var value or a safe default."""
    return os.getenv(key, default)


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------

ALPACA_API_KEY: str = _require("ALPACA_API_KEY")
ALPACA_SECRET_KEY: str = _require("ALPACA_SECRET_KEY")
ALPACA_BASE_URL: str = _optional(
    "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
)
PAPER: bool = _optional("PAPER", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Notifications (optional)
# ---------------------------------------------------------------------------

SMTP_HOST: str = _optional("SMTP_HOST")
SMTP_PORT: int = int(_optional("SMTP_PORT", "587"))
SMTP_USER: str = _optional("SMTP_USER")
SMTP_PASSWORD: str = _optional("SMTP_PASSWORD")
ALERT_EMAIL_FROM: str = _optional("ALERT_EMAIL_FROM")
ALERT_EMAIL_TO: str = _optional("ALERT_EMAIL_TO")

WEBHOOK_URL: str = _optional("WEBHOOK_URL")
