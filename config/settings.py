"""
Global configuration for regime_trader.

All tunable parameters live here. Import this module instead of
scattering magic numbers across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------

@dataclass
class BrokerConfig:
    """Alpaca connection settings."""
    base_url: str = "https://paper-api.alpaca.markets"
    data_feed: str = "crypto"       # "crypto" | "iex" (stocks, free) | "sip" (stocks, paid)
    paper_trading: bool = True
    request_timeout_s: int = 10

BROKER = BrokerConfig()

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

@dataclass
class UniverseConfig:
    """Tradeable tickers."""
    tickers: List[str] = field(default_factory=lambda: [
        "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD", "LINK/USD", "LTC/USD"
    ])
    benchmark: str = "BTC/USD"

UNIVERSE = UniverseConfig()

# ---------------------------------------------------------------------------
# Hidden Markov Model
# ---------------------------------------------------------------------------

@dataclass
class HMMConfig:
    """Parameters for the regime classifier."""
    n_regimes_min: int = 3
    n_regimes_max: int = 7
    n_regimes_default: int = 4        # used when auto-selection is off
    auto_select_regimes: bool = True  # AIC/BIC search in [min, max]
    training_window_days: int = 365   # 1 year of calendar days (crypto is 24/7)
    retrain_interval_days: int = 7    # re-fit weekly (crypto moves faster)
    features: List[str] = field(default_factory=lambda: [
        "log_return",
        "realized_vol_5d",
        "realized_vol_20d",
        "rsi_14",
        "macd_signal",
        "atr_14_pct",
    ])
    n_iter: int = 200                 # EM iterations
    tol: float = 1e-4
    covariance_type: str = "full"     # "full" | "diag" | "spherical"
    random_state: int = 42

HMM = HMMConfig()

# ---------------------------------------------------------------------------
# Per-regime allocation strategies
# ---------------------------------------------------------------------------

# Keys map to integer regime labels (0-indexed).
# Values are target allocations (sum <= 1.0) and leverage cap.
# Extend when n_regimes > 4 by adding more entries.
REGIME_STRATEGIES: Dict[int, Dict] = {
    0: {  # Bull / low-vol
        "label": "bull",
        "equity_allocation": 1.0,
        "bond_allocation": 0.0,
        "cash_allocation": 0.0,
        "leverage_max": 1.0,
    },
    1: {  # Neutral / mean-reverting
        "label": "neutral",
        "equity_allocation": 0.6,
        "bond_allocation": 0.2,
        "cash_allocation": 0.2,
        "leverage_max": 1.0,
    },
    2: {  # Bear / high-vol
        "label": "bear",
        "equity_allocation": 0.2,
        "bond_allocation": 0.4,
        "cash_allocation": 0.4,
        "leverage_max": 0.5,
    },
    3: {  # Crisis / crash
        "label": "crisis",
        "equity_allocation": 0.0,
        "bond_allocation": 0.3,
        "cash_allocation": 0.7,
        "leverage_max": 0.0,
    },
}

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    """Circuit breakers, sizing constraints, and drawdown limits."""
    max_capital_usd: float = 700.0        # hard cap on total capital used (0 = no cap)
    max_position_pct: float = 1.00       # max single position as % of NAV
    max_sector_pct: float = 1.00         # crypto — single asset, no sector limit
    max_drawdown_pct: float = 0.20       # daily drawdown hard stop (20% — crypto is volatile)
    max_portfolio_drawdown_pct: float = 0.35  # peak-to-trough hard stop
    daily_loss_limit_pct: float = 0.10   # intraday loss limit (10% for crypto)
    circuit_breaker_cooldown_minutes: int = 60
    vol_target_annual: float = 0.60      # 60% annualised vol target (crypto baseline)
    vol_lookback_days: int = 21
    min_trade_size_usd: float = 10.0     # crypto supports small fractional orders
    slippage_bps: float = 15.0           # wider spreads on crypto

RISK = RiskConfig()

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Walk-forward backtest settings."""
    in_sample_days: int = 252
    out_of_sample_days: int = 126
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001        # 0.1 % per side
    slippage_bps: float = 5.0
    benchmark: str = "BTC/USD"
    walk_forward_anchored: bool = False  # False = rolling, True = expanding

BACKTEST = BacktestConfig()

# ---------------------------------------------------------------------------
# Monitoring & alerting
# ---------------------------------------------------------------------------

@dataclass
class MonitoringConfig:
    """Logging, dashboard, and alert thresholds."""
    log_level: str = "INFO"              # DEBUG | INFO | WARNING | ERROR
    log_dir: str = "logs"
    log_rotation_mb: int = 50
    dashboard_refresh_s: int = 30
    # Alert thresholds
    alert_drawdown_pct: float = 0.10     # warn at 10 % drawdown
    alert_vol_spike_factor: float = 2.0  # warn when vol > 2× rolling avg
    alert_regime_change: bool = True
    alert_order_failure: bool = True
    # Notification channels (populated via credentials.py / .env)
    email_enabled: bool = False
    webhook_enabled: bool = False

MONITORING = MonitoringConfig()
