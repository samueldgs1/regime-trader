# regime_trader

A fully automated, regime-aware equity trading bot built in Python.

Uses a Hidden Markov Model (HMM) to classify the current market
regime (bull, neutral, bear, crisis) and dynamically shifts portfolio
allocations via Alpaca's paper-trading API.

---

## Architecture overview

```
market_data  →  feature_eng  →  hmm_engine
                                    ↓
                           regime_strategies
                                    ↓
                  risk_manager  →  order_executor
                                    ↓
                           position_tracker
                                    ↓
                    alerts  /  dashboard  /  logs
```

---

## Project structure

```
regime_trader/
├── config/
│   ├── settings.py          # All tunable parameters
│   └── credentials.py       # Reads from .env only
├── core/
│   ├── hmm_engine.py        # Gaussian HMM regime classifier
│   ├── feature_eng.py       # Technical indicator pipeline
│   ├── regime_strategies.py # Per-regime allocation logic
│   ├── risk_manager.py      # Circuit breakers and sizing
│   ├── order_executor.py    # Alpaca order management
│   ├── position_tracker.py  # Live position state
│   ├── market_data.py       # REST + WebSocket data feeds
│   ├── backtester.py        # Walk-forward backtest engine
│   ├── performance.py       # Risk-adjusted metrics
│   └── alerts.py            # Email / webhook notifications
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── tests/
│   ├── test_hmm.py
│   ├── test_strategies.py
│   ├── test_risk.py
│   ├── test_orders.py
│   └── test_backtest.py
├── logs/
├── main.py
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Quick start

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url> regime_trader
cd regime_trader
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **TA-Lib note:** `ta-lib` requires a compiled C library.
> - Windows: download the wheel from
>   https://github.com/cgohlke/talib-build/releases
> - macOS: `brew install ta-lib`
> - Linux: `sudo apt-get install libta-lib-dev`
>
> If you cannot install TA-Lib, the bot will fall back to `pandas-ta`
> for indicator computation.

### 3. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in your Alpaca API key and secret
```

### 4. Run a walk-forward backtest

```bash
python main.py --backtest --tickers SPY
```

### 5. Start the live paper-trading bot

```bash
python main.py --tickers SPY
```

### 6. Open the dashboard

```bash
streamlit run dashboard/app.py
```

---

## Configuration

All parameters are in `config/settings.py`.  Key sections:

| Section | Purpose |
|---|---|
| `BROKER` | Alpaca endpoint, data feed, timeout |
| `UNIVERSE` | Tickers to trade |
| `HMM` | Regime count, training window, features |
| `REGIME_STRATEGIES` | Per-regime allocation targets |
| `RISK` | Circuit breakers, sizing, drawdown limits |
| `BACKTEST` | IS/OOS window lengths, initial capital |
| `MONITORING` | Log level, dashboard refresh, alert thresholds |

---

## Running tests

```bash
pytest tests/ -v --cov=core --cov-report=term-missing
```

---

## Regimes

| Label | Name | Description |
|---|---|---|
| 0 | Bull | Low volatility, positive trend |
| 1 | Neutral | Mean-reverting, moderate vol |
| 2 | Bear | Elevated vol, negative trend |
| 3 | Crisis | Extreme vol, drawdown environment |

The number of regimes is tunable (`HMM.n_regimes_default`) or
auto-selected via AIC/BIC search (`HMM.auto_select_regimes = True`).

---

## Disclaimer

This software is for educational and research purposes only.
It is not financial advice. Paper trading carries no real financial risk,
but live trading does. Use at your own risk.
