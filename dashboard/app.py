"""
Streamlit dashboard for regime_trader.
"""

from __future__ import annotations
import streamlit as st
import traceback as _tb

st.set_page_config(page_title="Regime Trader", page_icon="📈", layout="wide")

try:
    import json, os, sys, time
    from datetime import datetime, timedelta, timezone
    from pathlib import Path
    from typing import Dict, List, Optional, Tuple
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _IMPORTS_OK = True
except Exception as _e:
    st.error(f"Import failed: {_e}")
    st.code(_tb.format_exc())
    _IMPORTS_OK = False

if _IMPORTS_OK:
    import subprocess, signal, psutil  # noqa: E401

    # -----------------------------------------------------------------------
    # Path setup — make project root importable
    # -----------------------------------------------------------------------
    _ROOT = Path(__file__).parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    _LOGS_DIR  = _ROOT / "logs"
    _STATE_FILE = _LOGS_DIR / "current_state.json"
    _TRADES_FILE = _LOGS_DIR / "trade_log.jsonl"
    _PRICE_FILE  = _LOGS_DIR / "price_history.json"
    _HALT_FILE   = _LOGS_DIR / "TRADING_HALTED.lock"
    _PID_FILE    = _LOGS_DIR / "bot.pid"

    # -----------------------------------------------------------------------
    # Regime colour palette
    # -----------------------------------------------------------------------
    _REGIME_COLORS: Dict[str, str] = {
        "crash":        "#c0392b",
        "deep_bear":    "#e74c3c",
        "bear":         "#e67e22",
        "neutral":      "#f39c12",
        "bull":         "#27ae60",
        "euphoria":     "#1abc9c",
        "extreme_bull": "#2980b9",
        "unknown":      "#95a5a6",
    }

    _CB_COLORS = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}
    _CB_LABELS = {0: "NONE", 1: "REDUCE_SIZES", 2: "HALT_DAY", 3: "FULL_STOP"}

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  .metric-card {
    background: #1e2130;
    border-radius: 10px;
    padding: 16px 20px;
    border: 1px solid #2d3149;
    margin-bottom: 8px;
  }
  .metric-label { color: #8891b4; font-size: 0.78rem; font-weight: 600;
                  letter-spacing: 0.06em; text-transform: uppercase; }
  .metric-value { color: #e8eaf6; font-size: 1.6rem; font-weight: 700; margin: 4px 0 2px; }
  .metric-delta-pos { color: #27ae60; font-size: 0.85rem; }
  .metric-delta-neg { color: #e74c3c; font-size: 0.85rem; }
  .regime-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-weight: 700; font-size: 0.9rem; letter-spacing: 0.04em;
  }
  .risk-ok     { color: #27ae60; font-weight: 700; }
  .risk-warn   { color: #f39c12; font-weight: 700; }
  .risk-danger { color: #e74c3c; font-weight: 700; }
  div[data-testid="stDataFrame"] { border-radius: 8px; }
  .stMetric label { color: #8891b4 !important; }
  .section-header {
    color: #8891b4; font-size: 0.78rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    border-bottom: 1px solid #2d3149; padding-bottom: 4px;
    margin: 16px 0 12px;
  }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Demo data generation
# ===========================================================================

def _make_demo_price_history(ticker: str = "SPY", n_bars: int = 390) -> pd.DataFrame:
    """Generate realistic synthetic intraday bars with regime labels."""
    rng = np.random.default_rng(42)
    now = datetime.now(tz=timezone.utc).replace(hour=16, minute=0, second=0, microsecond=0)
    market_open = now.replace(hour=9, minute=30)
    timestamps = [market_open + timedelta(minutes=i) for i in range(n_bars)]

    # Regime blocks — bull for most of the day, bear patch mid-day
    regimes = (
        ["bull"] * 120 + ["neutral"] * 60 + ["bear"] * 80 + ["neutral"] * 50 + ["bull"] * 80
    )[:n_bars]

    regime_drift = {"crash": -0.0008, "bear": -0.0004, "neutral": 0.0001,
                    "bull": 0.0003, "euphoria": 0.0002, "unknown": 0.0}
    base = 420.0
    closes, opens, highs, lows, volumes = [], [], [], [], []
    price = base
    for i, r in enumerate(regimes):
        drift = regime_drift.get(r, 0.0001)
        ret = drift + rng.normal(0, 0.0012)
        o = price
        price = price * (1 + ret)
        noise = abs(rng.normal(0, 0.15))
        h = max(o, price) + noise
        l = min(o, price) - noise
        vol = int(rng.integers(800_000, 3_000_000))
        opens.append(round(o, 2)); closes.append(round(price, 2))
        highs.append(round(h, 2)); lows.append(round(l, 2))
        volumes.append(vol)

    confidences = [round(min(0.99, max(0.51, 0.75 + rng.normal(0, 0.1))), 3) for _ in regimes]

    return pd.DataFrame({
        "timestamp":   timestamps,
        "open":        opens,
        "high":        highs,
        "low":         lows,
        "close":       closes,
        "volume":      volumes,
        "regime_name": regimes,
        "confidence":  confidences,
    })


def _make_demo_trades() -> pd.DataFrame:
    """Generate a realistic demo trade log."""
    rng = np.random.default_rng(7)
    now = datetime.now(tz=timezone.utc)
    regimes = ["bull", "neutral", "bear", "bull", "neutral"]
    allocs  = [0.95, 0.50, 0.20, 0.95, 0.50]
    sides   = ["buy", "sell", "sell", "buy", "sell"]
    prices  = [418.2, 421.5, 419.0, 417.8, 423.1]
    stops   = [p * 0.98 for p in prices]
    notionals = [9500, 5000, 2000, 9500, 5000]
    current_price = 420.5

    rows = []
    for i in range(5):
        ts = now - timedelta(minutes=int(rng.integers(10, 300)))
        entry = prices[i]
        qty = notionals[i] / entry
        pnl = (current_price - entry) * qty * (1 if sides[i] == "buy" else -1)
        rows.append({
            "timestamp":      ts.strftime("%H:%M:%S"),
            "ticker":         "SPY",
            "side":           sides[i].upper(),
            "allocation_%":   f"{allocs[i]*100:.0f}%",
            "entry_price":    f"${entry:.2f}",
            "stop":           f"${stops[i]:.2f}",
            "current_pnl":    pnl,
            "regime_at_entry": regimes[i],
            "confidence":     f"{(0.7 + rng.random() * 0.25):.0%}",
        })
    return pd.DataFrame(rows).sort_values("timestamp", ascending=False)


def _make_demo_state() -> dict:
    """Generate realistic demo bot state."""
    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "regime": {
            "label": 2, "name": "bull", "confidence": 0.87,
            "posteriors": [0.03, 0.10, 0.87], "is_uncertain": False, "n_regimes": 3,
        },
        "circuit_breaker_level": 0,
        "size_multiplier": 1.0,
        "session_start_nav": 100_000.0,
        "peak_nav": 101_200.0,
    }


def _make_demo_portfolio() -> dict:
    """Generate realistic demo portfolio snapshot."""
    return {
        "nav":          101_850.0,
        "cash":          5_200.0,
        "buying_power": 10_400.0,
        "gross_exposure": 96_650.0,
        "net_exposure":   96_650.0,
        "unrealised_pnl": 1_850.0,
        "daily_pnl":      1_200.0,
        "positions": {
            "SPY": {"qty": 230.0, "avg_entry_price": 418.5, "current_price": 420.5,
                    "market_value": 96_715.0, "unrealised_pnl": 460.0,
                    "unrealised_pnl_pct": 0.0048, "side": "long"},
        },
        "weights": {"SPY": 0.949},
    }


# ===========================================================================
# Data loaders
# ===========================================================================

@st.cache_data(ttl=28)
def load_current_state() -> dict:
    """Read bot state from logs/current_state.json, else use demo data."""
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text())
        except Exception:
            pass
    return _make_demo_state()


@st.cache_data(ttl=28)
def load_trade_log() -> pd.DataFrame:
    """Read trade log from logs/trade_log.jsonl, else use demo data."""
    if _TRADES_FILE.exists():
        try:
            rows = []
            for line in _TRADES_FILE.read_text().splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            if rows:
                df = pd.DataFrame(rows)
                # Normalise columns for display
                col_map = {
                    "filled_price": "entry_price",
                    "filled_avg_price": "entry_price",
                    "regime_name": "regime_at_entry",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                return df
        except Exception:
            pass
    return _make_demo_trades()


@st.cache_data(ttl=28)
def load_price_history(ticker: str = "SPY") -> pd.DataFrame:
    """Read price+regime history from logs/price_history.json, else demo."""
    if _PRICE_FILE.exists():
        try:
            raw = json.loads(_PRICE_FILE.read_text())
            df = pd.DataFrame(raw.get("bars", []))
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df
        except Exception:
            pass
    return _make_demo_price_history(ticker)


@st.cache_data(ttl=28)
def load_portfolio() -> dict:
    """
    Try Alpaca live data first; fall back to most recent snapshot JSON,
    then demo data.
    """
    # Try Alpaca
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env", override=True)
        api_key    = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        base_url   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        if api_key and secret_key:
            from alpaca.trading.client import TradingClient
            client  = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
            account = client.get_account()
            positions_raw = client.get_all_positions()

            nav           = float(account.equity)
            cash          = float(account.cash)
            buying_power  = float(account.buying_power)
            positions: Dict[str, dict] = {}
            gross_exp = 0.0
            total_upnl = 0.0
            for p in positions_raw:
                mv     = float(p.market_value)
                upnl   = float(p.unrealized_pl)
                upnl_pc = float(p.unrealized_plpc)
                gross_exp  += abs(mv)
                total_upnl += upnl
                positions[str(p.symbol)] = {
                    "qty":               float(p.qty),
                    "avg_entry_price":   float(p.avg_entry_price),
                    "current_price":     float(p.current_price),
                    "market_value":      mv,
                    "unrealised_pnl":    upnl,
                    "unrealised_pnl_pct": upnl_pc,
                    "side": (p.side.value if hasattr(p.side, "value") else str(p.side)),
                }
            weights = {t: d["market_value"] / nav for t, d in positions.items()} if nav > 0 else {}
            return {
                "nav": nav, "cash": cash, "buying_power": buying_power,
                "gross_exposure": gross_exp, "net_exposure": gross_exp,
                "unrealised_pnl": total_upnl,
                "daily_pnl": float(getattr(account, "equity", nav)) - float(getattr(account, "last_equity", nav)),
                "positions": positions,
                "weights": weights,
                "_source": "alpaca",
            }
    except Exception:
        pass

    # Try most recent snapshot file
    snapshots = sorted(_LOGS_DIR.glob("snapshot_*.json"), reverse=True)
    if snapshots:
        try:
            data = json.loads(snapshots[0].read_text())
            data["_source"] = "snapshot"
            data.setdefault("buying_power", data.get("cash", 0))
            data.setdefault("daily_pnl", 0.0)
            return data
        except Exception:
            pass

    demo = _make_demo_portfolio()
    demo["_source"] = "demo"
    return demo


# ===========================================================================
# Chart builders
# ===========================================================================

def build_price_regime_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Candlestick chart with coloured regime background bands.
    Expects columns: timestamp, open, high, low, close, regime_name.
    """
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Regime background bands
    if "regime_name" in df.columns and len(df) > 1:
        prev_regime = df["regime_name"].iloc[0]
        band_start  = df["timestamp"].iloc[0]
        for i in range(1, len(df)):
            cur = df["regime_name"].iloc[i]
            if cur != prev_regime or i == len(df) - 1:
                color = _REGIME_COLORS.get(prev_regime, "#95a5a6")
                fig.add_vrect(
                    x0=band_start, x1=df["timestamp"].iloc[i],
                    fillcolor=color, opacity=0.15,
                    layer="below", line_width=0,
                )
                band_start  = df["timestamp"].iloc[i]
                prev_regime = cur

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["timestamp"],
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name=ticker,
        increasing_line_color="#27ae60",
        decreasing_line_color="#e74c3c",
        increasing_fillcolor="#27ae60",
        decreasing_fillcolor="#e74c3c",
    ))

    fig.update_layout(
        title=dict(text=f"{ticker} — Price with Regime Overlay", font_size=14),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#c8ccd8",
        margin=dict(l=10, r=10, t=36, b=10),
        height=340,
        showlegend=False,
        xaxis=dict(gridcolor="#1e2130", showgrid=True),
        yaxis=dict(gridcolor="#1e2130", showgrid=True, side="right"),
    )
    return fig


def build_volume_chart(df: pd.DataFrame) -> go.Figure:
    """Volume bar chart coloured by up/down bar."""
    colors = [
        "#27ae60" if c >= o else "#e74c3c"
        for c, o in zip(df["close"], df["open"])
    ]
    fig = go.Figure(go.Bar(
        x=df["timestamp"], y=df["volume"],
        marker_color=colors, name="Volume",
    ))
    fig.update_layout(
        title=dict(text="Volume", font_size=14),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#c8ccd8",
        margin=dict(l=10, r=10, t=36, b=10),
        height=200,
        showlegend=False,
        xaxis=dict(gridcolor="#1e2130"),
        yaxis=dict(gridcolor="#1e2130", side="right"),
    )
    return fig


def build_confidence_chart(df: pd.DataFrame) -> go.Figure:
    """Confidence score over time as a filled area chart."""
    if "confidence" not in df.columns:
        df = df.copy()
        df["confidence"] = 0.75

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["confidence"],
        fill="tozeroy", fillcolor="rgba(52, 152, 219, 0.18)",
        line=dict(color="#3498db", width=1.5),
        name="Confidence",
    ))
    fig.add_hline(y=0.70, line_dash="dash", line_color="#f39c12",
                  annotation_text="70%", annotation_position="right")

    fig.update_layout(
        title=dict(text="Regime Confidence Score", font_size=14),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#c8ccd8",
        margin=dict(l=10, r=10, t=36, b=10),
        height=200,
        showlegend=False,
        yaxis=dict(range=[0, 1.05], gridcolor="#1e2130", tickformat=".0%", side="right"),
        xaxis=dict(gridcolor="#1e2130"),
    )
    return fig


def build_regime_pie(df: pd.DataFrame) -> go.Figure:
    """Pie chart of time spent in each regime."""
    if "regime_name" not in df.columns or df.empty:
        return go.Figure()

    counts = df["regime_name"].value_counts()
    colors = [_REGIME_COLORS.get(r, "#95a5a6") for r in counts.index]

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        marker_colors=colors,
        hole=0.45,
        textinfo="label+percent",
        textfont_size=11,
        pull=[0.04 if r == df["regime_name"].iloc[-1] else 0 for r in counts.index],
    ))
    fig.update_layout(
        title=dict(text="Regime Distribution", font_size=14),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#c8ccd8",
        margin=dict(l=10, r=10, t=36, b=10),
        height=340,
        showlegend=True,
        legend=dict(orientation="v", x=1.0, y=0.5, font_size=10),
    )
    return fig


# ===========================================================================
# UI section renderers
# ===========================================================================

def _bot_pid() -> int | None:
    """Return bot PID if the process is alive, else None."""
    if not _PID_FILE.exists():
        return None
    try:
        pid = int(_PID_FILE.read_text().strip())
        if psutil.pid_exists(pid):
            p = psutil.Process(pid)
            if p.status() != psutil.STATUS_ZOMBIE:
                return pid
    except Exception:
        pass
    _PID_FILE.unlink(missing_ok=True)
    return None


def _start_bot() -> None:
    proc = subprocess.Popen(
        ["python", str(_ROOT / "main.py")],
        cwd=str(_ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    _PID_FILE.write_text(str(proc.pid))


def _stop_bot(pid: int) -> None:
    try:
        p = psutil.Process(pid)
        p.terminate()
        p.wait(timeout=5)
    except Exception:
        pass
    _PID_FILE.unlink(missing_ok=True)


def render_sidebar() -> dict:
    """Render sidebar controls and return settings dict."""
    st.sidebar.title("⚙️ Settings")

    refresh_s = st.sidebar.selectbox(
        "Auto-refresh interval", [15, 30, 60, 120], index=1,
        format_func=lambda s: f"{s}s",
    )
    ticker = st.sidebar.text_input("Chart ticker", value="BTC/USD").upper()

    # --- Bot controls ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Bot Controls**")
    pid = _bot_pid()
    if pid:
        st.sidebar.success(f"🟢 Bot running (PID {pid})")
        if st.sidebar.button("⏹ Stop Bot", type="primary", use_container_width=True):
            _stop_bot(pid)
            st.cache_data.clear()
            st.rerun()
    else:
        st.sidebar.warning("🔴 Bot not running")
        if st.sidebar.button("▶ Start Bot", type="primary", use_container_width=True):
            _start_bot()
            st.cache_data.clear()
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Regime Legend**")
    for name, color in _REGIME_COLORS.items():
        if name != "unknown":
            st.sidebar.markdown(
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'background:{color};border-radius:3px;margin-right:6px;"></span>'
                f'<span style="font-size:0.85rem">{name.replace("_"," ").title()}</span>',
                unsafe_allow_html=True,
            )

    st.sidebar.markdown("---")
    # Check if running on real data
    _state = load_current_state()
    _port  = load_portfolio()
    source = _port.get("_source", "demo")
    if source == "alpaca":
        st.sidebar.success("🟢 Live — Alpaca connected")
    elif source == "snapshot":
        st.sidebar.warning("🟡 Snapshot — bot offline")
    else:
        st.sidebar.info("🔵 Demo mode — no credentials")

    halt = _HALT_FILE.exists()
    if halt:
        st.sidebar.error("🚨 TRADING_HALTED.lock present")

    return {"refresh_s": refresh_s, "ticker": ticker}


def render_header(state: dict, source: str) -> None:
    """Top banner with title and last-update timestamp."""
    ts = state.get("timestamp", datetime.now(tz=timezone.utc).isoformat())
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        age_s = (datetime.now(tz=timezone.utc) - dt).total_seconds()
        age_str = f"{int(age_s)}s ago" if age_s < 120 else f"{int(age_s/60)}m ago"
    except Exception:
        age_str = "—"

    source_badge = {"alpaca": "🟢 Live", "snapshot": "🟡 Snapshot", "demo": "🔵 Demo"}.get(source, "—")

    cols = st.columns([3, 1, 1])
    with cols[0]:
        st.markdown("## 📈 Regime Trader Dashboard")
    with cols[1]:
        st.markdown(f"<div style='text-align:right;color:#8891b4;font-size:0.85rem;padding-top:14px'>"
                    f"Updated {age_str}</div>", unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"<div style='text-align:right;padding-top:12px;font-size:0.9rem'>"
                    f"{source_badge}</div>", unsafe_allow_html=True)
    st.markdown("---")


def render_top_row(state: dict, portfolio: dict) -> None:
    """Four KPI cards: regime, portfolio value, active regimes, risk status."""
    c1, c2, c3, c4 = st.columns(4)

    # --- Regime card ---
    regime_info   = state.get("regime", {})
    regime_name   = regime_info.get("name", "unknown")
    regime_label  = regime_info.get("label", 0)
    confidence    = regime_info.get("confidence", 0.0)
    is_uncertain  = regime_info.get("is_uncertain", False)
    n_regimes     = regime_info.get("n_regimes", 4)
    regime_color  = _REGIME_COLORS.get(regime_name, "#95a5a6")
    uncertain_tag = " ⚠️" if is_uncertain else ""

    with c1:
        st.markdown('<p class="section-header">Current Regime</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">HMM Regime Label {regime_label}</div>'
            f'<div class="metric-value" style="color:{regime_color}">'
            f'{regime_name.replace("_"," ").upper()}{uncertain_tag}</div>'
            f'<div style="background:#2d3149;border-radius:4px;height:8px;margin-top:8px">'
            f'<div style="background:{regime_color};width:{confidence*100:.0f}%;'
            f'height:8px;border-radius:4px"></div></div>'
            f'<div style="color:#8891b4;font-size:0.8rem;margin-top:4px">'
            f'Confidence: {confidence:.1%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- Portfolio card ---
    nav           = portfolio.get("nav", 0.0)
    buying_power  = portfolio.get("buying_power", 0.0)
    daily_pnl     = portfolio.get("daily_pnl", 0.0)
    pnl_color     = "#27ae60" if daily_pnl >= 0 else "#e74c3c"
    pnl_sign      = "+" if daily_pnl >= 0 else ""

    with c2:
        st.markdown('<p class="section-header">Portfolio</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Net Asset Value</div>'
            f'<div class="metric-value">${nav:,.0f}</div>'
            f'<div style="color:#8891b4;font-size:0.8rem">'
            f'Buying power: <span style="color:#c8ccd8">${buying_power:,.0f}</span></div>'
            f'<div style="color:{pnl_color};font-size:0.85rem;margin-top:4px">'
            f'Daily P&L: {pnl_sign}${daily_pnl:,.0f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- Active regimes card ---
    positions = portfolio.get("positions", {})
    n_positions = len(positions)
    gross_exp = portfolio.get("gross_exposure", 0.0)
    leverage  = gross_exp / nav if nav > 0 else 0.0

    # Build per-position rows
    pos_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:0.78rem;margin-top:3px">'
        f'<span style="color:#c8ccd8">{ticker}</span>'
        f'<span style="color:{"#27ae60" if d.get("unrealised_pnl",0)>=0 else "#e74c3c"}">'
        f'${d.get("market_value",0):,.0f} '
        f'({"+" if d.get("unrealised_pnl",0)>=0 else ""}${d.get("unrealised_pnl",0):,.0f})'
        f'</span></div>'
        for ticker, d in positions.items()
    ) if positions else '<div style="color:#8891b4;font-size:0.78rem">No open positions</div>'

    with c3:
        st.markdown('<p class="section-header">Exposure</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Open Positions ({n_positions})</div>'
            f'<div style="color:#8891b4;font-size:0.75rem;margin-bottom:4px">'
            f'Leverage: <span style="color:#c8ccd8">{leverage:.2f}×</span></div>'
            f'{pos_rows}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- Risk status card ---
    cb_level = state.get("circuit_breaker_level", 0)
    size_mul = state.get("size_multiplier", 1.0)
    peak_nav = state.get("peak_nav", nav or 1.0)
    drawdown = (peak_nav - nav) / peak_nav if peak_nav > 0 else 0.0
    halt_exists = _HALT_FILE.exists()

    if halt_exists or cb_level >= 3:
        risk_color, risk_label, risk_cls = "#e74c3c", "FULL STOP 🚨", "risk-danger"
    elif cb_level == 2:
        risk_color, risk_label, risk_cls = "#e67e22", "HALT DAY 🟠", "risk-danger"
    elif cb_level == 1:
        risk_color, risk_label, risk_cls = "#f39c12", "REDUCE SIZES 🟡", "risk-warn"
    else:
        risk_color, risk_label, risk_cls = "#27ae60", "NORMAL 🟢", "risk-ok"

    with c4:
        st.markdown('<p class="section-header">Risk Status</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Circuit Breaker</div>'
            f'<div class="metric-value" style="color:{risk_color};font-size:1.2rem">'
            f'{risk_label}</div>'
            f'<div style="color:#8891b4;font-size:0.8rem;margin-top:4px">'
            f'Drawdown: <span style="color:{"#e74c3c" if drawdown > 0.05 else "#c8ccd8"}">'
            f'{drawdown:.2%}</span></div>'
            f'<div style="color:#8891b4;font-size:0.8rem">'
            f'Size scalar: <span style="color:#c8ccd8">{size_mul:.1f}×</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_charts(price_df: pd.DataFrame, ticker: str) -> None:
    """Two rows of charts."""
    # Row 1: price+regime | regime pie
    col_price, col_pie = st.columns([2, 1])
    with col_price:
        st.plotly_chart(
            build_price_regime_chart(price_df, ticker),
            width="stretch", config={"displayModeBar": False},
        )
    with col_pie:
        st.plotly_chart(
            build_regime_pie(price_df),
            width="stretch", config={"displayModeBar": False},
        )

    # Row 2: volume | confidence
    col_vol, col_conf = st.columns(2)
    with col_vol:
        st.plotly_chart(
            build_volume_chart(price_df),
            width="stretch", config={"displayModeBar": False},
        )
    with col_conf:
        st.plotly_chart(
            build_confidence_chart(price_df),
            width="stretch", config={"displayModeBar": False},
        )


def render_signal_feed(trades: pd.DataFrame) -> None:
    """Historical trade table with P&L colouring."""
    st.markdown('<p class="section-header">Signal Feed — Historical Trades</p>',
                unsafe_allow_html=True)

    if trades.empty:
        st.info("No trades recorded yet.")
        return

    # Normalise display columns
    display_cols = [
        "timestamp", "ticker", "side", "allocation_%",
        "entry_price", "stop", "current_pnl",
        "regime_at_entry", "confidence",
    ]
    show = trades.copy()
    for col in display_cols:
        if col not in show.columns:
            show[col] = "—"

    # Format current_pnl
    def _fmt_pnl(v):
        try:
            f = float(v)
            return f"+${f:,.2f}" if f >= 0 else f"-${abs(f):,.2f}"
        except Exception:
            return str(v)

    if "current_pnl" in show.columns:
        show["current_pnl"] = show["current_pnl"].apply(_fmt_pnl)

    # Regime badge colour helper (done via column config)
    show_df = show[display_cols].reset_index(drop=True)
    show_df.columns = [c.replace("_", " ").title() for c in show_df.columns]

    st.dataframe(
        show_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Side": st.column_config.TextColumn("Side", width="small"),
            "Current Pnl": st.column_config.TextColumn("P&L", width="medium"),
            "Regime At Entry": st.column_config.TextColumn("Regime", width="medium"),
        },
    )


def render_risk_panel(state: dict, portfolio: dict) -> None:
    """Risk panel: circuit breaker table + drawdown + leverage + daily P&L."""
    st.markdown('<p class="section-header">Risk Panel</p>', unsafe_allow_html=True)

    cb_level = state.get("circuit_breaker_level", 0)
    nav      = portfolio.get("nav", 0.0)
    peak_nav = state.get("peak_nav", nav or 1.0)
    sess_nav = state.get("session_start_nav", nav or 1.0)
    gross    = portfolio.get("gross_exposure", 0.0)
    leverage = gross / nav if nav > 0 else 0.0
    drawdown = (peak_nav - nav) / peak_nav if peak_nav > 0 else 0.0
    intraday = (sess_nav - nav) / sess_nav if sess_nav > 0 else 0.0
    daily_pnl = portfolio.get("daily_pnl", 0.0)

    # Circuit breaker status table
    cb_rows = []
    checks = [
        ("Intraday 2% Loss",    cb_level >= 1, "REDUCE_SIZES",  "↓ sizes ×0.5"),
        ("Daily 5% Loss",       cb_level >= 2, "HALT_DAY",      "No new orders"),
        ("7-Day 5% Rolling",    cb_level >= 1, "REDUCE_SIZES",  "↓ sizes ×0.5 + alert"),
        ("10% From Peak",       cb_level >= 3, "FULL_STOP",     "Write lock file"),
    ]
    for label, triggered, level, action in checks:
        icon = "🔴 TRIGGERED" if triggered else "🟢 CLEAR"
        cb_rows.append({"Trigger": label, "Status": icon, "Level": level, "Action": action})

    st.dataframe(
        pd.DataFrame(cb_rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn("Status", width="medium"),
        },
    )

    # Metric grid
    m1, m2, m3, m4 = st.columns(4)
    dd_color  = "inverse" if drawdown > 0.05 else "normal"
    pnl_color = "normal"
    m1.metric("Peak Drawdown", f"{drawdown:.2%}",
              delta=f"{intraday:.2%} intraday", delta_color="inverse")
    m2.metric("Leverage In Use",  f"{leverage:.2f}×",
              delta=f"max 1.5×", delta_color="off")
    m3.metric("Daily P&L",
              f"${daily_pnl:+,.0f}",
              delta_color="normal" if daily_pnl >= 0 else "inverse")
    m4.metric("Size Scalar",
              f"{state.get('size_multiplier', 1.0):.1f}×",
              delta="REDUCE_SIZES active" if cb_level == 1 else "normal",
              delta_color="inverse" if cb_level >= 1 else "off")


def render_positions_table(portfolio: dict) -> None:
    """Open positions table."""
    positions = portfolio.get("positions", {})
    nav       = portfolio.get("nav", 1.0)

    st.markdown('<p class="section-header">Open Positions</p>', unsafe_allow_html=True)

    if not positions:
        st.info("No open positions.")
        return

    rows = []
    for ticker, p in positions.items():
        mv   = p.get("market_value", 0.0)
        upnl = p.get("unrealised_pnl", 0.0)
        upct = p.get("unrealised_pnl_pct", 0.0)
        rows.append({
            "Ticker":          ticker,
            "Side":            p.get("side", "—").upper(),
            "Qty":             f'{p.get("qty", 0):.2f}',
            "Avg Entry":       f'${p.get("avg_entry_price", 0):.2f}',
            "Last Price":      f'${p.get("current_price", 0):.2f}',
            "Market Value":    f'${mv:,.0f}',
            "Weight":          f'{mv/nav:.1%}' if nav > 0 else "—",
            "Unrealised P&L":  f'${upnl:+,.0f}  ({upct:.2%})',
        })

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    """Compose and render the full dashboard layout."""
    # Sidebar first (controls page settings)
    settings = render_sidebar()
    ticker   = settings["ticker"]
    refresh_s = settings["refresh_s"]

    # Load all data
    state     = load_current_state()
    portfolio = load_portfolio()
    trades    = load_trade_log()
    price_df  = load_price_history(ticker)

    source = portfolio.get("_source", "demo")

    # Header
    render_header(state, source)

    # Top KPI row
    render_top_row(state, portfolio)

    st.markdown("")  # spacer

    # Charts
    render_charts(price_df, ticker)

    st.markdown("")  # spacer

    # Signal feed + positions side by side
    tab_trades, tab_positions, tab_risk = st.tabs(
        ["📋 Signal Feed", "💼 Open Positions", "🛡️ Risk Panel"]
    )

    with tab_trades:
        render_signal_feed(trades)

    with tab_positions:
        render_positions_table(portfolio)

    with tab_risk:
        render_risk_panel(state, portfolio)

    # Auto-refresh footer
    st.markdown("---")
    refresh_col, _ = st.columns([1, 3])
    with refresh_col:
        st.caption(f"⟳ Auto-refreshing every {refresh_s}s  •  {datetime.now().strftime('%H:%M:%S')}")

    # Auto-refresh trigger
    time.sleep(refresh_s)
    st.rerun()


if _IMPORTS_OK:
    try:
        main()
    except Exception as _exc:
        st.error(f"Dashboard crashed: {_exc}")
        st.code(_tb.format_exc())

