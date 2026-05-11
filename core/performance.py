"""
Performance analytics.

All methods are pure functions implemented as static methods on
PerformanceAnalyser, so they can be used standalone or composed freely.

Metrics
-------
Total return, CAGR, Sharpe, Sortino, Calmar, max/avg drawdown,
drawdown duration (recovery time), win rate, profit factor,
beta, Jensen's alpha, information ratio.

Attribution
-----------
regime_breakdown  — per-regime stats from a returns × regime pair.
confidence_breakdown — returns split into high/medium/low confidence buckets
                       (>70 %, 40–70 %, <40 %).

Rolling
-------
rolling_sharpe, rolling_volatility.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_TRADING_DAYS: int = 252


class PerformanceAnalyser:
    """Pure-function performance metrics for regime_trader."""

    TRADING_DAYS_PER_YEAR: int = _TRADING_DAYS

    # ------------------------------------------------------------------
    # Top-level summary
    # ------------------------------------------------------------------

    @staticmethod
    def summary(
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate_annual: float = 0.05,
    ) -> Dict[str, float]:
        """
        Compute a full set of performance statistics.

        Parameters
        ----------
        returns:
            Daily arithmetic return series.
        benchmark_returns:
            Optional aligned benchmark returns for relative metrics.
        risk_free_rate_annual:
            Annualised risk-free rate (default 5 %).

        Returns
        -------
        Dict containing: total_return, cagr, sharpe, sortino, calmar,
        max_drawdown, avg_drawdown, max_dd_duration_days, volatility,
        win_rate, profit_factor, and (if benchmark supplied)
        beta, alpha, information_ratio.
        """
        pa = PerformanceAnalyser
        stats: Dict[str, float] = {
            "total_return":        pa.total_return(returns),
            "cagr":                pa.cagr(returns),
            "sharpe":              pa.sharpe_ratio(returns, risk_free_rate_annual),
            "sortino":             pa.sortino_ratio(returns, risk_free_rate_annual),
            "calmar":              pa.calmar_ratio(returns),
            "max_drawdown":        pa.max_drawdown(returns),
            "avg_drawdown":        pa.avg_drawdown(returns),
            "max_dd_duration_days": float(pa.max_drawdown_duration(returns)),
            "volatility":          float(returns.std() * math.sqrt(_TRADING_DAYS)),
            "win_rate":            pa.win_rate(returns),         # treat each day as a "trade"
            "profit_factor":       pa.profit_factor(returns),
        }
        if benchmark_returns is not None:
            bm = benchmark_returns.reindex(returns.index).dropna()
            r_aligned = returns.reindex(bm.index).dropna()
            stats["beta"]              = pa.beta(r_aligned, bm)
            stats["alpha"]             = pa.alpha(r_aligned, bm, risk_free_rate_annual)
            stats["information_ratio"] = pa.information_ratio(r_aligned, bm)
        return stats

    # ------------------------------------------------------------------
    # Individual return metrics
    # ------------------------------------------------------------------

    @staticmethod
    def total_return(returns: pd.Series) -> float:
        """Compound total return over the full period."""
        if returns.empty:
            return 0.0
        return float((1.0 + returns).prod() - 1.0)

    @staticmethod
    def cagr(returns: pd.Series) -> float:
        """Compound annual growth rate (annualises total_return by length)."""
        if returns.empty or len(returns) < 2:
            return 0.0
        n_years = len(returns) / _TRADING_DAYS
        total = (1.0 + returns).prod()
        if total <= 0:
            return -1.0
        return float(total ** (1.0 / n_years) - 1.0)

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate_annual: float = 0.05,
    ) -> float:
        """Annualised Sharpe ratio."""
        if returns.empty:
            return 0.0
        rf_daily = risk_free_rate_annual / _TRADING_DAYS
        excess = returns - rf_daily
        std = float(excess.std())
        if std == 0.0:
            return 0.0
        return float(excess.mean() / std * math.sqrt(_TRADING_DAYS))

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate_annual: float = 0.05,
    ) -> float:
        """
        Annualised Sortino ratio.

        Downside deviation uses only negative excess returns.
        Returns +inf when there are no losing days.
        """
        if returns.empty:
            return 0.0
        rf_daily = risk_free_rate_annual / _TRADING_DAYS
        excess = returns - rf_daily
        downside = excess[excess < 0.0]
        if downside.empty:
            return float("inf")
        downside_std = float(downside.std())
        if downside_std == 0.0:
            return float("inf")
        return float(excess.mean() / downside_std * math.sqrt(_TRADING_DAYS))

    @staticmethod
    def calmar_ratio(returns: pd.Series) -> float:
        """CAGR / |max_drawdown|.  Returns +inf when max_drawdown == 0."""
        mdd = PerformanceAnalyser.max_drawdown(returns)
        if mdd == 0.0:
            return float("inf")
        return float(PerformanceAnalyser.cagr(returns) / mdd)

    # ------------------------------------------------------------------
    # Drawdown metrics
    # ------------------------------------------------------------------

    @staticmethod
    def drawdown_series(returns: pd.Series) -> pd.Series:
        """
        Running drawdown at each bar (always ≤ 0).

        dd_t = (equity_t - peak_t) / peak_t
        """
        if returns.empty:
            return pd.Series(dtype=float)
        equity = (1.0 + returns).cumprod()
        peak = equity.cummax()
        return (equity - peak) / peak

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """Peak-to-trough maximum drawdown as a positive fraction."""
        if returns.empty:
            return 0.0
        dd = PerformanceAnalyser.drawdown_series(returns)
        return float(abs(dd.min()))

    @staticmethod
    def avg_drawdown(returns: pd.Series) -> float:
        """Mean of the drawdown series on days when a drawdown is active."""
        if returns.empty:
            return 0.0
        dd = PerformanceAnalyser.drawdown_series(returns)
        in_dd = dd[dd < 0.0]
        if in_dd.empty:
            return 0.0
        return float(abs(in_dd.mean()))

    @staticmethod
    def max_drawdown_duration(returns: pd.Series) -> int:
        """
        Longest number of bars from a peak until full equity recovery.

        If the series ends in a drawdown, the duration extends to the
        last bar (i.e. the drawdown is never recovered).
        """
        if returns.empty:
            return 0
        dd = PerformanceAnalyser.drawdown_series(returns)
        in_dd = (dd < 0.0).values
        if not in_dd.any():
            return 0

        max_dur = 0
        current_dur = 0
        for flag in in_dd:
            if flag:
                current_dur += 1
                if current_dur > max_dur:
                    max_dur = current_dur
            else:
                current_dur = 0
        return max_dur

    # ------------------------------------------------------------------
    # Trade-level metrics
    # ------------------------------------------------------------------

    @staticmethod
    def win_rate(trade_pnl: pd.Series) -> float:
        """Fraction of entries in trade_pnl that are strictly positive."""
        if trade_pnl.empty:
            return 0.0
        return float((trade_pnl > 0.0).mean())

    @staticmethod
    def profit_factor(trade_pnl: pd.Series) -> float:
        """Gross profit / gross loss.  Returns +inf when there are no losses."""
        if trade_pnl.empty:
            return 0.0
        gross_profit = float(trade_pnl[trade_pnl > 0.0].sum())
        gross_loss = float(abs(trade_pnl[trade_pnl < 0.0].sum()))
        if gross_loss == 0.0:
            return float("inf")
        return gross_profit / gross_loss

    # ------------------------------------------------------------------
    # Relative metrics
    # ------------------------------------------------------------------

    @staticmethod
    def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """OLS beta of returns on benchmark_returns (market beta)."""
        aligned = (
            pd.concat([returns, benchmark_returns], axis=1)
            .dropna()
        )
        if len(aligned) < 2:
            return 0.0
        r = aligned.iloc[:, 0].values
        b = aligned.iloc[:, 1].values
        cov_matrix = np.cov(r, b)
        bench_var = cov_matrix[1, 1]
        if bench_var == 0.0:
            return 0.0
        return float(cov_matrix[0, 1] / bench_var)

    @staticmethod
    def alpha(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate_annual: float = 0.05,
    ) -> float:
        """
        Jensen's alpha, annualised.

        α = (mean_excess_return − β × mean_bench_excess) × 252
        """
        rf_daily = risk_free_rate_annual / _TRADING_DAYS
        b = PerformanceAnalyser.beta(returns, benchmark_returns)
        aligned = (
            pd.concat([returns, benchmark_returns], axis=1)
            .dropna()
        )
        if aligned.empty:
            return 0.0
        r_mean = float(aligned.iloc[:, 0].mean()) - rf_daily
        b_mean = float(aligned.iloc[:, 1].mean()) - rf_daily
        return float((r_mean - b * b_mean) * _TRADING_DAYS)

    @staticmethod
    def information_ratio(
        returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """Active return / tracking error, annualised."""
        aligned = (
            pd.concat([returns, benchmark_returns], axis=1)
            .dropna()
        )
        if aligned.empty:
            return 0.0
        active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        te = float(active.std())
        if te == 0.0:
            return 0.0
        return float(active.mean() / te * math.sqrt(_TRADING_DAYS))

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------

    @staticmethod
    def regime_breakdown(
        returns: pd.Series,
        regimes: pd.Series,
    ) -> Dict[int, Dict[str, float]]:
        """
        Per-regime performance statistics.

        Parameters
        ----------
        returns:
            Daily returns indexed by date.
        regimes:
            Integer regime label per date, aligned to returns.

        Returns
        -------
        Dict of regime_label → {total_return, avg_daily_return, volatility,
        sharpe, max_drawdown, n_days, pct_of_time}.
        """
        pa = PerformanceAnalyser
        result: Dict[int, Dict[str, float]] = {}
        total_days = len(returns)

        for label in sorted(regimes.unique()):
            mask = regimes == label
            r = returns[mask].dropna()
            if r.empty:
                continue
            result[int(label)] = {
                "total_return":     pa.total_return(r),
                "avg_daily_return": float(r.mean()),
                "volatility":       float(r.std() * math.sqrt(_TRADING_DAYS)),
                "sharpe":           pa.sharpe_ratio(r),
                "max_drawdown":     pa.max_drawdown(r),
                "n_days":           int(len(r)),
                "pct_of_time":      float(len(r) / total_days) if total_days > 0 else 0.0,
            }
        return result

    @staticmethod
    def confidence_breakdown(
        returns: pd.Series,
        confidences: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns split by HMM confidence bucket.

        Buckets
        -------
        high   : confidence > 0.70
        medium : 0.40 ≤ confidence ≤ 0.70
        low    : confidence < 0.40

        Returns
        -------
        Dict of bucket_name → {n_days, avg_daily_return, sharpe, total_return}.
        """
        pa = PerformanceAnalyser
        aligned = pd.concat(
            [returns.rename("r"), confidences.rename("c")], axis=1
        ).dropna()

        buckets: Dict[str, pd.Series] = {
            "high":   aligned.loc[aligned["c"] > 0.70, "r"],
            "medium": aligned.loc[(aligned["c"] >= 0.40) & (aligned["c"] <= 0.70), "r"],
            "low":    aligned.loc[aligned["c"] < 0.40, "r"],
        }

        result: Dict[str, Dict[str, float]] = {}
        for name, r in buckets.items():
            if r.empty:
                result[name] = {
                    "n_days": 0,
                    "avg_daily_return": 0.0,
                    "sharpe": 0.0,
                    "total_return": 0.0,
                }
            else:
                result[name] = {
                    "n_days":           int(len(r)),
                    "avg_daily_return": float(r.mean()),
                    "sharpe":           pa.sharpe_ratio(r),
                    "total_return":     pa.total_return(r),
                }
        return result

    # ------------------------------------------------------------------
    # Rolling metrics
    # ------------------------------------------------------------------

    @staticmethod
    def rolling_sharpe(
        returns: pd.Series,
        window: int = 63,
        risk_free_rate_annual: float = 0.05,
    ) -> pd.Series:
        """Rolling annualised Sharpe over a trailing window of bars."""
        rf_daily = risk_free_rate_annual / _TRADING_DAYS
        excess = returns - rf_daily
        roll_mean = excess.rolling(window).mean()
        roll_std  = excess.rolling(window).std()
        return (roll_mean / roll_std.replace(0, np.nan)) * math.sqrt(_TRADING_DAYS)

    @staticmethod
    def rolling_volatility(
        returns: pd.Series,
        window: int = 21,
    ) -> pd.Series:
        """Rolling annualised standard deviation of returns."""
        return returns.rolling(window).std() * math.sqrt(_TRADING_DAYS)

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------

    @staticmethod
    def equity_curve(
        returns: pd.Series,
        initial_capital: float = 1.0,
    ) -> pd.Series:
        """Cumulative NAV series: initial_capital × (1+r_1)(1+r_2)…"""
        if returns.empty:
            return pd.Series(dtype=float)
        return initial_capital * (1.0 + returns).cumprod()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def format_report(stats: Dict[str, float]) -> str:
        """Return a human-readable, left-aligned multi-line report string."""
        _PCT = "{:>10.2%}"
        _FLT = "{:>10.3f}"
        _INT = "{:>10d}"

        fmt_map = {
            "total_return":        ("Total Return",          _PCT),
            "cagr":                ("CAGR",                  _PCT),
            "sharpe":              ("Sharpe Ratio",          _FLT),
            "sortino":             ("Sortino Ratio",         _FLT),
            "calmar":              ("Calmar Ratio",          _FLT),
            "max_drawdown":        ("Max Drawdown",          _PCT),
            "avg_drawdown":        ("Avg Drawdown",          _PCT),
            "max_dd_duration_days": ("Max DD Duration (days)", "{:>10.0f}"),
            "volatility":          ("Ann. Volatility",       _PCT),
            "win_rate":            ("Win Rate",              _PCT),
            "profit_factor":       ("Profit Factor",         _FLT),
            "beta":                ("Beta",                  _FLT),
            "alpha":               ("Alpha (ann.)",          _PCT),
            "information_ratio":   ("Information Ratio",     _FLT),
        }

        lines: List[str] = ["=" * 44, "  Performance Report", "=" * 44]
        for key, (label, fmt) in fmt_map.items():
            if key in stats:
                val = stats[key]
                try:
                    lines.append(f"  {label:<26s} {fmt.format(val)}")
                except (ValueError, TypeError):
                    lines.append(f"  {label:<26s} {'N/A':>10s}")
        lines.append("=" * 44)
        return "\n".join(lines)
