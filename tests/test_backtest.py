"""
Tests for core/backtester.py and core/performance.py.

Covers
------
* Window generation — rolling and anchored modes
* Transaction-cost calculation
* Full walk-forward run — smoke tests with lightweight stub sub-components
* PerformanceAnalyser — every public metric
* BenchmarkComparison — buy-and-hold, SMA trend-following, random entry
* StressTestInjector — crash injection, bear-market injection
* Circuit-breaker and portfolio hard-stop verification
* Trade tracking — regime label and confidence captured in TradeRecord
* Confidence-breakdown and regime-breakdown attribution
"""

from __future__ import annotations

import types
from datetime import date

import numpy as np
import pandas as pd
import pytest

from config.settings import BACKTEST, UNIVERSE
from core.backtester import (
    Backtester,
    BacktestResult,
    BacktestWindow,
    BenchmarkComparison,
    FoldResult,
    SimulationResult,
    StressTestInjector,
    TradeRecord,
)
from core.performance import PerformanceAnalyser
from core.regime_strategies import StrategyOrchestrator


# ---------------------------------------------------------------------------
# Lightweight mock sub-components (deepcopy-safe plain Python classes)
# ---------------------------------------------------------------------------


class _FixedHMM:
    """HMM stub: always returns the 'neutral' regime with 80 % confidence."""

    n_regimes = 3
    regime_names = ["bear", "neutral", "bull"]

    def fit(self, X: np.ndarray) -> "_FixedHMM":
        return self

    def detect_regime(self, X: np.ndarray):  # noqa: ANN001
        return types.SimpleNamespace(
            regime=1,
            regime_name="neutral",
            confidence=0.80,
            is_uncertain=False,
            posteriors=np.array([0.1, 0.8, 0.1]),
            flicker_count=0,
            raw_regime=1,
        )


class _FixedFeatureEng:
    """FeatureEngineer stub: returns deterministic random features."""

    _COLS = ["log_return", "log_volatility", "volume_zscore", "realized_vol_20d"]

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def build_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        n = len(ohlcv)
        data = self._rng.standard_normal((n, 4))
        return pd.DataFrame(data, index=ohlcv.index, columns=self._COLS)

    def fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        return features.values

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        return features.values


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(idx: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    """Wrap a close-price array in a minimal OHLCV DataFrame."""
    n = len(idx)
    return pd.DataFrame(
        {
            "open":   close * 0.999,
            "high":   close * 1.005,
            "low":    close * 0.995,
            "close":  close,
            "volume": np.ones(n) * 1_000_000.0,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backtester() -> Backtester:
    """Backtester with BACKTEST config defaults (rolling, IS=252, OOS=126)."""
    return Backtester(
        in_sample_days=BACKTEST.in_sample_days,
        out_of_sample_days=BACKTEST.out_of_sample_days,
        initial_capital=BACKTEST.initial_capital,
        commission_pct=BACKTEST.commission_pct,
        slippage_bps=BACKTEST.slippage_bps,
        anchored=BACKTEST.walk_forward_anchored,
    )


@pytest.fixture
def synthetic_price_data() -> pd.DataFrame:
    """3 years of synthetic SPY OHLCV data (756 bars)."""
    rng = np.random.default_rng(1)
    n = 756
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = 400 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    return pd.DataFrame(
        {
            "open":   close * rng.uniform(0.998, 1.002, n),
            "high":   close * rng.uniform(1.001, 1.015, n),
            "low":    close * rng.uniform(0.985, 0.999, n),
            "close":  close,
            "volume": rng.integers(50_000_000, 150_000_000, n),
        },
        index=idx,
    )


@pytest.fixture
def trading_days(synthetic_price_data: pd.DataFrame) -> pd.DatetimeIndex:
    return synthetic_price_data.index


@pytest.fixture
def small_price_data() -> pd.DataFrame:
    """400-bar OHLCV — yields exactly 1 fold for IS=252, OOS=126."""
    rng = np.random.default_rng(7)
    n = 400
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = 300 * np.exp(np.cumsum(rng.normal(0.0003, 0.010, n)))
    return pd.DataFrame(
        {
            "open":   close * rng.uniform(0.999, 1.001, n),
            "high":   close * rng.uniform(1.001, 1.010, n),
            "low":    close * rng.uniform(0.990, 0.999, n),
            "close":  close,
            "volume": rng.integers(50_000_000, 100_000_000, n),
        },
        index=idx,
    )


@pytest.fixture
def mock_hmm() -> _FixedHMM:
    return _FixedHMM()


@pytest.fixture
def mock_feature_eng() -> _FixedFeatureEng:
    return _FixedFeatureEng()


@pytest.fixture
def strategy_orch() -> StrategyOrchestrator:
    return StrategyOrchestrator(tickers=["SPY"])


# ---------------------------------------------------------------------------
# Window generation (implementing original stubs)
# ---------------------------------------------------------------------------


def test_generate_windows_returns_non_empty_list(
    backtester: Backtester, trading_days: pd.DatetimeIndex
) -> None:
    """generate_windows() returns at least one BacktestWindow."""
    windows = backtester.generate_windows(trading_days)
    assert isinstance(windows, list)
    assert len(windows) >= 1
    assert all(isinstance(w, BacktestWindow) for w in windows)


def test_oos_windows_do_not_overlap(
    backtester: Backtester, trading_days: pd.DatetimeIndex
) -> None:
    """No two OOS windows share a date."""
    windows = backtester.generate_windows(trading_days)
    for i in range(1, len(windows)):
        prev = windows[i - 1]
        curr = windows[i]
        assert curr.out_of_sample_start > prev.out_of_sample_end, (
            f"Fold {i} OOS start {curr.out_of_sample_start} "
            f"overlaps with fold {i-1} OOS end {prev.out_of_sample_end}"
        )


def test_is_window_precedes_oos_window(
    backtester: Backtester, trading_days: pd.DatetimeIndex
) -> None:
    """Every in-sample window ends before its out-of-sample window begins."""
    windows = backtester.generate_windows(trading_days)
    for w in windows:
        assert w.in_sample_end < w.out_of_sample_start, (
            f"Fold {w.fold_id}: IS end {w.in_sample_end} "
            f">= OOS start {w.out_of_sample_start}"
        )


def test_rolling_window_is_length_constant(
    backtester: Backtester, trading_days: pd.DatetimeIndex
) -> None:
    """In rolling mode, each IS window has the same number of trading days."""
    # Ensure we are in rolling mode
    bt = Backtester(in_sample_days=252, out_of_sample_days=63, anchored=False)
    windows = bt.generate_windows(trading_days)
    assert len(windows) >= 2, "Need ≥ 2 folds to compare IS lengths"

    date_to_idx = {ts.date(): i for i, ts in enumerate(trading_days)}
    lengths = [
        date_to_idx[w.in_sample_end] - date_to_idx[w.in_sample_start] + 1
        for w in windows
    ]
    assert all(l == lengths[0] for l in lengths), (
        f"IS window lengths are not constant in rolling mode: {lengths}"
    )


def test_anchored_window_grows(trading_days: pd.DatetimeIndex) -> None:
    """In anchored mode, each IS window is strictly longer than the previous."""
    bt = Backtester(in_sample_days=252, out_of_sample_days=63, anchored=True)
    windows = bt.generate_windows(trading_days)
    assert len(windows) >= 2, "Need ≥ 2 folds to compare IS lengths"

    date_to_idx = {ts.date(): i for i, ts in enumerate(trading_days)}
    lengths = [
        date_to_idx[w.in_sample_end] - date_to_idx[w.in_sample_start] + 1
        for w in windows
    ]
    for i in range(1, len(lengths)):
        assert lengths[i] > lengths[i - 1], (
            f"Anchored IS window should grow: "
            f"fold {i} length {lengths[i]} <= fold {i-1} length {lengths[i-1]}"
        )


def test_insufficient_data_raises(backtester: Backtester) -> None:
    """generate_windows() raises ValueError when not enough data for one fold."""
    short = pd.bdate_range("2020-01-01", periods=10)
    with pytest.raises(ValueError, match="trading days"):
        backtester.generate_windows(short)


# ---------------------------------------------------------------------------
# Transaction costs (implementing original stubs)
# ---------------------------------------------------------------------------


def test_transaction_cost_reduces_dollar_amount(backtester: Backtester) -> None:
    """_apply_transaction_costs() returns less than the gross dollar amount."""
    gross = 10_000.0
    net = backtester._apply_transaction_costs(gross)
    assert net < gross


def test_zero_slippage_zero_commission_is_unchanged() -> None:
    """With zero costs, _apply_transaction_costs() returns the original amount."""
    bt = Backtester(slippage_bps=0.0, commission_pct=0.0)
    gross = 10_000.0
    net = bt._apply_transaction_costs(gross)
    assert net == pytest.approx(gross)


# ---------------------------------------------------------------------------
# Full run smoke tests (implementing original stubs)
# ---------------------------------------------------------------------------


def test_run_returns_backtest_result(
    backtester: Backtester,
    small_price_data: pd.DataFrame,
    mock_hmm: _FixedHMM,
    mock_feature_eng: _FixedFeatureEng,
    strategy_orch: StrategyOrchestrator,
) -> None:
    """run() returns a BacktestResult with at least one fold."""
    result = backtester.run(
        price_data=small_price_data,
        hmm_engine=mock_hmm,
        feature_engineer=mock_feature_eng,
        strategy_router=strategy_orch,
        run_benchmarks=False,
    )
    assert isinstance(result, BacktestResult)
    assert len(result.folds) >= 1


def test_combined_equity_starts_at_initial_capital(
    backtester: Backtester,
    small_price_data: pd.DataFrame,
    mock_hmm: _FixedHMM,
    mock_feature_eng: _FixedFeatureEng,
    strategy_orch: StrategyOrchestrator,
) -> None:
    """The combined equity curve's first value is close to initial_capital."""
    result = backtester.run(
        price_data=small_price_data,
        hmm_engine=mock_hmm,
        feature_engineer=mock_feature_eng,
        strategy_router=strategy_orch,
        run_benchmarks=False,
    )
    assert not result.combined_equity.empty
    # First bar: no position yet → NAV ≈ initial_capital minus first-trade cost.
    # One-way cost ≈ 50 % × 0.15 % ≈ 0.075 %, so within 1 % of initial capital.
    assert result.combined_equity.iloc[0] == pytest.approx(
        backtester.initial_capital, rel=0.01
    )


def test_fold_count_matches_windows(
    backtester: Backtester,
    small_price_data: pd.DataFrame,
    mock_hmm: _FixedHMM,
    mock_feature_eng: _FixedFeatureEng,
    strategy_orch: StrategyOrchestrator,
) -> None:
    """The number of FoldResult objects equals the number of generated windows."""
    windows = backtester.generate_windows(small_price_data.index)
    result = backtester.run(
        price_data=small_price_data,
        hmm_engine=mock_hmm,
        feature_engineer=mock_feature_eng,
        strategy_router=strategy_orch,
        run_benchmarks=False,
    )
    assert len(result.folds) == len(windows)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


class TestPerformanceMetrics:

    # -- total_return --

    def test_total_return_empty_series(self) -> None:
        assert PerformanceAnalyser.total_return(pd.Series(dtype=float)) == 0.0

    def test_total_return_flat(self) -> None:
        returns = pd.Series([0.0, 0.0, 0.0])
        assert PerformanceAnalyser.total_return(returns) == pytest.approx(0.0)

    def test_total_return_compound(self) -> None:
        # (1.10)(1.10) − 1 = 0.21
        returns = pd.Series([0.10, 0.10])
        assert PerformanceAnalyser.total_return(returns) == pytest.approx(0.21)

    # -- cagr --

    def test_cagr_one_year(self) -> None:
        r_daily = 1.10 ** (1.0 / 252) - 1
        returns = pd.Series([r_daily] * 252)
        assert PerformanceAnalyser.cagr(returns) == pytest.approx(0.10, rel=1e-4)

    def test_cagr_empty(self) -> None:
        assert PerformanceAnalyser.cagr(pd.Series(dtype=float)) == 0.0

    # -- sharpe ratio --

    def test_sharpe_zero_std_returns_zero(self) -> None:
        # All excess returns equal zero → std = 0 → Sharpe = 0
        rf_daily = 0.05 / 252
        returns = pd.Series([rf_daily] * 100)
        assert PerformanceAnalyser.sharpe_ratio(returns, 0.05) == pytest.approx(0.0)

    def test_sharpe_positive_mean(self) -> None:
        # Positive mean with non-zero std → positive Sharpe
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.002, 0.01, 252))
        sharpe = PerformanceAnalyser.sharpe_ratio(returns, risk_free_rate_annual=0.0)
        assert sharpe > 0.0

    # -- sortino ratio --

    def test_sortino_no_losing_days_is_inf(self) -> None:
        returns = pd.Series([0.01] * 50)
        result = PerformanceAnalyser.sortino_ratio(returns, risk_free_rate_annual=0.0)
        assert result == float("inf")

    def test_sortino_negative_returns(self) -> None:
        # Varying negative returns → downside std > 0 → Sortino < 0
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(-0.01, 0.005, 252))
        result = PerformanceAnalyser.sortino_ratio(returns, risk_free_rate_annual=0.0)
        assert result < 0.0

    # -- drawdown --

    def test_max_drawdown_known(self) -> None:
        # returns = [0.0, -0.2] → equity = [1.0, 0.8] → dd_min = -0.2
        returns = pd.Series([0.0, -0.2])
        assert PerformanceAnalyser.max_drawdown(returns) == pytest.approx(0.2)

    def test_max_drawdown_no_drawdown(self) -> None:
        # Monotonically rising equity → max_drawdown = 0
        returns = pd.Series([0.01, 0.02, 0.005])
        assert PerformanceAnalyser.max_drawdown(returns) == pytest.approx(0.0)

    def test_max_drawdown_empty(self) -> None:
        assert PerformanceAnalyser.max_drawdown(pd.Series(dtype=float)) == 0.0

    def test_avg_drawdown_no_drawdown(self) -> None:
        returns = pd.Series([0.01, 0.02])
        assert PerformanceAnalyser.avg_drawdown(returns) == pytest.approx(0.0)

    def test_avg_drawdown_positive_when_in_dd(self) -> None:
        returns = pd.Series([0.0, -0.10, -0.05])
        assert PerformanceAnalyser.avg_drawdown(returns) > 0.0

    def test_max_drawdown_duration_known(self) -> None:
        # equity = [1.0, 0.9, 0.855, 1.026]
        # in_dd  = [F,   T,   T,     F]  → max consecutive = 2
        returns = pd.Series([0.0, -0.10, -0.05, 0.20])
        assert PerformanceAnalyser.max_drawdown_duration(returns) == 2

    def test_max_drawdown_duration_no_drawdown(self) -> None:
        returns = pd.Series([0.01, 0.02, 0.03])
        assert PerformanceAnalyser.max_drawdown_duration(returns) == 0

    # -- win rate & profit factor --

    def test_win_rate_all_positive(self) -> None:
        pnl = pd.Series([1.0, 2.0, 3.0])
        assert PerformanceAnalyser.win_rate(pnl) == pytest.approx(1.0)

    def test_win_rate_half_positive(self) -> None:
        pnl = pd.Series([1.0, -1.0, 1.0, -1.0])
        assert PerformanceAnalyser.win_rate(pnl) == pytest.approx(0.5)

    def test_win_rate_empty(self) -> None:
        assert PerformanceAnalyser.win_rate(pd.Series(dtype=float)) == 0.0

    def test_profit_factor_known(self) -> None:
        # Gross profit = 300, gross loss = 150 → PF = 2.0
        pnl = pd.Series([100.0, -50.0, 200.0, -100.0])
        assert PerformanceAnalyser.profit_factor(pnl) == pytest.approx(2.0)

    def test_profit_factor_no_losses_is_inf(self) -> None:
        pnl = pd.Series([100.0, 200.0])
        assert PerformanceAnalyser.profit_factor(pnl) == float("inf")

    # -- relative metrics --

    def test_beta_perfect_correlation(self) -> None:
        r = pd.Series([0.01, -0.01, 0.02, -0.02, 0.005])
        assert PerformanceAnalyser.beta(r, r.copy()) == pytest.approx(1.0)

    def test_information_ratio_identical_series(self) -> None:
        r = pd.Series([0.01, -0.01, 0.02])
        # Active return = 0 everywhere → IR = 0
        assert PerformanceAnalyser.information_ratio(r, r.copy()) == pytest.approx(0.0)

    # -- summary --

    def test_summary_has_required_keys(self) -> None:
        rng = np.random.default_rng(99)
        returns = pd.Series(rng.normal(0.0005, 0.01, 252))
        stats = PerformanceAnalyser.summary(returns)
        required = {
            "total_return", "cagr", "sharpe", "sortino", "calmar",
            "max_drawdown", "avg_drawdown", "max_dd_duration_days",
            "volatility", "win_rate", "profit_factor",
        }
        assert required <= set(stats.keys())

    def test_summary_with_benchmark_adds_relative_metrics(self) -> None:
        rng = np.random.default_rng(99)
        returns = pd.Series(rng.normal(0.0005, 0.01, 252))
        bench = pd.Series(rng.normal(0.0003, 0.012, 252))
        stats = PerformanceAnalyser.summary(returns, bench)
        for key in ("beta", "alpha", "information_ratio"):
            assert key in stats, f"Missing key '{key}' in summary with benchmark"

    # -- rolling metrics --

    def test_rolling_sharpe_correct_length(self) -> None:
        returns = pd.Series(np.ones(100) * 0.001)
        rs = PerformanceAnalyser.rolling_sharpe(returns, window=21)
        assert len(rs) == 100

    def test_rolling_sharpe_nan_warmup(self) -> None:
        returns = pd.Series(np.ones(100) * 0.001)
        rs = PerformanceAnalyser.rolling_sharpe(returns, window=21)
        assert rs.iloc[:20].isna().all()

    def test_rolling_volatility_constant_series(self) -> None:
        # Constant returns → zero rolling vol
        returns = pd.Series(np.ones(63) * 0.001)
        rv = PerformanceAnalyser.rolling_volatility(returns, window=21)
        assert rv.dropna().iloc[-1] == pytest.approx(0.0, abs=1e-10)

    # -- equity curve --

    def test_equity_curve_starts_at_initial_capital(self) -> None:
        returns = pd.Series([0.0, 0.01, -0.01])
        eq = PerformanceAnalyser.equity_curve(returns, initial_capital=10_000.0)
        assert eq.iloc[0] == pytest.approx(10_000.0)

    def test_equity_curve_length_matches_returns(self) -> None:
        returns = pd.Series([0.01, -0.005, 0.02])
        eq = PerformanceAnalyser.equity_curve(returns, initial_capital=1.0)
        assert len(eq) == len(returns)


# ---------------------------------------------------------------------------
# Confidence breakdown
# ---------------------------------------------------------------------------


class TestConfidenceBreakdown:

    @pytest.fixture
    def sample(self):
        # confs: 0.8→high, 0.9→high, 0.55→medium, 0.3→low, 0.65→medium
        returns = pd.Series([0.01, 0.02, -0.01, 0.005, -0.005])
        confs = pd.Series([0.8, 0.9, 0.55, 0.3, 0.65])
        return returns, confs

    def test_all_three_buckets_present(self, sample) -> None:
        returns, confs = sample
        result = PerformanceAnalyser.confidence_breakdown(returns, confs)
        assert set(result.keys()) == {"high", "medium", "low"}

    def test_high_bucket_n_days(self, sample) -> None:
        returns, confs = sample
        result = PerformanceAnalyser.confidence_breakdown(returns, confs)
        assert result["high"]["n_days"] == 2   # 0.8, 0.9

    def test_medium_bucket_n_days(self, sample) -> None:
        returns, confs = sample
        result = PerformanceAnalyser.confidence_breakdown(returns, confs)
        assert result["medium"]["n_days"] == 2  # 0.55, 0.65

    def test_low_bucket_n_days(self, sample) -> None:
        returns, confs = sample
        result = PerformanceAnalyser.confidence_breakdown(returns, confs)
        assert result["low"]["n_days"] == 1    # 0.3

    def test_empty_bucket_zero_days(self) -> None:
        # All confidence > 0.70 → medium and low are empty
        returns = pd.Series([0.01, 0.02])
        confs = pd.Series([0.8, 0.9])
        result = PerformanceAnalyser.confidence_breakdown(returns, confs)
        assert result["medium"]["n_days"] == 0
        assert result["low"]["n_days"] == 0

    def test_boundary_70_pct_is_medium_not_high(self) -> None:
        # c = 0.70 → medium (c > 0.70 is required for high)
        returns = pd.Series([0.01])
        confs = pd.Series([0.70])
        result = PerformanceAnalyser.confidence_breakdown(returns, confs)
        assert result["high"]["n_days"] == 0
        assert result["medium"]["n_days"] == 1

    def test_required_keys_in_each_bucket(self, sample) -> None:
        returns, confs = sample
        result = PerformanceAnalyser.confidence_breakdown(returns, confs)
        expected = {"n_days", "avg_daily_return", "sharpe", "total_return"}
        for bucket in ("high", "medium", "low"):
            assert expected <= set(result[bucket].keys())


# ---------------------------------------------------------------------------
# Regime breakdown
# ---------------------------------------------------------------------------


class TestRegimeBreakdown:

    def test_returns_correct_regime_keys(self) -> None:
        returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
        regimes = pd.Series([0, 0, 1, 1, 1])
        result = PerformanceAnalyser.regime_breakdown(returns, regimes)
        assert 0 in result and 1 in result

    def test_n_days_per_regime(self) -> None:
        returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
        regimes = pd.Series([0, 0, 1, 1, 1])
        result = PerformanceAnalyser.regime_breakdown(returns, regimes)
        assert result[0]["n_days"] == 2
        assert result[1]["n_days"] == 3

    def test_pct_of_time_sums_to_one(self) -> None:
        returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
        regimes = pd.Series([0, 0, 1, 1, 1])
        result = PerformanceAnalyser.regime_breakdown(returns, regimes)
        total = sum(v["pct_of_time"] for v in result.values())
        assert total == pytest.approx(1.0)

    def test_required_stat_keys_present(self) -> None:
        returns = pd.Series([0.01, 0.02])
        regimes = pd.Series([0, 0])
        result = PerformanceAnalyser.regime_breakdown(returns, regimes)
        expected = {
            "total_return", "avg_daily_return", "volatility",
            "sharpe", "max_drawdown", "n_days", "pct_of_time",
        }
        assert expected <= set(result[0].keys())


# ---------------------------------------------------------------------------
# Benchmark comparisons
# ---------------------------------------------------------------------------


class TestBenchmarkComparison:

    def test_buy_and_hold_length(self, synthetic_price_data: pd.DataFrame) -> None:
        close = synthetic_price_data["close"]
        nav = BenchmarkComparison.buy_and_hold(close, initial_capital=100_000)
        assert len(nav) == len(close)

    def test_buy_and_hold_first_value(self, synthetic_price_data: pd.DataFrame) -> None:
        # pct_change().fillna(0) → first bar has 0 return → NAV[0] = capital
        close = synthetic_price_data["close"]
        nav = BenchmarkComparison.buy_and_hold(close, initial_capital=100_000)
        assert nav.iloc[0] == pytest.approx(100_000.0)

    def test_sma_length_matches_close(self, synthetic_price_data: pd.DataFrame) -> None:
        close = synthetic_price_data["close"]
        nav = BenchmarkComparison.sma_trend_following(close, sma_window=200)
        assert len(nav) == len(close)

    def test_sma_nav_always_positive(self, synthetic_price_data: pd.DataFrame) -> None:
        close = synthetic_price_data["close"]
        nav = BenchmarkComparison.sma_trend_following(close)
        assert (nav > 0).all()

    def test_sma_starts_at_initial_capital(self, synthetic_price_data: pd.DataFrame) -> None:
        # First bar: signal is 0 (shift(1).fillna(0)) → strategy is flat → NAV=capital
        close = synthetic_price_data["close"]
        nav = BenchmarkComparison.sma_trend_following(close, initial_capital=100_000)
        assert nav.iloc[0] == pytest.approx(100_000.0)

    def test_random_entry_reproducible_with_same_seed(
        self, synthetic_price_data: pd.DataFrame
    ) -> None:
        close = synthetic_price_data["close"]
        nav1 = BenchmarkComparison.random_entry(close, seed=42)
        nav2 = BenchmarkComparison.random_entry(close, seed=42)
        pd.testing.assert_series_equal(nav1, nav2)

    def test_random_entry_differs_with_different_seeds(
        self, synthetic_price_data: pd.DataFrame
    ) -> None:
        close = synthetic_price_data["close"]
        nav1 = BenchmarkComparison.random_entry(close, seed=42)
        nav2 = BenchmarkComparison.random_entry(close, seed=99)
        assert not nav1.equals(nav2)

    def test_run_all_returns_three_strategies(
        self, synthetic_price_data: pd.DataFrame
    ) -> None:
        close = synthetic_price_data["close"]
        results = BenchmarkComparison.run_all(close)
        assert set(results.keys()) == {"buy_and_hold", "sma_trend_following", "random_entry"}

    def test_run_all_nav_lengths_match_close(
        self, synthetic_price_data: pd.DataFrame
    ) -> None:
        close = synthetic_price_data["close"]
        results = BenchmarkComparison.run_all(close)
        for name, nav in results.items():
            assert len(nav) == len(close), (
                f"{name}: expected {len(close)} bars, got {len(nav)}"
            )


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------


class TestStressTests:

    def test_inject_crash_returns_same_shape(
        self, synthetic_price_data: pd.DataFrame
    ) -> None:
        stressed = StressTestInjector.inject_crash_events(synthetic_price_data)
        assert stressed.shape == synthetic_price_data.shape

    def test_inject_crash_does_not_modify_original(
        self, synthetic_price_data: pd.DataFrame
    ) -> None:
        original = synthetic_price_data["close"].copy()
        StressTestInjector.inject_crash_events(synthetic_price_data)
        pd.testing.assert_series_equal(synthetic_price_data["close"], original)

    def test_inject_crash_lowers_close_at_crash_bars(
        self, synthetic_price_data: pd.DataFrame
    ) -> None:
        stressed = StressTestInjector.inject_crash_events(
            synthetic_price_data, n_events=3, seed=42
        )
        # Overall prices should be ≤ original (crashes only go down)
        assert (stressed["close"] <= synthetic_price_data["close"] + 1e-9).all()
        # At least one bar must actually be lower
        assert (stressed["close"] < synthetic_price_data["close"]).any()

    def test_inject_crash_n_events(self, synthetic_price_data: pd.DataFrame) -> None:
        original = synthetic_price_data["close"].copy()
        stressed = StressTestInjector.inject_crash_events(
            synthetic_price_data, n_events=5, seed=0
        )
        n_different = (stressed["close"] != original).sum()
        assert n_different == 5

    def test_inject_bear_market_negative_cumulative_return(
        self, synthetic_price_data: pd.DataFrame
    ) -> None:
        n = len(synthetic_price_data)
        start = n // 3
        stressed = StressTestInjector.inject_bear_market(
            synthetic_price_data,
            start_idx=start,
            duration_days=30,
            daily_decline_pct=0.005,
        )
        cum_ret = StressTestInjector.cumulative_return_in_window(
            stressed, start, start + 30
        )
        assert cum_ret < 0.0

    def test_inject_bear_market_does_not_modify_original(
        self, synthetic_price_data: pd.DataFrame
    ) -> None:
        original = synthetic_price_data["close"].copy()
        StressTestInjector.inject_bear_market(synthetic_price_data)
        pd.testing.assert_series_equal(synthetic_price_data["close"], original)

    def test_inject_bear_market_pre_window_unchanged(
        self, synthetic_price_data: pd.DataFrame
    ) -> None:
        n = len(synthetic_price_data)
        start = n // 2
        original_before = synthetic_price_data["close"].iloc[:start].copy()
        stressed = StressTestInjector.inject_bear_market(
            synthetic_price_data, start_idx=start, duration_days=20
        )
        pd.testing.assert_series_equal(
            stressed["close"].iloc[:start], original_before
        )


# ---------------------------------------------------------------------------
# Circuit breaker and portfolio hard stop
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """Verify that risk controls fire correctly inside simulate_oos."""

    N_IS = 50
    N_OOS = 20

    @staticmethod
    def _make_is_oos(n_is: int, n_oos: int, oos_close: np.ndarray):
        idx_is = pd.bdate_range("2020-01-01", periods=n_is)
        idx_oos = pd.bdate_range(idx_is[-1] + pd.offsets.BDay(1), periods=n_oos)
        is_data = _make_ohlcv(idx_is, np.ones(n_is) * 100.0)
        oos_data = _make_ohlcv(idx_oos, oos_close)
        return is_data, oos_data

    def test_circuit_breaker_triggers_on_large_single_day_loss(
        self,
        mock_hmm: _FixedHMM,
        mock_feature_eng: _FixedFeatureEng,
        strategy_orch: StrategyOrchestrator,
    ) -> None:
        """10 % crash with 50 % position → ~5 % NAV loss → CB fires."""
        # Bar 0: flat (regime = neutral → 50 % invested)
        # Bar 1: -10 % crash → daily NAV loss ≈ 5 % → circuit breaker
        oos_close = np.array([100.0] + [90.0] * (self.N_OOS - 1))
        is_data, oos_data = self._make_is_oos(self.N_IS, self.N_OOS, oos_close)

        bt = Backtester(
            in_sample_days=self.N_IS,
            out_of_sample_days=self.N_OOS,
            daily_loss_limit_pct=0.05,
            initial_capital=100_000.0,
        )
        sim = bt.simulate_oos(
            is_data=is_data,
            oos_data=oos_data,
            fitted_hmm=mock_hmm,
            feature_eng=mock_feature_eng,
            strategy_orch=strategy_orch,
            starting_nav=100_000.0,
        )
        assert len(sim.circuit_breaker_events) >= 1

    def test_no_circuit_breaker_on_small_loss(
        self,
        mock_hmm: _FixedHMM,
        mock_feature_eng: _FixedFeatureEng,
        strategy_orch: StrategyOrchestrator,
    ) -> None:
        """1 % decline → NAV loss < 5 % → no circuit breaker."""
        oos_close = np.array([100.0] + [99.0] * (self.N_OOS - 1))
        is_data, oos_data = self._make_is_oos(self.N_IS, self.N_OOS, oos_close)

        bt = Backtester(
            in_sample_days=self.N_IS,
            out_of_sample_days=self.N_OOS,
            daily_loss_limit_pct=0.05,
            initial_capital=100_000.0,
        )
        sim = bt.simulate_oos(
            is_data=is_data,
            oos_data=oos_data,
            fitted_hmm=mock_hmm,
            feature_eng=mock_feature_eng,
            strategy_orch=strategy_orch,
            starting_nav=100_000.0,
        )
        assert len(sim.circuit_breaker_events) == 0

    def test_portfolio_hard_stop_limits_further_loss(
        self,
        mock_hmm: _FixedHMM,
        mock_feature_eng: _FixedFeatureEng,
        strategy_orch: StrategyOrchestrator,
    ) -> None:
        """Sustained 3 % daily decline eventually trips the 25 % portfolio hard stop."""
        n_is, n_oos = 50, 40
        # Price declines 3 % per bar — after ~10 bars, cumulative loss ≈ 26 %
        oos_close = 100.0 * (0.97 ** np.arange(n_oos))
        is_data, oos_data = self._make_is_oos(n_is, n_oos, oos_close)

        bt = Backtester(
            in_sample_days=n_is,
            out_of_sample_days=n_oos,
            portfolio_dd_limit_pct=0.25,
            daily_loss_limit_pct=1.0,   # disable daily CB so hard stop can fire
            initial_capital=100_000.0,
        )
        sim = bt.simulate_oos(
            is_data=is_data,
            oos_data=oos_data,
            fitted_hmm=mock_hmm,
            feature_eng=mock_feature_eng,
            strategy_orch=strategy_orch,
            starting_nav=100_000.0,
        )
        # After liquidation NAV should be positive but significantly below start
        assert sim.equity_curve.iloc[-1] > 0
        assert sim.equity_curve.iloc[-1] < 100_000.0


# ---------------------------------------------------------------------------
# Trade tracking
# ---------------------------------------------------------------------------


class TestTradeTracking:
    """Verify that regime, confidence, and other fields are captured in trades."""

    @staticmethod
    def _run_sim(
        mock_hmm: _FixedHMM,
        mock_feature_eng: _FixedFeatureEng,
        strategy_orch: StrategyOrchestrator,
        n_oos: int = 10,
    ) -> SimulationResult:
        rng = np.random.default_rng(3)
        n_is = 50
        idx_is = pd.bdate_range("2020-01-01", periods=n_is)
        idx_oos = pd.bdate_range(idx_is[-1] + pd.offsets.BDay(1), periods=n_oos)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n_is + n_oos)))
        is_data = _make_ohlcv(idx_is, close[:n_is])
        oos_data = _make_ohlcv(idx_oos, close[n_is:])
        bt = Backtester(in_sample_days=n_is, out_of_sample_days=n_oos)
        return bt.simulate_oos(
            is_data=is_data,
            oos_data=oos_data,
            fitted_hmm=mock_hmm,
            feature_eng=mock_feature_eng,
            strategy_orch=strategy_orch,
            starting_nav=100_000.0,
        )

    def test_trade_dataframe_has_required_columns(
        self,
        mock_hmm: _FixedHMM,
        mock_feature_eng: _FixedFeatureEng,
        strategy_orch: StrategyOrchestrator,
    ) -> None:
        sim = self._run_sim(mock_hmm, mock_feature_eng, strategy_orch)
        required = {"regime", "confidence", "regime_name", "timestamp",
                    "ticker", "side", "dollar_amount", "cost", "pnl"}
        assert required <= set(sim.trades.columns)

    def test_initial_rebalance_trade_is_recorded(
        self,
        mock_hmm: _FixedHMM,
        mock_feature_eng: _FixedFeatureEng,
        strategy_orch: StrategyOrchestrator,
    ) -> None:
        # Starting with 0 % allocation → neutral → 50 % allocation → rebalance fires
        sim = self._run_sim(mock_hmm, mock_feature_eng, strategy_orch)
        assert not sim.trades.empty, "Expected at least one trade (initial rebalance)"

    def test_trade_regime_matches_hmm_output(
        self,
        mock_hmm: _FixedHMM,
        mock_feature_eng: _FixedFeatureEng,
        strategy_orch: StrategyOrchestrator,
    ) -> None:
        # _FixedHMM always returns regime=1
        sim = self._run_sim(mock_hmm, mock_feature_eng, strategy_orch)
        if not sim.trades.empty:
            assert (sim.trades["regime"] == 1).all()

    def test_trade_confidence_in_unit_interval(
        self,
        mock_hmm: _FixedHMM,
        mock_feature_eng: _FixedFeatureEng,
        strategy_orch: StrategyOrchestrator,
    ) -> None:
        sim = self._run_sim(mock_hmm, mock_feature_eng, strategy_orch)
        if not sim.trades.empty:
            assert (sim.trades["confidence"] >= 0.0).all()
            assert (sim.trades["confidence"] <= 1.0).all()

    def test_trade_side_is_buy_or_sell(
        self,
        mock_hmm: _FixedHMM,
        mock_feature_eng: _FixedFeatureEng,
        strategy_orch: StrategyOrchestrator,
    ) -> None:
        sim = self._run_sim(mock_hmm, mock_feature_eng, strategy_orch)
        if not sim.trades.empty:
            assert sim.trades["side"].isin({"buy", "sell"}).all()

    def test_trade_cost_is_positive(
        self,
        mock_hmm: _FixedHMM,
        mock_feature_eng: _FixedFeatureEng,
        strategy_orch: StrategyOrchestrator,
    ) -> None:
        sim = self._run_sim(mock_hmm, mock_feature_eng, strategy_orch)
        if not sim.trades.empty:
            assert (sim.trades["cost"] >= 0.0).all()
