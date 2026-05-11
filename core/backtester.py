"""
Walk-forward backtest engine.

Architecture
------------
Backtester.run()
    └── generate_windows()                 → [BacktestWindow, …]
    └── run_single_fold() × n_folds
            ├── fit FeatureEngineer (scaler) on IS data
            ├── fit HMMEngine on IS scaled features
            └── simulate_oos()             → SimulationResult
    └── aggregate_folds()                  → BacktestResult

Simulation design
-----------------
* Strictly causal: at bar t we may only use o_{1:t-1} for decisions
  and o_{1:t} for the return on existing positions.
* Regime classification uses HMMEngine.detect_regime() (forward algorithm,
  no Viterbi) on a rolling window of IS + OOS-seen-so-far features.
* Rebalance gate: only trade when target allocation differs by > 5 pp.
* Transaction costs: one-way slippage (5 bps) + commission deducted
  from NAV on every executed trade.
* Circuit breaker: when daily NAV loss exceeds daily_loss_limit_pct
  (default 5 %), trading is halted for that bar and the event is logged.

Benchmarks (BenchmarkComparison)
---------------------------------
* Buy-and-hold
* 200-day SMA trend following
* Random entry with same allocation rules

Stress testing (StressTestInjector)
-------------------------------------
* inject_crash_events()  — insert n random 10–15 % single-day crashes
* inject_bear_market()   — impose 30 days of steady daily declines
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.performance import PerformanceAnalyser

logger = logging.getLogger(__name__)

_DEFAULT_DAILY_LOSS_LIMIT: float = 0.05       # 5 % intraday circuit breaker
_DEFAULT_PORTFOLIO_DD_LIMIT: float = 0.25     # 25 % peak-to-trough hard stop
_MAX_FEATURE_HISTORY: int = 350               # rolling window for forward algorithm
_COST_RATE_SCALE: float = 10_000.0           # bps divisor


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BacktestWindow:
    """One walk-forward fold — both date ranges are inclusive."""
    fold_id: int
    in_sample_start: date
    in_sample_end: date
    out_of_sample_start: date
    out_of_sample_end: date


@dataclass
class TradeRecord:
    """Record of a single rebalance execution."""
    timestamp: pd.Timestamp
    ticker: str
    side: str                # 'buy' | 'sell'
    dollar_amount: float     # absolute notional
    price: float             # execution price (bar close)
    regime: int              # HMM sorted label at entry
    regime_name: str         # e.g. 'bull', 'bear'
    confidence: float        # HMM forward-posterior confidence at entry
    cost: float              # slippage + commission deducted from NAV
    pnl: float = 0.0        # filled in post-hoc by _compute_trade_pnl()


@dataclass
class SimulationResult:
    """Raw outputs of simulate_oos()."""
    equity_curve: pd.Series
    returns: pd.Series
    trades: pd.DataFrame              # one row per TradeRecord
    regime_sequence: pd.Series
    confidence_sequence: pd.Series
    circuit_breaker_events: List[pd.Timestamp] = field(default_factory=list)


@dataclass
class FoldResult:
    """Performance metrics for a single out-of-sample fold."""
    fold_id: int
    window: BacktestWindow
    equity_curve: pd.Series
    returns: pd.Series
    regime_sequence: pd.Series
    confidence_sequence: pd.Series
    trades: pd.DataFrame
    sharpe: float
    max_drawdown: float
    total_return: float
    win_rate: float
    regime_breakdown: Dict[int, Dict]
    circuit_breaker_events: List[pd.Timestamp] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Aggregated results across all walk-forward folds."""
    folds: List[FoldResult] = field(default_factory=list)
    combined_equity: pd.Series = field(default_factory=pd.Series)
    combined_returns: pd.Series = field(default_factory=pd.Series)
    aggregate_stats: Dict = field(default_factory=dict)
    benchmark_results: Dict[str, pd.Series] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Benchmark strategies
# ---------------------------------------------------------------------------

class BenchmarkComparison:
    """
    Three benchmark strategies that the walk-forward result can be compared
    against.  All are causal (no look-ahead).
    """

    @staticmethod
    def buy_and_hold(
        close: pd.Series,
        initial_capital: float = 100_000.0,
    ) -> pd.Series:
        """
        100 % long from day 1.

        Returns
        -------
        Daily NAV series indexed identically to close.
        """
        returns = close.pct_change().fillna(0.0)
        return initial_capital * (1.0 + returns).cumprod()

    @staticmethod
    def sma_trend_following(
        close: pd.Series,
        sma_window: int = 200,
        initial_capital: float = 100_000.0,
    ) -> pd.Series:
        """
        100 % long when close > 200-day SMA, otherwise 100 % cash.

        Signal is computed on close and applied at the NEXT bar (causal).
        Before the SMA period is warm, the strategy stays in cash.

        Returns
        -------
        Daily NAV series.
        """
        returns = close.pct_change().fillna(0.0)
        sma = close.rolling(sma_window).mean()
        # Signal: invested when above SMA, shifted +1 for causality
        invested = (close > sma).astype(float).shift(1).fillna(0.0)
        strategy_returns = returns * invested
        return initial_capital * (1.0 + strategy_returns).cumprod()

    @staticmethod
    def random_entry(
        close: pd.Series,
        allocation_pct: float = 0.95,
        initial_capital: float = 100_000.0,
        seed: int = 42,
    ) -> pd.Series:
        """
        Randomly decide each day whether to be fully invested or flat.
        Uses the same allocation_pct as the regime strategy.

        The random choice is shifted +1 for causality (decision made
        at prior close, executed next open).

        Returns
        -------
        Daily NAV series.
        """
        rng = np.random.default_rng(seed)
        n = len(close)
        invested = pd.Series(
            rng.choice([0.0, allocation_pct], size=n),
            index=close.index,
        ).shift(1).fillna(0.0)

        returns = close.pct_change().fillna(0.0)
        strategy_returns = returns * invested
        return initial_capital * (1.0 + strategy_returns).cumprod()

    @staticmethod
    def run_all(
        close: pd.Series,
        initial_capital: float = 100_000.0,
        sma_window: int = 200,
        random_seed: int = 42,
    ) -> Dict[str, pd.Series]:
        """
        Run all three benchmarks and return a dict of name → NAV series.
        """
        bc = BenchmarkComparison
        return {
            "buy_and_hold":        bc.buy_and_hold(close, initial_capital),
            "sma_trend_following": bc.sma_trend_following(close, sma_window, initial_capital),
            "random_entry":        bc.random_entry(close, initial_capital=initial_capital,
                                                    seed=random_seed),
        }


# ---------------------------------------------------------------------------
# Stress test injector
# ---------------------------------------------------------------------------

class StressTestInjector:
    """
    Utilities for inserting adverse market conditions into a price series
    so that circuit-breaker and drawdown logic can be verified.
    """

    @staticmethod
    def inject_crash_events(
        price_data: pd.DataFrame,
        n_events: int = 3,
        min_crash_pct: float = 0.10,
        max_crash_pct: float = 0.15,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Insert n_events single-bar crashes of random magnitude in
        [min_crash_pct, max_crash_pct].

        Crash bars are chosen randomly from the middle 80 % of the series
        (avoiding the warm-up and tail periods).  All OHLCV columns are
        scaled consistently.

        Returns
        -------
        Modified deep copy of price_data.
        """
        data = price_data.copy()
        rng = np.random.default_rng(seed)
        n = len(data)
        margin = max(1, int(n * 0.10))
        eligible = list(range(margin, n - margin))
        crash_indices = sorted(rng.choice(eligible, size=min(n_events, len(eligible)),
                                          replace=False).tolist())

        ohlcv_cols = [c for c in ["open", "high", "low", "close"] if c in data.columns]

        for idx in crash_indices:
            magnitude = rng.uniform(min_crash_pct, max_crash_pct)
            multiplier = 1.0 - magnitude
            data.iloc[idx, data.columns.get_indexer(ohlcv_cols)] *= multiplier
            # Ensure high ≥ close, low ≤ close after scaling
            if "high" in data.columns and "close" in data.columns:
                data.iat[idx, data.columns.get_loc("high")] = max(
                    data.iat[idx, data.columns.get_loc("high")],
                    data.iat[idx, data.columns.get_loc("close")],
                )
            if "low" in data.columns and "close" in data.columns:
                data.iat[idx, data.columns.get_loc("low")] = min(
                    data.iat[idx, data.columns.get_loc("low")],
                    data.iat[idx, data.columns.get_loc("close")],
                )

        logger.info(
            "Injected %d crash events at indices %s", n_events, crash_indices
        )
        return data

    @staticmethod
    def inject_bear_market(
        price_data: pd.DataFrame,
        start_idx: Optional[int] = None,
        duration_days: int = 30,
        daily_decline_pct: float = 0.005,
    ) -> pd.DataFrame:
        """
        Impose a sustained bear market: apply a constant daily_decline_pct
        to close (and proportionally to open/high/low) for duration_days bars
        starting at start_idx.

        Cumulative loss ≈ (1 − daily_decline_pct)^duration_days − 1.

        Returns
        -------
        Modified deep copy of price_data.
        """
        data = price_data.copy()
        n = len(data)
        if start_idx is None:
            start_idx = n // 3
        end_idx = min(start_idx + duration_days, n)

        ohlcv_cols = [c for c in ["open", "high", "low", "close"] if c in data.columns]

        for i in range(start_idx, end_idx):
            factor = 1.0 - daily_decline_pct
            data.iloc[i, data.columns.get_indexer(ohlcv_cols)] *= factor

        logger.info(
            "Injected bear market: bars %d–%d, daily_decline=%.2f%%",
            start_idx, end_idx - 1, daily_decline_pct * 100,
        )
        return data

    @staticmethod
    def cumulative_return_in_window(
        price_data: pd.DataFrame,
        start_idx: int,
        end_idx: int,
    ) -> float:
        """Return the compound return of the close price in [start_idx, end_idx)."""
        close = price_data["close"].iloc[start_idx:end_idx]
        if len(close) < 2:
            return 0.0
        returns = close.pct_change().dropna()
        return float((1.0 + returns).prod() - 1.0)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class Backtester:
    """
    Walk-forward backtest engine.

    Wires together FeatureEngineer, HMMEngine, and StrategyOrchestrator
    to produce a BacktestResult with per-fold metrics, combined equity curve,
    and benchmark comparisons.
    """

    def __init__(
        self,
        in_sample_days: int = 252,
        out_of_sample_days: int = 126,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.001,
        slippage_bps: float = 5.0,
        anchored: bool = False,
        daily_loss_limit_pct: float = _DEFAULT_DAILY_LOSS_LIMIT,
        portfolio_dd_limit_pct: float = _DEFAULT_PORTFOLIO_DD_LIMIT,
        ticker: str = "SPY",
    ) -> None:
        """
        Parameters
        ----------
        in_sample_days:
            Training window length in trading days.
        out_of_sample_days:
            Evaluation window length in trading days.
        initial_capital:
            Starting NAV in dollars.
        commission_pct:
            One-way commission as a decimal (0.001 = 0.1 %).
        slippage_bps:
            One-way slippage in basis points (5 bps = 0.05 %).
        anchored:
            If True, IS window always starts from the first bar (expanding).
            If False, rolling fixed-length window.
        daily_loss_limit_pct:
            Intraday NAV loss that triggers the circuit breaker (halts trading).
        portfolio_dd_limit_pct:
            Peak-to-trough drawdown that triggers a hard-stop (liquidate).
        ticker:
            Primary ticker symbol used in trade records.
        """
        self.in_sample_days = in_sample_days
        self.out_of_sample_days = out_of_sample_days
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_bps = slippage_bps
        self.anchored = anchored
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.portfolio_dd_limit_pct = portfolio_dd_limit_pct
        self.ticker = ticker

        self._one_way_cost_rate: float = (
            slippage_bps / _COST_RATE_SCALE + commission_pct
        )

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def run(
        self,
        price_data: pd.DataFrame,
        hmm_engine,
        feature_engineer,
        strategy_router,
        risk_manager=None,
        run_benchmarks: bool = True,
    ) -> BacktestResult:
        """
        Execute the full walk-forward backtest.

        Parameters
        ----------
        price_data:
            OHLCV DataFrame covering the full history.
        hmm_engine:
            Unfitted HMMEngine (deep-copied per fold so folds don't share state).
        feature_engineer:
            FeatureEngineer (deep-copied per fold).
        strategy_router:
            StrategyOrchestrator (shared — it is stateless).
        risk_manager:
            Currently unused (reserved for future RiskManager integration).
        run_benchmarks:
            If True, compute buy-and-hold, SMA, and random-entry benchmarks
            on the combined OOS period.

        Returns
        -------
        BacktestResult with per-fold FoldResults and combined equity curve.
        """
        windows = self.generate_windows(price_data.index)
        folds: List[FoldResult] = []
        nav = self.initial_capital

        for window in windows:
            fold_hmm = copy.deepcopy(hmm_engine)
            fold_fe = copy.deepcopy(feature_engineer)
            fold = self.run_single_fold(
                window=window,
                price_data=price_data,
                hmm_engine=fold_hmm,
                feature_engineer=fold_fe,
                strategy_router=strategy_router,
                starting_nav=nav,
            )
            folds.append(fold)
            if len(fold.equity_curve) > 0:
                nav = float(fold.equity_curve.iloc[-1])

        result = self.aggregate_folds(folds)

        if run_benchmarks and not price_data.empty:
            oos_start = result.combined_equity.index[0] if not result.combined_equity.empty else None
            if oos_start is not None:
                close = price_data["close"].loc[oos_start:]
                result.benchmark_results = BenchmarkComparison.run_all(
                    close, self.initial_capital
                )

        return result

    def run_single_fold(
        self,
        window: BacktestWindow,
        price_data: pd.DataFrame,
        hmm_engine,
        feature_engineer,
        strategy_router,
        starting_nav: Optional[float] = None,
    ) -> FoldResult:
        """
        Fit and evaluate a single BacktestWindow.

        IS data is used to fit the scaler + HMM.
        OOS data is simulated bar-by-bar using the fitted components.
        """
        if starting_nav is None:
            starting_nav = self.initial_capital

        is_data = price_data.loc[
            str(window.in_sample_start): str(window.in_sample_end)
        ]
        oos_data = price_data.loc[
            str(window.out_of_sample_start): str(window.out_of_sample_end)
        ]

        if is_data.empty or oos_data.empty:
            raise ValueError(
                f"Fold {window.fold_id}: IS or OOS data is empty. "
                "Check that price_data covers the full window range."
            )

        # ---- Train on IS ----
        is_features = feature_engineer.build_features(is_data)
        is_scaled = feature_engineer.fit_transform(is_features)
        hmm_engine.fit(is_scaled)

        logger.info(
            "Fold %d trained: IS=%s→%s  OOS=%s→%s  n_regimes=%d",
            window.fold_id,
            window.in_sample_start, window.in_sample_end,
            window.out_of_sample_start, window.out_of_sample_end,
            hmm_engine.n_regimes,
        )

        # ---- Simulate OOS ----
        sim = self.simulate_oos(
            is_data=is_data,
            oos_data=oos_data,
            fitted_hmm=hmm_engine,
            feature_eng=feature_engineer,
            strategy_orch=strategy_router,
            starting_nav=starting_nav,
        )

        # ---- Metrics ----
        pa = PerformanceAnalyser
        r = sim.returns

        regime_bkdn = pa.regime_breakdown(r, sim.regime_sequence)

        trade_pnl = (
            pd.Series(sim.trades["pnl"].values, dtype=float)
            if not sim.trades.empty and "pnl" in sim.trades.columns
            else pd.Series(dtype=float)
        )

        return FoldResult(
            fold_id=window.fold_id,
            window=window,
            equity_curve=sim.equity_curve,
            returns=r,
            regime_sequence=sim.regime_sequence,
            confidence_sequence=sim.confidence_sequence,
            trades=sim.trades,
            sharpe=pa.sharpe_ratio(r),
            max_drawdown=pa.max_drawdown(r),
            total_return=pa.total_return(r),
            win_rate=pa.win_rate(trade_pnl) if not trade_pnl.empty else 0.0,
            regime_breakdown=regime_bkdn,
            circuit_breaker_events=sim.circuit_breaker_events,
        )

    # ------------------------------------------------------------------
    # Window generation
    # ------------------------------------------------------------------

    def generate_windows(
        self,
        trading_days: pd.DatetimeIndex,
    ) -> List[BacktestWindow]:
        """
        Split the index into non-overlapping OOS walk-forward folds.

        In rolling mode (anchored=False), the IS window advances by
        out_of_sample_days on each fold.
        In anchored mode (anchored=True), the IS window always starts at
        index 0 and grows fold by fold.

        Raises
        ------
        ValueError if there are not enough bars for at least one fold.
        """
        n = len(trading_days)
        min_needed = self.in_sample_days + self.out_of_sample_days
        if n < min_needed:
            raise ValueError(
                f"Need at least {min_needed} trading days "
                f"(in_sample={self.in_sample_days} + out_of_sample="
                f"{self.out_of_sample_days}), got {n}."
            )

        windows: List[BacktestWindow] = []
        fold_id = 0
        oos_start = self.in_sample_days

        while oos_start + self.out_of_sample_days <= n:
            is_start = 0 if self.anchored else oos_start - self.in_sample_days
            is_end   = oos_start - 1
            oos_end  = oos_start + self.out_of_sample_days - 1

            windows.append(
                BacktestWindow(
                    fold_id=fold_id,
                    in_sample_start=trading_days[is_start].date(),
                    in_sample_end=trading_days[is_end].date(),
                    out_of_sample_start=trading_days[oos_start].date(),
                    out_of_sample_end=trading_days[oos_end].date(),
                )
            )
            oos_start += self.out_of_sample_days
            fold_id   += 1

        return windows

    # ------------------------------------------------------------------
    # OOS simulation
    # ------------------------------------------------------------------

    def simulate_oos(
        self,
        is_data: pd.DataFrame,
        oos_data: pd.DataFrame,
        fitted_hmm,
        feature_eng,
        strategy_orch,
        starting_nav: float,
    ) -> SimulationResult:
        """
        Step through OOS bars one at a time, applying regime signals.

        Design guarantees
        -----------------
        * Causality: decisions at bar t use only IS + OOS[0:t] features.
        * Circuit breaker: if daily NAV loss ≥ daily_loss_limit_pct,
          trading halts for that bar and the timestamp is recorded.
        * Portfolio hard-stop: if peak-to-trough drawdown ≥ portfolio_dd_limit_pct,
          the position is fully liquidated and no further trades occur.
        * Transaction costs: deducted from NAV on every trade execution.

        Returns
        -------
        SimulationResult with all time-series and trade records.
        """
        nav            = starting_nav
        cash           = nav
        pos_value      = 0.0    # current equity exposure in dollars
        peak_nav       = nav
        last_regime    = 0
        last_conf      = 0.5
        hard_stop_hit  = False

        nav_list    : List[float]         = []
        ret_list    : List[float]         = []
        regime_list : List[int]           = []
        conf_list   : List[float]         = []
        trade_recs  : List[TradeRecord]   = []
        cb_events   : List[pd.Timestamp]  = []

        # Rolling price history for feature computation (IS seed + OOS accumulation)
        rolling_prices = is_data.copy()

        prev_close = float(is_data["close"].iloc[-1])
        prev_nav   = nav

        regime_names = (
            fitted_hmm.regime_names
            if hasattr(fitted_hmm, "regime_names") and fitted_hmm.regime_names
            else ["regime_0"]
        )

        for i, (timestamp, bar) in enumerate(oos_data.iterrows()):
            bar_close = float(bar["close"])

            # ---- Step 1: apply today's return to existing position ----
            bar_return = (bar_close - prev_close) / prev_close if prev_close > 0 else 0.0
            pos_value *= (1.0 + bar_return)
            nav = cash + pos_value

            # ---- Step 2: update peak, check portfolio hard-stop ----
            if nav > peak_nav:
                peak_nav = nav
            portfolio_dd = (peak_nav - nav) / peak_nav if peak_nav > 0 else 0.0

            if not hard_stop_hit and portfolio_dd >= self.portfolio_dd_limit_pct:
                hard_stop_hit = True
                logger.warning(
                    "Portfolio hard-stop: drawdown=%.1f%% at %s",
                    portfolio_dd * 100, timestamp,
                )
                # Liquidate position
                cost = abs(pos_value) * self._one_way_cost_rate
                cash = nav - cost
                pos_value = 0.0
                nav = cash

            # ---- Step 3: daily circuit breaker ----
            daily_loss = (prev_nav - nav) / prev_nav if prev_nav > 0 else 0.0
            if daily_loss >= self.daily_loss_limit_pct and not hard_stop_hit:
                cb_events.append(timestamp)
                logger.info(
                    "Circuit breaker triggered: daily_loss=%.1f%% at %s",
                    daily_loss * 100, timestamp,
                )
                nav_list.append(nav)
                ret_list.append(bar_return)
                regime_list.append(last_regime)
                conf_list.append(last_conf)
                prev_close = bar_close
                prev_nav = nav
                continue   # skip trading today

            if not hard_stop_hit:
                # ---- Step 4: compute features (causal) ----
                rolling_prices = pd.concat(
                    [rolling_prices, oos_data.iloc[[i]]]
                ).iloc[-_MAX_FEATURE_HISTORY:]

                try:
                    features = feature_eng.build_features(rolling_prices)
                    scaled   = feature_eng.transform(features)
                    # Use a shorter recent window for the forward algorithm
                    recent   = scaled[-min(len(scaled), 60):]
                    state    = fitted_hmm.detect_regime(recent)
                    last_regime = state.regime
                    last_conf   = state.confidence
                    regime_name = (
                        regime_names[last_regime]
                        if last_regime < len(regime_names)
                        else f"regime_{last_regime}"
                    )
                    is_uncertain = state.is_uncertain
                except Exception as exc:
                    logger.debug("Feature/HMM error at %s: %s", timestamp, exc)
                    regime_name  = regime_names[last_regime] if last_regime < len(regime_names) else "unknown"
                    is_uncertain = False

                # ---- Step 5: generate signal ----
                signal = strategy_orch.get_signal(
                    regime=last_regime,
                    regime_name=regime_name,
                    confidence=last_conf,
                    is_uncertain=is_uncertain,
                )

                # ---- Step 6: rebalance if needed ----
                current_alloc = pos_value / nav if nav > 0 else 0.0
                if strategy_orch.should_rebalance(signal.allocation_pct, current_alloc):
                    target_pos   = signal.allocation_pct * nav
                    trade_amt    = target_pos - pos_value
                    cost         = abs(trade_amt) * self._one_way_cost_rate

                    side = "buy" if trade_amt > 0 else "sell"
                    trade_recs.append(
                        TradeRecord(
                            timestamp=timestamp,
                            ticker=self.ticker,
                            side=side,
                            dollar_amount=abs(trade_amt),
                            price=bar_close,
                            regime=last_regime,
                            regime_name=regime_name,
                            confidence=last_conf,
                            cost=cost,
                        )
                    )

                    pos_value = target_pos
                    cash      = nav - pos_value - cost
                    nav       = cash + pos_value    # NAV net of cost

            # ---- Step 7: record state ----
            nav_list.append(nav)
            ret_list.append(bar_return)
            regime_list.append(last_regime)
            conf_list.append(last_conf)
            prev_close = bar_close
            prev_nav   = nav

        idx = oos_data.index

        trade_df = (
            pd.DataFrame(
                [
                    {
                        "timestamp":    t.timestamp,
                        "ticker":       t.ticker,
                        "side":         t.side,
                        "dollar_amount": t.dollar_amount,
                        "price":        t.price,
                        "regime":       t.regime,
                        "regime_name":  t.regime_name,
                        "confidence":   t.confidence,
                        "cost":         t.cost,
                        "pnl":          t.pnl,
                    }
                    for t in trade_recs
                ]
            )
            if trade_recs
            else pd.DataFrame(
                columns=[
                    "timestamp", "ticker", "side", "dollar_amount",
                    "price", "regime", "regime_name", "confidence", "cost", "pnl",
                ]
            )
        )

        if not trade_df.empty:
            trade_df = self._compute_trade_pnl(trade_df)

        return SimulationResult(
            equity_curve=pd.Series(nav_list, index=idx, name="nav"),
            returns=pd.Series(ret_list, index=idx, name="return"),
            trades=trade_df,
            regime_sequence=pd.Series(regime_list, index=idx, name="regime"),
            confidence_sequence=pd.Series(conf_list, index=idx, name="confidence"),
            circuit_breaker_events=cb_events,
        )

    # ------------------------------------------------------------------
    # Transaction costs
    # ------------------------------------------------------------------

    def _apply_transaction_costs(
        self,
        dollar_traded: float,
        price: float = 0.0,
    ) -> float:
        """
        Return the net dollar amount received after one-way slippage and
        commission are deducted.

        net = dollar_traded × (1 − slip_rate − commission_rate)
        """
        return dollar_traded * (1.0 - self._one_way_cost_rate)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_folds(self, folds: List[FoldResult]) -> BacktestResult:
        """
        Concatenate fold equity curves and returns; compute aggregate stats.
        """
        if not folds:
            return BacktestResult()

        all_equity  = pd.concat([f.equity_curve  for f in folds])
        all_returns = pd.concat([f.returns       for f in folds])
        all_regimes = pd.concat([f.regime_sequence for f in folds])

        pa = PerformanceAnalyser
        agg_stats = pa.summary(all_returns)
        agg_stats["n_folds"]       = len(folds)
        agg_stats["total_trades"]  = sum(len(f.trades) for f in folds)
        agg_stats["total_cb_events"] = sum(len(f.circuit_breaker_events) for f in folds)

        return BacktestResult(
            folds=folds,
            combined_equity=all_equity,
            combined_returns=all_returns,
            aggregate_stats=agg_stats,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_trade_pnl(trade_df: pd.DataFrame) -> pd.DataFrame:
        """
        Approximate per-trade PnL from sequential position changes.

        Pairs consecutive buy→sell trades and computes
        pnl ≈ (exit_price − entry_price) / entry_price × dollar_amount − cost.
        Unmatched trades get pnl = 0.
        """
        df = trade_df.copy()
        df["pnl"] = 0.0

        entry_price = None
        entry_amount = 0.0
        entry_cost = 0.0

        for idx, row in df.iterrows():
            if row["side"] == "buy":
                entry_price  = row["price"]
                entry_amount = row["dollar_amount"]
                entry_cost   = row["cost"]
            elif row["side"] == "sell" and entry_price is not None and entry_price > 0:
                price_ret = (row["price"] - entry_price) / entry_price
                pnl = price_ret * entry_amount - entry_cost - row["cost"]
                df.at[idx, "pnl"] = pnl
                entry_price = None

        return df
