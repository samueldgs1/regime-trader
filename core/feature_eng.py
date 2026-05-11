"""
Technical-indicator feature pipeline.

Transforms raw OHLCV price data into the scaled feature matrix consumed
by the HMM engine and any ML-based strategy components.

Required HMM features (must appear first in feature_names for regime sorting):
    log_return      — daily log return (causal, no look-ahead)
    log_volatility  — log of 20-day realised vol (more Gaussian-distributed)
    volume_zscore   — volume standardised against rolling window
    realized_vol_20d — 20-day rolling realised volatility (annualised)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

_REQUIRED_COLS: List[str] = ["open", "high", "low", "close", "volume"]
_ANNUALISE: float = np.sqrt(252)


class FeatureEngineer:
    """Compute and scale technical indicators from OHLCV data."""

    def __init__(self, feature_names: List[str]) -> None:
        """
        Parameters
        ----------
        feature_names:
            Ordered list of feature keys to include in the output matrix.
            The first entry should be 'log_return' so that HMMEngine can
            sort regimes by mean return using index 0.
        """
        self.feature_names = feature_names
        self._scaler: Optional[RobustScaler] = None

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def build_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from an OHLCV DataFrame.

        Parameters
        ----------
        ohlcv:
            DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            and a DatetimeIndex.  Columns may be lower- or upper-case.

        Returns
        -------
        DataFrame with one column per feature, NaN rows dropped.
        """
        ohlcv = ohlcv.copy()
        ohlcv.columns = [c.lower() for c in ohlcv.columns]
        self._validate_ohlcv(ohlcv)

        close = ohlcv["close"]
        high = ohlcv["high"]
        low = ohlcv["low"]
        volume = ohlcv["volume"]

        all_feats = pd.DataFrame(index=ohlcv.index)

        # --- four primary HMM features ---
        all_feats["log_return"] = self.log_return(close)
        all_feats["log_volatility"] = self.log_volatility(close, window=20)
        all_feats["volume_zscore"] = self.volume_zscore(volume, window=21)
        all_feats["realized_vol_20d"] = self.realized_vol(close, window=20)

        # --- supplementary features ---
        all_feats["realized_vol_5d"] = self.realized_vol(close, window=5)
        all_feats["rsi_14"] = self.rsi(close, period=14)
        all_feats["macd_signal"] = self.macd_signal(close)
        all_feats["atr_14_pct"] = self.atr(high, low, close, period=14)
        all_feats["bb_bandwidth"] = self.bollinger_bandwidth(close)

        selected = self._select_features(all_feats)
        return selected.dropna()

    def fit_scaler(self, features: pd.DataFrame) -> "FeatureEngineer":
        """
        Fit a RobustScaler on the training window.

        RobustScaler is preferred over StandardScaler because financial
        features (especially vol and volume) have heavy tails.

        Returns
        -------
        self
        """
        self._scaler = RobustScaler()
        self._scaler.fit(features.values)
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted scaler to a feature DataFrame.

        Returns
        -------
        Scaled numpy array of shape (n_bars, n_features).
        """
        if self._scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit_scaler() first.")
        return self._scaler.transform(features.values)

    def fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        """Fit scaler and transform in one step."""
        return self.fit_scaler(features).transform(features)

    # ------------------------------------------------------------------
    # Individual indicators
    # ------------------------------------------------------------------

    def log_return(self, close: pd.Series) -> pd.Series:
        """Compute log return: ln(close_t / close_{t-1})."""
        return np.log(close / close.shift(1)).rename("log_return")

    def realized_vol(self, close: pd.Series, window: int = 20) -> pd.Series:
        """Rolling standard deviation of log returns, annualised."""
        lr = np.log(close / close.shift(1))
        rv = lr.rolling(window).std() * _ANNUALISE
        return rv.rename(f"realized_vol_{window}d")

    def log_volatility(self, close: pd.Series, window: int = 20) -> pd.Series:
        """
        Log of annualised realised volatility.

        Taking the log makes the feature more Gaussian-distributed, which
        improves HMM emission density fit.
        """
        rv = self.realized_vol(close, window)
        return np.log(rv.clip(lower=1e-8)).rename("log_volatility")

    def volume_zscore(self, volume: pd.Series, window: int = 21) -> pd.Series:
        """Z-score of volume relative to its rolling mean and std."""
        mean = volume.rolling(window).mean()
        std = volume.rolling(window).std().clip(lower=1e-8)
        return ((volume - mean) / std).rename("volume_zscore")

    def rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index (0–100)."""
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.clip(lower=1e-8)
        return (100 - 100 / (1 + rs)).rename(f"rsi_{period}")

    def macd_signal(
        self,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.Series:
        """MACD histogram: (MACD line) − (signal line)."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return (macd - signal_line).rename("macd_signal")

    def atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Average True Range as a percentage of close price."""
        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr_val = tr.ewm(span=period, adjust=False).mean()
        return (atr_val / close.clip(lower=1e-8)).rename(f"atr_{period}_pct")

    def bollinger_bandwidth(
        self,
        close: pd.Series,
        period: int = 20,
        n_std: float = 2.0,
    ) -> pd.Series:
        """(Upper band − lower band) / middle band."""
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = mid + n_std * std
        lower = mid - n_std * std
        return ((upper - lower) / mid.clip(lower=1e-8)).rename("bb_bandwidth")

    def adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Average Directional Index (trend strength, 0–100)."""
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)

        atr_s = tr.ewm(span=period, adjust=False).mean()
        plus_di = (
            pd.Series(plus_dm, index=close.index)
            .ewm(span=period, adjust=False)
            .mean()
            / atr_s.clip(lower=1e-8)
            * 100
        )
        minus_di = (
            pd.Series(minus_dm, index=close.index)
            .ewm(span=period, adjust=False)
            .mean()
            / atr_s.clip(lower=1e-8)
            * 100
        )
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower=1e-8)) * 100
        return dx.ewm(span=period, adjust=False).mean().rename("adx")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_ohlcv(self, ohlcv: pd.DataFrame) -> None:
        """Raise ValueError if required columns are missing."""
        missing = [c for c in _REQUIRED_COLS if c not in ohlcv.columns]
        if missing:
            raise ValueError(
                f"Missing required OHLCV columns: {missing}. "
                f"Found: {list(ohlcv.columns)}"
            )

    def _select_features(self, all_features: pd.DataFrame) -> pd.DataFrame:
        """Return only the columns listed in self.feature_names, in order."""
        missing = [f for f in self.feature_names if f not in all_features.columns]
        if missing:
            raise ValueError(
                f"The following feature_names were not computed: {missing}. "
                f"Available: {list(all_features.columns)}"
            )
        return all_features[self.feature_names]

    @property
    def is_fitted(self) -> bool:
        """True after fit_scaler() has been called."""
        return self._scaler is not None
