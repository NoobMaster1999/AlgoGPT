"""
Feature Engineering for AlgoGPT + Kronos
=========================================
Technical indicators and Kronos-friendly features for Indian derivatives.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    rsi_window: int = 14
    sma_short: int = 20
    sma_long: int = 50
    atr_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_window: int = 14
    bb_window: int = 20
    bb_std: float = 2.0
    ctx_window: int = 20


class FeatureEngineer:
    """
    Unified feature engineering for AlgoGPT + Kronos trading pipeline.

    Computes technical indicators compatible with both the Kronos foundation
    model and traditional ML models (RandomForest, XGBoost, etc.).
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add all features to a copy of data.

        Parameters
        ----------
        data : pd.DataFrame
            Must have at least: open, high, low, close
            Optional: volume, amount

        Returns
        -------
        pd.DataFrame
            DataFrame with all features added
        """
        df = data.copy()

        # Normalize column names
        df.columns = [c.lower() for c in df.columns]

        # Ensure required columns
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found")

        if "volume" not in df.columns:
            df["volume"] = 0.0
        if "amount" not in df.columns:
            df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)

        # Add all feature groups
        df = self._add_returns(df)
        df = self._add_moving_averages(df)
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_atr(df)
        df = self._add_stochastic(df)
        df = self._add_vwap(df)
        df = self._add_bollinger_bands(df)
        df = self._add_candlestick_patterns(df)
        df = self._add_kronos_context_features(df)

        return df

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["price_change"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["hl_range"] = (df["high"] - df["low"]) / df["close"]
        df["oc_change"] = (df["close"] - df["open"]) / df["open"]
        df["near_day_low"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-9)
        df["near_day_high"] = (df["high"] - df["close"]) / (df["high"] - df["low"] + 1e-9)
        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        df[f"sma_{c.sma_short}"] = df["close"].rolling(c.sma_short).mean()
        df[f"sma_{c.sma_long}"] = df["close"].rolling(c.sma_long).mean()
        df[f"ema_{c.sma_short}"] = df["close"].ewm(span=c.sma_short, adjust=False).mean()
        df[f"ema_{c.sma_long}"] = df["close"].ewm(span=c.sma_long, adjust=False).mean()

        # Normalized cross-over signals
        df["sma_cross"] = (df[f"sma_{c.sma_short}"] - df[f"sma_{c.sma_long}"]) / df["close"]
        df["ema_cross"] = (df[f"ema_{c.sma_short}"] - df[f"ema_{c.sma_long}"]) / df["close"]

        # Legacy names for compatibility
        df["SMA_50"] = df["close"].rolling(50).mean()
        df["SMA_200"] = df["close"].rolling(200).mean()
        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(c.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(c.rsi_window).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))
        df["rsi_norm"] = df["RSI"] / 100.0
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        ema_fast = df["close"].ewm(span=c.macd_fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=c.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal = macd_line.ewm(span=c.macd_signal, adjust=False).mean()
        df["macd"] = macd_line / df["close"]
        df["macd_signal"] = signal / df["close"]
        df["macd_hist"] = (macd_line - signal) / df["close"]
        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        hl = df["high"] - df["low"]
        hpc = (df["high"] - df["close"].shift(1)).abs()
        lpc = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        df["atr"] = tr.rolling(c.atr_window).mean()
        df["atr_norm"] = df["atr"] / df["close"]
        return df

    def _add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        low_n = df["low"].rolling(c.stoch_window).min()
        high_n = df["high"].rolling(c.stoch_window).max()
        df["stoch_k"] = (df["close"] - low_n) / (high_n - low_n + 1e-9) * 100
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()
        return df

    def _add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        tp_vol = typical_price * df["volume"]
        cum_vol = df["volume"].cumsum()
        cum_tp_vol = tp_vol.cumsum()
        df["vwap"] = cum_tp_vol / (cum_vol + 1e-9)
        df["vwap_norm"] = (df["close"] - df["vwap"]) / df["close"]
        return df

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        sma = df["close"].rolling(c.bb_window).mean()
        std = df["close"].rolling(c.bb_window).std()
        upper = sma + c.bb_std * std
        lower = sma - c.bb_std * std
        df["bb_upper"] = upper
        df["bb_lower"] = lower
        df["bb_width"] = (upper - lower) / sma
        df["bb_position"] = (df["close"] - lower) / (upper - lower + 1e-9)
        return df

    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Bullish Engulfing
        df["Bullish_Engulfing"] = (
            (df["close"] > df["open"]) &
            (df["close"].shift(1) < df["open"].shift(1)) &
            (df["close"] > df["open"].shift(1)) &
            (df["open"] < df["close"].shift(1))
        ).astype(int)

        # Bearish Engulfing
        df["Bearish_Engulfing"] = (
            (df["close"] < df["open"]) &
            (df["close"].shift(1) > df["open"].shift(1)) &
            (df["close"] < df["open"].shift(1)) &
            (df["open"] > df["close"].shift(1))
        ).astype(int)

        # Doji
        body = (df["close"] - df["open"]).abs()
        range_ = df["high"] - df["low"] + 1e-9
        df["Doji"] = (body / range_ < 0.1).astype(int)

        # Hammer
        lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
        upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
        df["Hammer"] = (
            (lower_wick > 2 * body) &
            (upper_wick < body) &
            (body > 0)
        ).astype(int)

        # Shooting Star
        df["Shooting_Star"] = (
            (upper_wick > 2 * body) &
            (lower_wick < body) &
            (body > 0)
        ).astype(int)

        return df

    def _add_kronos_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        df["ctx_close_mean"] = df["close"].rolling(c.ctx_window).mean() / df["close"]
        df["ctx_close_std"] = df["close"].rolling(c.ctx_window).std() / df["close"]
        df["ctx_vol_mean"] = df["volume"].rolling(c.ctx_window).mean()
        df["ctx_vol_std"] = df["volume"].rolling(c.ctx_window).std()
        df["ctx_trend"] = (df["close"] - df["close"].shift(c.ctx_window)) / (df["close"].shift(c.ctx_window) + 1e-9)
        df["ctx_momentum"] = df["close"].pct_change(c.ctx_window)
        return df

    @property
    def feature_columns(self) -> List[str]:
        """Canonical list of feature columns produced by compute()."""
        c = self.config
        return [
            "price_change", "log_return", "hl_range", "oc_change",
            "near_day_low", "near_day_high",
            f"sma_{c.sma_short}", f"sma_{c.sma_long}",
            f"ema_{c.sma_short}", f"ema_{c.sma_long}",
            "sma_cross", "ema_cross",
            "SMA_50", "SMA_200",
            "RSI", "rsi_norm",
            "macd", "macd_signal", "macd_hist",
            "atr", "atr_norm",
            "stoch_k", "stoch_d",
            "vwap", "vwap_norm",
            "bb_upper", "bb_lower", "bb_width", "bb_position",
            "Bullish_Engulfing", "Bearish_Engulfing",
            "Doji", "Hammer", "Shooting_Star",
            "ctx_close_mean", "ctx_close_std",
            "ctx_vol_mean", "ctx_vol_std",
            "ctx_trend", "ctx_momentum",
        ]

    @staticmethod
    def to_kronos_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Convert yfinance-style DataFrame to Kronos format."""
        col_map = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
            "Adj Close": "close",
        }
        out = df.rename(columns=col_map).copy()
        out.columns = [c.lower() for c in out.columns]
        return out