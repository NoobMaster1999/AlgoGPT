"""
Signal Generation for AlgoGPT + Kronos
=======================================
Fuses Kronos forecasts with technical analysis for trading decisions.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from .features import FeatureEngineer, FeatureConfig

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Container for a trading signal."""
    direction: int          # +1 = BUY, 0 = HOLD, -1 = SELL
    strength: float         # continuous in [-1, +1]
    confidence: float       # 0-1 from ensemble agreement
    kronos_direction: int   # raw Kronos signal
    ta_direction: int       # raw TA signal
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        action = {1: "BUY", 0: "HOLD", -1: "SELL"}[self.direction]
        return (
            f"Signal={action} | "
            f"strength={self.strength:+.3f} | "
            f"confidence={self.confidence:.2f} | "
            f"kronos={self.kronos_direction:+d} | "
            f"ta={self.ta_direction:+d}"
        )


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    kronos_weight: float = 0.65
    buy_threshold: float = 0.20
    sell_threshold: float = -0.20
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    use_macd_filter: bool = True
    min_confidence: float = 0.5


class SignalGenerator:
    """
    Fuses Kronos price forecasts with technical analysis signals.

    Strategy:
    1. Kronos directional signal from forecast close price
    2. Technical indicator confluence (RSI, MACD, BB, MA cross, candles)
    3. Weighted ensemble (Kronos gets higher weight as foundation model)
    4. Optional MACD filter for confirmation
    """

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self.kronos_weight = float(np.clip(self.config.kronos_weight, 0.0, 1.0))
        self.ta_weight = 1.0 - self.kronos_weight

    def generate(
        self,
        current_row: pd.Series,
        pred_df: pd.DataFrame,
    ) -> TradeSignal:
        """
        Generate a trade signal by combining Kronos forecast with TA.

        Parameters
        ----------
        current_row : pd.Series
            Most recent bar from feature-engineered DataFrame
        pred_df : pd.DataFrame
            Kronos prediction DataFrame (output of KronosPredictor.forecast())

        Returns
        -------
        TradeSignal
        """
        # 1. Kronos directional signal
        last_close = current_row["close"]
        pred_close_avg = pred_df["close"].mean()
        pct_change = (pred_close_avg - last_close) / (last_close + 1e-9)

        # Soft Kronos score in [-1, +1] using tanh scaling
        kronos_score = float(np.tanh(pct_change * 20))
        kronos_dir = int(np.sign(kronos_score)) if abs(kronos_score) > 0.05 else 0

        # 2. Technical indicator signals
        ta_scores: Dict[str, float] = {}

        # RSI
        rsi = current_row.get("RSI", 50.0)
        if pd.isna(rsi):
            ta_scores["rsi"] = 0.0
        elif rsi < self.config.rsi_oversold:
            ta_scores["rsi"] = +(1 - rsi / self.config.rsi_oversold)
        elif rsi > self.config.rsi_overbought:
            ta_scores["rsi"] = -(rsi - self.config.rsi_overbought) / (100 - self.config.rsi_overbought)
        else:
            ta_scores["rsi"] = 0.0

        # MACD histogram
        macd_hist = current_row.get("macd_hist", 0.0)
        ta_scores["macd"] = float(np.tanh(macd_hist * 100)) if not pd.isna(macd_hist) else 0.0

        # Moving average cross-over
        sma_cross = current_row.get("sma_cross", 0.0)
        ta_scores["sma_cross"] = float(np.tanh(sma_cross * 50)) if not pd.isna(sma_cross) else 0.0

        # Bollinger Band position (mean reversion)
        bb_pos = current_row.get("bb_position", 0.5)
        ta_scores["bb"] = float(np.tanh((0.5 - bb_pos) * 4)) if not pd.isna(bb_pos) else 0.0

        # Candlestick patterns
        bullish = current_row.get("Bullish_Engulfing", 0) + current_row.get("Hammer", 0)
        bearish = current_row.get("Bearish_Engulfing", 0) + current_row.get("Shooting_Star", 0)
        ta_scores["candle"] = float(np.tanh((bullish - bearish) * 0.5))

        # Aggregate TA score
        ta_score = float(np.mean(list(ta_scores.values())))
        ta_dir = int(np.sign(ta_score)) if abs(ta_score) > 0.05 else 0

        # 3. Ensemble score
        ensemble = self.kronos_weight * kronos_score + self.ta_weight * ta_score

        # 4. MACD filter (optional)
        if self.config.use_macd_filter:
            macd_dir = int(np.sign(ta_scores.get("macd", 0.0)))
            if macd_dir != 0 and macd_dir != kronos_dir:
                ensemble *= 0.5  # Discount when MACD and Kronos disagree

        # 5. Discrete direction + confidence
        if ensemble >= self.config.buy_threshold:
            direction = 1
        elif ensemble <= self.config.sell_threshold:
            direction = -1
        else:
            direction = 0

        # Confidence: agreement fraction among sub-signals
        sub_signals = [kronos_dir] + [int(np.sign(v)) for v in ta_scores.values() if v != 0]
        if sub_signals:
            agree = sum(s == direction for s in sub_signals)
            confidence = agree / len(sub_signals)
        else:
            confidence = 0.5

        return TradeSignal(
            direction=direction,
            strength=float(np.clip(ensemble, -1, 1)),
            confidence=float(np.clip(confidence, 0, 1)),
            kronos_direction=kronos_dir,
            ta_direction=ta_dir,
            details={
                "kronos_score": kronos_score,
                "ta_score": ta_score,
                "ta_breakdown": ta_scores,
                "pct_change": pct_change,
                "pred_close": pred_close_avg,
                "last_close": last_close,
            },
        )

    def generate_series(
        self,
        feature_df: pd.DataFrame,
        pred_df_series: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate signal series over a full DataFrame."""
        records = []
        for idx in feature_df.index:
            if idx not in pred_df_series.index:
                continue
            row = feature_df.loc[idx]
            pred_row = pred_df_series.loc[[idx]] if isinstance(pred_df_series.loc[idx], pd.Series) else pred_df_series.loc[idx:idx]
            sig = self.generate(row, pred_row)
            records.append({
                "timestamp": idx,
                "direction": sig.direction,
                "strength": sig.strength,
                "confidence": sig.confidence,
                "kronos_dir": sig.kronos_direction,
                "ta_dir": sig.ta_direction,
            })

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records).set_index("timestamp")

    @staticmethod
    def position_size(
        signal: TradeSignal,
        capital: float,
        price: float,
        max_risk_pct: float = 0.02,
        min_confidence: float = 0.5,
    ) -> float:
        """Compute position size based on signal strength and confidence."""
        if signal.direction == 0 or signal.confidence < min_confidence:
            return 0.0

        risk_capital = capital * max_risk_pct
        scaled_risk = risk_capital * signal.confidence * abs(signal.strength)
        shares = scaled_risk / (price + 1e-9)
        return float(np.floor(shares))