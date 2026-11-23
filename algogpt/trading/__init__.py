"""
AlgoGPT Trading Package
=======================
Feature engineering, signal generation, and backtesting for Indian derivatives.
"""
from .features import FeatureEngineer, FeatureConfig
from .signals import SignalGenerator, SignalConfig, TradeSignal
from .backtest import KronosBacktester, BacktestConfig, BacktestResults, run_backtest

__all__ = [
    "FeatureEngineer",
    "FeatureConfig",
    "SignalGenerator",
    "SignalConfig",
    "TradeSignal",
    "KronosBacktester",
    "BacktestConfig",
    "BacktestResults",
    "run_backtest",
]