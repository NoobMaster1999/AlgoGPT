"""
AlgoGPT + Kronos: AI-Powered Options Intelligence for Indian Derivatives
=========================================================================

Main package exposing all public APIs.
"""
from .model import KronosTokenizer, Kronos, KronosPredictor
from .data import (
    create_fetcher,
    IndianDerivativesDataset,
    create_train_val_datasets,
    SymbolConfig,
    INDIAN_INDEX_SYMBOLS,
)
from .training import FinetuneConfig, load_config_from_env
from .trading import (
    FeatureEngineer,
    SignalGenerator,
    KronosBacktester,
    run_backtest,
    BacktestConfig,
    BacktestResults,
    TradeSignal,
)

__version__ = "0.1.0"
__author__ = "Soumen Basak"

__all__ = [
    # Model
    "KronosTokenizer",
    "Kronos",
    "KronosPredictor",
    # Data
    "create_fetcher",
    "IndianDerivativesDataset",
    "create_train_val_datasets",
    "SymbolConfig",
    "INDIAN_INDEX_SYMBOLS",
    # Training
    "FinetuneConfig",
    "load_config_from_env",
    # Trading
    "FeatureEngineer",
    "SignalGenerator",
    "KronosBacktester",
    "run_backtest",
    "BacktestConfig",
    "BacktestResults",
    "TradeSignal",
]