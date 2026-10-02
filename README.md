<div align="center">
<img align="center" width="30%" alt="image" src="https://github.com/NoobMaster1999/AlgoGPT/blob/main/Assests/AlgoGPT.png">
</div>

# AlgoGPT: Open-Source AI Foundation Model(Kronos) for Indian Derivatives and Equity Market Data
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Kronos-yellow.svg)](https://huggingface.co/NeoQuasar/Kronos-small)
[![pandas 2.0+](https://img.shields.io/badge/pandas-2.0+-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit_learn-1.3+-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.7+-green.svg?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



A clean, modular codebase for finetuning the Kronos foundation model on NIFTY/BANKNIFTY data and running systematic trading strategies.

## Architecture

```
algogpt/
├── model/           # Kronos foundation model (tokenizer, model, predictor)
├── data/            # Data fetching & dataset creation for Indian derivatives
├── training/        # Finetuning pipelines (tokenizer + predictor)
├── trading/         # Feature engineering, signals, backtesting
├── main.py          # Main trading pipeline entry point
├── quickstart.py    # Quick test with Yahoo Finance
└── requirements.txt # Dependencies
```

## Core Algorithem:
 - [Release] Jun, 2024: Released [AlgoGPT-Broker Setup Module](https://github.com/NoobMaster1999/AlgoGPT/blob/main/Broker%20Setup.ipynb)!  🔥[Base Model](https://github.com/NoobMaster1999/AlgoGPT/tree/main/Basic%20Version%20V1.0), [Core Structure](https://arxiv.org/abs/2306.06031)!
 -  [Release v2.0] October, 2026: Upgraded with [Kronos Foundation Model](https://github.com/shiyu-coder/Kronos) (BSQ K-Line Tokenizer + Causal Autoregressive Transformer), multi-symbol walk-forward backtesting, and automated market data feeds (Yahoo Finance, Upstox, NSE).

## Model Performace:
 - [Version 1.0] April, 2024: [Base Model](https://github.com/NoobMaster1999/AlgoGPT/blob/main/Assests/Screenshot%202024-07-02%20171707.png) Model Accuracy: 0.43
 -  [Version 2.0] October, 2026: [Kronos Foundation Engine](https://github.com/shiyu-coder/Kronos) (`NeoQuasar/Kronos-small` & `Kronos-Tokenizer-base`) with continuous BSQ candlestick tokenization, multi-step autoregressive price path generation, and systematic options derivative backtesting with risk metrics (Sharpe Ratio, Alpha, Max Drawdown).

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run quick test (uses Yahoo Finance for NIFTY daily data)
python quickstart.py
```

## Data Sources

| Source | Description | Setup |
|--------|-------------|-------|
| `csv` | Local CSV files in `./data/` | Place `{SYMBOL}_5min.csv` files |
| `yahoo` | Yahoo Finance (free, daily/intraday) | No setup needed |
| `upstox` | Upstox API (live + historical) | Requires API credentials |
| `nse` | NSE via nsepython (daily only) | `pip install nsepython` |

## Finetuning on Indian Derivatives

### 1. Prepare Data
```bash
# Download NIFTY/BANKNIFTY 5min data to ./data/
# Or use Upstox API with credentials
```

### 2. Configure Training
Edit `training/config.py` or use environment variables:
```bash
export SYMBOLS="NIFTY,BANKNIFTY"
export DATA_SOURCE="csv"
export DATA_DIR="./data"
export INTERVAL="5min"
export START_DATE="2020-01-01"
export END_DATE="2024-12-31"
export PREDICTOR_EPOCHS=30
export BATCH_SIZE=50
```

### 3. Run Distributed Finetuning
```bash
# Tokenizer finetuning (run first)
torchrun --standalone --nproc_per_node=1 training/train_tokenizer.py

# Predictor finetuning (uses finetuned tokenizer)
torchrun --standalone --nproc_per_node=1 training/train_predictor.py
```

For multi-GPU:
```bash
torchrun --standalone --nproc_per_node=4 training/train_predictor.py
```

### 4. Use Finetuned Models
```python
from algogpt.model import KronosPredictor

predictor = KronosPredictor.from_local(
    model_path="./outputs/models/finetune_predictor_indian/checkpoints/best_model",
    tokenizer_path="./outputs/models/finetune_tokenizer_indian/checkpoints/best_model",
)
```

## Trading Pipeline

```bash
# Basic backtest on NIFTY
python main.py --symbol NIFTY --start 2023-01-01 --end 2024-06-01

# With Upstox data
python main.py --symbol BANKNIFTY --data-source upstox \
  --upstox-api-key YOUR_KEY --upstox-api-secret YOUR_SECRET \
  --upstox-access-token YOUR_TOKEN

# Using finetuned model
python main.py --model ./outputs/models/finetune_predictor_indian/checkpoints/best_model \
  --tokenizer ./outputs/models/finetune_tokenizer_indian/checkpoints/best_model
```

## Key Components

### Model (`algogpt/model/`)
- **KronosTokenizer**: Hybrid quantization (BSQ) encoder-decoder
- **Kronos**: Autoregressive transformer with hierarchical tokens (s1+s2)
- **KronosPredictor**: High-level inference with normalization, sampling, batch prediction

### Data (`algogpt/data/`)
- **DataFetcher**: Abstract base + Upstox/NSE/Yahoo/CSV implementations
- **IndianDerivativesDataset**: Sliding window dataset with proper normalization
- **create_train_val_datasets**: Time-aware train/val split (no lookahead bias)

### Training (`algogpt/training/`)
- **FinetuneConfig**: Centralized dataclass config with env overrides
- **train_tokenizer.py**: DDP tokenizer finetuning (BSQ loss)
- **train_predictor.py**: DDP predictor finetuning (next-token CE loss)
- **utils.py**: DDP setup, seeding, checkpointing, Comet logging

### Trading (`algogpt/trading/`)
- **FeatureEngineer**: 30+ technical indicators (RSI, MACD, BB, candles, etc.)
- **SignalGenerator**: Kronos + TA weighted ensemble with MACD filter
- **KronosBacktester**: Walk-forward with realistic execution (next-bar open, slippage, commission)

## Indian Derivatives Specifics

| Symbol | Lot Size | Expiry | Tick Size |
|--------|----------|--------|-----------|
| NIFTY | 50 | Thursday | 0.05 |
| BANKNIFTY | 15 | Wednesday | 0.05 |
| FINNIFTY | 40 | Tuesday | 0.05 |
| MIDCPNIFTY | 75 | Monday | 0.05 |

Time features: minute, hour, weekday, day, month (IST timezone)

## Configuration

All configs use Python dataclasses with:
- Type hints and defaults
- Environment variable overrides (`export PARAM=value`)
- Validation in `__post_init__`
- Dict-like access for backward compatibility

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- 8GB+ VRAM for finetuning (batch_size=50)
- 16GB+ RAM for data processing

## License

MIT License - See LICENSE file for details.
