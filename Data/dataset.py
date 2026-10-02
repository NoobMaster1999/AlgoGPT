"""
Dataset for Kronos Finetuning on Indian Derivatives
====================================================
PyTorch Dataset with sliding windows for NIFTY/BANKNIFTY data.
"""
import logging
import pickle
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.fetcher import DataFetcher, create_fetcher, add_time_features, normalize_window

logger = logging.getLogger(__name__)


class IndianDerivativesDataset(Dataset):
    """
    PyTorch Dataset for Indian index derivatives (NIFTY/BANKNIFTY).

    Pre-computes all valid sliding window indices for efficient sampling.
    Handles multiple symbols with proper normalization per window.
    """

    def __init__(
        self,
        symbols: List[str],
        data_dir: Union[str, Path],
        lookback_window: int = 90,
        predict_window: int = 10,
        max_context: int = 512,
        clip: float = 5.0,
        feature_list: Optional[List[str]] = None,
        time_feature_list: Optional[List[str]] = None,
        data_source: str = "csv",
        fetcher_kwargs: Optional[dict] = None,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        interval: str = "5min",
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        symbols : List[str]
            List of symbols to include (e.g., ["NIFTY", "BANKNIFTY"])
        data_dir : str or Path
            Directory containing cached/data files
        lookback_window : int
            Number of historical bars for input
        predict_window : int
            Number of future bars to predict
        max_context : int
            Maximum context length for model
        clip : float
            Clipping value for normalized data
        feature_list : List[str]
            Price/volume features to use
        time_feature_list : List[str]
            Time features to generate
        data_source : str
            Data source: "csv", "upstox", "nse", "yahoo"
        fetcher_kwargs : dict
            Additional arguments for data fetcher
        start_date : str
            Start date for data
        end_date : str
            End date for data
        interval : str
            Bar interval
        seed : int
            Random seed
        """
        self.symbols = symbols
        self.data_dir = Path(data_dir)
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.max_context = max_context
        self.clip = clip
        self.seed = seed

        self.feature_list = feature_list or ["open", "high", "low", "close", "volume", "amount"]
        self.time_feature_list = time_feature_list or ["minute", "hour", "weekday", "day", "month"]

        self.window_size = lookback_window + predict_window + 1
        self.py_rng = random.Random(seed)

        # Initialize fetcher
        fetcher_kwargs = fetcher_kwargs or {}
        if data_source == "csv":
            fetcher_kwargs.setdefault("data_dir", self.data_dir)
        self.fetcher = create_fetcher(data_source, **fetcher_kwargs)

        # Load or build dataset
        self.data = {}
        self.indices = []  # List of (symbol, start_idx)

        self._load_or_build_data(start_date, end_date, interval)

        self.n_samples = min(len(self.indices), self.n_samples_per_epoch)
        logger.info(f"Dataset built: {len(self.indices)} total samples, {self.n_samples} per epoch")

    @property
    def n_samples_per_epoch(self) -> int:
        """Samples per epoch - can be overridden for faster epochs."""
        return 2000 * 50  # batch_size * iterations

    def _load_or_build_data(self, start_date: str, end_date: str, interval: str):
        """Load cached processed data or build from raw."""
        cache_file = self.data_dir / f"processed_{'-'.join(self.symbols)}_{interval}_{start_date}_{end_date}.pkl"

        if cache_file.exists():
            logger.info(f"Loading cached dataset from {cache_file}")
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            self.data = cached["data"]
            self.indices = cached["indices"]
            return

        logger.info("Building dataset from raw data...")
        self._build_data(start_date, end_date, interval)

        logger.info(f"Caching dataset to {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump({"data": self.data, "indices": self.indices}, f)

    def _build_data(self, start_date: str, end_date: str, interval: str):
        """Fetch and preprocess data for all symbols."""
        for symbol in self.symbols:
            logger.info(f"Processing {symbol}...")
            try:
                df = self.fetcher.fetch_ohlcv(symbol, start_date, end_date, interval)
                if df.empty:
                    logger.warning(f"No data for {symbol}, skipping")
                    continue

                df = add_time_features(df)
                df = df[self.feature_list + self.time_feature_list]
                self.data[symbol] = df

                # Build sliding window indices
                series_len = len(df)
                num_samples = series_len - self.window_size + 1
                if num_samples > 0:
                    for i in range(num_samples):
                        self.indices.append((symbol, i))
                else:
                    logger.warning(f"{symbol}: series too short ({series_len} < {self.window_size})")

            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                raise

        if not self.indices:
            raise ValueError("No valid samples found for any symbol")

    def set_epoch_seed(self, epoch: int):
        """Set seed for this epoch (for distributed training reproducibility)."""
        self.py_rng.seed(self.seed + epoch)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random sample from all indices
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]

        df = self.data[symbol]
        end_idx = start_idx + self.window_size
        win_df = df.iloc[start_idx:end_idx]

        # Extract features and time features
        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # Normalize using lookback window only (no future leakage)
        x_norm, _, _ = normalize_window(x, self.lookback_window, self.clip)

        return torch.from_numpy(x_norm), torch.from_numpy(x_stamp)


class MultiSymbolDataset(Dataset):
    """
    Dataset that ensures balanced sampling across symbols.
    Useful when symbols have very different data volumes.
    """

    def __init__(
        self,
        symbols: List[str],
        data_dir: Union[str, Path],
        lookback_window: int = 90,
        predict_window: int = 10,
        max_context: int = 512,
        clip: float = 5.0,
        feature_list: Optional[List[str]] = None,
        time_feature_list: Optional[List[str]] = None,
        data_source: str = "csv",
        fetcher_kwargs: Optional[dict] = None,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        interval: str = "5min",
        seed: int = 42,
        samples_per_symbol: Optional[int] = None,
    ):
        self.symbols = symbols
        self.samples_per_symbol = samples_per_symbol
        self.seed = seed
        self.py_rng = random.Random(seed)

        # Build individual datasets per symbol
        self.symbol_datasets = {}
        self.symbol_indices = {}

        for symbol in symbols:
            ds = IndianDerivativesDataset(
                symbols=[symbol],
                data_dir=data_dir,
                lookback_window=lookback_window,
                predict_window=predict_window,
                max_context=max_context,
                clip=clip,
                feature_list=feature_list,
                time_feature_list=time_feature_list,
                data_source=data_source,
                fetcher_kwargs=fetcher_kwargs,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                seed=seed,
            )
            self.symbol_datasets[symbol] = ds
            self.symbol_indices[symbol] = ds.indices

        # Build balanced index list
        self.indices = []
        for symbol in symbols:
            indices = self.symbol_indices[symbol]
            if samples_per_symbol and len(indices) > samples_per_symbol:
                indices = self.py_rng.sample(indices, samples_per_symbol)
            for idx in indices:
                self.indices.append((symbol, idx))

        self.n_samples = min(len(self.indices), 2000 * 50)
        logger.info(f"MultiSymbolDataset: {len(self.indices)} total balanced samples")

    def set_epoch_seed(self, epoch: int):
        self.py_rng.seed(self.seed + epoch)
        for ds in self.symbol_datasets.values():
            ds.set_epoch_seed(epoch)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]
        return self.symbol_datasets[symbol][start_idx]


def create_train_val_datasets(
    symbols: List[str],
    data_dir: Union[str, Path],
    lookback_window: int = 90,
    predict_window: int = 10,
    max_context: int = 512,
    clip: float = 5.0,
    train_ratio: float = 0.8,
    data_source: str = "csv",
    fetcher_kwargs: Optional[dict] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    interval: str = "5min",
    seed: int = 42,
) -> Tuple[IndianDerivativesDataset, IndianDerivativesDataset]:
    """
    Create train/validation split datasets.

    Splits by time (not random) to avoid lookahead bias.
    """
    # Build full dataset first
    full_dataset = IndianDerivativesDataset(
        symbols=symbols,
        data_dir=data_dir,
        lookback_window=lookback_window,
        predict_window=predict_window,
        max_context=max_context,
        clip=clip,
        data_source=data_source,
        fetcher_kwargs=fetcher_kwargs,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        seed=seed,
    )

    # Split indices by time
    symbol_to_indices = {}
    for symbol, start_idx in full_dataset.indices:
        if symbol not in symbol_to_indices:
            symbol_to_indices[symbol] = []
        symbol_to_indices[symbol].append(start_idx)

    train_indices = []
    val_indices = []

    for symbol, indices in symbol_to_indices.items():
        indices = sorted(indices)
        split_idx = int(len(indices) * train_ratio)
        train_indices.extend([(symbol, i) for i in indices[:split_idx]])
        val_indices.extend([(symbol, i) for i in indices[split_idx:]])

    # Create subset datasets
    train_dataset = IndianDerivativesDataset(
        symbols=symbols,
        data_dir=data_dir,
        lookback_window=lookback_window,
        predict_window=predict_window,
        max_context=max_context,
        clip=clip,
        data_source=data_source,
        fetcher_kwargs=fetcher_kwargs,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        seed=seed,
    )
    train_dataset.indices = train_indices
    train_dataset.n_samples = min(len(train_indices), train_dataset.n_samples_per_epoch)

    val_dataset = IndianDerivativesDataset(
        symbols=symbols,
        data_dir=data_dir,
        lookback_window=lookback_window,
        predict_window=predict_window,
        max_context=max_context,
        clip=clip,
        data_source=data_source,
        fetcher_kwargs=fetcher_kwargs,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        seed=seed,
    )
    val_dataset.indices = val_indices
    val_dataset.n_samples = min(len(val_indices), 400 * 50)

    logger.info(f"Train: {len(train_indices)} samples, Val: {len(val_indices)} samples")
    return train_dataset, val_dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    dataset = IndianDerivativesDataset(
        symbols=["NIFTY", "BANKNIFTY"],
        data_dir="./data",
        data_source="csv",
        start_date="2023-01-01",
        end_date="2024-06-01",
        interval="5min",
    )

    print(f"Dataset size: {len(dataset)}")
    x, x_stamp = dataset[0]
    print(f"Sample shapes: x={x.shape}, x_stamp={x_stamp.shape}")