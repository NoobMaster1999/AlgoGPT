"""
Kronos Predictor - High-level inference wrapper for the Kronos model.
"""
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch

from .tokenizer import KronosTokenizer
from .model import Kronos, auto_regressive_inference

logger = logging.getLogger(__name__)


def calc_time_stamps(x_timestamp: pd.Series) -> pd.DataFrame:
    """Calculate time features from timestamps."""
    time_df = pd.DataFrame()
    time_df["minute"] = x_timestamp.dt.minute
    time_df["hour"] = x_timestamp.dt.hour
    time_df["weekday"] = x_timestamp.dt.weekday
    time_df["day"] = x_timestamp.dt.day
    time_df["month"] = x_timestamp.dt.month
    return time_df


class KronosPredictor:
    """
    High-level predictor for Kronos foundation model.

    Handles:
    - Model loading from HuggingFace Hub or local paths
    - Data preprocessing (normalization, timestamp features)
    - Autoregressive inference with sampling
    - Single and batch prediction
    """

    def __init__(
        self,
        model: Kronos,
        tokenizer: KronosTokenizer,
        device: Optional[str] = None,
        max_context: int = 512,
        clip: float = 5.0,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.max_context = max_context
        self.clip = clip
        self.price_cols = ["open", "high", "low", "close"]
        self.vol_col = "volume"
        self.amt_col = "amount"
        self.time_cols = ["minute", "hour", "weekday", "day", "month"]

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)
        self.tokenizer.eval()
        self.model.eval()

    @property
    def is_kronos_loaded(self) -> bool:
        """Whether the Kronos model is loaded and ready."""
        return True

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "NeoQuasar/Kronos-small",
        tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: Optional[str] = None,
        max_context: int = 512,
        clip: float = 5.0,
    ) -> "KronosPredictor":
        """Load model and tokenizer from HuggingFace Hub."""
        logger.info("Loading Kronos tokenizer: %s", tokenizer_id)
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
        logger.info("Loading Kronos model: %s", model_id)
        model = Kronos.from_pretrained(model_id)
        return cls(model, tokenizer, device, max_context, clip)

    @classmethod
    def from_local(
        cls,
        model_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
        device: Optional[str] = None,
        max_context: int = 512,
        clip: float = 5.0,
    ) -> "KronosPredictor":
        """Load model and tokenizer from local paths."""
        logger.info("Loading Kronos tokenizer from: %s", tokenizer_path)
        tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
        logger.info("Loading Kronos model from: %s", model_path)
        model = Kronos.from_pretrained(str(model_path))
        return cls(model, tokenizer, device, max_context, clip)

    def generate(
        self,
        x: torch.Tensor,
        x_stamp: torch.Tensor,
        y_stamp: torch.Tensor,
        pred_len: int,
        T: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = False,
    ) -> np.ndarray:
        """Core autoregressive generation."""
        preds = auto_regressive_inference(
            self.tokenizer,
            self.model,
            x,
            x_stamp,
            y_stamp,
            self.max_context,
            pred_len,
            self.clip,
            T,
            top_k,
            top_p,
            sample_count,
            verbose,
        )
        return preds[:, -pred_len:, :]

    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int,
        T: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Generate a price forecast for the next `pred_len` K-lines.

        Parameters
        ----------
        df : pd.DataFrame
            Historical OHLCV DataFrame. Required: open, high, low, close.
            Optional: volume, amount.
        x_timestamp : pd.Series
            DatetimeSeries aligned with `df` rows.
        y_timestamp : pd.Series
            Future timestamps to predict (must have exactly `pred_len` elements).
        pred_len : int
            Number of future bars to forecast.
        T : float
            Sampling temperature.
        top_k : int
            Top-k filtering.
        top_p : float
            Nucleus sampling threshold.
        sample_count : int
            Monte Carlo samples to average.
        verbose : bool
            Show progress bar.

        Returns
        -------
        pd.DataFrame
            Predicted OHLCV DataFrame indexed by `y_timestamp`.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if not all(col in df.columns for col in self.price_cols):
            raise ValueError(f"Price columns {self.price_cols} not found in DataFrame.")

        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0
            df[self.amt_col] = 0.0
        if self.amt_col not in df.columns and self.vol_col in df.columns:
            df[self.amt_col] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

        if df[self.price_cols + [self.vol_col, self.amt_col]].isnull().values.any():
            raise ValueError("Input DataFrame contains NaN values in price or volume columns.")

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        x = df[self.price_cols + [self.vol_col, self.amt_col]].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        x = torch.from_numpy(x[np.newaxis, :]).float()
        x_stamp = torch.from_numpy(x_stamp[np.newaxis, :]).float()
        y_stamp = torch.from_numpy(y_stamp[np.newaxis, :]).float()

        preds = self.generate(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose)

        preds = preds.squeeze(0)
        preds = preds * (x_std + 1e-5) + x_mean

        pred_df = pd.DataFrame(preds, columns=self.price_cols + [self.vol_col, self.amt_col], index=y_timestamp)
        return pred_df

    def predict_batch(
        self,
        df_list: List[pd.DataFrame],
        x_timestamp_list: List[pd.Series],
        y_timestamp_list: List[pd.Series],
        pred_len: int,
        T: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = True,
    ) -> List[pd.DataFrame]:
        """Batch prediction for multiple assets in a single GPU pass."""
        if not isinstance(df_list, (list, tuple)) or not isinstance(x_timestamp_list, (list, tuple)) or not isinstance(y_timestamp_list, (list, tuple)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must be list or tuple types.")
        if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must have consistent lengths.")

        num_series = len(df_list)

        x_list = []
        x_stamp_list = []
        y_stamp_list = []
        means = []
        stds = []
        seq_lens = []
        y_lens = []

        for i in range(num_series):
            df = df_list[i]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Input at index {i} is not a pandas DataFrame.")
            if not all(col in df.columns for col in self.price_cols):
                raise ValueError(f"DataFrame at index {i} is missing price columns {self.price_cols}.")

            df = df.copy()
            if self.vol_col not in df.columns:
                df[self.vol_col] = 0.0
                df[self.amt_col] = 0.0
            if self.amt_col not in df.columns and self.vol_col in df.columns:
                df[self.amt_col] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

            if df[self.price_cols + [self.vol_col, self.amt_col]].isnull().values.any():
                raise ValueError(f"DataFrame at index {i} contains NaN values in price or volume columns.")

            x_timestamp = x_timestamp_list[i]
            y_timestamp = y_timestamp_list[i]

            x_time_df = calc_time_stamps(x_timestamp)
            y_time_df = calc_time_stamps(y_timestamp)

            x = df[self.price_cols + [self.vol_col, self.amt_col]].values.astype(np.float32)
            x_stamp = x_time_df.values.astype(np.float32)
            y_stamp = y_time_df.values.astype(np.float32)

            if x.shape[0] != x_stamp.shape[0]:
                raise ValueError(f"Inconsistent lengths at index {i}: x has {x.shape[0]} vs x_stamp has {x_stamp.shape[0]}.")
            if y_stamp.shape[0] != pred_len:
                raise ValueError(f"y_timestamp length at index {i} should equal pred_len={pred_len}, got {y_stamp.shape[0]}.")

            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = (x - x_mean) / (x_std + 1e-5)
            x_norm = np.clip(x_norm, -self.clip, self.clip)

            x_list.append(x_norm)
            x_stamp_list.append(x_stamp)
            y_stamp_list.append(y_stamp)
            means.append(x_mean)
            stds.append(x_std)
            seq_lens.append(x_norm.shape[0])
            y_lens.append(y_stamp.shape[0])

        if len(set(seq_lens)) != 1:
            raise ValueError(f"Parallel prediction requires consistent historical lengths, got: {seq_lens}")
        if len(set(y_lens)) != 1:
            raise ValueError(f"Parallel prediction requires consistent prediction lengths, got: {y_lens}")

        x_batch = torch.from_numpy(np.stack(x_list, axis=0)).float()
        x_stamp_batch = torch.from_numpy(np.stack(x_stamp_list, axis=0)).float()
        y_stamp_batch = torch.from_numpy(np.stack(y_stamp_list, axis=0)).float()

        preds = self.generate(x_batch, x_stamp_batch, y_stamp_batch, pred_len, T, top_k, top_p, sample_count, verbose)

        pred_dfs = []
        for i in range(num_series):
            preds_i = preds[i] * (stds[i] + 1e-5) + means[i]
            pred_df = pd.DataFrame(preds_i, columns=self.price_cols + [self.vol_col, self.amt_col], index=y_timestamp_list[i])
            pred_dfs.append(pred_df)

        return pred_dfs