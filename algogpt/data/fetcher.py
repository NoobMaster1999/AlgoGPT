"""
Data Pipeline for Indian Derivatives (NIFTY/BANKNIFTY)
=======================================================
Handles data fetching from Upstox/NSE, preprocessing, and dataset creation
for Kronos finetuning on Indian index options.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SymbolConfig:
    """Configuration for a trading symbol."""
    symbol: str              # e.g., "NIFTY", "BANKNIFTY"
    exchange: str = "NSE"    # Exchange code
    lot_size: int = 50       # Lot size for options
    tick_size: float = 0.05  # Minimum price tick
    expiry_day: str = "Thursday"  # Weekly expiry day


# Indian index symbols
INDIAN_INDEX_SYMBOLS = {
    "NIFTY": SymbolConfig("NIFTY", "NSE", 50, 0.05, "Thursday"),
    "BANKNIFTY": SymbolConfig("BANKNIFTY", "NSE", 15, 0.05, "Wednesday"),
    "FINNIFTY": SymbolConfig("FINNIFTY", "NSE", 40, 0.05, "Tuesday"),
    "MIDCPNIFTY": SymbolConfig("MIDCPNIFTY", "NSE", 75, 0.05, "Monday"),
}


class DataFetcher:
    """
    Base data fetcher - implement specific fetchers for different sources.
    Supports: Upstox WebSocket, NSE historical, Yahoo Finance, local CSV.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("./data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "5min",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data. Override in subclasses.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "NIFTY", "BANKNIFTY")
        start : str
            Start date (YYYY-MM-DD)
        end : str
            End date (YYYY-MM-DD)
        interval : str
            Bar interval (e.g., "1min", "5min", "15min", "1h", "1d")
        use_cache : bool
            Whether to use cached data

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: open, high, low, close, volume, amount, timestamp
        """
        raise NotImplementedError


class UpstoxDataFetcher(DataFetcher):
    """
    Fetch data from Upstox API (WebSocket for live, REST for historical).
    Requires Upstox API credentials.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        access_token: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        super().__init__(cache_dir)
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self._client = None

    def _get_client(self):
        """Lazy-load Upstox client."""
        if self._client is None:
            try:
                from upstox_client import ApiClient, Configuration, HistoryApi
                config = Configuration()
                config.access_token = self.access_token
                self._client = HistoryApi(ApiClient(config))
            except ImportError:
                raise ImportError("upstox-client not installed. Run: pip install upstox-client")
        return self._client

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "5min",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_file = self.cache_dir / f"{symbol}_{interval}_{start}_{end}.parquet"

        if use_cache and cache_file.exists():
            logger.info("Loading cached data from %s", cache_file)
            return pd.read_parquet(cache_file)

        # Map interval to Upstox format
        interval_map = {
            "1min": "I1", "5min": "I5", "15min": "I15",
            "30min": "I30", "1h": "I60", "1d": "D"
        }
        upstox_interval = interval_map.get(interval, "I5")

        # Get instrument key for symbol
        instrument_key = self._get_instrument_key(symbol)

        logger.info("Fetching %s data from Upstox: %s to %s", symbol, start, end)
        client = self._get_client()

        try:
            response = client.get_historical_candle_data(
                instrument_key=instrument_key,
                interval=upstox_interval,
                to_date=end,
                from_date=start,
            )

            candles = response.data.candles
            if not candles:
                logger.warning("No data returned for %s", symbol)
                return pd.DataFrame()

            # Upstox returns: [timestamp, open, high, low, close, volume, oi]
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
            df.set_index("timestamp", inplace=True)
            df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)

            # Keep standard columns
            df = df[["open", "high", "low", "close", "volume", "amount"]]

            if use_cache:
                df.to_parquet(cache_file)
                logger.info("Cached data to %s", cache_file)

            return df

        except Exception as e:
            logger.error("Upstox fetch failed: %s", e)
            raise

    def _get_instrument_key(self, symbol: str) -> str:
        """Map symbol to Upstox instrument key."""
        # These are example keys - replace with actual instrument keys
        instrument_map = {
            "NIFTY": "NSE_INDEX|Nifty 50",
            "BANKNIFTY": "NSE_INDEX|Nifty Bank",
            "FINNIFTY": "NSE_INDEX|Nifty Fin Service",
            "MIDCPNIFTY": "NSE_INDEX|Nifty Midcap Select",
        }
        key = instrument_map.get(symbol.upper())
        if not key:
            raise ValueError(f"Unknown symbol: {symbol}. Add instrument key mapping.")
        return key


class NSEDataFetcher(DataFetcher):
    """
    Fetch historical data from NSE (via nsepython or direct API).
    Good for daily data, limited for intraday.
    """

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_file = self.cache_dir / f"{symbol}_{interval}_{start}_{end}.parquet"

        if use_cache and cache_file.exists():
            return pd.read_parquet(cache_file)

        try:
            import nsepython as nse
        except ImportError:
            raise ImportError("nsepython not installed. Run: pip install nsepython")

        logger.info("Fetching %s daily data from NSE: %s to %s", symbol, start, end)

        # NSE symbols for indices
        nse_symbol_map = {
            "NIFTY": "NIFTY 50",
            "BANKNIFTY": "NIFTY BANK",
            "FINNIFTY": "NIFTY FIN SERVICE",
            "MIDCPNIFTY": "NIFTY MIDCAP SELECT",
        }

        nse_symbol = nse_symbol_map.get(symbol.upper(), symbol)

        try:
            df = nse.index_history(nse_symbol, start, end)
            df = df.rename(columns={
                "OPEN": "open", "HIGH": "high", "LOW": "low",
                "CLOSE": "close", "VOLUME": "volume"
            })
            df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
            df.index = pd.to_datetime(df.index)
            df = df[["open", "high", "low", "close", "volume", "amount"]]

            if use_cache:
                df.to_parquet(cache_file)

            return df

        except Exception as e:
            logger.error("NSE fetch failed: %s", e)
            raise


class YahooDataFetcher(DataFetcher):
    """Fetch data from Yahoo Finance (fallback for testing)."""

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_file = self.cache_dir / f"{symbol}_{interval}_{start}_{end}.parquet"

        if use_cache and cache_file.exists():
            return pd.read_parquet(cache_file)

        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        # Yahoo symbols for Indian indices
        yahoo_map = {
            "NIFTY": "^NSEI",
            "BANKNIFTY": "^NSEBANK",
            "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
            "MIDCPNIFTY": "NIFTY_MIDCAP_SELECT.NS",
        }

        yahoo_symbol = yahoo_map.get(symbol.upper(), symbol)

        logger.info("Fetching %s from Yahoo Finance: %s to %s", yahoo_symbol, start, end)

        df = yf.download(yahoo_symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False)

        if df.empty:
            raise ValueError(f"No data returned for {yahoo_symbol}")

        # Standardize columns - handle MultiIndex from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"adj close": "close"})
        if "amount" not in df.columns:
            df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)

        df = df[["open", "high", "low", "close", "volume", "amount"]]

        if use_cache:
            df.to_parquet(cache_file)

        return df


class CSVDataFetcher(DataFetcher):
    """Load data from local CSV files."""

    def __init__(self, data_dir: Union[str, Path], cache_dir: Optional[Path] = None):
        super().__init__(cache_dir)
        self.data_dir = Path(data_dir)

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "5min",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        # Try different file patterns
        patterns = [
            f"{symbol}_{interval}.csv",
            f"{symbol}.csv",
            f"{symbol}_data.csv",
        ]

        df = None
        for pattern in patterns:
            file_path = self.data_dir / pattern
            if file_path.exists():
                df = pd.read_csv(file_path)
                break

        if df is None:
            raise FileNotFoundError(f"No CSV file found for {symbol} in {self.data_dir}")

        # Parse timestamps
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        elif "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"])
        elif "datetime" in df.columns:
            df["timestamp"] = pd.to_datetime(df["datetime"])
        else:
            raise ValueError("CSV must have a timestamp/date/datetime column")

        df.set_index("timestamp", inplace=True)
        df = df.sort_index()

        # Filter date range
        df = df.loc[start:end]

        # Standardize columns
        df.columns = [c.lower() for c in df.columns]
        if "amount" not in df.columns and "volume" in df.columns:
            price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
            df["amount"] = df["volume"] * df[price_cols].mean(axis=1)

        required = ["open", "high", "low", "close", "volume", "amount"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        return df[required]


def create_fetcher(source: str, **kwargs) -> DataFetcher:
    """Factory function to create data fetcher."""
    fetchers = {
        "upstox": UpstoxDataFetcher,
        "nse": NSEDataFetcher,
        "yahoo": YahooDataFetcher,
        "csv": CSVDataFetcher,
    }

    if source not in fetchers:
        raise ValueError(f"Unknown data source: {source}. Options: {list(fetchers.keys())}")

    return fetchers[source](**kwargs)


def _normalize_freq_for_pandas(interval: str) -> str:
    """Normalize interval string (e.g. '5m', '15m', '1d') to pandas offset alias."""
    s = interval.strip()
    import re
    m = re.match(r"^(\d+)?([a-zA-Z]+)$", s)
    if m:
        n, unit = m.groups()
        n = n or "1"
        unit_lower = unit.lower()
        if unit_lower in ("m", "min"):
            return f"{n}min"
        elif unit_lower == "h":
            return f"{n}h"
        elif unit_lower == "d":
            return f"{n}D"
        elif unit_lower == "w":
            return f"{n}W"
        elif unit_lower in ("mo", "mon", "me"):
            return f"{n}ME"
    return s


def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample OHLCV data to a different interval."""
    if df.empty:
        return df

    ohlcv_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "amount": "sum",
    }
    agg_dict = {col: agg for col, agg in ohlcv_dict.items() if col in df.columns}

    pandas_freq = _normalize_freq_for_pandas(interval)
    resampled = df.resample(pandas_freq).apply(agg_dict).dropna()
    return resampled


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features to DataFrame."""
    df = df.copy()
    df["minute"] = df.index.minute
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["day"] = df.index.day
    df["month"] = df.index.month
    return df


def normalize_window(
    x: np.ndarray,
    lookback: int,
    clip: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize a window using lookback period statistics.
    Returns normalized array, mean, std.
    """
    past_x = x[:lookback]
    x_mean = np.mean(past_x, axis=0)
    x_std = np.std(past_x, axis=0)
    x_norm = (x - x_mean) / (x_std + 1e-5)
    x_norm = np.clip(x_norm, -clip, clip)
    return x_norm, x_mean, x_std


def denormalize_window(
    x_norm: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> np.ndarray:
    """Denormalize predictions back to original scale."""
    return x_norm * (x_std + 1e-5) + x_mean