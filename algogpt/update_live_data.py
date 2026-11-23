"""Build NSE index forecasts from closed market bars and publish a versioned dashboard feed."""

import sys
import json
import time as _time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
import yfinance as yf
import pandas_market_calendars as market_calendars

# Set up project root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algogpt.forecast_archive import IST, MINUTES, forecast_key, publish, publication_lock, write_json
from algogpt.score_forecasts import write_scorecard
from algogpt.model import KronosPredictor
from algogpt.model.predictor import calc_time_stamps
from algogpt.model.model import sample_from_logits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AlgoGPT.LiveUpdate")

# ---------------------------------------------------------------------------
# Deterministic seeding for reproducible runs
# ---------------------------------------------------------------------------
_RNG_SEED = 42
np.random.seed(_RNG_SEED)
torch.manual_seed(_RNG_SEED)

# ---------------------------------------------------------------------------
# Production constants
# ---------------------------------------------------------------------------
_YAHOO_MAX_RETRIES = 3
_YAHOO_RETRY_DELAY = 2.0          # seconds between retries
_MC_SAMPLE_COUNT = 50             # Monte Carlo paths (was 25)
_TEMPERATURE = 0.72               # lower → more conservative predictions
_TOP_P = 0.85                     # tighter nucleus sampling
_MIN_BARS_REQUIRED = 10           # minimum historical bars before we forecast

# Ticker mappings
SYMBOLS_CONFIG = {
    "BANKNIFTY": {
        "ticker": "^NSEBANK",
        "name": "NIFTY Bank Index",
        "exchange": "NSE (India)",
        "market_hours": "09:15 – 15:30 IST (Mon–Fri)",
        "default_volume": 180000,
    },
    "NIFTY": {
        "ticker": "^NSEI",
        "name": "NIFTY 50 Index",
        "exchange": "NSE (India)",
        "market_hours": "09:15 – 15:30 IST (Mon–Fri)",
        "default_volume": 250000,
    },
    "FINNIFTY": {
        "ticker": "NIFTY_FIN_SERVICE.NS",
        "name": "NIFTY Financial Services Index",
        "exchange": "NSE (India)",
        "market_hours": "09:15 – 15:30 IST (Mon–Fri)",
        "default_volume": 120000,
    }
}


def _safe_tz_convert(df: pd.DataFrame, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    """Normalize a DataFrame index to the given timezone."""
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)
    return df


def _fetch_with_retry(ticker: yf.Ticker, **kwargs) -> pd.DataFrame:
    """Fetch data from Yahoo Finance with retry logic."""
    last_err = None
    for attempt in range(1, _YAHOO_MAX_RETRIES + 1):
        try:
            df = ticker.history(**kwargs)
            if not df.empty:
                return df
            logger.warning("Empty response (attempt %d/%d) for %s",
                           attempt, _YAHOO_MAX_RETRIES, kwargs)
        except Exception as e:
            last_err = e
            logger.warning("Yahoo Finance error (attempt %d/%d): %s",
                           attempt, _YAHOO_MAX_RETRIES, e)
        if attempt < _YAHOO_MAX_RETRIES:
            _time.sleep(_YAHOO_RETRY_DELAY)
    raise ValueError(
        f"Failed to fetch data after {_YAHOO_MAX_RETRIES} attempts "
        f"(kwargs={kwargs}): {last_err}"
    )


@lru_cache(maxsize=64)
def trading_sessions(start, end):
    """Use scheduled NSE opens and closes, including exchange holidays."""
    schedule = market_calendars.get_calendar("NSE").schedule(start, end)
    return {row.market_open.tz_convert("Asia/Kolkata").date():
            (row.market_open.tz_convert("Asia/Kolkata"), row.market_close.tz_convert("Asia/Kolkata"))
            for row in schedule.itertuples()}


def closed_market_bars(df, interval, now, default_volume):
    df = _safe_tz_convert(df.copy()).sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    sessions = trading_sessions(df.index.min().date(), df.index.max().date())
    rows = []
    for ts in df.index:
        session = sessions.get(ts.date())
        if not session:
            rows.append(False)
            continue
        opening, closing = session
        if interval == "1d":
            rows.append(closing <= now)
        else:
            length = pd.Timedelta(minutes=MINUTES[interval])
            aligned = (ts - opening) % length == pd.Timedelta(0)
            rows.append(opening <= ts < closing and aligned and min(ts + length, closing) <= now)
    df = df.loc[rows].copy()
    prices = df[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
    valid &= (prices["High"] >= prices[["Open", "Close"]].max(axis=1))
    valid &= (prices["Low"] <= prices[["Open", "Close"]].min(axis=1))
    df = df.loc[valid].copy()
    if len(df) < _MIN_BARS_REQUIRED:
        raise ValueError(f"Insufficient valid closed {interval} bars: {len(df)}")
    df[prices.columns] = prices.loc[valid]
    volume = pd.to_numeric(df["Volume"], errors="coerce") if "Volume" in df else pd.Series(np.nan, index=df.index)
    volume = volume.where(np.isfinite(volume) & (volume > 0))
    df["reported_volume"] = volume
    df["Volume"] = volume.fillna(volume.median() if volume.notna().any() else default_volume)
    return df


def fetch_symbol_data(symbol_key, ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    frames = {}
    for interval, period in (("15m", "5d"), ("1h", "1mo"), ("1d", "3mo")):
        try:
            df = _fetch_with_retry(ticker, period=period, interval=interval)
            frames[interval] = closed_market_bars(df, interval, datetime.now(IST),
                                                  SYMBOLS_CONFIG[symbol_key]["default_volume"])
        except Exception:
            logger.exception("Unable to refresh %s/%s", symbol_key, interval)
    return frames


def bar_end(stamp, interval):
    session = trading_sessions(stamp.date(), stamp.date())[stamp.date()]
    return session[1] if interval == "1d" else min(stamp + pd.Timedelta(minutes=MINUTES[interval]), session[1])


def generate_future_timestamps(last_ts, interval_key, count):
    sessions = trading_sessions(last_ts.date(), last_ts.date() + timedelta(days=max(35, 3 * count)))
    future = []
    for opening, closing in sessions.values():
        bars = [opening.normalize()] if interval_key == "1d" else pd.date_range(
            opening, closing, inclusive="left", freq=pd.Timedelta(minutes=MINUTES[interval_key]))
        for bar in bars:
            if bar > last_ts:
                future.append(bar)
                if len(future) == count:
                    return future
    raise ValueError("Insufficient future sessions in the exchange calendar")


def next_update(first_close, issued_at):
    start = max(first_close, issued_at)
    sessions = trading_sessions(start.date(), start.date() + timedelta(days=20))
    for opening, _closing in sessions.values():
        # GitHub Actions uses UTC 04:17..09:17 and 11:17 (IST 09:47..14:47 and 16:47).
        for hour in (9, 10, 11, 12, 13, 14, 16):
            attempt = opening.replace(hour=hour, minute=47)
            if attempt >= first_close and attempt > issued_at:
                return attempt
    raise ValueError("No scheduled update found")


def run_kronos_mc_trajectories(
    predictor: KronosPredictor,
    hist_df: pd.DataFrame,
    future_timestamps: list,
    sample_count: int = _MC_SAMPLE_COUNT,
    T: float = _TEMPERATURE,
    top_p: float = _TOP_P,
) -> np.ndarray:
    """
    Run Kronos autoregressive generation across `sample_count` Monte Carlo paths.
    Returns array of shape: [sample_count, pred_len, 6] (OHLCVA).

    Production improvements over v2.0:
    - Robust median/IQR normalization (resistant to outlier candles)
    - Tighter temperature + top_p for realistic Indian index forecasts
    """
    pred_len = len(future_timestamps)
    tokenizer = predictor.tokenizer
    model = predictor.model
    max_context = predictor.max_context
    clip = predictor.clip

    price_cols = ["open", "high", "low", "close"]
    vol_col = "volume"
    amt_col = "amount"

    df = hist_df.copy()
    if vol_col not in df.columns:
        df[vol_col] = 1000.0
    if amt_col not in df.columns:
        df[amt_col] = df[vol_col] * df[price_cols].mean(axis=1)

    x_timestamp = pd.Series(df.index)
    y_timestamp = pd.Series(future_timestamps)

    x_time_df = calc_time_stamps(x_timestamp)
    y_time_df = calc_time_stamps(y_timestamp)

    x = df[price_cols + [vol_col, amt_col]].values.astype(np.float32)
    x_stamp = x_time_df.values.astype(np.float32)
    y_stamp = y_time_df.values.astype(np.float32)

    # ── Robust normalization: median + IQR-based scale ──────────────────
    # Mean/std is sensitive to outlier candles common in Indian indices.
    # Median/IQR gives much tighter, more realistic denormalized output.
    x_median = np.median(x, axis=0)
    q75 = np.percentile(x, 75, axis=0)
    q25 = np.percentile(x, 25, axis=0)
    x_iqr = q75 - q25
    # Fall back to std if IQR is degenerate (e.g. constant volume)
    x_scale = np.where(x_iqr > 1e-5, x_iqr, np.std(x, axis=0) + 1e-5)
    x_norm = (x - x_median) / x_scale
    x_norm = np.clip(x_norm, -clip, clip)

    x_tensor = torch.from_numpy(x_norm[np.newaxis, :]).float()
    x_stamp_tensor = torch.from_numpy(x_stamp[np.newaxis, :]).float()
    y_stamp_tensor = torch.from_numpy(y_stamp[np.newaxis, :]).float()

    with torch.no_grad():
        x_tensor = torch.clip(x_tensor, -clip, clip)
        device = predictor.device

        x_tile = x_tensor.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_tensor.size(1), x_tensor.size(2)).to(device)
        x_stamp_tile = x_stamp_tensor.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp_tensor.size(1), x_stamp_tensor.size(2)).to(device)
        y_stamp_tile = y_stamp_tensor.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp_tensor.size(1), y_stamp_tensor.size(2)).to(device)

        x_token = tokenizer.encode(x_tile, half=True)
        initial_seq_len = x_tile.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp_tile, y_stamp_tile], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

        for i in range(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)
            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=0, top_p=top_p, sample_logits=True)

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=0, top_p=top_p, sample_logits=True)

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous()
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        raw_trajs = z.cpu().numpy()[0, :, -pred_len:, :]  # [sample_count, pred_len, 6]

        # ── Denormalize using the same robust stats ─────────────────────
        trajs = raw_trajs * x_scale + x_median
        return trajs


def _anchor_to_spot(
    raw_paths: np.ndarray,
    spot_close: float,
    pred_len: int,
    col_idx: int = 3,
    decay_half_life: int = 3,
) -> np.ndarray:
    """
    Exponential-decay anchoring: smoothly transition from the known spot close
    to the model's raw prediction, so bar-0 starts near spot and later bars
    converge fully to the model output.

    This avoids the "jump gap" artifact that makes forecasts look unrealistic.

    Parameters
    ----------
    raw_paths : ndarray [sample_count, pred_len, 6]
    spot_close : float – last known real close price
    pred_len : int
    col_idx : int – column index to anchor (default 3 = close)
    decay_half_life : int – bars until anchor weight halves
    """
    paths = raw_paths.copy()
    # Weight decays exponentially: w(t) = 0.95 * 0.5^(t / half_life)
    alpha0 = 0.95
    weights = alpha0 * np.power(0.5, np.arange(pred_len) / max(decay_half_life, 1))

    for col in range(4):  # O, H, L, C
        bar0_mean = np.mean(paths[:, 0, col])
        offset = spot_close - bar0_mean
        for t in range(pred_len):
            paths[:, t, col] += offset * weights[t]
    return paths


def _enforce_ohlc_constraints(paths: np.ndarray) -> np.ndarray:
    """
    Enforce strict OHLC ordering on every bar of every MC path:
      low  <= min(open, close)
      high >= max(open, close)
    Also ensures high >= low always.
    """
    out = paths.copy()
    o, h, l, c = out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]
    # high must be >= max(open, close)
    out[:, :, 1] = np.maximum(h, np.maximum(o, c))
    # low must be <= min(open, close)
    out[:, :, 2] = np.minimum(l, np.minimum(o, c))
    # Final sanity: high >= low
    mask = out[:, :, 1] < out[:, :, 2]
    out[:, :, 1][mask], out[:, :, 2][mask] = out[:, :, 2][mask], out[:, :, 1][mask]
    return out


def _make_forecast(get_predictor, symbol, cfg, intv_key, intv_label, hist_len, pred_len, raw_df, key):
    df_hist = raw_df.tail(hist_len).rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
    })

    spot_close = float(df_hist["close"].iloc[-1])
    last_timestamp = df_hist.index[-1]
    observed_at = bar_end(last_timestamp, intv_key)
    update_str = observed_at.strftime("%Y-%m-%d %H:%M IST")

    # Format historical list (show last 35 bars max in JSON for display)
    display_hist = df_hist.tail(35)
    historical_list = []
    for ts, row in display_hist.iterrows():
        if intv_key == "1d":
            time_str = ts.strftime("%Y-%m-%d")
        else:
            time_str = ts.strftime("%Y-%m-%d %H:%M")
        reported_volume = row.get("reported_volume", row["volume"])
        historical_list.append({
            "time": time_str,
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
            "volume": int(reported_volume) if pd.notna(reported_volume) else None
        })

    # Generate future timestamps
    future_timestamps = generate_future_timestamps(last_timestamp, intv_key, pred_len)
    first_close = bar_end(future_timestamps[0], intv_key)
    if first_close <= datetime.now(IST):
        logger.warning("Skipping stale input for %s/%s", symbol, intv_key)
        return None
    target_at = bar_end(future_timestamps[-1], intv_key)
    horizon_desc = f"Next {pred_len} bars through {target_at:%d %b %Y, %H:%M IST}"

    # Run Kronos Monte Carlo trajectories
    trajs = run_kronos_mc_trajectories(
        predictor=get_predictor(),
        hist_df=df_hist,
        future_timestamps=future_timestamps,
        sample_count=_MC_SAMPLE_COUNT,
        T=_TEMPERATURE,
        top_p=_TOP_P,
    )  # shape: [N, pred_len, 6] (O, H, L, C, V, A)

    # ── Post-processing: anchor + constrain ─────────────────────
    # 1. Exponential-decay anchoring to spot close
    decay_hl = 3 if intv_key == "15m" else (4 if intv_key == "1h" else 5)
    trajs = _anchor_to_spot(trajs, spot_close, pred_len,
                            decay_half_life=decay_hl)
    # 2. Enforce OHLC ordering
    trajs = _enforce_ohlc_constraints(trajs)
    if not np.isfinite(trajs).all() or (trajs[:, :, :4] <= 0).any():
        raise ValueError("Invalid simulated prices")

    # Extract closes across samples
    sample_closes = trajs[:, :, 3]  # [N, pred_len]
    sample_opens = trajs[:, :, 0]
    sample_highs = trajs[:, :, 1]
    sample_lows = trajs[:, :, 2]

    # ── Percentiles & Median (more robust than mean) ────────────
    median_close = np.median(sample_closes, axis=0)
    median_open = np.median(sample_opens, axis=0)
    median_high = np.maximum(
        np.median(sample_highs, axis=0),
        np.maximum(median_open, median_close)
    )
    median_low = np.minimum(
        np.median(sample_lows, axis=0),
        np.minimum(median_open, median_close)
    )

    lower_05, upper_95 = np.percentile(sample_closes, [5, 95], axis=0)
    u90 = np.percentile(sample_closes, 90, axis=0)
    u75 = np.percentile(sample_closes, 75, axis=0)
    l25 = np.percentile(sample_closes, 25, axis=0)
    l10 = np.percentile(sample_closes, 10, axis=0)

    # ── Build prediction list with strict OHLC ──────────────────
    predictions_list = []
    for i, f_ts in enumerate(future_timestamps):
        if intv_key == "1d":
            f_time_str = f_ts.strftime("%Y-%m-%d")
        else:
            f_time_str = f_ts.strftime("%Y-%m-%d %H:%M")

        # 5 evenly-spaced representative sample paths
        n_samples = trajs.shape[0]
        sample_indices = np.linspace(0, n_samples - 1, 5, dtype=int)
        step_samples = [round(float(sample_closes[s, i]), 2)
                        for s in sample_indices]

        # Strict OHLC ordering for the central estimate
        o_val = round(float(
            spot_close if i == 0 else median_close[i - 1]
        ), 2)
        c_val = round(float(median_close[i]), 2)
        h_val = round(float(max(o_val, c_val, median_high[i])), 2)
        l_val = round(float(min(o_val, c_val, median_low[i])), 2)
        # Store quantiles of simulated closes independently of the illustrative OHLC candles.
        u90_val = round(float(u90[i]), 2)
        u75_val = round(float(max(c_val, u75[i])), 2)
        l25_val = round(float(min(c_val, l25[i])), 2)
        l10_val = round(float(l10[i]), 2)

        predictions_list.append({
            "time": f_time_str,
            "open": o_val,
            "high": h_val,
            "low": l_val,
            "close": c_val,
            "upper_95": round(float(upper_95[i]), 2),
            "lower_05": round(float(lower_05[i]), 2),
            "upper_90": u90_val,
            "lower_10": l10_val,
            "upper_75": u75_val,
            "lower_25": l25_val,
            "samples": step_samples
        })

    # ── Quantitative Metrics (honest, no artificial clamps) ─────
    terminal_prices = sample_closes[:, -1]
    # Upside probability: proportion of MC paths that end above spot
    upside_prob = round(
        float(np.mean(terminal_prices > spot_close) * 100), 1
    )

    # Volatility Amplification
    hist_returns = df_hist["close"].pct_change().dropna()
    hist_vol = float(np.std(hist_returns))
    forecast_returns = np.diff(sample_closes, axis=1) / (
        sample_closes[:, :-1] + 1e-8
    )
    forecast_vols = np.std(forecast_returns, axis=1)
    vol_amp_prob = round(
        float(np.mean(forecast_vols > hist_vol) * 100), 1
    )

    # Expected Return %
    exp_return = round(
        float(((np.mean(terminal_prices) - spot_close) / spot_close) * 100), 2
    )

    # 95% VaR (5th percentile return) — honest, no forced floor
    var_95 = round(
        float(np.percentile(
            (terminal_prices - spot_close) / spot_close, 5
        ) * 100),
        2,
    )

    metrics = {
        "upside_prob": upside_prob,
        "vol_amp_prob": vol_amp_prob,
        "expected_return": exp_return,
        "var_95": var_95
    }

    issued_at = datetime.now(IST)
    if issued_at >= first_close:
        logger.warning("Forecast missed its first close for %s/%s", symbol, intv_key)
        return None
    return {
        "forecast_id": key,
        "issued_at": issued_at.isoformat(timespec="seconds"),
        "observed_at": observed_at.isoformat(),
        "target_at": target_at.isoformat(),
        "next_update_at": next_update(first_close, issued_at).isoformat(),
        "sample_count": _MC_SAMPLE_COUNT,
        "source": f"Yahoo Finance ({cfg['ticker']})",
        "volume_policy": "Missing index volume is imputed for model input only.",
        "model": "NeoQuasar/Kronos-small",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "symbol": symbol,
        "interval": intv_label,
        "interval_key": intv_key,
        "exchange": cfg["exchange"],
        "market_hours": cfg["market_hours"],
        "last_updated": update_str,
        "spot_close": round(spot_close, 2),
        "horizon_desc": horizon_desc,
        "metrics": metrics,
        "historical": historical_list,
        "predictions": predictions_list
    }



def build_payload(observations=None, data_dir=None):
    """Return independently refreshed series and collect verified closes for scoring."""
    observations = {} if observations is None else observations
    data_dir = Path(data_dir or REPO_ROOT / "docs" / "data")
    forecast_data = {}
    failures = 0
    specs = (("15m", "15-Minute", 80, 12), ("1h", "1-Hour", 60, 10), ("1d", "Daily", 50, 10))

    @lru_cache(maxsize=1)
    def get_predictor():
        return KronosPredictor.from_pretrained(device="cpu")

    for symbol, cfg in SYMBOLS_CONFIG.items():
        raw_dfs = fetch_symbol_data(symbol, cfg["ticker"])
        forecasts = {}
        for interval, label, hist_len, pred_len in specs:
            frame = raw_dfs.get(interval)
            if frame is None:
                continue
            fmt = "%Y-%m-%d" if interval == "1d" else "%Y-%m-%d %H:%M"
            observations[symbol, interval] = {
                ts.strftime(fmt): round(float(close), 2) for ts, close in frame["Close"].items()
            }
            key = forecast_key(symbol, interval, frame.index[-1].strftime(fmt))
            archive = data_dir / "archive" / f"{key}.json"
            if archive.exists():
                forecasts[interval] = json.loads(archive.read_text(encoding="utf-8"))
                continue
            try:
                entry = _make_forecast(get_predictor, symbol, cfg, interval, label, hist_len, pred_len, frame, key)
            except Exception:
                failures += 1
                logger.exception("Could not issue forecast for %s/%s", symbol, interval)
                continue
            if entry:
                forecasts[interval] = entry
        if forecasts:
            forecast_data[symbol] = forecasts
    if failures and not forecast_data:
        raise RuntimeError("All forecast generation attempts failed")
    return forecast_data


def main():
    data_dir = REPO_ROOT / "docs" / "data"
    observations = {}
    payload = build_payload(observations, data_dir)
    if not observations:
        raise RuntimeError("No verified observations available from any market feed")
    with publication_lock(data_dir):
        latest = publish(data_dir, payload, observations)
        if not any(latest.values()):
            raise RuntimeError("No publishable forecasts available")
        report = write_scorecard(data_dir)
        demo_data = REPO_ROOT / "demo" / "data"
        if demo_data.exists():
            write_json(demo_data / "forecast_data.json", latest)
    logger.info("Updated %d forecast records", len(report["outcomes"]))


if __name__ == "__main__":
    main()
