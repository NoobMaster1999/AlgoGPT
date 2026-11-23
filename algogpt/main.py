"""
AlgoGPT Main Trading Script
============================
End-to-end pipeline: data -> Kronos forecast -> signals -> backtest.

Usage examples:
    # Single symbol, Yahoo Finance (no CSV needed)
    python -m algogpt.main --symbols BANKNIFTY --start 2026-01-01 --end 2026-10-01 --data-source yahoo

    # Multiple symbols
    python -m algogpt.main --symbols NIFTY BANKNIFTY FINNIFTY --start 2025-01-01 --end 2026-01-01 --data-source yahoo

    # Comma-separated symbols also work
    python -m algogpt.main --symbols NIFTY,BANKNIFTY --start 2025-01-01 --end 2026-01-01 --data-source yahoo

    # Custom model size, more samples, longer lookback
    python -m algogpt.main --symbols NIFTY --start 2024-01-01 --end 2026-10-01 --data-source yahoo \\
        --model NeoQuasar/Kronos-base --lookback 300 --pred-len 10 --sample-count 5

    # From local CSV files
    python -m algogpt.main --symbols NIFTY --start 2024-01-01 --end 2026-01-01 \\
        --data-source csv --data-dir ./data --interval 5min
"""
import argparse
import logging
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

from algogpt.data import create_fetcher, resample_ohlcv
from algogpt.model import KronosPredictor
from algogpt.trading import KronosBacktester, BacktestConfig, run_backtest

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AlgoGPT")

# ============================================================================
# HELPERS
# ============================================================================

def parse_symbols(symbol_values: List[str]) -> List[str]:
    """
    Parse symbols supplied as either:
        --symbols NIFTY BANKNIFTY FINNIFTY
        --symbols NIFTY,BANKNIFTY,FINNIFTY
        --symbols NIFTY BANKNIFTY,FINNIFTY   (mixed)
    """
    symbols: List[str] = []
    for value in symbol_values:
        if not value:
            continue
        for symbol in value.split(","):
            symbol = symbol.strip().upper()
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    if not symbols:
        raise ValueError("At least one trading symbol must be provided.")
    return symbols


def validate_date(date_string: str, argument_name: str) -> str:
    """Validate a YYYY-MM-DD date string and return it unchanged."""
    try:
        parsed = pd.to_datetime(date_string, format="%Y-%m-%d")
        if pd.isna(parsed):
            raise ValueError
        return date_string
    except Exception as exc:
        raise ValueError(
            f"Invalid {argument_name}: '{date_string}'. Expected format YYYY-MM-DD."
        ) from exc


def validate_date_range(start: str, end: str) -> Tuple[str, str]:
    """Validate and compare start/end dates."""
    start = validate_date(start, "start date")
    end   = validate_date(end,   "end date")
    if pd.to_datetime(start) >= pd.to_datetime(end):
        raise ValueError(f"Start date ({start}) must be earlier than end date ({end}).")
    return start, end


def build_fetcher_kwargs(
    data_source: str,
    data_dir: str = "./data",
    upstox_api_key: Optional[str] = None,
    upstox_api_secret: Optional[str] = None,
    upstox_access_token: Optional[str] = None,
) -> Dict:
    """Build data-fetcher configuration based on the selected source."""
    if data_source == "csv":
        return {"data_dir": data_dir}
    if data_source == "upstox":
        if not all([upstox_api_key, upstox_api_secret, upstox_access_token]):
            raise ValueError(
                "Upstox requires all three credentials:\n"
                "  --upstox-api-key\n"
                "  --upstox-api-secret\n"
                "  --upstox-access-token"
            )
        return {
            "api_key":      upstox_api_key,
            "api_secret":   upstox_api_secret,
            "access_token": upstox_access_token,
        }
    # yahoo / nse — no extra kwargs needed
    return {}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(
    symbol: str,
    start: str,
    end: str,
    interval: str = "15m",
    data_source: str = "yahoo",
    fetcher_kwargs: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare OHLCV data for a single symbol."""
    fetcher_kwargs = fetcher_kwargs or {}

    logger.info(
        "Fetching %s from %s | %s -> %s | interval=%s",
        symbol, data_source, start, end, interval,
    )

    fetcher = create_fetcher(data_source, **fetcher_kwargs)
    df = fetcher.fetch_ohlcv(symbol, start, end, interval)

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for symbol={symbol}, "
            f"source={data_source}, start={start}, end={end}"
        )

    # Normalise index to DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as exc:
            raise ValueError(
                f"Cannot convert index to DatetimeIndex for {symbol}"
            ) from exc

    df = df.sort_index()

    # Resample to 5-min only if a finer 1-minute interval was requested
    if interval in ("1m", "1min"):
        logger.info("Resampling %s from %s to 5min", symbol, interval)
        df = resample_ohlcv(df, "5min")

        if df.empty:
            raise ValueError(f"Data became empty after resampling for {symbol}")

    # Ensure required columns exist
    required = ["open", "high", "low", "close", "volume", "amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {symbol}: {missing}")

    timestamps = pd.Series(df.index, index=df.index)
    logger.info(
        "Loaded %d bars for %s (%s -> %s)",
        len(df), symbol, timestamps.iloc[0], timestamps.iloc[-1],
    )
    return df, timestamps


# ============================================================================
# FORECAST
# ============================================================================

def generate_forecast(
    predictor: KronosPredictor,
    df: pd.DataFrame,
    timestamps: pd.Series,
    lookback: int,
    pred_len: int,
    temperature: float,
    top_p: float,
    sample_count: int,
    verbose: bool,
) -> pd.DataFrame:
    """Run a single Kronos forecast on the last `lookback` bars."""
    if len(df) < lookback:
        raise ValueError(
            f"Insufficient data for forecast: need lookback={lookback}, have {len(df)} bars."
        )

    x_df        = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = timestamps.iloc[-lookback:]

    # Infer bar frequency
    freq = pd.infer_freq(df.index)
    if freq is None:
        freq = "B"   # business day fallback for daily data
    logger.info("Inferred data frequency: %s", freq)

    try:
        offset       = pd.tseries.frequencies.to_offset(freq)
        future_start = timestamps.iloc[-1] + offset
        future_index = pd.date_range(start=future_start, periods=pred_len, freq=freq)
    except Exception:
        logger.warning("Cannot infer offset for freq=%s – falling back to business days.", freq)
        future_index = pd.date_range(
            start=timestamps.iloc[-1] + pd.Timedelta(days=1),
            periods=pred_len, freq="B",
        )

    y_timestamp = pd.Series(future_index)

    pred_df = predictor.predict(
        df          = x_df,
        x_timestamp = x_timestamp,
        y_timestamp = y_timestamp,
        pred_len    = pred_len,
        T           = temperature,
        top_p       = top_p,
        sample_count= sample_count,
        verbose     = verbose,
    )
    return pred_df


# ============================================================================
# SINGLE-SYMBOL PIPELINE
# ============================================================================

def run_trading_pipeline(
    symbol: str,
    start: str,
    end: str,
    interval: str = "15m",
    data_source: str = "yahoo",
    fetcher_kwargs: Optional[Dict] = None,
    model_id: str = "NeoQuasar/Kronos-small",
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
    max_context: int = 512,
    pred_len: int = 5,
    sample_count: int = 3,
    temperature: float = 1.0,
    top_p: float = 0.9,
    kronos_weight: float = 0.65,
    lookback: int = 200,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    plot: bool = True,
    verbose: bool = True,
    predictor: Optional[KronosPredictor] = None,
):
    """Run the complete AlgoGPT + Kronos pipeline for a single symbol."""

    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE  |  %s  |  %s -> %s  |  source=%s", symbol, start, end, data_source)
    logger.info("=" * 70)

    # ── 1. Data ────────────────────────────────────────────────────────────
    df, timestamps = load_data(
        symbol         = symbol,
        start          = start,
        end            = end,
        interval       = interval,
        data_source    = data_source,
        fetcher_kwargs = fetcher_kwargs,
    )

    # ── 2. Kronos predictor ────────────────────────────────────────────────
    if predictor is None:
        logger.info("Initializing KronosPredictor...")
        predictor = KronosPredictor.from_pretrained(
            model_id     = model_id,
            tokenizer_id = tokenizer_id,
            max_context  = max_context,
        )

    # ── 3. Demo forecast ───────────────────────────────────────────────────
    # Use the smaller of lookback or len(df)//2 for the demo so we always have data
    demo_lookback = min(lookback, max(10, len(df) // 2))
    logger.info("Running demo forecast on latest %d bars...", demo_lookback)
    pred_df = generate_forecast(
        predictor    = predictor,
        df           = df,
        timestamps   = timestamps,
        lookback     = demo_lookback,
        pred_len     = pred_len,
        temperature  = temperature,
        top_p        = top_p,
        sample_count = sample_count,
        verbose      = verbose,
    )

    logger.info(
        "\n%s\nForecast for %s (next %d bars):\n%s\n%s",
        "=" * 60, symbol, pred_len,
        pred_df[["open", "high", "low", "close"]].to_string(),
        "=" * 60,
    )

    # ── 4. Walk-forward backtest ───────────────────────────────────────────
    effective_lookback = min(lookback, max_context)
    if effective_lookback != lookback:
        logger.info("Capping lookback %d -> %d to match max_context", lookback, effective_lookback)
    lookback = effective_lookback

    # Auto-adjust lookback if data is shorter than required
    min_required = lookback + pred_len + 1
    if len(df) < min_required:
        adjusted = max(10, len(df) - pred_len - 1)
        logger.info(
            "Adjusting lookback %d -> %d (only %d bars available for %s)",
            lookback, adjusted, len(df), symbol,
        )
        lookback = adjusted

    logger.info("Running walk-forward backtest for %s...", symbol)
    bt_config = BacktestConfig(
        initial_capital = initial_capital,
        commission      = commission,
        slippage        = slippage,
        lookback        = lookback,
        pred_len        = pred_len,
        kronos_weight   = kronos_weight,
    )


    results = run_backtest(predictor, df, timestamps, bt_config, verbose=verbose)

    # ── 5. Print results ───────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"  BACKTEST RESULTS  |  {symbol}  |  {start} -> {end}")
    print("=" * 70)
    print(results.summary())

    # ── 6. Plot ────────────────────────────────────────────────────────────
    if plot:
        try:
            results.plot(title=f"AlgoGPT + Kronos  [{symbol} {interval}]  {start} -> {end}")
        except Exception as exc:
            logger.warning("Plot failed for %s: %s", symbol, exc)

    return results


# ============================================================================
# MULTI-SYMBOL RUNNER
# ============================================================================

def run_multiple_symbols(
    symbols: List[str],
    **kwargs,   # passed directly to run_trading_pipeline
) -> Dict:
    """Run the same pipeline configuration for each symbol with one shared model."""
    all_results: Dict = {}

    logger.info("")
    logger.info("#" * 70)
    logger.info("MULTI-SYMBOL RUN  |  %s", ", ".join(symbols))
    logger.info("#" * 70)

    logger.info("Initializing one shared KronosPredictor for %d symbol(s)...", len(symbols))
    predictor = KronosPredictor.from_pretrained(
        model_id=kwargs["model_id"],
        tokenizer_id=kwargs["tokenizer_id"],
        max_context=kwargs["max_context"],
    )

    for idx, symbol in enumerate(symbols, 1):
        logger.info("")
        logger.info("[%d/%d] Processing %s", idx, len(symbols), symbol)
        try:
            all_results[symbol] = run_trading_pipeline(
                symbol=symbol,
                predictor=predictor,
                **kwargs,
            )
        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", symbol, exc, exc_info=True)

    # Summary
    success = list(all_results.keys())
    failed  = [s for s in symbols if s not in all_results]

    logger.info("")
    logger.info("#" * 70)
    logger.info("DONE  |  OK: %s  |  FAILED: %s",
                ", ".join(success) or "none",
                ", ".join(failed)  or "none")
    logger.info("#" * 70)

    return all_results


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "algogpt",
        description = "AlgoGPT + Kronos — Full Backtest CLI for Indian Derivatives",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m algogpt.main --symbols BANKNIFTY --start 2026-01-01 --end 2026-10-01 --data-source yahoo\n"
            "  python -m algogpt.main --symbols NIFTY BANKNIFTY --start 2025-01-01 --end 2026-01-01 --data-source yahoo\n"
            "  python -m algogpt.main --symbols NIFTY,BANKNIFTY,FINNIFTY --start 2024-01-01 --end 2026-01-01 --data-source yahoo\n"
        ),
    )

    # ── Data ──────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Data")
    g.add_argument(
        "--symbols", nargs="+", required=True,
        metavar="SYM",
        help=(
            "Trading symbol(s).  Space- or comma-separated.  "
            "Examples: --symbols NIFTY  |  --symbols NIFTY BANKNIFTY  |  --symbols NIFTY,BANKNIFTY,FINNIFTY"
        ),
    )
    g.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    g.add_argument("--end",   required=True, help="End date   YYYY-MM-DD")
    g.add_argument(
        "--interval", default="15m",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="OHLCV bar interval.  Use '15m' for daily (recommended with Yahoo).",
    )
    g.add_argument(
        "--data-source", default="yahoo",
        choices=["csv", "upstox", "nse", "yahoo"],
        help="Market data source.  'yahoo' requires no credentials.",
    )
    g.add_argument("--data-dir", default="./data", help="Directory for CSV files")

    # ── Upstox credentials (optional) ────────────────────────────────────
    g2 = parser.add_argument_group("Upstox Credentials (only needed with --data-source upstox)")
    g2.add_argument("--upstox-api-key",      default=None)
    g2.add_argument("--upstox-api-secret",   default=None)
    g2.add_argument("--upstox-access-token", default=None)

    # ── Kronos model ──────────────────────────────────────────────────────
    g3 = parser.add_argument_group("Kronos Model")
    g3.add_argument("--model",       default="NeoQuasar/Kronos-small",
                    help="HuggingFace model ID.  Options: Kronos-mini / Kronos-small / Kronos-base")
    g3.add_argument("--tokenizer",   default="NeoQuasar/Kronos-Tokenizer-base",
                    help="HuggingFace tokenizer ID")
    g3.add_argument("--max-context", type=int,   default=512,  help="Max context window (bars)")
    g3.add_argument("--pred-len",    type=int,   default=5,    help="Forecast horizon (bars)")
    g3.add_argument("--sample-count",type=int,   default=3,    help="Monte Carlo samples to average")
    g3.add_argument("--temperature", type=float, default=1.0,  help="Sampling temperature")
    g3.add_argument("--top-p",       type=float, default=0.9,  help="Nucleus sampling probability")

    # ── Backtest ──────────────────────────────────────────────────────────
    g4 = parser.add_argument_group("Backtest / Trading")
    g4.add_argument("--lookback",      type=int,   default=200,       help="Historical bars fed to Kronos per step")
    g4.add_argument("--kronos-weight", type=float, default=0.65,      help="Kronos signal weight in ensemble [0-1]")
    g4.add_argument("--capital",       type=float, default=100_000.0, help="Initial capital per symbol (INR)")
    g4.add_argument("--commission",    type=float, default=0.001,     help="Commission per trade (fraction)")
    g4.add_argument("--slippage",      type=float, default=0.0005,    help="Slippage per trade (fraction)")

    # ── Output ────────────────────────────────────────────────────────────
    g5 = parser.add_argument_group("Output")
    g5.add_argument("--no-plot", action="store_true", help="Suppress matplotlib chart")
    g5.add_argument("--quiet",   action="store_true", help="Reduce log verbosity")

    return parser


# ============================================================================
# VALIDATION
# ============================================================================

def validate_args(args: argparse.Namespace) -> None:
    """Validate parsed CLI arguments and raise ValueError on bad inputs."""
    validate_date_range(args.start, args.end)

    checks = [
        (args.max_context  > 0,        "--max-context must be > 0"),
        (args.pred_len     > 0,        "--pred-len must be > 0"),
        (args.sample_count > 0,        "--sample-count must be > 0"),
        (args.lookback     > 0,        "--lookback must be > 0"),
        (0 <= args.kronos_weight <= 1, "--kronos-weight must be in [0, 1]"),
        (args.capital      > 0,        "--capital must be > 0"),
        (args.commission   >= 0,       "--commission cannot be negative"),
        (args.slippage     >= 0,       "--slippage cannot be negative"),
        (args.temperature  > 0,        "--temperature must be > 0"),
        (0 < args.top_p   <= 1,        "--top-p must be in (0, 1]"),
    ]
    for condition, message in checks:
        if not condition:
            raise ValueError(message)


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    try:
        if args.quiet:
            logging.getLogger().setLevel(logging.WARNING)

        validate_args(args)

        symbols = parse_symbols(args.symbols)

        fetcher_kwargs = build_fetcher_kwargs(
            data_source         = args.data_source,
            data_dir            = args.data_dir,
            upstox_api_key      = args.upstox_api_key,
            upstox_api_secret   = args.upstox_api_secret,
            upstox_access_token = args.upstox_access_token,
        )

        logger.info("")
        logger.info("=" * 70)
        logger.info("ALGOGPT + KRONOS  —  Full Backtest")
        logger.info("=" * 70)
        logger.info("Symbols    : %s", ", ".join(symbols))
        logger.info("Period     : %s  ->  %s", args.start, args.end)
        logger.info("Interval   : %s", args.interval)
        logger.info("Source     : %s", args.data_source)
        logger.info("Model      : %s", args.model)
        logger.info("Lookback   : %d bars", args.lookback)
        logger.info("Pred len   : %d bars", args.pred_len)
        logger.info("Samples    : %d", args.sample_count)
        logger.info("Capital    : %.0f", args.capital)
        logger.info("=" * 70)

        results = run_multiple_symbols(
            symbols        = symbols,
            start          = args.start,
            end            = args.end,
            interval       = args.interval,
            data_source    = args.data_source,
            fetcher_kwargs = fetcher_kwargs,
            model_id       = args.model,
            tokenizer_id   = args.tokenizer,
            max_context    = args.max_context,
            pred_len       = args.pred_len,
            sample_count   = args.sample_count,
            temperature    = args.temperature,
            top_p          = args.top_p,
            kronos_weight  = args.kronos_weight,
            lookback       = args.lookback,
            initial_capital= args.capital,
            commission     = args.commission,
            slippage       = args.slippage,
            plot           = not args.no_plot,
            verbose        = not args.quiet,
        )

        logger.info("Completed %d / %d symbol(s).", len(results), len(symbols))
        return 0 if len(results) == len(symbols) else 1

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130

    except Exception as exc:
        logger.error("Fatal: %s", exc, exc_info=True)
        return 1


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())