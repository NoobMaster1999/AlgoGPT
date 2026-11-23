"""Score immutable forecast snapshots at their terminal close in index points.

Run from the repository root with `python algogpt/score_forecasts.py`.
Only the publisher's ledger of closed, observed bars supplies outcomes.
"""
import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean


def _read(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _instant(value):
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.utcoffset() is not None else None


def _price(value):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _coverage(bar, actual, low_key, high_key):
    low, high = bar.get(low_key), bar.get(high_key)
    if _price(low) and _price(high) and low <= high:
        return low <= actual <= high
    return None


def score_forecast(payload, observations, now):
    """Use the exact last predicted bar, including symbol and interval scope."""
    result = {
        "symbol": payload["symbol"],
        "interval": payload["interval_key"],
        "issued_at": payload.get("issued_at"),
        "observed_at": payload.get("observed_at"),
        "target_at": payload.get("target_at"),
        "status": "excluded",
        "reason": None,
        "actual_close": None,
        "model_abs_error": None,
        "baseline_abs_error": None,
        "inside_50": None,
        "inside_90": None,
    }
    issued, observed, target = (_instant(result[k]) for k in
                                ("issued_at", "observed_at", "target_at"))
    predictions = payload.get("predictions") or []
    if issued is None:
        result["reason"] = "Forecast issue time is unverified."
    elif observed is None or target is None or not observed <= issued < target:
        result["reason"] = "Issue time must follow the market observation and precede the target close."
    elif (not predictions or not isinstance(predictions[-1], dict)
          or not predictions[-1].get("time") or not _price(predictions[-1].get("close"))
          or not _price(payload.get("spot_close"))):
        result["reason"] = "The archived forecast lacks a usable terminal or baseline price."
    elif now < target:
        result.update(status="pending", reason="The target bar has not closed.")
    else:
        terminal = predictions[-1]
        actual = observations.get(terminal["time"])
        if not _price(actual):
            result.update(status="pending", reason="Awaiting the observed close for the exact target bar.")
        else:
            result.update(
                status="completed",
                actual_close=actual,
                model_abs_error=round(abs(terminal["close"] - actual), 6),
                baseline_abs_error=round(abs(payload["spot_close"] - actual), 6),
                inside_50=_coverage(terminal, actual, "lower_25", "upper_75"),
                inside_90=_coverage(terminal, actual, "lower_05", "upper_95"),
            )
    return result


def _summarize(rows):
    completed = [row for row in rows if row["status"] == "completed"]
    stats = {
        "issued_count": len(rows),
        "completed_count": len(completed),
        "pending_count": sum(row["status"] == "pending" for row in rows),
        "excluded_count": sum(row["status"] == "excluded" for row in rows),
        "model_mae": None,
        "baseline_mae": None,
    }
    if completed:
        stats["model_mae"] = round(fmean(row["model_abs_error"] for row in completed), 6)
        stats["baseline_mae"] = round(fmean(row["baseline_abs_error"] for row in completed), 6)
    for level in (50, 90):
        values = [row[f"inside_{level}"] for row in completed if row[f"inside_{level}"] is not None]
        stats[f"coverage_{level}_count"] = len(values)
        stats[f"coverage_{level}"] = round(fmean(values), 6) if values else None
    return stats


def write_scorecard(data_dir, now=None):
    """Rebuild the dashboard's scorecard and publish it with an atomic replace."""
    now = now or datetime.now(timezone.utc)
    if now.utcoffset() is None:
        raise ValueError("Scorecard generation time must be timezone-aware")
    data_dir = Path(data_dir)
    batches = defaultdict(list)
    observations = {}
    outcomes = {}
    for path in sorted((data_dir / "archive").glob("*.json")):
        payload = _read(path)
        if payload["forecast_id"] != path.stem:
            raise ValueError(f"Forecast ID differs from its archive filename in {path}")
        key = (payload["symbol"], payload["interval_key"])
        if key not in observations:
            ledger = data_dir / "observations" / f"{key[0]}-{key[1]}.json"
            observations[key] = _read(ledger) if ledger.exists() else {}
        outcome = score_forecast(payload, observations[key], now)
        outcomes[path.stem] = outcome
        batches[key].append(outcome)
    groups = {}
    for (symbol, interval), rows in batches.items():
        groups.setdefault(symbol, {})[interval] = _summarize(rows)
    report = {
        "schema_version": 1,
        "generated_at": now.astimezone(timezone.utc).isoformat(),
        "groups": groups,
        "outcomes": outcomes,
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=data_dir,
                                     delete=False) as output:
        tmp = Path(output.name)
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        os.replace(tmp, data_dir / "scorecard.json")
    finally:
        tmp.unlink(missing_ok=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs" / "data")
    args = parser.parse_args()
    scorecard = write_scorecard(args.data_dir)
    print(f"Scored archive index with {len(scorecard['outcomes'])} forecast records.")
