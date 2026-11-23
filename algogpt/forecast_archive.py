"""Immutable forecast snapshots and first-observed market closes for GitHub Pages."""
import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30))
MINUTES = {"15m": 15, "1h": 60, "1d": 0}
ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == encoded:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False, newline="\n") as handle:
        temporary = Path(handle.name)
        try:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def publication_lock(data_dir):
    """Serialize read/merge/write operations, including local and scheduled retries."""
    key = hashlib.sha256(str(Path(data_dir).resolve()).encode()).hexdigest()
    with (Path(tempfile.gettempdir()) / f"algogpt-publish-{key}.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            if not lock.read(1):
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def stamp(label):
    value = datetime.fromisoformat(label)
    return value.replace(tzinfo=IST) if value.tzinfo is None else value.astimezone(IST)


def regular_close(label, interval):
    start = stamp(label)
    end = start.replace(hour=15, minute=30)
    return end if interval == "1d" else min(end, start + timedelta(minutes=MINUTES[interval]))


def forecast_key(symbol, interval, last_bar):
    if not re.fullmatch(r"[A-Z0-9]+", symbol) or interval not in MINUTES:
        raise ValueError("Invalid symbol or forecast interval")
    return f"{symbol}-{interval}-{stamp(last_bar):%Y%m%dT%H%M}"


def save_snapshot(data_dir, entry):
    interval = entry["interval_key"]
    last_bar = entry["historical"][-1]["time"]
    key = forecast_key(entry["symbol"], interval, last_bar)
    path = data_dir / "archive" / f"{key}.json"
    if path.exists():
        return read_json(path)
    first = dict(entry)
    first["forecast_id"] = key
    if "issued_at" not in first:
        first.update(
            issued_at=None, next_update_at=None, sample_count=None,
            observed_at=regular_close(last_bar, interval).isoformat(),
            target_at=regular_close(first["predictions"][-1]["time"], interval).isoformat(),
            source="Historical snapshot (original source unverified)",
        )
    write_json(path, first)
    return first


def publish(data_dir, incoming=None, observations=None):
    """Caller holds publication_lock. Archive before replacing the latest pointer."""
    data_dir = Path(data_dir)
    latest_path = data_dir / "forecast_data.json"
    latest = read_json(latest_path) if latest_path.exists() else {}
    for dataset in (latest, incoming or {}):
        for symbol, intervals in dataset.items():
            for interval, entry in intervals.items():
                if entry["symbol"] != symbol or entry["interval_key"] != interval:
                    raise ValueError("Mismatched symbol or interval in forecast")
                snapshot = save_snapshot(data_dir, entry)
                current = latest.setdefault(symbol, {}).get(interval)
                if not current or snapshot["historical"][-1]["time"] >= current["historical"][-1]["time"]:
                    latest[symbol][interval] = snapshot
    for (symbol, interval), bars in (observations or {}).items():
        path = data_dir / "observations" / f"{symbol}-{interval}.json"
        history = read_json(path) if path.exists() else {}
        for label, close in bars.items():
            if not math.isfinite(close) or close <= 0:
                raise ValueError(f"Invalid closing price for {symbol} on {label}")
            history.setdefault(label, close)
        write_json(path, dict(sorted(history.items())))
    write_json(latest_path, latest)
    return latest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Archive existing dashboard snapshots with unknown issue time.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "docs" / "data")
    data_dir = parser.parse_args().data_dir
    if not (data_dir / "forecast_data.json").exists():
        parser.error("forecast_data.json must already exist")
    with publication_lock(data_dir):
        publish(data_dir)
