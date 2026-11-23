"""Check the static Pages bundle and cross-check its scorecard before deployment."""
import json
import math
import sys
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime
from shutil import copytree
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algogpt"))
from score_forecasts import write_scorecard


class PageAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        path = attrs.get("href") if tag == "link" else attrs.get("src")
        if path and not urlsplit(path).scheme and not path.startswith("//"):
            self.assets.append(urlsplit(path).path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify():
    site = ROOT / "docs"
    assets = PageAssets()
    assets.feed((site / "index.html").read_text(encoding="utf-8"))
    for asset in assets.assets:
        if asset.startswith("/") or not (site / asset).is_file():
            raise ValueError(f"Missing or non-relative page asset: {asset}")

    data = site / "data"
    feed = read(data / "forecast_data.json")
    report = read(data / "scorecard.json")
    for symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY"):
        for interval in ("15m", "1h", "1d"):
            latest = feed[symbol][interval]
            forecast_id = latest["forecast_id"]
            original = read(data / "archive" / f"{forecast_id}.json")
            for key in ("symbol", "interval_key", "spot_close", "predictions", "historical"):
                if original[key] != latest[key]:
                    raise ValueError(f"Latest forecast disagrees with its archive: {forecast_id}/{key}")
            if (latest["symbol"], latest["interval_key"]) != (symbol, interval):
                raise ValueError(f"Incorrect series in {forecast_id}")
            for series in ("historical", "predictions"):
                bars = latest[series]
                if not bars or any(not math.isfinite(bar["close"]) or bar["close"] <= 0 for bar in bars):
                    raise ValueError(f"Invalid closes in {forecast_id}/{series}")
            if forecast_id not in report["outcomes"]:
                raise ValueError(f"Missing scorecard entry: {forecast_id}")

    # Regenerate at the original evaluation instant so validation is reproducible.
    with TemporaryDirectory() as scratch:
        comparison = Path(scratch)
        for name in ("archive", "observations"):
            if (data / name).is_dir():
                copytree(data / name, comparison / name)
        regenerated = write_scorecard(comparison, datetime.fromisoformat(report["generated_at"]))
        if regenerated != report:
            raise ValueError("Published scorecard disagrees with archived forecasts and observations")
    print(f"Verified {len(feed)} indices and {len(report['outcomes'])} archived forecasts.")


if __name__ == "__main__":
    verify()
