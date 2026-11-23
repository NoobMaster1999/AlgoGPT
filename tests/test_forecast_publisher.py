import copy
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from algogpt import update_live_data as live
from algogpt.forecast_archive import IST, publication_lock, publish


def price_bars(day="2026-11-23"):
    times = pd.date_range(f"{day} 09:15", periods=25, freq="15min", tz="Asia/Kolkata")
    return pd.DataFrame({"Open": 100., "High": 105., "Low": 98., "Close": 101., "Volume": 0}, index=times)


class MarketBarTests(unittest.TestCase):
    def test_excludes_incomplete_malformed_and_duplicate_bars(self):
        bars = price_bars()
        bars.iloc[2, bars.columns.get_loc("Low")] = 200
        now = pd.Timestamp("2026-11-23 12:30", tz="Asia/Kolkata")
        clean = live.closed_market_bars(pd.concat([bars, bars]), "15m", now, 30)
        self.assertEqual(12, len(clean))
        self.assertEqual("12:15", clean.index[-1].strftime("%H:%M"))
        self.assertTrue((clean.Volume == 30).all())
        self.assertTrue(clean.index.is_unique)
        self.assertNotIn(bars.index[2], clean.index)

    def test_calendar_uses_trading_hours_holiday_and_actual_session_close(self):
        last_bar = pd.Timestamp("2026-11-23 14:15", tz="Asia/Kolkata")
        future = live.generate_future_timestamps(last_bar, "1h", 2)
        self.assertEqual(["23 15:15", "25 09:15"], [x.strftime("%d %H:%M") for x in future])
        self.assertEqual("15:30", live.bar_end(future[0], "1h").strftime("%H:%M"))
        self.assertEqual("25 10:47", live.next_update(live.bar_end(future[1], "1h"), last_bar).strftime("%d %H:%M"))

    def test_publish_retries_preserve_first_issue_and_observed_close(self):
        bars = price_bars()
        key = "NIFTY-15m-20261123T1515"
        now = datetime(2026, 11, 23, 16, 47, tzinfo=IST)
        with patch.object(live, "datetime", wraps=datetime) as clock, patch.object(live, "run_kronos_mc_trajectories") as model:
            clock.now.return_value = now
            sampled = np.full((50, 12, 6), 100.)
            sampled[:, :, :4] += np.arange(50)[:, None, None] / 50
            model.return_value = sampled
            forecast = live._make_forecast(lambda: None, "NIFTY", live.SYMBOLS_CONFIG["NIFTY"],
                                           "15m", "15-Minute", 80, 12, bars, key)
        self.assertEqual("2026-11-25 09:15", forecast["predictions"][0]["time"])
        self.assertEqual(now.isoformat(), forecast["issued_at"])
        self.assertEqual("2026-11-25T12:15:00+05:30", forecast["target_at"])
        for bar in forecast["predictions"]:
            self.assertLessEqual(bar["lower_05"], bar["lower_10"])
            self.assertLessEqual(bar["upper_90"], bar["upper_95"])
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            observations = {("NIFTY", "15m"): {"2026-11-23 15:15": 101}}
            retry = copy.deepcopy(forecast)
            retry["spot_close"] = 12345
            with publication_lock(data):
                publish(data, {"NIFTY": {"15m": forecast}}, observations)
                path = data / "archive" / f"{key}.json"
                before = path.read_bytes()
                current = publish(data, {"NIFTY": {"15m": retry}},
                                  {("NIFTY", "15m"): {"2026-11-23 15:15": 42}})
                self.assertEqual(current["NIFTY"]["15m"], forecast)
                self.assertEqual(before, path.read_bytes())
                ledger = json.loads((data / "observations" / "NIFTY-15m.json").read_text())
                self.assertEqual(101, ledger["2026-11-23 15:15"])

    def test_stale_input_skips_model(self):
        bars = price_bars()
        with patch.object(live, "datetime", wraps=datetime) as clock:
            clock.now.return_value = datetime(2026, 11, 25, 9, 31, tzinfo=IST)
            result = live._make_forecast(self.fail, "NIFTY", live.SYMBOLS_CONFIG["NIFTY"],
                                         "15m", "15-Minute", 80, 12, bars, "NIFTY-15m-20261123T1515")
        self.assertIsNone(result)
