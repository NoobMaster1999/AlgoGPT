import json
import runpy
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


# The standalone scorer has no model or training dependencies.
scorer = runpy.run_path(str(Path(__file__).resolve().parents[1] / "algogpt" / "score_forecasts.py"))
score_forecast = scorer["score_forecast"]
write_scorecard = scorer["write_scorecard"]
AS_OF = datetime.fromisoformat("2026-11-23T12:00:00+05:30")


def forecast(record_id="NIFTY-15m-20261123T0915"):
    return {
        "forecast_id": record_id, "symbol": "NIFTY", "interval_key": "15m",
        "observed_at": "2026-11-23T09:30:00+05:30",
        "issued_at": "2026-11-23T09:32:00+05:30",
        "target_at": "2026-11-23T10:00:00+05:30",
        "spot_close": 100,
        "predictions": [
            {"time": "2026-11-23 09:30", "close": 101},
            {"time": "2026-11-23 09:45", "close": 103, "lower_25": 102, "upper_75": 104,
             "lower_05": 99, "upper_95": 105, "lower_10": 100, "upper_90": 104},
        ],
    }


class TerminalCloseTests(unittest.TestCase):
    def test_comparison_uses_same_terminal_close_and_baseline(self):
        result = score_forecast(forecast(), {"2026-11-23 09:45": 105}, AS_OF)
        self.assertEqual("completed", result["status"])
        self.assertEqual(2, result["model_abs_error"])
        self.assertEqual(5, result["baseline_abs_error"])
        self.assertIs(result["inside_50"], False)
        self.assertIs(result["inside_90"], True)

    def test_exact_match_required_even_when_adjacent_bars_exist(self):
        bars = {"2026-11-23 09:30": 101, "2026-11-23 10:00": 110}
        result = score_forecast(forecast(), bars, AS_OF)
        self.assertEqual("pending", result["status"])
        self.assertIsNone(result["actual_close"])

    def test_incomplete_target_bar_never_scored(self):
        now = datetime.fromisoformat("2026-11-23T09:59:59+05:30")
        result = score_forecast(forecast(), {"2026-11-23 09:45": 104}, now)
        self.assertEqual("pending", result["status"])
        self.assertIn("not closed", result["reason"])

    def test_unverified_or_late_forecasts_cannot_earn_a_score(self):
        for issued in (None, "2026-11-23 09:33", "2026-11-23T09:29:00+05:30",
                       "2026-11-23T10:00:00+05:30"):
            with self.subTest(issued=issued):
                payload = dict(forecast(), issued_at=issued)
                result = score_forecast(payload, {"2026-11-23 09:45": 103}, AS_OF)
                self.assertEqual("excluded", result["status"])
                self.assertTrue(result["reason"])

    def test_invalid_close_is_never_scored(self):
        for close in (None, 0, -1, float("nan"), True):
            with self.subTest(close=close):
                result = score_forecast(forecast(), {"2026-11-23 09:45": close}, AS_OF)
                self.assertEqual("pending", result["status"])

    def test_end_to_end_counts_exclude_missing_bands_and_unverified_issues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "archive"
            archive.mkdir()
            ledger = root / "observations"
            ledger.mkdir()
            (ledger / "NIFTY-15m.json").write_text(json.dumps({"2026-11-23 09:45": 104}))
            first = forecast()
            second = forecast("second")
            del second["predictions"][-1]["lower_05"]
            third = dict(forecast("third"), issued_at=None)
            fourth = dict(forecast("fourth"), symbol="BANKNIFTY")
            for payload in (first, second, third, fourth):
                (archive / (payload["forecast_id"] + ".json")).write_text(json.dumps(payload))
            report = write_scorecard(root, AS_OF)
            self.assertEqual(report, write_scorecard(root, AS_OF))
            self.assertEqual(report, json.loads((root / "scorecard.json").read_text()))
            group = report["groups"]["NIFTY"]["15m"]
            self.assertEqual((3, 2, 1), (group["issued_count"], group["completed_count"], group["excluded_count"]))
            self.assertEqual((1, 4), (group["model_mae"], group["baseline_mae"]))
            self.assertEqual((2, 1), (group["coverage_50_count"], group["coverage_90_count"]))
            self.assertEqual((1, 1), (group["coverage_50"], group["coverage_90"]))
            pending = report["groups"]["BANKNIFTY"]["15m"]
            self.assertEqual(1, pending["pending_count"])
            self.assertIsNone(pending["model_mae"])
            self.assertIsNone(pending["coverage_50"])


if __name__ == "__main__":
    unittest.main()
