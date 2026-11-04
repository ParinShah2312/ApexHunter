"""Integration test: the real Isolation Forest pipeline on real data (2023 Bahrain race, car #44).

No mocks: reads data_lake/clean_data/2023_1_R.parquet and FastF1 laps from the project cache, writes
to a temp folder. Skipped when the local data is not present.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from detect_mistakes import run_pipeline
from utils import DATA_LAKE_DIR

RUN_SLOW = os.environ.get("APEXHUNTER_RUN_SLOW", "0") == "1"
SESSION = DATA_LAKE_DIR / "clean_data" / "2023_1_R.parquet"
DRIVER = "44"


@unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")
@unittest.skipUnless(SESSION.exists(), f"Needs local data: {SESSION}")
class TestMistakesOnRealData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name)
        run_pipeline(str(SESSION), DRIVER, out, force=True, compare=True)
        cls.df = pd.read_parquet(out / f"2023_1_R_{DRIVER}_mistakes.parquet")
        with open(out / f"2023_1_R_{DRIVER}_mistakes_meta.json", encoding="utf-8") as f:
            cls.meta = json.load(f)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_outputs_have_what_the_dashboard_and_mongodb_read(self):
        """Output schema (replaces old audit item A20): columns, meta keys and event fields read by
        frontend/components (strategy_tab, mistakes_tab, app.py) and bigdata/mongo_manager.py."""
        self.assertTrue({"X", "Y", "Speed", "SessionTime", "LapNumber", "is_mistake", "anomaly_score",
                         "event_id"} <= set(self.df.columns))
        self.assertTrue({"session_file", "driver", "best_contamination", "reference_drivers",
                         "evaluation", "n_events"} <= set(self.meta))
        with open(Path(self.tmp.name) / f"2023_1_R_{DRIVER}_mistakes_events.json", encoding="utf-8") as f:
            events = json.load(f)["events"]
        self.assertTrue({"rank", "lap", "x", "y", "track_pos_m", "location", "duration_s", "speed_kmh",
                         "reason", "time_lost_s", "costly", "likely_saving", "saving_note"} <= set(events[0]))
        self.assertEqual(self.meta["n_events"], len(events))

    def test_reference_is_five_other_drivers(self):
        """The field reference never contains the driver being analysed (old method: it could)."""
        refs = self.meta["reference_drivers"]
        self.assertEqual(len(refs), 5)
        self.assertNotIn(DRIVER, refs)

    def test_braking_does_not_dominate_the_flags(self):
        """§B checkpoint: the old features flagged braking (95-99 % of flags, base rate ~20 %).
        Measured on this run: 37 % of flags are braking samples, base rate 22 %."""
        flagged = self.df[self.df["is_mistake"]]
        self.assertGreater(len(flagged), 0)
        self.assertLess((flagged["Brake"] > 0.5).mean(), 0.60)

    def test_planted_mistakes_are_found(self):
        """Per-mistake score on planted mistakes. Measured: precision 0.52, recall 0.25.
        If planting stopped changing the features, almost no event would touch a planted mistake."""
        ev = self.meta["evaluation"]
        self.assertGreater(ev["precision"], 0.3)
        self.assertGreater(ev["recall"], 0.1)

    def test_events_are_long_flag_runs_ranked_by_time_lost(self):
        """Events are the flag runs of at least 0.5 s (shorter runs keep their flags but are not
        events), ranked by time lost (likely saving last), each with a corner location."""
        with open(Path(self.tmp.name) / f"2023_1_R_{DRIVER}_mistakes_events.json", encoding="utf-8") as f:
            events = json.load(f)["events"]
        in_event = self.df["event_id"] >= 0
        self.assertTrue(self.df.loc[in_event, "is_mistake"].all())
        self.assertEqual(len(events), self.df.loc[in_event, "event_id"].nunique())
        self.assertGreater(self.meta["flags_outside_events"], 0)
        self.assertTrue(all(e["duration_s"] >= 0.5 for e in events))
        lost = [e["time_lost_s"] for e in events if not e["likely_saving"] and e["time_lost_s"] is not None]
        self.assertEqual(lost, sorted(lost, reverse=True))
        self.assertTrue(all(e["reason"] and e["location"].startswith(("Turn", "Approaching", "After"))
                            for e in events))

    def test_drs_and_close_following_samples_are_left_out(self):
        """Review: in races the samples with DRS open or < 1 s behind a car are not analysed
        (Hamilton spent much of the race behind other cars)."""
        self.assertGreater(self.meta["traffic_samples_removed"], 0.02 * self.meta["total_rows"])

    def test_events_cost_time_more_often_than_random_spots(self):
        """Real-data check: events lose clearly more time than the driver's own nearby laps more
        often than random spots do (measured 19 % vs 12 %)."""
        check = self.meta["time_lost_check"]
        self.assertGreater(check["isolation_forest"]["costly_share"], check["random_spots"]["costly_share"])

    def test_baseline_comparison_uses_the_same_test_and_flag_rate(self):
        """§B phase 4: Z-score and Isolation Forest are scored on the same planted mistakes and flag
        the same share of samples; both find the planted mistakes well above chance
        (measured per-mistake F1: Z-score 0.32, Isolation Forest 0.32)."""
        cmp = self.meta["comparison"]
        z, iso = cmp["planted"]["z_score"], cmp["planted"]["isolation_forest"]
        self.assertEqual(z["counts"]["planted"], iso["counts"]["planted"])
        self.assertAlmostEqual(iso["f1"], self.meta["evaluation"]["f1"], places=6)
        for m in (z, iso):
            self.assertGreater(m["recall"], 0.1)
        flagged = int(self.df["is_mistake"].sum())
        self.assertAlmostEqual(cmp["real_laps"]["z_score_flagged_samples"], flagged, delta=0.01 * flagged)

    def test_lstm_autoencoder_joins_the_same_comparison(self):
        """§D: the LSTM autoencoder is scored on the same planted mistakes and flag rate as the other
        two detectors, finds them well above chance, and flags 2 % of the real laps too."""
        cmp = self.meta["comparison"]
        ae, iso = cmp["planted"]["lstm_autoencoder"], cmp["planted"]["isolation_forest"]
        self.assertEqual(ae["counts"]["planted"], iso["counts"]["planted"])
        self.assertGreater(ae["recall"], 0.1)
        self.assertGreater(ae["precision"], 0.2)
        flagged = int(self.df["is_mistake"].sum())
        self.assertAlmostEqual(cmp["real_laps"]["lstm_autoencoder_flagged_samples"], flagged, delta=0.02 * flagged)
        self.assertIn("costly_share", cmp["lstm_autoencoder_time_lost_check"])

    def test_only_representative_laps_are_analysed(self):
        """Lap 1, pit laps and the last lap are excluded (Hamilton: in-laps 12 and 30, out-laps 13
        and 31, last lap 57, where drivers ease off)."""
        laps = set(self.df["LapNumber"].unique())
        self.assertNotIn(1, laps)
        self.assertTrue({12, 13, 30, 31, 57}.isdisjoint(laps), sorted(laps))


if __name__ == "__main__":
    unittest.main()
