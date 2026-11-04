"""Integration test: Ridge tyre pace baseline (train_ridge.py) on real data (ROADMAP §C step 5).

No mocks: four real 2024 races from data_lake/clean_data (copied to a temp folder) and FastF1 laps
from the project cache. The model is saved to a temp models folder; models/ is never touched.
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from utils import DATA_LAKE_DIR

RUN_SLOW = os.environ.get("APEXHUNTER_RUN_SLOW", "0") == "1"
RACES = [DATA_LAKE_DIR / "clean_data" / f"2024_{r}_R.parquet" for r in (1, 2, 3, 4)]


@unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")
@unittest.skipUnless(all(p.exists() for p in RACES), "Needs local 2024 R1-R4 race data")
class TestRidgeTrainingOnRealData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from train_ridge import run_ridge_training

        cls.tmp = Path(tempfile.mkdtemp())
        sessions = cls.tmp / "clean_data"
        sessions.mkdir()
        for p in RACES:
            shutil.copy(p, sessions / p.name)
        cls.models_dir = cls.tmp / "models"
        log = logging.getLogger("test_ridge_training")
        log.addHandler(logging.NullHandler())
        log.propagate = False
        # test_split 0.25 of 4 races -> the latest race (2024 R4, Japan) is the test set
        cls.config = run_ridge_training(sessions, cls.models_dir, ["2024"], 0.25, log)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_holds_out_the_latest_race_and_reports_against_no_change(self):
        self.assertEqual((self.config["n_train_sessions"], self.config["n_test_sessions"]), (3, 1))
        m = self.config["metrics"]
        for part in ("train", "test", "train_no_change", "test_no_change"):
            self.assertEqual(set(m[part]), {"mae_s", "rmse_s", "r2", "mape_pct"})
        # real lap-time changes: errors of a few tenths, MAPE well under 1 % of a ~90 s lap
        self.assertLess(m["test"]["mae_s"], 1.0)
        self.assertLess(m["test"]["mape_pct"], 1.0)
        self.assertAlmostEqual(m["test_no_change"]["r2"], 0.0, delta=0.05)

    def test_saved_model_reloads_and_is_rejected_for_another_target(self):
        import numpy as np
        from lstm.tyre_ridge import CONFIG_FILENAME, load_ridge

        model, config = load_ridge(self.models_dir)
        self.assertEqual(config["alpha"], self.config["alpha"])
        self.assertEqual(model.predict(np.zeros((1, 45))).shape, (1,))
        path = self.models_dir / CONFIG_FILENAME
        saved = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**saved, "target": "next_lap_mean_speed_change_kmh"}), encoding="utf-8")
        try:
            with self.assertRaisesRegex(ValueError, "Retrain"):
                load_ridge(self.models_dir)
        finally:
            path.write_text(json.dumps(saved), encoding="utf-8")

    def test_strategy_check_on_the_held_out_race(self):
        """§C step 6: the saved model's test race goes through the real pit window, Ridge, "no
        change" and "laps driven" are scored on the same stints, and Ridge only advises once it has
        5 laps."""
        from check_strategy import check_races
        from lstm.tyre_ridge import load_ridge
        from strategy.strategy_check import summarise

        model, config = load_ridge(self.models_dir)
        self.assertEqual(config["test_races"], ["2024_4_R"])
        log = logging.getLogger("test_ridge_training")
        from lstm.tyre_ridge import flatten
        rows = check_races(self.tmp / "clean_data", config["test_races"],
                           {"ridge": lambda w: model.predict(flatten(w))}, log)
        s = summarise(rows)
        type(self).summary = s
        self.assertEqual(set(s), {"ridge", "no_change", "laps_driven"})
        self.assertGreater(s["ridge"]["stints"], 10)                 # 20 drivers, mostly 1-2 stops
        self.assertEqual(s["ridge"]["stints"], s["no_change"]["stints"])
        for src in s.values():
            self.assertGreater(src["hit_pct"], 0.0)
            self.assertLess(src["random_pct"], 100.0)
        # no advice before a model prediction exists (needs 5 clean laps), so Ridge advises on
        # fewer laps than "no change", which has advice from lap 1
        ridge_most = rows[rows["source"] == "ridge"]["advised_laps"].max()
        self.assertLess(ridge_most, rows[rows["source"] == "no_change"]["advised_laps"].max())

    def test_strategy_output_for_a_driver(self):
        """§C step 7: predict_strategy output for Verstappen (#1) in the held-out race."""
        import json
        from predict_strategy import run_prediction

        out = run_prediction(self.tmp / "clean_data" / "2024_4_R.parquet", "1", self.models_dir,
                             source="ridge", check_file=self.tmp / "no_check.json")
        json.dumps(out)
        self.assertEqual((out["driver_code"], out["model"], out["in_training_data"]), ("VER", "ridge", False))
        laps = out["laps"]
        predicted = [r for r in laps if r["predicted_lap_time"] is not None]
        # lap 1 is never used, so the first prediction needs laps 2-6 -> lap 7 at the earliest
        self.assertGreaterEqual(min(r["lap"] for r in predicted), 7)
        self.assertGreater(len(predicted), 20)
        # predicted = previous real lap + a change of tenths, never the old speed-based estimate
        for r in predicted:
            prev = laps[r["lap"] - 2]["lap_time"]
            self.assertLess(abs(r["predicted_lap_time"] - prev), 2.0)
        planned = [s for s in out["stints"] if "summary" in s]
        self.assertTrue(planned)
        self.assertTrue(all(s["summary"][-1].startswith(f"Actual: pitted lap {s['actual_stop']}") for s in planned))

    def test_best_source_is_taken_from_the_strategy_check(self):
        """User, 2026-11-04: the output uses whichever source scored best in check_strategy."""
        import json
        from predict_strategy import RULE_BASED, run_prediction
        from strategy.strategy_check import best_source

        if not hasattr(type(self), "summary"):
            self.test_strategy_check_on_the_held_out_race()
        check = self.tmp / "strategy_check.json"
        best = best_source(self.summary)
        check.write_text(json.dumps({"summary": self.summary, "best_source": best}), encoding="utf-8")
        out = run_prediction(self.tmp / "clean_data" / "2024_4_R.parquet", "1", self.models_dir,
                             check_file=check)
        self.assertEqual((out["model"], out["best_source"]), (best, best))
        self.assertEqual(set(out["strategy_scores"]), {"ridge", "no_change", "laps_driven"})
        has_predictions = any(r["predicted_lap_time"] is not None for r in out["laps"])
        self.assertEqual(has_predictions, best not in RULE_BASED)


if __name__ == "__main__":
    unittest.main()
