"""Integration test: the real LSTM tyre pipeline (train -> save -> load -> predict) on real data.

No mocks: four real 2024 races from data_lake/clean_data (copied to a temp folder) and FastF1 laps
from the project cache. A small, short training run goes to a temp models folder; the project's
models/ folder is never touched. Skipped when the local data is not present.
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
PREDICT_DRIVER = "1"


@unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")
@unittest.skipUnless(all(p.exists() for p in RACES), "Needs local 2024 R1-R4 race data")
class TestTyrePipelineOnRealData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import fastf1
        import pandas as pd
        from lstm.tyre_data import extract_stints
        from lstm.tyre_io import load_model_artifacts
        from lstm.tyre_model import predict_single_stint
        from train_lstm import run_training
        from utils import enable_fastf1_cache

        cls.tmp = Path(tempfile.mkdtemp())
        sessions_dir = cls.tmp / "clean_data"
        sessions_dir.mkdir()
        for p in RACES:
            shutil.copy(p, sessions_dir / p.name)
        cls.models_dir = cls.tmp / "models"
        log = logging.getLogger("test_tyre_pipeline")
        log.handlers = [logging.NullHandler()]
        log.propagate = False

        cls.config = run_training(sessions_dir, cls.models_dir, ["2024"], test_split=0.25, log=log,
                                  grid=[{"hidden_size": 32, "learning_rate": 0.001}],
                                  n_epochs_search=5, n_epochs_final=5)
        model, scaler, config = load_model_artifacts(cls.models_dir, log)

        enable_fastf1_cache()
        session = fastf1.get_session(2024, 4, "R")
        session.load(telemetry=False, weather=False, messages=False)
        df = pd.read_parquet(RACES[-1])
        df_driver = df[df["Driver"] == PREDICT_DRIVER].sort_values("SessionTime").reset_index(drop=True)
        stints = extract_stints(session.laps.pick_drivers(PREDICT_DRIVER), df_driver, log)
        cls.results = [predict_single_stint(s, model, scaler, config["y_mean"], config["y_std"],
                                            config["sequence_length"]) for s in stints]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_holds_out_the_latest_whole_race(self):
        """#3: 4 races with test_split 0.25 -> 3 training races, the latest (R4) held out."""
        self.assertEqual((self.config["n_train_sessions"], self.config["n_test_sessions"]), (3, 1))

    def test_saved_model_matches_current_features(self):
        """#3: the saved config records the feature list, and the model reloads with it."""
        from lstm.tyre_data import LAP_FEATURES
        self.assertEqual(self.config["features"], LAP_FEATURES)
        with open(self.models_dir / "tyre_config.json", encoding="utf-8") as f:
            self.assertEqual(json.load(f)["input_size"], len(LAP_FEATURES))

    def test_predictions_only_where_a_five_lap_history_exists(self):
        """#20/#21: a lap is predicted iff the 5 laps before it are consecutive kept laps."""
        self.assertGreaterEqual(len(self.results), 1)
        for stint in self.results:
            laps = stint["lap_numbers"]
            for i, pred in enumerate(stint["predicted_laps"]):
                has_history = i >= 5 and laps[i] - laps[i - 5] == 5
                self.assertEqual(pred is not None, has_history, (stint["stint_index"], laps[i]))

    def test_predicted_lap_times_are_realistic(self):
        """Predicted lap times are within 10 % of the real ones (Japan 2024: ~95 s laps)."""
        for stint in self.results:
            for actual, pred in zip(stint["actual_laps"], stint["predicted_laps"]):
                if pred is not None and actual is not None:
                    self.assertAlmostEqual(pred, actual, delta=0.10 * actual)


if __name__ == "__main__":
    unittest.main()
