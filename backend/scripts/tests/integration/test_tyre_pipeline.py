"""Integration test: the real LSTM / vanilla RNN tyre pipeline (train -> save -> load -> predict ->
pit-window check) on real data (ROADMAP §D).

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
        from lstm.tyre_model import sequence_predictor
        from strategy.strategy_check import predicted_changes
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
                                  n_epochs_search=5, n_epochs_final=5, seeds=(42, 43))
        cls.rnn_config = run_training(sessions_dir, cls.models_dir, ["2024"], test_split=0.25, log=log,
                                      grid=[{"hidden_size": 32, "learning_rate": 0.001}],
                                      n_epochs_search=5, n_epochs_final=5, cell="rnn", seeds=(42,))
        cls.log, cls.sessions_dir = log, sessions_dir
        model, scaler, config = load_model_artifacts(cls.models_dir, log)

        enable_fastf1_cache()
        session = fastf1.get_session(2024, 4, "R")
        session.load(telemetry=False, weather=False, messages=False)
        df = pd.read_parquet(RACES[-1])
        df_driver = df[df["Driver"] == PREDICT_DRIVER].sort_values("SessionTime").reset_index(drop=True)
        stints = extract_stints(session.laps.pick_drivers(PREDICT_DRIVER), df_driver, log)
        cls.stints = stints
        cls.changes = predicted_changes(stints, sequence_predictor(model, scaler, config["y_mean"], config["y_std"]))

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

    def test_config_reports_the_saved_model_and_metrics_vs_no_change(self):
        """§D / #46: val_mse is the saved final model's (the best of its seeds), not the search
        run's; test metrics are on the real scale, next to "no change"; test races recorded."""
        c = self.config
        self.assertEqual(c["val_mse"], min(r["val_mse"] for r in c["seed_runs"]))
        self.assertEqual([r["seed"] for r in c["seed_runs"]], [42, 43])
        self.assertEqual(c["test_races"], ["2024_4_R"])
        for part in ("test", "test_no_change"):
            self.assertLess(c["metrics"][part]["mae_s"], 2.0)          # seconds, not normalised
        self.assertAlmostEqual(c["metrics"]["test_no_change"]["r2"], 0.0, delta=0.05)

    def test_rnn_saved_next_to_the_lstm_and_both_go_through_the_pit_window(self):
        """§D: the RNN has its own files and RNN layer; both models are scored by the strategy check
        on the same real stints as "no change" and "laps driven"."""
        import torch.nn as nn
        from check_strategy import check_races
        from lstm.tyre_io import load_model_artifacts
        from lstm.tyre_model import sequence_predictor
        from strategy.strategy_check import summarise

        self.assertTrue((self.models_dir / "tyre_rnn.pt").exists())
        self.assertEqual(self.rnn_config["cell"], "rnn")
        predictors = {}
        for cell in ("lstm", "rnn"):
            model, scaler, cfg = load_model_artifacts(self.models_dir, self.log, cell=cell)
            self.assertIsInstance(model.lstm, nn.LSTM if cell == "lstm" else nn.RNN)
            predictors[cell] = sequence_predictor(model, scaler, cfg["y_mean"], cfg["y_std"])
        s = summarise(check_races(self.sessions_dir, ["2024_4_R"], predictors, self.log))
        self.assertEqual(set(s), {"lstm", "rnn", "no_change", "laps_driven"})
        self.assertEqual(len({v["stints"] for v in s.values()}), 1)        # same stints for all
        self.assertGreater(s["lstm"]["stints"], 10)

    def test_predictions_only_where_a_five_lap_history_exists(self):
        """#20/#21: a lap is predicted iff the 5 laps before it are consecutive kept laps."""
        expected = set()
        for stint in self.stints:
            laps = [int(n) for n in stint["lap_features"]["race_lap_number"]]
            expected |= {laps[i] for i in range(5, len(laps)) if laps[i] - laps[i - 5] == 5}
            expected |= {laps[i] + 1 for i in range(4, len(laps))      # the lap after a 5-lap run (#52)
                         if laps[i] - laps[i - 4] == 4 and (i + 1 == len(laps) or laps[i + 1] != laps[i] + 1)}
        self.assertTrue(expected)
        self.assertEqual(set(self.changes), expected)

    def test_predicted_changes_are_realistic(self):
        """Predicted lap-time changes are tenths of a second (Japan 2024: ~95 s laps), not lap times."""
        self.assertLess(max(abs(v) for v in self.changes.values()), 3.0)


if __name__ == "__main__":
    unittest.main()
