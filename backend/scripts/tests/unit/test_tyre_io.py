"""Unit tests for lstm/tyre_io.py: model/scaler/config persistence and prediction output.
(Feature-set compatibility checks on load are in test_tyre_training.py.)"""

import json
import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import torch

from lstm.tyre_data import LAP_FEATURES, SEQUENCE_LENGTH
from lstm.tyre_io import build_prediction_output, load_model_artifacts, save_model_artifacts, save_prediction
from lstm.tyre_model import DROPOUT_RATE, INPUT_SIZE, NUM_LAYERS, TyreCliffLSTM

LOG = logging.getLogger("test_tyre_io")
LOG.addHandler(logging.NullHandler())
LOG.propagate = False


class TestModelArtifacts(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        torch.manual_seed(0)
        self.model = TyreCliffLSTM(INPUT_SIZE, 16, NUM_LAYERS, DROPOUT_RATE).eval()
        self.scaler = StandardScaler().fit(np.random.RandomState(0).rand(50, INPUT_SIZE))
        self.config = {"hidden_size": 16, "input_size": INPUT_SIZE, "features": LAP_FEATURES,
                       "sequence_length": SEQUENCE_LENGTH, "y_mean": 230.0, "y_std": 12.0}

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_round_trip_preserves_model_scaler_and_config(self) -> None:
        save_model_artifacts(self.model, self.scaler, self.config, self.tmp, LOG)
        model, scaler, config = load_model_artifacts(self.tmp, LOG)
        x = torch.rand(4, SEQUENCE_LENGTH, INPUT_SIZE)
        with torch.no_grad():
            torch.testing.assert_close(model(x), self.model(x))
        self.assertFalse(model.training)
        np.testing.assert_array_equal(scaler.mean_, self.scaler.mean_)
        np.testing.assert_array_equal(scaler.scale_, self.scaler.scale_)
        self.assertEqual(config, self.config)

    def test_missing_artifact_named_in_error(self) -> None:
        save_model_artifacts(self.model, self.scaler, self.config, self.tmp, LOG)
        (self.tmp / "tyre_scaler.pkl").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "tyre_scaler.pkl"):
            load_model_artifacts(self.tmp, LOG)


class TestPredictionOutput(unittest.TestCase):

    def test_build_and_save_round_trip(self) -> None:
        stints = [{"stint_index": 0, "n_laps": 2, "lap_numbers": [5, 6], "actual_laps": [90.1, None],
                   "predicted_laps": [90.0, 90.4], "confidence_upper": [90.5, 91.0],
                   "confidence_lower": [89.5, 89.8], "cliff_lap": None, "laps_remaining": None}]
        data = build_prediction_output("data_lake/clean_data/2024_1_R.parquet", "44", stints)
        self.assertEqual((data["session_file"], data["driver"], data["stints"]),
                         ("data_lake/clean_data/2024_1_R.parquet", "44", stints))
        self.assertIn("+05:30", data["timestamp"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "out.json"            # parent created on save
            save_prediction(data, path, LOG)
            with open(path) as f:
                self.assertEqual(json.load(f), data)


if __name__ == "__main__":
    unittest.main()
