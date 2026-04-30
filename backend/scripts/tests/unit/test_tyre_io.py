"""Unit tests for tyre_io.py — model artifact I/O and prediction output."""

import json
import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import torch
from sklearn.preprocessing import StandardScaler

from tyre_model import TyreCliffLSTM, INPUT_SIZE, NUM_LAYERS, DROPOUT_RATE
from tyre_io import (
    build_prediction_output,
    load_model_artifacts,
    save_model_artifacts,
    save_prediction,
)


def _silent_logger() -> logging.Logger:
    """Return a logger that writes nowhere."""
    lg = logging.getLogger("test_tyre_io_silent")
    lg.handlers = [logging.NullHandler()]
    lg.setLevel(logging.CRITICAL)
    return lg


class TestModelArtifacts(unittest.TestCase):
    """Tests for save/load model artifacts."""

    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.logger = _silent_logger()

        self.dummy_model = TyreCliffLSTM(INPUT_SIZE, 32, NUM_LAYERS, DROPOUT_RATE)
        self.dummy_scaler = StandardScaler()
        self.dummy_scaler.fit(np.random.rand(50, INPUT_SIZE).astype(np.float32))
        self.dummy_config = {
            "hidden_size": 32,
            "learning_rate": 0.001,
            "input_size": INPUT_SIZE,
            "sequence_length": 5,
            "num_layers": NUM_LAYERS,
            "dropout_rate": DROPOUT_RATE,
            "y_mean": 250.0,
            "y_std": 20.0,
            "val_mse": 0.05,
            "test_mae_kmh": 1.5,
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_model_artifacts_creates_files(self) -> None:
        save_model_artifacts(
            self.dummy_model, self.dummy_scaler, self.dummy_config,
            self.tmp_dir, self.logger,
        )
        self.assertTrue((self.tmp_dir / "tyre_lstm.pt").exists())
        self.assertTrue((self.tmp_dir / "tyre_scaler.pkl").exists())
        self.assertTrue((self.tmp_dir / "tyre_config.json").exists())

    def test_load_model_artifacts_returns_three(self) -> None:
        save_model_artifacts(
            self.dummy_model, self.dummy_scaler, self.dummy_config,
            self.tmp_dir, self.logger,
        )
        model, scaler, config = load_model_artifacts(self.tmp_dir, self.logger)
        self.assertIsInstance(config, dict)
        self.assertIn("hidden_size", config)

    def test_load_model_artifacts_missing_file_raises(self) -> None:
        # Only save the scaler
        import pickle
        with open(self.tmp_dir / "tyre_scaler.pkl", "wb") as f:
            pickle.dump(self.dummy_scaler, f)
        with self.assertRaises(FileNotFoundError):
            load_model_artifacts(self.tmp_dir, self.logger)


class TestPredictionOutput(unittest.TestCase):
    """Tests for prediction output building and saving."""

    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.logger = _silent_logger()
        self.dummy_stint = {
            "stint_index": 0,
            "n_laps": 10,
            "actual_laps": [250.0 + i for i in range(10)],
            "predicted_laps": [None] * 5 + [252.0, 251.0, 250.0, 249.0, 248.0],
            "confidence_upper": [None] * 5 + [254.0, 253.0, 252.0, 251.0, 250.0],
            "confidence_lower": [None] * 5 + [250.0, 249.0, 248.0, 247.0, 246.0],
            "cliff_lap": 8,
            "laps_remaining": 2,
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_build_prediction_output_structure(self) -> None:
        data = build_prediction_output("session.parquet", "44", [self.dummy_stint])
        self.assertIn("session_file", data)
        self.assertIn("driver", data)
        self.assertIn("stints", data)
        self.assertIsInstance(data["stints"], list)

    def test_stint_result_has_required_keys(self) -> None:
        data = build_prediction_output("session.parquet", "44", [self.dummy_stint])
        required_keys = [
            "stint_index", "n_laps", "actual_laps", "predicted_laps",
            "confidence_upper", "confidence_lower", "cliff_lap", "laps_remaining",
        ]
        for key in required_keys:
            self.assertIn(key, data["stints"][0])

    def test_save_prediction_creates_file(self) -> None:
        data = build_prediction_output("session.parquet", "44", [self.dummy_stint])
        output_path = self.tmp_dir / "out.json"
        save_prediction(data, output_path, self.logger)
        self.assertTrue(output_path.exists())
        with open(output_path, "r") as f:
            loaded = json.load(f)
        self.assertIn("stints", loaded)


if __name__ == "__main__":
    unittest.main()
