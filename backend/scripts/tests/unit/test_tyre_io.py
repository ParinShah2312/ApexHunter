"""Unit tests for lstm/tyre_io.py: model/scaler/config persistence.
(Feature-set compatibility checks on load are in test_tyre_training.py.)"""

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

from lstm.tyre_data import LAP_FEATURES, SEQUENCE_LENGTH, TARGET
from lstm.tyre_io import load_model_artifacts, save_model_artifacts
from lstm.tyre_model import DROPOUT_RATE, INPUT_SIZE, NUM_LAYERS, TyrePaceRNN

LOG = logging.getLogger("test_tyre_io")
LOG.addHandler(logging.NullHandler())
LOG.propagate = False


class TestModelArtifacts(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        torch.manual_seed(0)
        self.model = TyrePaceRNN(INPUT_SIZE, 16, NUM_LAYERS, DROPOUT_RATE).eval()
        self.scaler = StandardScaler().fit(np.random.RandomState(0).rand(50, INPUT_SIZE))
        self.config = {"hidden_size": 16, "input_size": INPUT_SIZE, "features": LAP_FEATURES, "target": TARGET,
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


if __name__ == "__main__":
    unittest.main()
