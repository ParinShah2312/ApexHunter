"""Regression tests for LSTM training correctness: chronological ordering, session-based
splitting, leak-free scaling, best-epoch weight restore and model/feature compatibility."""

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
import torch.nn as nn

from lstm.tyre_data import SEQUENCE_LENGTH, session_sort_key
from lstm.tyre_io import load_model_artifacts, save_model_artifacts
from lstm.tyre_model import (
    DROPOUT_RATE, INPUT_SIZE, NUM_LAYERS, TyreCliffLSTM, split_by_groups, train_one_config,
)
from train_lstm import fit_scalers


def _silent_logger() -> logging.Logger:
    lg = logging.getLogger("test_tyre_training_silent")
    lg.handlers = [logging.NullHandler()]
    lg.setLevel(logging.CRITICAL)
    return lg


class TestChronologicalOrdering(unittest.TestCase):

    def test_session_sort_key_orders_by_year_then_round(self) -> None:
        names = ["2024_10_R.parquet", "2023_5_R.parquet", "2024_2_R.parquet", "2023_22_R.parquet"]
        ordered = sorted((Path(n) for n in names), key=session_sort_key)
        self.assertEqual([p.name for p in ordered],
                         ["2023_5_R.parquet", "2023_22_R.parquet", "2024_2_R.parquet", "2024_10_R.parquet"])


class TestSplitByGroups(unittest.TestCase):

    def setUp(self) -> None:
        # 5 sessions × 4 sequences each
        self.groups = np.repeat(np.arange(5), 4)

    def test_holds_out_latest_whole_sessions(self) -> None:
        keep, hold = split_by_groups(len(self.groups), self.groups, 0.2)
        self.assertEqual(set(self.groups[hold]), {4})
        self.assertEqual(set(self.groups[keep]), {0, 1, 2, 3})

    def test_no_session_on_both_sides(self) -> None:
        keep, hold = split_by_groups(len(self.groups), self.groups, 0.4)
        self.assertFalse(set(self.groups[keep]) & set(self.groups[hold]))

    def test_never_holds_out_every_session(self) -> None:
        keep, hold = split_by_groups(len(self.groups), self.groups, 0.99)
        self.assertGreater(len(keep), 0)


class TestScalingWithoutLeakage(unittest.TestCase):

    def test_scalers_fitted_on_training_data_only(self) -> None:
        rng = np.random.RandomState(0)
        X_train = rng.normal(0, 1, (40, SEQUENCE_LENGTH, INPUT_SIZE)).astype(np.float32)
        y_train = rng.normal(200, 5, 40).astype(np.float32)
        scaler, y_mean, y_std = fit_scalers(X_train, y_train)
        np.testing.assert_allclose(scaler.mean_, X_train.reshape(-1, INPUT_SIZE).mean(axis=0), rtol=1e-5)
        self.assertAlmostEqual(y_mean, float(y_train.mean()), places=4)
        self.assertAlmostEqual(y_std, float(y_train.std()), places=4)


class TestBestEpochRestore(unittest.TestCase):

    def test_returned_model_matches_reported_best_val_mse(self) -> None:
        rng = np.random.RandomState(1)
        X = rng.rand(120, SEQUENCE_LENGTH, INPUT_SIZE).astype(np.float32)
        y = rng.rand(120).astype(np.float32)
        model, best_mse = train_one_config(
            X[:90], y[:90], X[90:], y[90:], hidden_size=16, learning_rate=0.05,
            n_epochs=8, logger=_silent_logger(),
        )
        model.eval()
        with torch.no_grad():
            mse = float(nn.MSELoss()(model(torch.FloatTensor(X[90:])), torch.FloatTensor(y[90:])).item())
        self.assertAlmostEqual(mse, best_mse, places=5)


class TestModelCompatibilityCheck(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.logger = _silent_logger()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _save(self, input_size: int, features=None) -> None:
        model = TyreCliffLSTM(input_size, 16, NUM_LAYERS, DROPOUT_RATE)
        scaler = StandardScaler().fit(np.random.rand(20, input_size))
        config = {"hidden_size": 16, "input_size": input_size, "sequence_length": SEQUENCE_LENGTH,
                  "y_mean": 0.0, "y_std": 1.0}
        if features is not None:
            config["features"] = features
        save_model_artifacts(model, scaler, config, self.tmp, self.logger)

    def test_old_input_size_raises_clear_error(self) -> None:
        self._save(INPUT_SIZE - 1)
        with self.assertRaisesRegex(ValueError, "train_lstm.py --force"):
            load_model_artifacts(self.tmp, self.logger)

    def test_feature_name_mismatch_raises(self) -> None:
        self._save(INPUT_SIZE, features=[f"f{i}" for i in range(INPUT_SIZE)])
        with self.assertRaises(ValueError):
            load_model_artifacts(self.tmp, self.logger)

    def test_matching_model_loads(self) -> None:
        from lstm.tyre_data import LAP_FEATURES
        self._save(INPUT_SIZE, features=LAP_FEATURES)
        model, scaler, config = load_model_artifacts(self.tmp, self.logger)
        self.assertIsInstance(model, TyreCliffLSTM)


if __name__ == "__main__":
    unittest.main()
