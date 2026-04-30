"""Unit tests for tyre_model.py — LSTM architecture and training loop."""

import logging
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import torch
import torch.nn as nn

from tyre_data import SEQUENCE_LENGTH
from tyre_model import (
    DROPOUT_RATE,
    INPUT_SIZE,
    NUM_LAYERS,
    TyreCliffLSTM,
    evaluate_on_test,
    monte_carlo_predict,
    run_hyperparameter_search,
    train_one_config,
)


def _silent_logger() -> logging.Logger:
    """Return a logger that writes nowhere."""
    lg = logging.getLogger("test_tyre_model_silent")
    lg.handlers = [logging.NullHandler()]
    lg.setLevel(logging.CRITICAL)
    return lg


class TestTyreCliffLSTM(unittest.TestCase):
    """Tests for the TyreCliffLSTM model class."""

    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.RandomState(42)
        N = 100
        cls.X = rng.rand(N, SEQUENCE_LENGTH, INPUT_SIZE).astype(np.float32)
        cls.y = (rng.rand(N).astype(np.float32) * 100 + 200)
        cls.logger = _silent_logger()

    def test_model_instantiation(self) -> None:
        model = TyreCliffLSTM(INPUT_SIZE, 32, NUM_LAYERS, DROPOUT_RATE)
        self.assertIsInstance(model, nn.Module)

    def test_model_forward_output_shape(self) -> None:
        model = TyreCliffLSTM(INPUT_SIZE, 32, NUM_LAYERS, DROPOUT_RATE)
        x = torch.FloatTensor(self.X[:8])
        out = model(x)
        self.assertEqual(out.shape, (8,))

    def test_train_one_config_returns_model_and_float(self) -> None:
        model, mse = train_one_config(
            self.X[:80], self.y[:80], self.X[80:], self.y[80:],
            hidden_size=32, learning_rate=0.001,
            n_epochs=2, logger=self.logger,
        )
        self.assertIsInstance(model, TyreCliffLSTM)
        self.assertIsInstance(mse, float)
        self.assertGreaterEqual(mse, 0.0)

    def test_hyperparameter_search_returns_valid_config(self) -> None:
        best_h, best_lr, best_mse = run_hyperparameter_search(
            self.X, self.y, self.logger
        )
        self.assertIn(best_h, [32, 64, 128])
        self.assertIn(best_lr, [0.001, 0.0005, 0.0001])
        self.assertGreaterEqual(best_mse, 0.0)

    def test_monte_carlo_predict_output_shapes(self) -> None:
        model = TyreCliffLSTM(INPUT_SIZE, 32, NUM_LAYERS, DROPOUT_RATE)
        mean, lower, upper = monte_carlo_predict(model, self.X[:20], n_samples=5)
        self.assertEqual(mean.shape, (20,))
        self.assertEqual(lower.shape, (20,))
        self.assertEqual(upper.shape, (20,))

    def test_monte_carlo_lower_le_mean_le_upper(self) -> None:
        model = TyreCliffLSTM(INPUT_SIZE, 32, NUM_LAYERS, DROPOUT_RATE)
        mean, lower, upper = monte_carlo_predict(model, self.X[:20], n_samples=10)
        # Allow tiny floating point tolerance
        self.assertTrue((lower <= mean + 0.001).all())
        self.assertTrue((mean <= upper + 0.001).all())

    def test_evaluate_on_test_returns_float(self) -> None:
        model, _ = train_one_config(
            self.X[:80], self.y[:80], self.X[80:], self.y[80:],
            32, 0.001, 2, self.logger,
        )
        mae = evaluate_on_test(model, self.X[80:], self.y[80:], 250.0, 20.0)
        self.assertIsInstance(mae, float)
        self.assertGreaterEqual(mae, 0.0)


if __name__ == "__main__":
    unittest.main()
