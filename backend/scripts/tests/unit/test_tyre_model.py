"""Unit tests for lstm/tyre_model.py: the LSTM learns, search reports truthfully, evaluation
de-normalises correctly, MC-dropout uncertainty, and the cliff-lap rule."""

import logging
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import torch
import torch.nn as nn

from lstm.tyre_data import SEQUENCE_LENGTH
from lstm.tyre_model import (
    INPUT_SIZE, NUM_LAYERS, TyreCliffLSTM, evaluate_on_test, monte_carlo_predict,
    run_hyperparameter_search, train_one_config,
)

LOG = logging.getLogger("test_tyre_model")
LOG.addHandler(logging.NullHandler())
LOG.propagate = False


def _learnable(n: int = 400, seed: int = 0):
    """Target = mean of feature 0 over the window: a simple, fully learnable signal."""
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (n, SEQUENCE_LENGTH, INPUT_SIZE)).astype(np.float32)
    y = X[:, :, 0].mean(axis=1).astype(np.float32)
    return X, y


class TestTraining(unittest.TestCase):

    def test_lstm_learns_a_learnable_signal(self) -> None:
        X, y = _learnable()
        model, val_mse = train_one_config(X[:320], y[:320], X[320:], y[320:], hidden_size=32,
                                          learning_rate=0.01, n_epochs=25, logger=LOG)
        baseline = float(np.mean((y[320:] - y[:320].mean()) ** 2))   # predict the mean
        self.assertLess(val_mse, 0.3 * baseline)

    def test_search_reports_the_true_score_of_its_choice(self) -> None:
        X, y = _learnable(200)
        grid = [{"hidden_size": 8, "learning_rate": 0.0001}, {"hidden_size": 32, "learning_rate": 0.01}]
        best_h, best_lr, best_mse = run_hyperparameter_search(X, y, LOG, grid=grid, n_epochs=8)
        self.assertEqual((best_h, best_lr), (32, 0.01))            # the clearly better config wins
        _, rerun_mse = train_one_config(X[:160], y[:160], X[160:], y[160:], best_h, best_lr, 8, LOG)
        self.assertAlmostEqual(best_mse, rerun_mse, places=5)       # training is seeded


class _Constant(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0])


class TestEvaluation(unittest.TestCase):

    def test_mae_is_reported_in_km_h(self) -> None:
        y_mean, y_std = 200.0, 10.0
        y_test_norm = np.array([1.0, -2.0, 0.5], dtype=np.float32)   # 210, 180, 205 km/h
        mae = evaluate_on_test(_Constant(), np.zeros((3, SEQUENCE_LENGTH, INPUT_SIZE), np.float32),
                               y_test_norm, y_mean, y_std)
        self.assertAlmostEqual(mae, (10 + 20 + 5) / 3, places=4)    # model always predicts 200


class TestMonteCarloDropout(unittest.TestCase):

    def test_band_contains_mean_varies_and_restores_eval_mode(self) -> None:
        torch.manual_seed(0)
        model = TyreCliffLSTM(INPUT_SIZE, 32, NUM_LAYERS, 0.5)
        X = np.random.RandomState(0).rand(10, SEQUENCE_LENGTH, INPUT_SIZE).astype(np.float32)
        mean, lower, upper = monte_carlo_predict(model, X, n_samples=20)
        self.assertEqual(mean.shape, (10,))
        self.assertTrue(np.all(lower <= mean) and np.all(mean <= upper))
        self.assertTrue(np.all(upper - lower > 0))                  # dropout really was active
        self.assertFalse(model.training)



if __name__ == "__main__":
    unittest.main()
