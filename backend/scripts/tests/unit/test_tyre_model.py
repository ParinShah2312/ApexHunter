"""Unit tests for lstm/tyre_model.py: the LSTM learns, search reports truthfully, evaluation
de-normalises correctly, MC-dropout uncertainty, and the cliff-lap rule."""

import logging
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import torch
import torch.nn as nn

from lstm.tyre_data import LAP_FEATURES, SEQUENCE_LENGTH
from lstm.tyre_model import (
    DROPOUT_RATE, INPUT_SIZE, NUM_LAYERS, TyreCliffLSTM, evaluate_on_test, monte_carlo_predict,
    predict_single_stint, run_hyperparameter_search, train_one_config,
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

    def test_model_output_shape(self) -> None:
        model = TyreCliffLSTM(INPUT_SIZE, 16, NUM_LAYERS, DROPOUT_RATE)
        self.assertEqual(model(torch.zeros(8, SEQUENCE_LENGTH, INPUT_SIZE)).shape, (8,))

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


class TestCliffLap(unittest.TestCase):

    def test_cliff_is_first_lap_over_best_plus_1_5s(self) -> None:
        n = 8
        speeds = [200.0] * n
        stint = {"stint_index": 0, "lap_features": {
            **{f: [0.0] * n for f in LAP_FEATURES},
            "race_lap_number": list(range(10, 10 + n)), "mean_speed": speeds,
            "lap_time_seconds": [90.0] * n}}
        # track distance = 200 × 90 → predicted time = 18000 / speed
        pred_speed = np.array([200, 202, 201, 200, 199.9, 197, 195, 190], dtype=float)
        scaler = mock.MagicMock()
        scaler.transform.side_effect = lambda a: a
        with mock.patch("lstm.tyre_model.monte_carlo_predict",
                        return_value=(pred_speed, pred_speed, pred_speed)):
            out = predict_single_stint(stint, mock.MagicMock(), scaler, y_mean=0.0, y_std=1.0, seq_len=5)
        # Predicted times (18000 / speed): best is index 1 at 89.11 s → threshold 90.61 s.
        # Index 4 = 90.05 s (under), index 5 = 91.37 s (first over) → cliff at index 5.
        self.assertEqual(out["cliff_lap"], 5)
        self.assertEqual(out["laps_remaining"], 3)
        self.assertEqual(out["lap_numbers"], list(range(10, 10 + n)))
        self.assertAlmostEqual(out["predicted_laps"][1], 89.1089, places=3)
        self.assertAlmostEqual(out["predicted_laps"][5], 91.3706, places=3)


if __name__ == "__main__":
    unittest.main()
