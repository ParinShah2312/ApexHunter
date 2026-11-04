"""Unit tests for lstm/tyre_model.py: the LSTM and the vanilla RNN learn, the search reports
truthfully, and predictions are de-normalised to lap-time changes in seconds."""

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
    INPUT_SIZE, NUM_LAYERS, TyrePaceRNN, run_hyperparameter_search, sequence_predictor, train_one_config,
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

    def test_rnn_cell_is_a_plain_rnn_and_learns_the_same_signal(self) -> None:
        """§D: cell="rnn" must really swap the recurrent layer (the RNN vs LSTM comparison)."""
        X, y = _learnable()
        model, val_mse = train_one_config(X[:320], y[:320], X[320:], y[320:], hidden_size=32,
                                          learning_rate=0.01, n_epochs=25, logger=LOG, cell="rnn")
        self.assertIsInstance(model.lstm, nn.RNN)
        self.assertNotIsInstance(model.lstm, nn.LSTM)
        baseline = float(np.mean((y[320:] - y[:320].mean()) ** 2))
        self.assertLess(val_mse, 0.3 * baseline)

    def test_search_reports_the_true_score_of_its_choice(self) -> None:
        X, y = _learnable(200)
        grid = [{"hidden_size": 8, "learning_rate": 0.0001}, {"hidden_size": 32, "learning_rate": 0.01}]
        best_h, best_lr, best_mse = run_hyperparameter_search(X, y, LOG, grid=grid, n_epochs=8)
        self.assertEqual((best_h, best_lr), (32, 0.01))            # the clearly better config wins
        _, rerun_mse = train_one_config(X[:160], y[:160], X[160:], y[160:], best_h, best_lr, 8, LOG)
        self.assertAlmostEqual(best_mse, rerun_mse, places=5)       # training is seeded


class TestPredictor(unittest.TestCase):

    def test_predictions_are_lap_time_changes_in_seconds(self) -> None:
        """The strategy check and output get changes in s: scaled inputs in, de-normalised out."""
        from sklearn.preprocessing import StandardScaler
        windows = np.random.RandomState(0).rand(3, SEQUENCE_LENGTH, INPUT_SIZE).astype(np.float32)
        scaler = StandardScaler().fit(windows.reshape(-1, INPUT_SIZE))
        model = TyrePaceRNN(INPUT_SIZE, 8, NUM_LAYERS, 0.0)
        with torch.no_grad():                                      # output layer set to 0: the real
            model.fc.weight.zero_(); model.fc.bias.zero_()         # model then predicts exactly 0
        out = sequence_predictor(model, scaler, y_mean=0.05, y_std=0.9)(windows)
        np.testing.assert_allclose(out, [0.05, 0.05, 0.05], atol=1e-7)   # 0 normalised = the mean
        model.fc.bias.data.fill_(1.0)                              # +1 normalised = mean + one std
        np.testing.assert_allclose(sequence_predictor(model, scaler, 0.05, 0.9)(windows), [0.95] * 3, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
