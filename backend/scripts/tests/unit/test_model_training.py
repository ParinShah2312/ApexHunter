"""Unit tests for train_final_model with MOCKED IsolationForest."""

import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))


def _make_silent_logger():
    lg = logging.getLogger("test_train_mock")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.DEBUG)
    return lg


class TestModelTraining(unittest.TestCase):

    def setUp(self):
        self.logger = _make_silent_logger()
        self.X = np.random.rand(100, 7).astype("float32")

    @patch("mistakes_model.IsolationForest")
    def test_train_returns_fitted_model(self, MockIF):
        instance = MagicMock()
        instance.fit.return_value = instance
        MockIF.return_value = instance
        from mistakes_model import train_final_model
        result = train_final_model(self.X, 0.10, self.logger)
        instance.fit.assert_called_once()
        # The first positional arg to fit should be our X
        actual_X = instance.fit.call_args[0][0]
        np.testing.assert_array_equal(actual_X, self.X)

    @patch("mistakes_model.IsolationForest")
    def test_train_uses_best_contamination(self, MockIF):
        instance = MagicMock()
        instance.fit.return_value = instance
        MockIF.return_value = instance
        from mistakes_model import train_final_model
        train_final_model(self.X, 0.10, self.logger)
        self.assertEqual(MockIF.call_args.kwargs["contamination"], 0.10)

    @patch("mistakes_model.IsolationForest")
    def test_train_uses_n_jobs_minus_one(self, MockIF):
        instance = MagicMock()
        instance.fit.return_value = instance
        MockIF.return_value = instance
        from mistakes_model import train_final_model
        train_final_model(self.X, 0.10, self.logger)
        self.assertEqual(MockIF.call_args.kwargs["n_jobs"], -1)

    @patch("mistakes_model.IsolationForest")
    def test_train_uses_max_samples_auto(self, MockIF):
        instance = MagicMock()
        instance.fit.return_value = instance
        MockIF.return_value = instance
        from mistakes_model import train_final_model
        train_final_model(self.X, 0.10, self.logger)
        self.assertEqual(MockIF.call_args.kwargs["max_samples"], "auto")

    @patch("mistakes_model.IsolationForest")
    def test_train_returns_the_model_instance(self, MockIF):
        instance = MagicMock()
        instance.fit.return_value = instance
        MockIF.return_value = instance
        from mistakes_model import train_final_model
        result = train_final_model(self.X, 0.10, self.logger)
        self.assertIs(result, instance)


if __name__ == "__main__":
    unittest.main()
