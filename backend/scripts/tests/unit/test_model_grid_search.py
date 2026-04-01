"""Unit tests for run_grid_search with MOCKED IsolationForest."""

import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))


def _make_silent_logger():
    lg = logging.getLogger("test_grid_mock")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.DEBUG)
    return lg


def _make_mock_model(n=50):
    mock = MagicMock()
    mock.decision_function.return_value = np.random.uniform(-0.3, 0.3, n)
    mock.predict.return_value = np.ones(n, dtype=int)
    mock.fit.return_value = mock
    return mock


class TestGridSearch(unittest.TestCase):

    def setUp(self):
        self.logger = _make_silent_logger()
        self.X = np.random.rand(100, 7).astype("float32")

    @patch("mistakes_model.IsolationForest")
    def test_grid_search_tries_all_contaminations(self, MockIF):
        instance = _make_mock_model(n=20)
        MockIF.return_value = instance
        from mistakes_model import run_grid_search, CONTAMINATION_VALUES, N_FOLDS
        best_c, scores, best_score = run_grid_search(self.X, self.logger)
        expected_calls = len(CONTAMINATION_VALUES) * N_FOLDS
        self.assertEqual(instance.fit.call_count, expected_calls)

    @patch("mistakes_model.IsolationForest")
    def test_grid_search_returns_tuple_of_three(self, MockIF):
        MockIF.return_value = _make_mock_model(n=20)
        from mistakes_model import run_grid_search
        result = run_grid_search(self.X, self.logger)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    @patch("mistakes_model.IsolationForest")
    def test_grid_search_best_contamination_is_from_list(self, MockIF):
        MockIF.return_value = _make_mock_model(n=20)
        from mistakes_model import run_grid_search, CONTAMINATION_VALUES
        best_c, _, _ = run_grid_search(self.X, self.logger)
        self.assertIn(best_c, CONTAMINATION_VALUES)

    @patch("mistakes_model.IsolationForest")
    def test_grid_search_cv_scores_dict_length(self, MockIF):
        MockIF.return_value = _make_mock_model(n=20)
        from mistakes_model import run_grid_search, CONTAMINATION_VALUES
        _, scores, _ = run_grid_search(self.X, self.logger)
        self.assertEqual(len(scores), len(CONTAMINATION_VALUES))

    @patch("mistakes_model.IsolationForest")
    def test_grid_search_cv_scores_dict_keys_are_strings(self, MockIF):
        MockIF.return_value = _make_mock_model(n=20)
        from mistakes_model import run_grid_search
        _, scores, _ = run_grid_search(self.X, self.logger)
        for key in scores:
            self.assertIsInstance(key, str)

    @patch("mistakes_model.IsolationForest")
    def test_grid_search_best_score_consistent(self, MockIF):
        MockIF.return_value = _make_mock_model(n=20)
        from mistakes_model import run_grid_search
        best_c, scores, best_score = run_grid_search(self.X, self.logger)
        self.assertAlmostEqual(best_score, scores[str(best_c)], places=5)

    @patch("mistakes_model.IsolationForest")
    def test_grid_search_uses_correct_n_estimators(self, MockIF):
        MockIF.return_value = _make_mock_model(n=20)
        from mistakes_model import run_grid_search, N_ESTIMATORS
        run_grid_search(self.X, self.logger)
        for c in MockIF.call_args_list:
            self.assertEqual(c.kwargs.get("n_estimators"), N_ESTIMATORS)

    @patch("mistakes_model.IsolationForest")
    def test_grid_search_uses_correct_random_state(self, MockIF):
        MockIF.return_value = _make_mock_model(n=20)
        from mistakes_model import run_grid_search, RANDOM_STATE
        run_grid_search(self.X, self.logger)
        for c in MockIF.call_args_list:
            self.assertEqual(c.kwargs.get("random_state"), RANDOM_STATE)

    @patch("mistakes_model.IsolationForest")
    def test_grid_search_logs_each_contamination(self, MockIF):
        MockIF.return_value = _make_mock_model(n=20)
        from mistakes_model import run_grid_search, CONTAMINATION_VALUES
        handler = logging.handlers.MemoryHandler(capacity=100)
        self.logger.addHandler(handler)
        run_grid_search(self.X, self.logger)
        # Check log records contain one per contamination
        log_messages = [r.getMessage() for r in handler.buffer]
        for c in CONTAMINATION_VALUES:
            found = any(f"contamination={c:.2f}" in m for m in log_messages)
            self.assertTrue(found, f"No log for contamination={c}")
        self.logger.removeHandler(handler)


# Need logging.handlers for the log test
import logging.handlers

if __name__ == "__main__":
    unittest.main()
