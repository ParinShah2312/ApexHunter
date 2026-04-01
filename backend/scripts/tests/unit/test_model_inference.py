"""Unit tests for run_inference with MOCKED IsolationForest."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from mistakes_model import run_inference


class TestModelInference(unittest.TestCase):

    def setUp(self):
        self.n = 40
        self.features_df = pd.DataFrame(
            np.random.rand(self.n, 7).astype("float32"),
            columns=[f"f{i}" for i in range(7)],
        )
        self.mock_model = MagicMock()
        self.mock_model.decision_function.return_value = np.random.uniform(
            -0.3, 0.3, self.n
        )
        self.mock_model.predict.return_value = np.ones(self.n, dtype=int)

    def test_run_inference_calls_decision_function(self):
        run_inference(self.mock_model, self.features_df)
        self.mock_model.decision_function.assert_called_once()

    def test_run_inference_calls_predict(self):
        run_inference(self.mock_model, self.features_df)
        self.mock_model.predict.assert_called_once()

    def test_run_inference_output_types(self):
        raw_scores, predictions = run_inference(self.mock_model, self.features_df)
        self.assertIsInstance(raw_scores, np.ndarray)
        self.assertIsInstance(predictions, np.ndarray)

    def test_run_inference_output_shapes_match_input(self):
        raw_scores, predictions = run_inference(self.mock_model, self.features_df)
        self.assertEqual(raw_scores.shape, (self.n,))
        self.assertEqual(predictions.shape, (self.n,))

    def test_run_inference_passes_numpy_array(self):
        run_inference(self.mock_model, self.features_df)
        arg = self.mock_model.decision_function.call_args[0][0]
        self.assertIsInstance(arg, np.ndarray)


if __name__ == "__main__":
    unittest.main()
