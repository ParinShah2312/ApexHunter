"""Unit tests for feature contract: column names, dtypes, row count, no mutation."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from mistakes_features import FEATURE_COLUMNS, engineer_features


def _make_df(n, **overrides):
    defaults = {
        "Driver": ["44"] * n,
        "Speed": np.linspace(100, 300, n).astype("float32"),
        "Throttle": np.tile([0.0, 80.0], (n + 1) // 2)[:n].astype("float32"),
        "Brake": np.tile([0.0, 60.0], (n + 1) // 2)[:n].astype("float32"),
        "RPM": np.full(n, 8000.0, dtype="float32"),
        "nGear": np.full(n, 4.0, dtype="float32"),
        "X": np.zeros(n, dtype="float32"),
        "Y": np.zeros(n, dtype="float32"),
        "Time": pd.to_timedelta(np.arange(n), unit="s"),
        "SessionTime": pd.to_timedelta(np.arange(n), unit="s"),
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


class TestFeaturesContract(unittest.TestCase):

    def setUp(self):
        self.df = _make_df(20)

    def test_output_columns(self):
        result = engineer_features(self.df)
        self.assertEqual(list(result.columns), FEATURE_COLUMNS)

    def test_output_dtypes(self):
        result = engineer_features(self.df)
        for col in result.columns:
            self.assertEqual(result[col].dtype, np.dtype("float32"),
                             f"Column {col} is {result[col].dtype}")

    def test_row_count_preserved(self):
        result = engineer_features(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_no_mutation(self):
        original_columns = list(self.df.columns)
        original_dtypes = dict(self.df.dtypes)
        _ = engineer_features(self.df)
        self.assertEqual(list(self.df.columns), original_columns)
        self.assertEqual(dict(self.df.dtypes), original_dtypes)

    def test_single_row_returns_valid_output(self):
        df = _make_df(1)
        result = engineer_features(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(list(result.columns), FEATURE_COLUMNS)
        self.assertFalse(result.isna().any().any())


if __name__ == "__main__":
    unittest.main()
