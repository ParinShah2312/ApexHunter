"""Unit tests for feature normalization: speed, throttle, brake, rpm."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from mistakes_features import engineer_features


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


class TestFeaturesNormalization(unittest.TestCase):

    def setUp(self):
        self.df = _make_df(20)

    def test_speed_normalized_range(self):
        result = engineer_features(self.df)
        self.assertTrue((result["speed_normalized"] >= 0.0).all())
        self.assertTrue((result["speed_normalized"] <= 1.0).all())

    def test_throttle_range(self):
        result = engineer_features(self.df)
        self.assertTrue((result["throttle_intensity"] >= 0.0).all())
        self.assertTrue((result["throttle_intensity"] <= 1.0).all())

    def test_brake_range(self):
        result = engineer_features(self.df)
        self.assertTrue((result["brake_intensity"] >= 0.0).all())
        self.assertTrue((result["brake_intensity"] <= 1.0).all())

    def test_all_zero_inputs(self):
        df = _make_df(10, Speed=np.zeros(10, dtype="float32"),
                      RPM=np.zeros(10, dtype="float32"),
                      Throttle=np.zeros(10, dtype="float32"),
                      Brake=np.zeros(10, dtype="float32"),
                      nGear=np.zeros(10, dtype="float32"))
        result = engineer_features(df)
        self.assertFalse(result.isna().any().any())
        for col in result.columns:
            self.assertTrue((result[col] == 0.0).all(),
                            f"Column {col} has non-zero values with all-zero inputs")

    def test_all_max_inputs(self):
        n = 10
        df = _make_df(n, Speed=np.full(n, 380.0, dtype="float32"),
                      RPM=np.full(n, 15000.0, dtype="float32"),
                      Throttle=np.full(n, 100.0, dtype="float32"),
                      Brake=np.full(n, 100.0, dtype="float32"))
        result = engineer_features(df)
        self.assertTrue((result["speed_normalized"] == 1.0).all())
        self.assertTrue((result["rpm_normalized"] == 1.0).all())
        self.assertTrue((result["throttle_intensity"] == 1.0).all())
        self.assertTrue((result["brake_intensity"] == 1.0).all())

    def test_features_independent_of_xy(self):
        df_a = _make_df(10, X=np.zeros(10, dtype="float32"),
                        Y=np.zeros(10, dtype="float32"))
        df_b = _make_df(10, X=np.full(10, 999.0, dtype="float32"),
                        Y=np.full(10, 999.0, dtype="float32"))
        result_a = engineer_features(df_a)
        result_b = engineer_features(df_b)
        pd.testing.assert_frame_equal(result_a, result_b)


if __name__ == "__main__":
    unittest.main()
