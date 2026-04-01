"""Unit tests for brake_throttle_overlap edge cases."""

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


class TestFeaturesFlags(unittest.TestCase):

    def test_overlap_binary(self):
        df = _make_df(20)
        result = engineer_features(df)
        unique_vals = set(result["brake_throttle_overlap"].unique())
        self.assertTrue(unique_vals.issubset({0.0, 1.0}),
                        f"Expected only 0.0 and 1.0, got {unique_vals}")

    def test_overlap_below_noise_threshold(self):
        n = 10
        df = _make_df(n, Brake=np.full(n, 3.0, dtype="float32"),
                      Throttle=np.full(n, 3.0, dtype="float32"))
        result = engineer_features(df)
        self.assertTrue((result["brake_throttle_overlap"] == 0.0).all())

    def test_overlap_above_noise_threshold(self):
        n = 10
        df = _make_df(n, Brake=np.full(n, 6.0, dtype="float32"),
                      Throttle=np.full(n, 6.0, dtype="float32"))
        result = engineer_features(df)
        self.assertTrue((result["brake_throttle_overlap"] == 1.0).all())

    def test_overlap_one_side_zero(self):
        n = 10
        df = _make_df(n, Brake=np.full(n, 50.0, dtype="float32"),
                      Throttle=np.zeros(n, dtype="float32"))
        result = engineer_features(df)
        self.assertTrue((result["brake_throttle_overlap"] == 0.0).all())


if __name__ == "__main__":
    unittest.main()
