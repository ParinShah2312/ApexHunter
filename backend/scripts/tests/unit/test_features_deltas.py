"""Unit tests for speed_delta and gear_change clipping."""

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


class TestFeaturesDeltas(unittest.TestCase):

    def setUp(self):
        self.df = _make_df(20)

    def test_speed_delta_clipped(self):
        df = self.df.copy()
        df.loc[5, "Speed"] = 0.0
        df.loc[6, "Speed"] = 380.0
        result = engineer_features(df)
        self.assertAlmostEqual(float(result.loc[6, "speed_delta"]), 50.0, places=1)

    def test_gear_change_clipped(self):
        df = self.df.copy()
        df.loc[5, "nGear"] = 1.0
        df.loc[6, "nGear"] = 8.0
        result = engineer_features(df)
        self.assertAlmostEqual(float(result.loc[6, "gear_change"]), 4.0, places=1)

    def test_single_row_deltas_zero(self):
        df = _make_df(1)
        result = engineer_features(df)
        self.assertAlmostEqual(float(result["speed_delta"].iloc[0]), 0.0)
        self.assertAlmostEqual(float(result["gear_change"].iloc[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
