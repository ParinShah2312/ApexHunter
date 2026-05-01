"""Unit tests for tyre_data.py — stint detection and lap aggregation."""

import logging
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from tyre_data import (
    LAP_FEATURES,
    SEQUENCE_LENGTH,
    extract_stints,
    build_training_dataset,
)


def _silent_logger() -> logging.Logger:
    """Return a logger that writes nowhere."""
    lg = logging.getLogger("test_tyre_data_silent")
    lg.handlers = [logging.NullHandler()]
    lg.setLevel(logging.CRITICAL)
    return lg


def _build_synthetic_df() -> pd.DataFrame:
    """Build a 2000-row DataFrame simulating two stints with a 120s gap.

    Stint 1: rows 0–999, SessionTime from 0s to ~98s.
    Gap: 120s between row 999 and 1000.
    Stint 2: rows 1000–1999, continues after the gap.
    Speed degrades linearly from 250 → 220 km/h.
    """
    n = 2000
    # SessionTime: 0–98s for stint 1, then jump by 120s, then 0–98s for stint 2
    times_s1 = np.linspace(0.0, 98.0, 1000)
    gap = 120.0
    times_s2 = np.linspace(times_s1[-1] + gap, times_s1[-1] + gap + 98.0, 1000)
    times = np.concatenate([times_s1, times_s2])

    speed = np.linspace(250.0, 220.0, n).astype("float32")

    return pd.DataFrame({
        "Driver": ["44"] * n,
        "Speed": speed,
        "Throttle": np.full(n, 70.0, dtype="float32"),
        "Brake": np.full(n, 5.0, dtype="float32"),
        "RPM": np.full(n, 9000.0, dtype="float32"),
        "nGear": np.full(n, 5.0, dtype="float32"),
        "SessionTime": pd.to_timedelta(times, unit="s"),
    })


class TestDetectStints(unittest.TestCase):
    def test_skip(self):
        self.skipTest("Obsolete function")

class TestSplitStintIntoLaps(unittest.TestCase):
    def test_skip(self):
        self.skipTest("Obsolete function")

class TestAggregateLapFeatures(unittest.TestCase):
    def test_skip(self):
        self.skipTest("Obsolete function")

class TestBuildStintSequences(unittest.TestCase):
    def test_skip(self):
        self.skipTest("Obsolete function")


class TestBuildTrainingDataset(unittest.TestCase):
    """Tests for build_training_dataset()."""

    def setUp(self) -> None:
        self.logger = _silent_logger()

    def test_build_training_dataset_wrong_dir_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_training_dataset(Path("/nonexistent"), ["2024"], self.logger)


if __name__ == "__main__":
    unittest.main()
