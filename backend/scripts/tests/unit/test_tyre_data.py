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


class TestConstants(unittest.TestCase):
    """Tests for module-level constants and LAP_FEATURES contract."""

    def test_sequence_length_positive(self) -> None:
        self.assertGreater(SEQUENCE_LENGTH, 0)

    def test_lap_features_is_non_empty_list(self) -> None:
        self.assertIsInstance(LAP_FEATURES, list)
        self.assertGreater(len(LAP_FEATURES), 0)

    def test_lap_features_are_strings(self) -> None:
        for feat in LAP_FEATURES:
            self.assertIsInstance(feat, str)

    def test_lap_features_no_duplicates(self) -> None:
        self.assertEqual(len(LAP_FEATURES), len(set(LAP_FEATURES)))


class TestExtractStintsEdgeCases(unittest.TestCase):
    """Tests for extract_stints with edge-case inputs (no FastF1 dependency)."""

    def setUp(self) -> None:
        self.logger = _silent_logger()

    def test_empty_driver_laps_returns_empty(self) -> None:
        """extract_stints should return [] when driver_laps is empty."""
        driver_laps = pd.DataFrame(columns=["LapNumber", "Stint", "LapStartTime", "Time", "Compound", "TyreLife", "LapTime"])
        telemetry = pd.DataFrame(columns=["SessionTime", "Speed", "Throttle", "Brake", "RPM"])
        result = extract_stints(driver_laps, telemetry, self.logger)
        self.assertEqual(result, [])

    def test_insufficient_laps_returns_empty(self) -> None:
        """extract_stints should return [] when stint has fewer laps than MIN_STINT_LAPS."""
        from tyre_data import MIN_STINT_LAPS
        # Create 1 lap (below threshold)
        driver_laps = pd.DataFrame({
            "LapNumber": [1],
            "Stint": [1],
            "LapStartTime": [pd.Timedelta(seconds=0)],
            "Time": [pd.Timedelta(seconds=90)],
            "Compound": ["SOFT"],
            "TyreLife": [1.0],
            "LapTime": [pd.Timedelta(seconds=90)],
        })
        telemetry = pd.DataFrame({
            "SessionTime": pd.to_timedelta(np.linspace(0, 90, 100), unit="s"),
            "Speed": np.full(100, 200.0),
            "Throttle": np.full(100, 70.0),
            "Brake": np.full(100, 5.0),
            "RPM": np.full(100, 9000.0),
        })
        result = extract_stints(driver_laps, telemetry, self.logger)
        self.assertEqual(result, [])

    def test_telemetry_too_sparse_skips_lap(self) -> None:
        """Laps with < 5 telemetry rows should be skipped."""
        driver_laps = pd.DataFrame({
            "LapNumber": [1, 2, 3, 4],
            "Stint": [1, 1, 1, 1],
            "LapStartTime": [pd.Timedelta(seconds=i * 90) for i in range(4)],
            "Time": [pd.Timedelta(seconds=(i + 1) * 90) for i in range(4)],
            "Compound": ["MEDIUM"] * 4,
            "TyreLife": [1.0, 2.0, 3.0, 4.0],
            "LapTime": [pd.Timedelta(seconds=90)] * 4,
        })
        # Only 2 telemetry rows total — every lap will have < 5 rows
        telemetry = pd.DataFrame({
            "SessionTime": pd.to_timedelta([10, 100], unit="s"),
            "Speed": [200.0, 210.0],
            "Throttle": [70.0, 75.0],
            "Brake": [5.0, 3.0],
            "RPM": [9000.0, 9500.0],
        })
        result = extract_stints(driver_laps, telemetry, self.logger)
        self.assertEqual(result, [])


class TestBuildTrainingDataset(unittest.TestCase):
    """Tests for build_training_dataset()."""

    def setUp(self) -> None:
        self.logger = _silent_logger()

    def test_build_training_dataset_wrong_dir_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_training_dataset(Path("/nonexistent"), ["2024"], self.logger)


if __name__ == "__main__":
    unittest.main()
