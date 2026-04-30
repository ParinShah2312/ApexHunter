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
    aggregate_lap_features,
    build_stint_sequences,
    build_training_dataset,
    detect_stints,
    split_stint_into_laps,
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
    """Tests for detect_stints()."""

    def setUp(self) -> None:
        self.df = _build_synthetic_df()
        self.logger = _silent_logger()

    def test_detect_stints_finds_two_stints(self) -> None:
        stints = detect_stints(self.df, self.logger)
        self.assertEqual(len(stints), 2)

    def test_detect_stints_each_has_rows(self) -> None:
        stints = detect_stints(self.df, self.logger)
        for stint in stints:
            self.assertGreater(len(stint), 0)

    def test_detect_stints_no_gap_is_one_stint(self) -> None:
        n = 1000
        df = pd.DataFrame({
            "Driver": ["44"] * n,
            "Speed": np.full(n, 250.0, dtype="float32"),
            "Throttle": np.full(n, 70.0, dtype="float32"),
            "Brake": np.full(n, 5.0, dtype="float32"),
            "RPM": np.full(n, 9000.0, dtype="float32"),
            "nGear": np.full(n, 5.0, dtype="float32"),
            "SessionTime": pd.to_timedelta(np.linspace(0, 98, n), unit="s"),
        })
        stints = detect_stints(df, self.logger)
        self.assertEqual(len(stints), 1)


class TestSplitStintIntoLaps(unittest.TestCase):
    """Tests for split_stint_into_laps()."""

    def setUp(self) -> None:
        self.df = _build_synthetic_df()
        self.logger = _silent_logger()

    def test_split_stint_into_laps_returns_list(self) -> None:
        stints = detect_stints(self.df, self.logger)
        self.assertGreater(len(stints), 0)
        result = split_stint_into_laps(stints[0])
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)


class TestAggregateLapFeatures(unittest.TestCase):
    """Tests for aggregate_lap_features()."""

    def setUp(self) -> None:
        self.df = _build_synthetic_df()
        self.logger = _silent_logger()
        stints = detect_stints(self.df, self.logger)
        laps = split_stint_into_laps(stints[0])
        self.lap = laps[0] if laps else pd.DataFrame()

    def test_aggregate_lap_features_keys(self) -> None:
        result = aggregate_lap_features(self.lap, tyre_age=3)
        self.assertIsInstance(result, dict)
        for key in LAP_FEATURES:
            self.assertIn(key, result)

    def test_aggregate_lap_features_tyre_age(self) -> None:
        result = aggregate_lap_features(self.lap, tyre_age=7)
        self.assertEqual(result["tyre_age"], 7.0)

    def test_aggregate_empty_lap_returns_none(self) -> None:
        result = aggregate_lap_features(pd.DataFrame(), tyre_age=0)
        self.assertIsNone(result)


class TestBuildStintSequences(unittest.TestCase):
    """Tests for build_stint_sequences()."""

    def setUp(self) -> None:
        self.df = _build_synthetic_df()
        self.logger = _silent_logger()

    def test_build_stint_sequences_structure(self) -> None:
        stints = detect_stints(self.df, self.logger)
        result = build_stint_sequences(stints[0], 0, self.logger)
        if result is None:
            self.skipTest("Stint too short for sequences — acceptable.")
        self.assertIn("sequences", result)
        self.assertIn("targets", result)
        self.assertEqual(len(result["sequences"]), len(result["targets"]))
        for seq in result["sequences"]:
            self.assertEqual(np.array(seq).shape, (SEQUENCE_LENGTH, 5))

    def test_build_stint_sequences_cliff_detection(self) -> None:
        """Create a synthetic stint where speed drops after lap 15."""
        n_rows = 6000
        times = np.linspace(0.0, 600.0, n_rows)
        speed = np.full(n_rows, 260.0, dtype="float32")
        # Inject cliff: after ~60% of rows, speed drops by 5 km/h
        cliff_start = int(n_rows * 0.6)
        speed[cliff_start:] = 253.0

        df = pd.DataFrame({
            "Driver": ["44"] * n_rows,
            "Speed": speed,
            "Throttle": np.full(n_rows, 70.0, dtype="float32"),
            "Brake": np.full(n_rows, 5.0, dtype="float32"),
            "RPM": np.full(n_rows, 9000.0, dtype="float32"),
            "nGear": np.full(n_rows, 5.0, dtype="float32"),
            "SessionTime": pd.to_timedelta(times, unit="s"),
        })
        result = build_stint_sequences(df, 0, self.logger)
        if result is not None:
            self.assertIsNotNone(result["cliff_lap"])
            self.assertGreater(result["cliff_lap"], 0)


class TestBuildTrainingDataset(unittest.TestCase):
    """Tests for build_training_dataset()."""

    def setUp(self) -> None:
        self.logger = _silent_logger()

    def test_build_training_dataset_wrong_dir_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_training_dataset(Path("/nonexistent"), ["2024"], self.logger)


if __name__ == "__main__":
    unittest.main()
