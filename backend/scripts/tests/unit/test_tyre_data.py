"""Unit tests for lstm/tyre_data.py: per-lap aggregation, training windows, stint rules and
dataset building. (Lap filtering is covered in test_tyre_lap_filtering.py.)"""

import logging
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from lstm.tyre_data import LAP_FEATURES, SEQUENCE_LENGTH, build_training_dataset, extract_stints

LAP_S = 10.0
LOG = logging.getLogger("test_tyre_data")
LOG.addHandler(logging.NullHandler())
LOG.propagate = False


def _laps(lap_numbers, compound="SOFT", stint=1) -> pd.DataFrame:
    """Clean laps (no FastF1 flag columns → nothing filtered except lap 1)."""
    return pd.DataFrame({
        "LapNumber": [float(n) for n in lap_numbers], "Stint": [float(stint)] * len(lap_numbers),
        "Compound": [compound] * len(lap_numbers), "TyreLife": [float(n + 2) for n in lap_numbers],
        "LapStartTime": [pd.Timedelta(seconds=(n - 1) * LAP_S) for n in lap_numbers],
        "Time": [pd.Timedelta(seconds=n * LAP_S) for n in lap_numbers],
        "LapTime": [pd.Timedelta(seconds=LAP_S + 0.1 * n) for n in lap_numbers],
    })


def _telemetry(last_lap: int, skip_lap: int = None) -> pd.DataFrame:
    """10 samples per lap. Lap n has speed 300 - n, throttle 50 + n, brake on in 2 of 10
    samples, RPM 10000 + n. Lap `skip_lap` gets only 2 samples (too sparse)."""
    rows = []
    for lap in range(1, last_lap + 1):
        n_samples = 2 if lap == skip_lap else 10
        for k in range(n_samples):
            rows.append({"SessionTime": pd.Timedelta(seconds=(lap - 1) * LAP_S + 0.5 + k * 0.9),
                         "Speed": 300.0 - lap, "Throttle": 50.0 + lap,
                         "Brake": 1.0 if k < 2 else 0.0, "RPM": 10000.0 + lap})
    return pd.DataFrame(rows)


class TestLapAggregation(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.stint = extract_stints(_laps(range(1, 11), compound="MEDIUM"), _telemetry(10), LOG)[0]
        cls.lf = pd.DataFrame(cls.stint["lap_features"])

    def test_per_lap_features(self) -> None:
        row = self.lf[self.lf["race_lap_number"] == 4.0].iloc[0]
        self.assertAlmostEqual(row["mean_speed"], 296.0)
        self.assertAlmostEqual(row["mean_throttle"], 54.0)
        self.assertAlmostEqual(row["mean_brake"], 0.2)          # fraction of the lap braking
        self.assertAlmostEqual(row["mean_rpm"], 10004.0)
        self.assertAlmostEqual(row["tyre_age"], 6.0)            # from FastF1 TyreLife
        self.assertEqual((row["is_soft"], row["is_medium"], row["is_hard"]), (0.0, 1.0, 0.0))
        self.assertAlmostEqual(row["lap_time_seconds"], 10.4)

    def test_windows_and_next_lap_targets(self) -> None:
        # laps 2..10 are kept (lap 1 is the start) → 9 laps → 9 - 5 = 4 windows
        self.assertEqual(self.stint["n_laps"], 9)
        self.assertEqual(len(self.stint["sequences"]), 4)
        first = self.stint["sequences"][0]
        self.assertEqual(first.shape, (SEQUENCE_LENGTH, len(LAP_FEATURES)))
        lap_col = LAP_FEATURES.index("race_lap_number")
        np.testing.assert_array_equal(first[:, lap_col], [2, 3, 4, 5, 6])
        self.assertAlmostEqual(self.stint["targets"][0], 300.0 - 7)  # mean speed of lap 7

    def test_stint_index_zero_based(self) -> None:
        self.assertEqual(self.stint["stint_index"], 0)


class TestStintRules(unittest.TestCase):

    def test_sparse_lap_is_skipped(self) -> None:
        dense = extract_stints(_laps(range(1, 12)), _telemetry(11), LOG)[0]
        sparse = extract_stints(_laps(range(1, 12)), _telemetry(11, skip_lap=11), LOG)[0]
        self.assertIn(11.0, dense["lap_features"]["race_lap_number"])
        self.assertNotIn(11.0, sparse["lap_features"]["race_lap_number"])
        self.assertEqual(sparse["n_laps"], dense["n_laps"] - 1)

    def test_stint_too_short_for_a_window_is_dropped(self) -> None:
        # laps 2..6 = 5 laps: enough for a stint, but a window needs 5 inputs + 1 target
        self.assertEqual(extract_stints(_laps(range(2, 7)), _telemetry(6), LOG), [])
        self.assertEqual(len(extract_stints(_laps(range(2, 8)), _telemetry(7), LOG)), 1)

    def test_each_stint_handled_separately(self) -> None:
        laps = pd.concat([_laps(range(2, 10), "SOFT", stint=1), _laps(range(10, 18), "HARD", stint=2)])
        stints = extract_stints(laps, _telemetry(17), LOG)
        self.assertEqual([s["stint_index"] for s in stints], [0, 1])
        self.assertEqual(stints[1]["lap_features"]["is_hard"][0], 1.0)

    def test_empty_laps(self) -> None:
        self.assertEqual(extract_stints(_laps([]), _telemetry(1), LOG), [])


class TestBuildTrainingDataset(unittest.TestCase):

    def test_no_race_files_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "No Race parquet files"):
            build_training_dataset(Path("/nonexistent"), ["2024"], LOG)


if __name__ == "__main__":
    unittest.main()
