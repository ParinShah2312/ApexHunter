"""Regression tests for LSTM lap filtering: out/in-laps, lap 1 and neutralised (SC/VSC/red)
laps must be excluded, windows must never bridge a removed lap, and prediction must not
overwrite the first lap's features."""

import logging
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from lstm.tyre_data import (
    LAP_FEATURES, SEQUENCE_LENGTH, _consecutive_runs, extract_stints, representative_lap_mask,
)
from lstm.tyre_model import DROPOUT_RATE, INPUT_SIZE, NUM_LAYERS, TyreCliffLSTM, predict_single_stint

LAP_SECONDS = 90.0


def _silent_logger() -> logging.Logger:
    lg = logging.getLogger("test_tyre_lap_filtering_silent")
    lg.handlers = [logging.NullHandler()]
    lg.setLevel(logging.CRITICAL)
    return lg


def _laps(lap_numbers, track_status=None, stint=1, **overrides) -> pd.DataFrame:
    """FastF1-style laps with every filtering column present and 'clean' defaults."""
    n = len(lap_numbers)
    df = pd.DataFrame({
        "LapNumber": [float(x) for x in lap_numbers],
        "Stint": [float(stint)] * n,
        "Compound": ["MEDIUM"] * n,
        "TyreLife": [float(i + 1) for i in range(n)],
        "LapStartTime": [pd.Timedelta(seconds=(x - 1) * LAP_SECONDS) for x in lap_numbers],
        "Time": [pd.Timedelta(seconds=x * LAP_SECONDS) for x in lap_numbers],
        "LapTime": [pd.Timedelta(seconds=LAP_SECONDS)] * n,
        "PitOutTime": pd.Series([pd.NaT] * n, dtype="timedelta64[ns]"),
        "PitInTime": pd.Series([pd.NaT] * n, dtype="timedelta64[ns]"),
        "TrackStatus": track_status or ["1"] * n,
        "Deleted": [False] * n,
        "FastF1Generated": [False] * n,
        "IsAccurate": [True] * n,
    })
    for col, values in overrides.items():
        df[col] = values
    return df


def _telemetry(last_lap: int) -> pd.DataFrame:
    """Telemetry at 10 Hz covering laps 1..last_lap, speed slowly decreasing."""
    t = np.arange(0.0, last_lap * LAP_SECONDS, 0.1)
    return pd.DataFrame({
        "SessionTime": pd.to_timedelta(t, unit="s"),
        "Speed": (250.0 - t / 100.0).astype("float32"),
        "Throttle": np.full(len(t), 70.0, dtype="float32"),
        "Brake": (np.arange(len(t)) % 5 == 0).astype("float32"),
        "RPM": np.full(len(t), 10000.0, dtype="float32"),
    })


class TestRepresentativeLapMask(unittest.TestCase):

    def test_excludes_each_non_representative_lap_type(self) -> None:
        laps = _laps(range(1, 12), track_status=["1", "1", "1", "1", "4", "6", "671", "5", "12", "1", "1"])
        laps.loc[2, "PitOutTime"] = pd.Timedelta(seconds=100)   # lap 3: out-lap
        laps.loc[3, "PitInTime"] = pd.Timedelta(seconds=200)    # lap 4: in-lap
        laps.loc[9, "IsAccurate"] = False                       # lap 10: inaccurate timing
        laps.loc[10, "LapTime"] = pd.NaT                        # lap 11: no lap time
        mask = representative_lap_mask(laps)
        kept = laps.loc[mask, "LapNumber"].astype(int).tolist()
        # lap 1 start, 3 out, 4 in, 5 SC, 6 VSC, 7 VSC ending, 8 red, 10 inaccurate, 11 no time
        self.assertEqual(kept, [2, 9])  # lap 9 is a yellow flag ("12") and is deliberately kept

    def test_deleted_and_generated_laps_excluded(self) -> None:
        laps = _laps([2, 3, 4], Deleted=[False, True, False], FastF1Generated=[False, False, True])
        self.assertEqual(representative_lap_mask(laps).tolist(), [True, False, False])

    def test_missing_columns_are_ignored(self) -> None:
        laps = pd.DataFrame({"LapNumber": [1.0, 2.0, 3.0]})
        self.assertEqual(representative_lap_mask(laps).tolist(), [False, True, True])


class TestConsecutiveRuns(unittest.TestCase):

    def test_splits_on_gaps(self) -> None:
        runs = _consecutive_runs(pd.Series([2, 3, 4, 7, 8, 10]))
        self.assertEqual([r.tolist() for r in runs], [[0, 1, 2], [3, 4], [5]])


class TestExtractStintsFiltering(unittest.TestCase):

    def test_windows_never_bridge_a_safety_car_lap(self) -> None:
        # Laps 1..16; lap 1 (start) and lap 9 (safety car) are removed → runs 2-8 and 10-16
        status = ["1"] * 16
        status[8] = "4"
        stints = extract_stints(_laps(range(1, 17), track_status=status), _telemetry(16), _silent_logger())
        self.assertEqual(len(stints), 1)
        stint = stints[0]
        self.assertNotIn(9.0, stint["lap_features"]["race_lap_number"])
        self.assertNotIn(1.0, stint["lap_features"]["race_lap_number"])
        # each run has 7 laps → 7 - 5 = 2 windows per run
        self.assertEqual(len(stint["sequences"]), 4)
        lap_col = LAP_FEATURES.index("race_lap_number")
        for seq in stint["sequences"]:
            np.testing.assert_array_equal(np.diff(seq[:, lap_col]), np.ones(SEQUENCE_LENGTH - 1))

    def test_out_lap_excluded_from_stint(self) -> None:
        laps = _laps(range(20, 31), stint=2)
        laps.loc[0, "PitOutTime"] = pd.Timedelta(seconds=1)
        stints = extract_stints(laps, _telemetry(30), _silent_logger())
        self.assertEqual(stints[0]["lap_features"]["race_lap_number"][0], 21.0)


class TestPredictionUsesRealFirstLap(unittest.TestCase):

    def test_first_lap_kept_and_lap_numbers_reported(self) -> None:
        stints = extract_stints(_laps(range(1, 12)), _telemetry(11), _silent_logger())
        stint = stints[0]
        model = TyreCliffLSTM(INPUT_SIZE, 16, NUM_LAYERS, DROPOUT_RATE)
        scaler = StandardScaler().fit(np.random.RandomState(0).rand(30, INPUT_SIZE))
        result = predict_single_stint(stint, model, scaler, 230.0, 10.0, SEQUENCE_LENGTH)
        self.assertEqual(result["lap_numbers"], list(range(2, 12)))
        self.assertEqual(len(result["predicted_laps"]), len(result["lap_numbers"]))
        # The input features must not have been mutated (old code copied lap 2 over lap 1)
        speeds = stint["lap_features"]["mean_speed"]
        self.assertNotAlmostEqual(speeds[0], speeds[1], places=3)


if __name__ == "__main__":
    unittest.main()
