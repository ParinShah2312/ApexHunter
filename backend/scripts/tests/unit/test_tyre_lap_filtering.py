"""Regression tests for LSTM lap filtering: out/in-laps, lap 1 and neutralised (SC/VSC/red)
laps must be excluded, windows must never bridge a removed lap, and predictions exist only where a
full 5-lap window does."""

import copy
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
    window_positions,
)
from lstm.tyre_model import DROPOUT_RATE, INPUT_SIZE, NUM_LAYERS, TyrePaceRNN, sequence_predictor
from strategy.strategy_check import predicted_changes

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


class TestWindowPositions(unittest.TestCase):

    def test_only_laps_with_five_consecutive_predecessors(self) -> None:
        laps = pd.Series([2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16])   # lap 9 removed
        # Laps 7, 8 (positions 5, 6) and 15, 16 (positions 12, 13)
        self.assertEqual(window_positions(laps, 5), [5, 6, 12, 13])


def _predict():
    """A real (untrained) recurrent model behind the same predictor the strategy code uses."""
    scaler = StandardScaler().fit(np.random.RandomState(0).rand(30, INPUT_SIZE))
    return sequence_predictor(TyrePaceRNN(INPUT_SIZE, 16, NUM_LAYERS, DROPOUT_RATE), scaler, 0.0, 1.0)


class TestPredictionWindows(unittest.TestCase):
    """Regression for #20/#21: predictions used padded copies of the first lap, bridged removed
    laps and dropped stints too short for a training window. Now checked on
    strategy_check.predicted_changes, which feeds the pit window and the dashboard."""

    def test_predictions_only_where_a_real_window_exists(self) -> None:
        status = ["1"] * 16
        status[8] = "4"                                     # lap 9 under safety car
        stints = extract_stints(_laps(range(1, 17), track_status=status), _telemetry(16), _silent_logger())
        before = copy.deepcopy(stints[0]["lap_features"])
        changes = predicted_changes(stints, _predict())
        # laps 7-8 and 15-16 have 5 clean laps before them; so do 9 (the safety-car lap) and 17 (the
        # lap after the stint): whether a lap turns out clean is only known after it (#52)
        self.assertEqual(sorted(changes), [7, 8, 9, 15, 16, 17])
        self.assertEqual(stints[0]["lap_features"], before)  # inputs not modified by prediction

    def test_short_stint_gets_no_prediction(self) -> None:
        stints = extract_stints(_laps(range(2, 6)), _telemetry(5), _silent_logger())
        self.assertEqual(stints[0]["n_laps"], 4)                  # fewer than 5 clean laps
        self.assertEqual(predicted_changes(stints, _predict()), {})


if __name__ == "__main__":
    unittest.main()
