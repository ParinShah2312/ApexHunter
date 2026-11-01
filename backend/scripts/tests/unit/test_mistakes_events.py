"""Tests for isolation_forest/mistakes_events.py: flags -> events, ranking and reasons."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from isolation_forest.mistakes_events import assign_events, build_events, location_label
from isolation_forest.mistakes_features import FEATURE_COLUMNS, INFO_COLUMNS


def _samples(times, laps, flagged, scores):
    """Scored samples with neutral features (no deviation from the field)."""
    n = len(times)
    df = pd.DataFrame({
        "SessionTime": pd.to_timedelta(times, unit="s"), "LapNumber": laps,
        "is_mistake": flagged, "anomaly_score": scores,
        "track_pos": np.arange(n) * 10.0, "X": np.arange(n, dtype=float), "Y": np.zeros(n), "Speed": np.full(n, 250.0),
    })
    for col in FEATURE_COLUMNS + INFO_COLUMNS:
        df[col] = 0.0
    df["field_brake_share"] = 0.5
    return df


class TestEvents(unittest.TestCase):

    def test_flags_merge_within_one_second_of_the_same_lap_and_short_runs_are_dropped(self) -> None:
        df = _samples(times=[0.0, 0.2, 0.6, 1.0, 1.5, 2.2, 2.5, 2.6, 2.9, 3.2],
                      laps=[2, 2, 2, 2, 2, 2, 2, 3, 3, 3],
                      flagged=[True, True, False, True, False, True, True, True, True, True],
                      scores=[-0.1] * 10)
        # 0.2 -> 1.0 is a 0.8 s gap, so 0.0-1.0 is one event (with a 0.5 s gap it was two pieces
        # too short to be events); 1.0 -> 2.2 is a 1.2 s gap, and 2.2-2.5 lasts only 0.3 s (no
        # event); 2.5 -> 2.6 crosses into lap 3, which starts a new event (0.6 s long)
        self.assertEqual(assign_events(df).tolist(), [0, 0, -1, 0, -1, -1, -1, 1, 1, 1])

    def test_sustained_deviation_ranks_first_with_its_reason(self) -> None:
        times = np.arange(12) * 0.2
        times[8:] += 0.6                                         # 1.2 s gap: two separate events
        flagged = [False, True, True, True, True, True, False, False, True, True, True, True]
        df = _samples(times, [5] * 12, flagged, scores=[0.1] + [-0.1] * 5 + [0.1] * 2 + [-0.05, -0.2, -0.05, -0.05])
        # Event 0 (0.8 s): lifted 80 % where the field is flat out. Event 1 (0.6 s): a bigger peak
        # score but less unusual overall; braking where only 10 % of the field brakes.
        df.loc[1:5, "throttle_diff"] = -16.0
        df.loc[1:5, "throttle_vs_field_pct"] = -80.0
        df.loc[8:11, "brake_mismatch"] = 0.9
        df.loc[8:11, "field_brake_share"] = 0.1
        df.loc[0, "throttle_diff"] = 1.0                         # some spread for every feature
        df.loc[0, "brake_mismatch"] = -0.1
        df["event_id"] = assign_events(df)

        events = build_events(df)

        self.assertEqual([e["rank"] for e in events], [1, 2])
        self.assertEqual((events[0]["lap"], events[0]["n_samples"]), (5, 5))
        self.assertEqual(events[0]["reason"], "Off the throttle: 80 % less than the field")
        self.assertEqual(events[1]["reason"],
                         "Braking where the field mostly doesn't (10 % of field samples brake here)")

    def test_location_names_the_nearest_corner_across_the_finish_line(self) -> None:
        corners = pd.DataFrame({"label": ["Turn 1", "Turn 2", "Turn 3"], "track_pos": [300.0, 1500.0, 4900.0]})
        labels = [location_label(p, corners, 5000.0) for p in (330.0, 150.0, 800.0, 20.0)]
        # 20 m into the lap is 120 m past Turn 3 (the lap wraps at 5000 m)
        self.assertEqual(labels, ["Turn 1", "Approaching Turn 1", "After Turn 1", "After Turn 3"])


if __name__ == "__main__":
    unittest.main()
