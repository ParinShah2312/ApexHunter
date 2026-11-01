"""Tests for isolation_forest/mistakes_timeloss.py (time lost per event, saving label) and the
per-mistake scoring in isolation_forest/mistakes_eval.py."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from isolation_forest.mistakes_eval import mistake_metrics
from isolation_forest.mistakes_timeloss import add_time_lost, build_lap_timing, time_lost

LAP_M = 1000.0


def _laps(n_laps=7, slow_lap=None, coast_laps=()):
    """Laps of a 1 km track sampled every 10 m at ~180 km/h. `slow_lap` drives 400-500 m at
    144 km/h (0.5 s lost); `coast_laps` coast (no pedals) from 600 to 700 m."""
    rows, t = [], 0.0
    for lap in range(1, n_laps + 1):
        for pos in np.arange(0.0, LAP_M, 10.0):
            speed = 144.0 if lap == slow_lap and 400 <= pos < 500 else 180.0 + 0.2 * lap
            coast = lap in coast_laps and 600 <= pos < 700
            rows.append((lap, t, pos, speed, 0.0 if coast else 100.0))
            t += 10.0 / (speed / 3.6)
    df = pd.DataFrame(rows, columns=["LapNumber", "t", "track_pos", "Speed", "Throttle"])
    df["SessionTime"] = pd.to_timedelta(df.pop("t"), unit="s")
    df["Brake"] = 0.0
    return df


class TestTimeLost(unittest.TestCase):

    def test_slow_stretch_costs_its_extra_time_and_a_normal_lap_does_not(self) -> None:
        timing = build_lap_timing(_laps(slow_lap=4), LAP_M)
        stint_of = {lap: 1 for lap in timing}

        slow = time_lost(timing, stint_of, 4, 420.0, 480.0, LAP_M)
        normal = time_lost(timing, stint_of, 2, 420.0, 480.0, LAP_M)

        # 100 m at 40 m/s instead of 50 m/s = 0.5 s
        self.assertAlmostEqual(slow["time_lost_s"], 0.5, delta=0.05)
        self.assertTrue(slow["costly"])
        self.assertFalse(normal["costly"])

    def test_coasting_repeated_on_most_laps_is_labelled_saving_and_ranked_last(self) -> None:
        df = _laps(slow_lap=6, coast_laps=(1, 2, 3, 4, 5, 7))
        timing = build_lap_timing(df, LAP_M)
        df["event_id"] = -1
        df.loc[(df["LapNumber"] == 4) & df["track_pos"].between(600, 690), "event_id"] = 0   # coasting
        df.loc[(df["LapNumber"] == 6) & df["track_pos"].between(400, 490), "event_id"] = 1   # slow, no coasting
        events = [{"event_id": 0, "lap": 4, "start_pos_m": 600.0, "end_pos_m": 690.0, "track_pos_m": 650.0},
                  {"event_id": 1, "lap": 6, "start_pos_m": 400.0, "end_pos_m": 490.0, "track_pos_m": 450.0}]

        ranked = add_time_lost(events, df, timing, {lap: 1 for lap in timing}, LAP_M)

        self.assertEqual([e["event_id"] for e in ranked], [1, 0])
        self.assertTrue(ranked[1]["likely_saving"])
        self.assertIn("5 of 6 other laps", ranked[1]["saving_note"])
        self.assertFalse(ranked[0]["likely_saving"])


class TestPerMistakeScore(unittest.TestCase):

    def test_a_mistake_counts_once_however_many_of_its_samples_are_flagged(self) -> None:
        t = np.arange(20) * 0.2
        frame = pd.DataFrame({"SessionTime": pd.to_timedelta(t, unit="s"), "LapNumber": 1})
        mistake_ids = np.full(20, -1)
        mistake_ids[2:6], mistake_ids[12:15] = 0, 1           # two planted mistakes
        kinds = np.where(mistake_ids >= 0, "lockup", "")
        flagged = np.zeros(20, dtype=bool)
        flagged[3:7] = True                                   # one event over part of mistake 0
        flagged[16:20] = True                                 # one event where nothing was planted

        m = mistake_metrics(flagged, frame, mistake_ids, kinds)

        self.assertEqual((m["precision"], m["recall"]), (0.5, 0.5))
        self.assertEqual(m["counts"], {"planted": 2, "found": 1, "events": 2, "correct_events": 1})


if __name__ == "__main__":
    unittest.main()
