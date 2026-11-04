"""Unit tests for strategy/pit_cost.py (ROADMAP §C step 2)."""

import sys
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from strategy.pit_cost import DEFAULT_PIT_COST_S, SC_VSC_FACTOR, pit_cost_for_lap, race_pit_cost, stop_costs


def _race() -> pd.DataFrame:
    """6 cars x 6 laps at 90 s. A stops on lap 3 (green), B on lap 5 under a safety car,
    C serves a drive-through on lap 2 (no new stint), D stops on lap 1."""
    rows = []
    for d in "ABCDEF":
        for n in range(1, 7):
            rows.append({"Driver": d, "LapNumber": n, "t": 90.0, "Stint": 1, "TrackStatus": "1",
                         "PitInTime": pd.NaT, "PitOutTime": pd.NaT})
    laps = pd.DataFrame(rows)

    def at(d, n):
        return (laps["Driver"] == d) & (laps["LapNumber"] == n)

    def stop(d, n, in_t, out_t, new_stint=True):
        laps.loc[at(d, n), ["t", "PitInTime"]] = [in_t, pd.Timedelta(seconds=1)]
        laps.loc[at(d, n + 1), ["t", "PitOutTime"]] = [out_t, pd.Timedelta(seconds=1)]
        if new_stint:
            laps.loc[(laps["Driver"] == d) & (laps["LapNumber"] > n), "Stint"] = 2

    stop("A", 3, 100.0, 102.0)
    stop("B", 5, 95.0, 97.0)
    laps.loc[laps["LapNumber"] == 5, "TrackStatus"] = "14"
    stop("C", 2, 105.0, 104.0, new_stint=False)
    stop("D", 1, 120.0, 110.0)
    laps["LapTime"] = pd.to_timedelta(laps.pop("t"), unit="s")
    return laps


class TestPitCost(unittest.TestCase):

    def test_stop_cost_is_in_and_out_lap_minus_the_cars_that_did_not_pit(self) -> None:
        costs = stop_costs(_race())
        # A: 202 - 180 = 22 s (green); B: 192 - 180 = 12 s (SC lap);
        # C's drive-through (same stint) and D's lap-1 stop are not tyre stops to measure
        self.assertEqual(costs[["Driver", "LapNumber", "kind"]].values.tolist(),
                         [["A", 3, "green"], ["B", 5, "sc_vsc"]])
        self.assertEqual(costs["cost_s"].tolist(), [22.0, 12.0])

    def test_race_cost_falls_back_to_default_and_scales_for_sc_vsc(self) -> None:
        cost = race_pit_cost(_race())          # one normal stop: too few for a race median
        self.assertEqual((cost["normal_s"], cost["n_normal_stops"]), (DEFAULT_PIT_COST_S, 1))
        self.assertAlmostEqual(pit_cost_for_lap(cost, "16"), DEFAULT_PIT_COST_S * SC_VSC_FACTOR)
        self.assertEqual(pit_cost_for_lap(cost, "1"), DEFAULT_PIT_COST_S)
        self.assertIsNone(pit_cost_for_lap(cost, "15"))   # red flag: tyres change for free


if __name__ == "__main__":
    unittest.main()
