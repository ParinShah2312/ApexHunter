"""Unit tests for strategy/pit_window.py (ROADMAP §C step 4)."""

import sys
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from strategy.pit_window import FUEL_GAIN_S_PER_LAP, driver_strategy, observed_changes, wear_rate


def _driver(status=None) -> pd.DataFrame:
    """One driver, 30 laps: new softs laps 1-14 (stop on lap 14), new hards laps 15-30."""
    laps = pd.DataFrame({"LapNumber": range(1, 31)})
    laps["Stint"] = (laps["LapNumber"] > 14).astype(int) + 1
    laps["Compound"] = laps["Stint"].map({1: "SOFT", 2: "HARD"})
    laps["TyreLife"] = laps["LapNumber"].where(laps["Stint"] == 1, laps["LapNumber"] - 14).astype(float)
    laps["TrackStatus"] = [(status or {}).get(n, "1") for n in laps["LapNumber"]]
    return laps


class TestPitWindow(unittest.TestCase):

    def test_window_around_the_balanced_stop_with_sc_and_red_flag_laps(self) -> None:
        # wear 0.1 s per lap of age on both sets; normal stop 20 s, SC/VSC stop 14 s
        known = {j: 0.1 - FUEL_GAIN_S_PER_LAP for j in range(1, 31)}
        laps = _driver(status={10: "14", 13: "15"})
        st = driver_strategy(laps, known, {"normal_s": 20.0, "sc_vsc_s": 14.0})[0]
        # best = stop on lap 15 (both sets 15 laps old at the end); window = best +-2 = laps 13-17
        # (user, 2026-11-04). Laps 15-17 come after the real stop (stayed out); 18+ are overdue.
        # Lap 10 is 2.5 s off the best but the SC saves 6 s, so stopping there is best; lap 13 is a
        # red flag (no advice).
        self.assertEqual(st["window"], [10, 14, 15, 16, 17])
        self.assertEqual((st["best_lap"], st["sc_vsc_calls"], st["hit"]), (15, [10], True))
        self.assertAlmostEqual(st["random_hit_chance"], 5 / 28)     # laps 1-29 without lap 13

    def test_wear_is_the_stint_average_not_the_last_five_laps(self) -> None:
        """User, 2026-11-04: averaging the last 5 changes made the wear jump with lap-to-lap noise."""
        known = {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}                    # previous stint
        known.update({j: (0.4 if j % 2 == 0 else -0.2) for j in range(6, 14)})   # stint from lap 6
        # whole stint (laps 6-13): mean 0.1; the last five alone (9-13) would give 0.04
        self.assertAlmostEqual(wear_rate(known, 13, stint_start=6), 0.1 + FUEL_GAIN_S_PER_LAP)
        # only 2 changes in the stint yet: average everything known so far
        self.assertAlmostEqual(wear_rate(known, 7, stint_start=6), 4.2 / 6 + FUEL_GAIN_S_PER_LAP)

    def test_no_wear_signal_keeps_only_the_cheaper_safety_car_stop(self) -> None:
        """User decision 2026-11-03: a model predicting laps getting faster gives wear 0; every lap
        then ties and used to be "in the window", e.g. right after a stop."""
        known = {j: -0.3 for j in range(1, 31)}                     # wear clipped to 0
        st = driver_strategy(_driver(status={10: "14"}), known, {"normal_s": 20.0, "sc_vsc_s": 14.0})[0]
        self.assertEqual(st["window"], [10])
        self.assertEqual((st["best_lap"], st["sc_vsc_calls"]), (None, [10]))
        self.assertTrue(all(a["no_wear_signal"] for a in st["advice"]))

    def test_observed_change_is_used_only_after_the_lap_is_complete(self) -> None:
        laps = _driver()
        laps["LapTime"] = pd.to_timedelta(90.0 + 0.2 * laps["LapNumber"], unit="s")
        changes = observed_changes(laps)
        # the change into lap 5 (lap 5 - lap 4) is known once lap 5 ends: first usable on lap 6
        self.assertAlmostEqual(changes[6], 0.2)
        self.assertNotIn(2, changes)                # lap 1 (standing start) is not representative


if __name__ == "__main__":
    unittest.main()
