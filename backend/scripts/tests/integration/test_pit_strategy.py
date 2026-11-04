"""Integration test: pit cost + pit window (ROADMAP §C steps 2 and 4) on real races.

No mocks: real FastF1 laps from the project cache for 2023 Bahrain (green race) and 2023 Singapore
(safety car on lap 20, VSC on lap 44). Wear is the "no change" baseline (fuel gain only), so the
results do not depend on a trained model. Skipped when the local data is not present.
"""

import os
import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from utils import DATA_LAKE_DIR

RUN_SLOW = os.environ.get("APEXHUNTER_RUN_SLOW", "0") == "1"
RACES = {"bahrain": (2023, 1), "singapore": (2023, 15)}
NO_CHANGE = {j: 0.0 for j in range(1, 100)}


@unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")
@unittest.skipUnless(all((DATA_LAKE_DIR / "clean_data" / f"{y}_{r}_R.parquet").exists()
                         for y, r in RACES.values()), "Needs local 2023 R1 and R15 race data")
class TestPitStrategyOnRealRaces(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import fastf1
        from strategy.pit_cost import race_pit_cost
        from strategy.pit_window import driver_strategy
        from utils import enable_fastf1_cache

        enable_fastf1_cache()
        fastf1.set_log_level("ERROR")
        cls.cost, cls.plans = {}, {}
        for name, (year, rnd) in RACES.items():
            session = fastf1.get_session(year, rnd, "R")
            session.load(telemetry=False, weather=False, messages=False)
            cls.cost[name] = race_pit_cost(session.laps)
            for driver in ("VER", "SAI", "RUS"):
                laps = session.laps.pick_drivers(driver)
                cls.plans[(name, driver)] = driver_strategy(laps, NO_CHANGE, cls.cost[name])

    def test_pit_cost_from_the_races_own_stops(self):
        bahrain, singapore = self.cost["bahrain"], self.cost["singapore"]
        # measured 2026-11-01: 44 normal stops, median 25.1 s; Singapore's long pit lane ~30 s
        self.assertEqual(bahrain["n_normal_stops"], 44)
        self.assertAlmostEqual(bahrain["normal_s"], 25.1, delta=0.1)
        self.assertAlmostEqual(bahrain["sc_vsc_s"], 0.7 * bahrain["normal_s"])
        self.assertGreater(singapore["normal_s"], bahrain["normal_s"])

    def test_green_race_window_brackets_the_real_stops(self):
        """Verstappen, Bahrain 2023: stops on laps 14 and 36. Each window must contain the real stop,
        reach past it (not cut off there) and close again (not stay open to the end)."""
        stints = self.plans[("bahrain", "VER")]
        self.assertEqual([s["actual_stop"] for s in stints], [14, 36])
        for s in stints:
            green = [lap for lap in s["window"] if lap not in s["sc_vsc_calls"]]
            self.assertTrue(s["hit"])
            self.assertGreater(max(green), s["actual_stop"])
            self.assertLess(max(green), s["horizon_end"] - 5)
            self.assertLessEqual(len(green), 10)

    def test_safety_car_stops_get_a_pit_now_call(self):
        """Singapore 2023: Sainz (1 stop) and Russell (2 stops) pitted under the lap-20 safety car;
        Russell again under the lap-44 VSC. The advice must say "pit now" on those laps."""
        sai, rus = self.plans[("singapore", "SAI")], self.plans[("singapore", "RUS")]
        for s in sai + rus:
            self.assertTrue(s["actual_under_sc_vsc"])
            self.assertIn(s["actual_stop"], s["sc_vsc_calls"])
        self.assertEqual([s["actual_stop"] for s in rus], [20, 44])

    def test_output_file_content(self):
        """§C step 7: the per-driver output (laps + stints + readable summary) for Russell,
        Singapore 2023, matches the plan's format and is valid JSON."""
        import json
        import fastf1
        from strategy.strategy_output import build_strategy_output

        session = fastf1.get_session(*RACES["singapore"], "R")
        session.load(telemetry=False, weather=False, messages=False)
        out = build_strategy_output(session.laps.pick_drivers("RUS"), NO_CHANGE,
                                    self.cost["singapore"], "no_change", None)
        json.dumps(out)
        self.assertEqual(len(out["laps"]), 62)
        self.assertEqual(out["laps"][19]["neutralised"], "Safety car")          # lap 20
        self.assertEqual([s["summary"] for s in out["stints"][:2]], [
            ["Normal window: laps 20-24, best: lap 22",
             "Safety car on laps 20-22 -> pit now (best option)",
             "Actual: pitted lap 20 (under the safety car) - inside the window"],
            ["Normal window: laps 39-43, best: lap 41",
             "VSC on laps 44-45 -> pit now (best option)",
             "Actual: pitted lap 44 (under VSC) - inside the window"]])
        self.assertNotIn("summary", out["stints"][2])                           # last stint


if __name__ == "__main__":
    unittest.main()
