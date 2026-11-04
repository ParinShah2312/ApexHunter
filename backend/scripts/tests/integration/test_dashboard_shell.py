"""Integration test: the redesigned dashboard shell (ROADMAP §3 step 1) on real 2023 Bahrain data.

No mocks: runs the whole app (frontend/app.py) with Streamlit's AppTest on the cleaned session file
and the FastF1 cache. Checks the opening view, the five tabs, a season switch, and that the header
shows the team the driver raced for in that season (#55: a fixed table showed 2024 team names).
"""

import os
import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = SCRIPTS_DIR.parent.parent / "frontend"
sys.path.insert(0, str(SCRIPTS_DIR))

from utils import DATA_LAKE_DIR

RUN_SLOW = os.environ.get("APEXHUNTER_RUN_SLOW", "0") == "1"
RACE = DATA_LAKE_DIR / "clean_data" / "2023_1_R.parquet"


def _header(at) -> str:
    return next(m.value for m in at.markdown if "Gap to fastest" in m.value)


@unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")
@unittest.skipUnless(RACE.exists(), "Needs local 2023 R1 race data")
class TestDashboardShell(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from streamlit.testing.v1 import AppTest
        sys.path.insert(0, str(FRONTEND_DIR))
        cls._cwd = os.getcwd()
        os.chdir(FRONTEND_DIR)
        cls.at = AppTest.from_file("app.py", default_timeout=300)
        cls.at.run()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)

    def test_opens_on_a_session_with_results(self):
        at = self.at
        self.assertFalse(at.exception)
        self.assertEqual([t.label for t in at.tabs],
                         ["Overview", "Driver Mistakes", "Tyres & Pit Window", "Telemetry", "More"])
        header = _header(at)
        self.assertIn("Bahrain Grand Prix", header)
        self.assertIn("Max Verstappen", header)
        # Verstappen's best race lap 1:36.236 vs Zhou's fastest lap 1:33.996 (official timing)
        self.assertIn("1:36.236", header)
        self.assertIn("+2.240 s", header)
        self.assertIn("Gap to fastest (ZHO)", header)      # says who set it (Zhou, fresh softs, lap 56)
        self.assertIn("P1", header)

    def test_header_team_is_the_one_for_that_season(self):
        at = self.at
        at.session_state["sel_driver_number"] = "24"      # Zhou: Alfa Romeo in 2023
        at.run()
        self.assertFalse(at.exception)
        self.assertIn("Alfa Romeo", _header(at))
        self.assertNotIn("Kick Sauber", _header(at))
        at.session_state["sel_driver_number"] = "1"
        at.run()

    def test_round_falls_back_when_the_season_has_fewer_rounds(self):
        at = self.at
        at.session_state["sel_year"] = 2024
        at.run()
        at.session_state["sel_round"] = 24
        at.run()
        self.assertIn("Abu Dhabi Grand Prix", _header(at))
        at.session_state["sel_year"] = 2023                # 2023 has no round 24 file
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["sel_round"], 1)


if __name__ == "__main__":
    unittest.main()
