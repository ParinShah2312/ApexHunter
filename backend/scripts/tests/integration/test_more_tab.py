"""Integration test: the More tab (ROADMAP §3 step 6) on real outputs. No mocks: the whole app in
Streamlit's AppTest.

- Apex detection: the cards count corners (they used to count video frames); on the processed
  2024 Bahrain pole lap 12 of the 15 corners are "Hitting Apex".
- Racing line: still drawn for the one session that has an output (2023 R1 qualifying, #1), and
  the other sessions say where it exists.
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
METRICS = DATA_LAKE_DIR / "processed_csv" / "01_bahrain_ver_pole - Trim_metrics.csv"
RACING_LINE = DATA_LAKE_DIR / "racing_lines" / "2023_1_Q_1_racing_line.json"


@unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")
@unittest.skipUnless(METRICS.exists() and RACING_LINE.exists(), "Needs the CV and racing-line outputs")
class TestMoreTab(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(FRONTEND_DIR))
        cls._cwd = os.getcwd()
        os.chdir(FRONTEND_DIR)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)

    def _app(self, part: str, session: str = "R"):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=300)
        at.session_state["sel_session_type"] = session
        at.session_state["main_tab"] = "More"
        at.session_state["more_part"] = part
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at

    def test_apex_cards_count_corners(self):
        at = self._app("Apex detection")
        cards = {m.label: m.value for m in at.metric}
        self.assertEqual(cards["Hitting Apex"], "12 of 15")
        table = at.dataframe[0].value
        self.assertEqual(len(table), 15)
        self.assertEqual(int((table["Status"] == "Hitting Apex").sum()), 12)

    def test_racing_line_drawn_where_it_exists_and_pointed_to_elsewhere(self):
        at = self._app("Racing line", session="Q")
        self.assertGreaterEqual(len(at.get("plotly_chart")), 1)
        self.assertFalse(at.info)
        at = self._app("Racing line", session="R")
        self.assertIn("2023 R1 Qualifying, Max Verstappen", " ".join(i.value for i in at.info))


if __name__ == "__main__":
    unittest.main()
