"""Integration test: the redesigned Telemetry tab (ROADMAP §3 step 2) on real 2023 Bahrain qualifying.

No mocks: the whole app runs in Streamlit's AppTest on the cleaned session file and the FastF1
cache. Hamilton's fastest lap is compared with Verstappen's pole lap; the official gap is 0.676 s.
"""

import json
import os
import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = SCRIPTS_DIR.parent.parent / "frontend"
sys.path.insert(0, str(SCRIPTS_DIR))

from utils import DATA_LAKE_DIR

RUN_SLOW = os.environ.get("APEXHUNTER_RUN_SLOW", "0") == "1"
QUALI = DATA_LAKE_DIR / "clean_data" / "2023_1_Q.parquet"
BAHRAIN_LENGTH_M = 5412


@unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")
@unittest.skipUnless(QUALI.exists(), "Needs local 2023 R1 qualifying data")
class TestTelemetryTab(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from streamlit.testing.v1 import AppTest
        sys.path.insert(0, str(FRONTEND_DIR))
        cls._cwd = os.getcwd()
        os.chdir(FRONTEND_DIR)
        at = AppTest.from_file("app.py", default_timeout=300)
        at.session_state["sel_session_type"] = "Q"
        at.session_state["sel_driver_number"] = "44"
        at.session_state["main_tab"] = "Telemetry"      # AppTest does not keep the open tab between runs
        cls.at = at.run()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)

    def test_opens_on_the_fastest_lap_against_the_pole_lap(self):
        at = self.at
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual(at.selectbox(key="tel_lap_2023_1_Q_44").value, 14)
        self.assertEqual(at.selectbox(key="tel_ref_driver_2023_1_Q_44").value, "1")
        cards = {m.label: m.value for m in at.metric}
        self.assertEqual(cards["Lap time"], "1:30.384")
        self.assertEqual(cards["Gap to VER lap 14"], "+0.676 s")

    def test_charts_are_on_a_distance_axis_with_the_corners(self):
        fig = json.loads(self.at.get("plotly_chart")[1].proto.spec)      # [0] = lap picker
        delta = next(d for d in fig["data"] if d.get("name") == "Time delta")
        y = delta["y"]
        if isinstance(y, dict):                                           # Plotly 6 binary arrays
            import base64
            import numpy as np
            y = np.frombuffer(base64.b64decode(y["bdata"]), dtype=y["dtype"])
        self.assertAlmostEqual(float(y[-1]), 0.676, places=3)            # delta ends at the lap gap
        length = fig["layout"]["xaxis"]["range"][1]
        self.assertLess(abs(length - BAHRAIN_LENGTH_M) / BAHRAIN_LENGTH_M, 0.02)
        corners = [a["text"] for a in fig["layout"]["annotations"]]
        self.assertEqual(corners, [f"T{n}" for n in range(1, 16)])       # Bahrain: 15 corners, in order


if __name__ == "__main__":
    unittest.main()
