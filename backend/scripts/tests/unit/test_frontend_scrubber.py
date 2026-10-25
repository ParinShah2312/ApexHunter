"""Tests for frontend/components/scrubber.py: the master scrubber and the Telemetry-tab slider must
always show the same session time, and so must the time card, chart cut-off and scrub line (#31)."""

import base64
import datetime
import json
import unittest
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

FRONTEND_DIR = Path(__file__).resolve().parents[4] / "frontend"
BASE = datetime.datetime(2000, 1, 1)

# Synthetic session: one sample per second from 100 s to 4000 s, laid out like app.py's tabs
APP = f"""
import sys
sys.path.insert(0, {str(FRONTEND_DIR)!r})
import numpy as np
import pandas as pd
import streamlit as st
from components.scrubber import render_master_scrubber
from components.telemetry_charts import render_telemetry

end = st.session_state.get("test_session_end", 4000)
t = np.arange(100, end + 1, dtype=float)
df = pd.DataFrame({{"SessionTime": pd.to_timedelta(t, unit="s"), "Speed": 200.0, "Throttle": 50.0,
                    "Brake": 0.0, "nGear": 6, "RPM": 10000.0}})
tab1, tab2 = st.tabs(["Race Intelligence", "Telemetry"])
with tab1:
    render_master_scrubber(float(t[0]), float(t[-1]))
with tab2:
    render_telemetry(df, "Test", "1")
"""


def _values(arr) -> np.ndarray:
    """Plotly 6 serialises numeric arrays as {"dtype", "bdata": base64}; plain lists otherwise."""
    if isinstance(arr, dict):
        return np.frombuffer(base64.b64decode(arr["bdata"]), dtype=arr["dtype"])
    return np.asarray(arr, dtype=float)


class TestLinkedScrubbers(unittest.TestCase):

    def setUp(self) -> None:
        self.at = AppTest.from_string(APP, default_timeout=60).run()
        self.assertFalse(self.at.exception, [e.value for e in self.at.exception])

    def _state(self) -> dict:
        at = self.at
        self.assertFalse(at.exception, [e.value for e in at.exception])
        fig = json.loads(at.get("plotly_chart")[0].proto.spec)
        return {
            "master": (at.slider(key="scrub_dt").value - BASE).total_seconds(),
            "telemetry": at.slider(key="telemetry_scrub_seconds").value,
            "shared": at.session_state["scrub_seconds"],
            "card": {m.label: m.value for m in at.metric}["Session Time"],
            "chart_end": float(_values(fig["data"][0]["x"]).max()),        # last sample drawn on the speed trace
            "scrub_line": fig["layout"]["shapes"][0]["x0"],
        }

    def _assert_all_at(self, seconds: float, card: str) -> None:
        s = self._state()
        for name in ("master", "telemetry", "shared", "chart_end", "scrub_line"):
            self.assertAlmostEqual(s[name], seconds, places=3, msg=f"{name}: {s}")
        self.assertEqual(s["card"], card)

    def test_moving_master_moves_telemetry_view(self) -> None:
        self.at.slider(key="scrub_dt").set_value(BASE + datetime.timedelta(seconds=1500)).run()
        self._assert_all_at(1500.0, "25:00.000")

    def test_moving_telemetry_slider_moves_master(self) -> None:
        """Before #31 the master overwrote this move on the next rerun."""
        self.at.slider(key="scrub_dt").set_value(BASE + datetime.timedelta(seconds=1500)).run()
        self.at.slider(key="telemetry_scrub_seconds").set_value(2500.0).run()
        self._assert_all_at(2500.0, "41:40.000")
        self.at.slider(key="scrub_dt").set_value(BASE + datetime.timedelta(seconds=900)).run()
        self._assert_all_at(900.0, "15:00.000")

    def test_position_clamped_when_session_is_shorter(self) -> None:
        self.at.session_state["test_session_end"] = 2000
        self.at.run()
        self._assert_all_at(2000.0, "33:20.000")


if __name__ == "__main__":
    unittest.main()
