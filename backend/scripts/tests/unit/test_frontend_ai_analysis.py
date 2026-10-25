"""Tests for frontend/components/ai_analysis.py: mistake-rate display, LSTM stint lap axis and
the projected-pace delta colour."""

import sys
import unittest
from pathlib import Path

import pandas as pd

FRONTEND_DIR = Path(__file__).resolve().parents[4] / "frontend"
sys.path.insert(0, str(FRONTEND_DIR))

from components.ai_analysis import mistake_rate_pct, stint_cliff_label, stint_lap_axis


def _render_if_panel():
    """Streamlit script: render the Isolation Forest panel on synthetic data (10 of 400 flagged)."""
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(Path.cwd() / "frontend"))
    from components.ai_analysis import _render_isolation_forest_column

    n = 400
    is_mistake = np.zeros(n, dtype=bool)
    is_mistake[:10] = True
    df = pd.DataFrame({
        "X": np.arange(n, dtype=float), "Y": np.arange(n, dtype=float),
        "Speed": np.full(n, 200.0), "is_mistake": is_mistake,
        "anomaly_score": np.zeros(n), "brake_intensity": np.zeros(n),
    })
    meta = {"best_contamination": 0.005, "reference_driver": "1"}
    _render_isolation_forest_column(df, meta)


# Streamlit renders lap-time deltas with these colours (Metric proto enum: RED = 0, GREEN = 1)
RED, GREEN = 0, 1

LSTM_PANEL = f"""
import sys
sys.path.insert(0, {str(FRONTEND_DIR)!r})
import streamlit as st
from components.ai_analysis import _render_lstm_column

last_pred = st.session_state["last_predicted"]
stint = {{"stint_index": 0, "n_laps": 3, "lap_numbers": [10, 11, 12], "cliff_lap": None,
          "laps_remaining": None, "actual_laps": [95.0, 95.2, 95.5],
          "predicted_laps": [95.1, 95.3, last_pred], "confidence_upper": [96.0] * 3,
          "confidence_lower": [94.0] * 3}}
_render_lstm_column({{"stints": [stint]}})
"""


def _end_pace_metric(last_predicted: float):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(LSTM_PANEL, default_timeout=60)
    at.session_state["last_predicted"] = last_predicted
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return next(m for m in at.metric if m.label == "Proj. End Pace")


class TestProjectedPaceColour(unittest.TestCase):
    """Regression for #32: both branches used to paint the delta red."""

    def test_faster_projection_is_green(self) -> None:
        m = _end_pace_metric(94.5)                     # 1.00 s quicker than the last lap (95.5)
        self.assertEqual((m.delta, m.proto.color), ("-1.00s", GREEN))


class TestMistakeRate(unittest.TestCase):

    def test_rate_is_measured_share_in_percent(self) -> None:
        df = pd.DataFrame({"is_mistake": [True, False, False, False]})
        self.assertAlmostEqual(mistake_rate_pct(df), 25.0)

    def test_card_shows_measured_rate_not_contamination(self) -> None:
        from streamlit.testing.v1 import AppTest
        project_root = FRONTEND_DIR.parent
        import os
        cwd = os.getcwd()
        os.chdir(project_root)
        try:
            at = AppTest.from_function(_render_if_panel, default_timeout=60).run()
        finally:
            os.chdir(cwd)
        self.assertFalse(at.exception, [e.value for e in at.exception])
        metrics = {m.label: m.value for m in at.metric}
        self.assertEqual(metrics["Mistake Rate"], "2.50%")       # 10 / 400
        self.assertNotEqual(metrics["Mistake Rate"], "0.005")    # the contamination setting
        caption = " ".join(c.value for c in at.caption)
        self.assertIn("10 anomalies (2.50% of samples)", caption)


class TestStintLapAxis(unittest.TestCase):

    def test_uses_race_lap_numbers_when_present(self) -> None:
        stint = {"n_laps": 3, "lap_numbers": [5, 6, 8], "cliff_lap": 2}
        self.assertEqual(stint_lap_axis(stint), ([5, 6, 8], "Race lap"))
        self.assertEqual(stint_cliff_label(stint), "8")

    def test_falls_back_for_old_prediction_files(self) -> None:
        stint = {"n_laps": 3, "cliff_lap": 1}
        self.assertEqual(stint_lap_axis(stint), ([1, 2, 3], "Lap in stint"))
        self.assertEqual(stint_cliff_label(stint), "2")

    def test_no_cliff(self) -> None:
        self.assertIsNone(stint_cliff_label({"n_laps": 3, "cliff_lap": None}))


if __name__ == "__main__":
    unittest.main()
