"""Tests for frontend/components/ai_analysis.py: mistake-rate display. (The tyre pit-window card is
tested on real data in integration/test_strategy_ui.py.)"""

import sys
import unittest
from pathlib import Path

import pandas as pd

FRONTEND_DIR = Path(__file__).resolve().parents[4] / "frontend"
sys.path.insert(0, str(FRONTEND_DIR))

from components.ai_analysis import mistake_rate_pct


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
        "anomaly_score": np.zeros(n),
    })
    meta = {"best_contamination": 0.005, "reference_drivers": ["1", "11"]}
    events = [{"rank": 1, "lap": 3, "track_pos_m": 520.0, "duration_s": 0.6, "speed_kmh": 280.0,
               "x": 1.0, "y": 1.0, "location": "Turn 4", "time_lost_s": 0.18, "costly": True, "saving_note": "", "reason": "Off the throttle: 78 % less than the field"}]
    _render_isolation_forest_column(df, meta, events)


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
        self.assertIn("1 events from 10 flagged samples (2.50% of samples)", caption)


if __name__ == "__main__":
    unittest.main()
