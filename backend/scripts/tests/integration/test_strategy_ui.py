"""Integration test: the Tyres & Pit Window tab (ROADMAP §C step 8, §3 step 4) on a real race.

No mocks: the output is built from real FastF1 laps (Russell, Singapore 2023, "no change" wear, so no
trained model is needed) and rendered with Streamlit's AppTest. Checks that what the tab shows (the
summary, the real stop, the shaded window and safety-car laps, the lap times) is what the output says
and what happened in the race. The model switch and comparison bars: test_strategy_tab.py.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = SCRIPTS_DIR.parent.parent / "frontend"
sys.path.insert(0, str(SCRIPTS_DIR))

from utils import DATA_LAKE_DIR

RUN_SLOW = os.environ.get("APEXHUNTER_RUN_SLOW", "0") == "1"
RACE = DATA_LAKE_DIR / "clean_data" / "2023_15_R.parquet"

CARD = f"""
import json, sys
sys.path.insert(0, {str(FRONTEND_DIR)!r})
import streamlit as st
from components.strategy_tab import render_strategy_tab
with open(st.session_state["path"], encoding="utf-8") as f:
    render_strategy_tab(json.load(f))
"""


@unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")
@unittest.skipUnless(RACE.exists(), "Needs local 2023 R15 race data")
class TestStrategyCardOnRealRace(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import fastf1
        from streamlit.testing.v1 import AppTest
        from strategy.pit_cost import race_pit_cost
        from strategy.strategy_output import build_strategy_output
        from utils import enable_fastf1_cache

        enable_fastf1_cache()
        fastf1.set_log_level("ERROR")
        session = fastf1.get_session(2023, 15, "R")
        session.load(telemetry=False, weather=False, messages=False)
        cls.laps = session.laps.pick_drivers("RUS")
        from strategy.pit_window import driver_strategy, observed_changes
        from strategy.strategy_check import best_source, stint_rows, summarise
        import pandas as pd
        cost = race_pit_cost(session.laps)
        cls.out = build_strategy_output(cls.laps, {j: 0.0 for j in range(1, 100)}, cost, "no_change",
                                        None, lap_predictions=False)
        # real scores for this race (every driver, two rule-based sources), as the card shows them
        rows = []
        for d in session.laps["Driver"].unique():
            dl = session.laps.pick_drivers(d)
            for src, known in (("no_change", {j: 0.0 for j in range(1, 100)}),
                               ("laps_driven", observed_changes(dl))):
                rows += stint_rows(src, "2023_15_R", d, driver_strategy(dl, known, cost))
        cls.scores = summarise(pd.DataFrame(rows))
        cls.out.update({"strategy_scores": cls.scores, "best_source": best_source(cls.scores)})
        cls.tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(cls.out, cls.tmp)
        cls.tmp.close()
        cls.at = AppTest.from_string(CARD, default_timeout=60)
        cls.at.session_state["path"] = cls.tmp.name
        cls.at.run()

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.tmp.name)

    def test_card_shows_the_first_stint_as_in_the_output_and_the_race(self):
        at = self.at
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual(at.session_state["tyre_stint_"], 1)              # opens on stint 1
        self.assertEqual(at.session_state["tyre_model_"], "no_change")    # the file's only source
        text = " ".join(m.value for m in at.markdown)
        for line in self.out["stints"][0]["summary"]:
            self.assertIn(line, text)
        metrics = {m.label: m.value for m in at.metric}
        # Russell really pitted on lap 20, under the safety car (FastF1 PitInTime)
        real_stop = int(self.laps.loc[self.laps["PitInTime"].notna(), "LapNumber"].min())
        self.assertEqual(real_stop, 20)
        self.assertEqual(metrics["Real stop"], "Lap 20")
        self.assertEqual(metrics["Real stop in the window"], "Yes")
        self.assertEqual(metrics["Lap-time error, this race"], "—")       # rule-based: no predictions

    def test_chart_shades_the_window_and_safety_car_laps_and_plots_real_lap_times(self):
        spec = json.loads(self.at.get("plotly_chart")[0].proto.spec)
        shapes = spec["layout"]["shapes"]
        rects = sorted((s["x0"] + 0.5, s["x1"] - 0.5, s["fillcolor"]) for s in shapes if s["type"] == "rect")
        window = [(a, b) for a, b, c in rects if "0,255,136" in c]
        neutral = [(a, b) for a, b, c in rects if "255,200,0" in c]
        # window 20-24 (best lap 22 +-2), shaded only within stint 1 (laps 1-20)
        self.assertEqual(window, [(20, 20)])
        self.assertIn((20, 20), neutral)                       # lap-20 safety car
        self.assertEqual(spec["data"][0]["name"], "Real lap time")
        caption = " ".join(c.value for c in self.at.caption)
        self.assertIn("Window laps after the real stop (21-24)", caption)
        # the plotted values (Plotly 6 encodes them in the spec, so read the same figure object)
        sys.path.insert(0, str(FRONTEND_DIR))
        from components.strategy_tab import _build_strategy_figure
        fig = _build_strategy_figure(self.out, self.out["stints"][0])
        real, other = fig.data[0], fig.data[1]
        lap5 = self.laps.loc[self.laps["LapNumber"] == 5, "LapTime"].dt.total_seconds().iloc[0]
        self.assertAlmostEqual(real.y[list(real.x).index(5)], lap5)
        # user, 2026-11-03: only the stint's laps, and the line does not jump up at lap 1 or at the
        # safety-car in-lap (lap 20): those are separate markers
        self.assertEqual((min(real.x), max(real.x)), (1, 20))
        self.assertIsNone(real.y[list(real.x).index(1)])
        self.assertIsNone(real.y[list(real.x).index(20)])
        self.assertEqual(list(other.x), [1, 20])


if __name__ == "__main__":
    unittest.main()
