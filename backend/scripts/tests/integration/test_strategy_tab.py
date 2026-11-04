"""Integration test: the Tyres & Pit Window tab's model switch (ROADMAP §3 step 4) on a real output.

No mocks: Verstappen in the 2024 Las Vegas GP (2024_22_R, a held-out test race), written by
predict_strategy.py --source all, rendered by the whole app in Streamlit's AppTest. Checks that each
model shows its own predictions and lap-time error, and that the comparison bars show the saved
numbers.
"""

import json
import os
import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = SCRIPTS_DIR.parent.parent / "frontend"
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from utils import DATA_LAKE_DIR

RUN_SLOW = os.environ.get("APEXHUNTER_RUN_SLOW", "0") == "1"
OUTPUT = DATA_LAKE_DIR / "tyre_predictions" / "2024_22_R_1_strategy.json"
CTX = "2024_22_R_1"


def _has_all_sources() -> bool:
    if not OUTPUT.exists():
        return False
    with open(OUTPUT, encoding="utf-8") as f:
        return "sources" in json.load(f)


@unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")
@unittest.skipUnless(_has_all_sources(), "Needs 2024_22_R_1_strategy.json from predict_strategy.py --source all")
class TestStrategyModelSwitch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from streamlit.testing.v1 import AppTest
        with open(OUTPUT, encoding="utf-8") as f:
            cls.data = json.load(f)
        sys.path.insert(0, str(FRONTEND_DIR))
        cls._cwd = os.getcwd()
        os.chdir(FRONTEND_DIR)
        at = AppTest.from_file("app.py", default_timeout=300)
        at.session_state["sel_year"] = 2024
        at.run()
        at.session_state["sel_round"] = 22
        at.session_state["sel_session_type"] = "R"
        at.session_state["sel_driver_number"] = "1"
        cls.at = at

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)

    def _open(self, model: str):
        at = self.at
        at.session_state["main_tab"] = "Tyres & Pit Window"     # AppTest does not keep the open tab
        at.session_state[f"tyre_model_{CTX}"] = model
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        charts = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
        return {m.label: m.value for m in at.metric}, charts

    def test_each_model_shows_its_own_predictions_and_error(self):
        self.assertFalse(self.data["sources"]["lstm"]["in_training_data"])       # held-out race
        for model in ("ridge", "rnn", "lstm"):
            cards, charts = self._open(model)
            names = [t.get("name", "") for t in charts[0]["data"]]
            laps = self.data["sources"][model]["laps"]
            errs = [abs(r["predicted_lap_time"] - r["lap_time"]) for r in laps
                    if r["predicted_lap_time"] is not None and r["lap_time"] is not None]
            self.assertIn(f"Predicted by {model.upper() if model != 'ridge' else 'Ridge'}", names)
            self.assertEqual(cards["Lap-time error, this race"], f"{sum(errs) / len(errs):.3f} s")

    def test_switch_offers_the_three_trained_models_and_opens_on_the_best_margin(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=300)              # fresh: no model picked yet
        at.session_state["sel_year"] = 2024
        at.run()
        at.session_state["sel_round"] = 22
        at.session_state["sel_session_type"] = "R"
        at.session_state["sel_driver_number"] = "1"
        at.session_state["main_tab"] = "Tyres & Pit Window"
        at.run()
        scores = self.data["strategy_scores"]
        best = max(("ridge", "rnn", "lstm"), key=lambda m: scores[m]["hit_pct"] - scores[m]["random_pct"])
        self.assertEqual(at.session_state[f"tyre_model_{CTX}"], best)
        switch = next(b for b in at.get("button_group") if b.proto.label == "Tyre model")
        self.assertEqual(len(switch.proto.options), 3)
        stint = self.data["sources"][best]["stints"][0]
        cards = {m.label: m.value for m in at.metric}
        self.assertEqual(cards["Real stop"], f"Lap {stint['actual_stop']}")
        self.assertEqual(cards["Real stop in the window"], "Yes" if stint["hit"] else "No")

    def test_error_bars_show_the_model_configs(self):
        _, charts = self._open("lstm")
        bars = next(c for c in charts if c["data"] and c["data"][0].get("type") == "bar"
                    and len(c["data"]) == 1)
        with open(PROJECT_ROOT / "models" / "tyre_rnn_config.json", encoding="utf-8") as f:
            rnn = json.load(f)["metrics"]
        shown = dict(zip(bars["data"][0]["x"], bars["data"][0]["text"]))
        self.assertEqual(list(shown), ["Ridge", "RNN", "LSTM"])           # the trained models only
        self.assertEqual(shown["RNN"], f"{rnn['test']['mae_s']:.3f} s")


if __name__ == "__main__":
    unittest.main()
