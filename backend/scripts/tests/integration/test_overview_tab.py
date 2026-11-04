"""Integration test: the Overview tab (ROADMAP §3 step 5) on real outputs.

No mocks: the whole app in Streamlit's AppTest, Verstappen in the 2023 Bahrain race (mistake and
pit-window outputs) and in 2023 Bahrain qualifying (mistakes: none; pit windows: races only). The
Overview must show the same numbers as the saved files and the model configs.
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
MISTAKES = DATA_LAKE_DIR / "mistake_data" / "2023_1_R_1_mistakes_events.json"
STRATEGY = DATA_LAKE_DIR / "tyre_predictions" / "2023_1_R_1_strategy.json"


@unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")
@unittest.skipUnless(MISTAKES.exists() and STRATEGY.exists(), "Needs the 2023 R1 #1 outputs")
class TestOverviewTab(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from streamlit.testing.v1 import AppTest
        with open(MISTAKES, encoding="utf-8") as f:
            cls.events = sorted(json.load(f)["events"], key=lambda e: e["rank"])
        sys.path.insert(0, str(FRONTEND_DIR))
        cls._cwd = os.getcwd()
        os.chdir(FRONTEND_DIR)
        cls.at = AppTest.from_file("app.py", default_timeout=300).run()     # opens on the Overview

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)

    def test_session_summary_matches_the_outputs(self):
        at = self.at
        self.assertFalse(at.exception, [e.value for e in at.exception])
        cards = {m.label: m.value for m in at.metric}
        self.assertEqual(cards["Mistakes"], str(len(self.events)))
        text = " ".join(m.value for m in at.markdown)
        for e in self.events[:3]:
            self.assertIn(f"Lap {e['lap']} · {e['location']}", text)
        sys.path.insert(0, str(FRONTEND_DIR))
        from components.strategy_tab import default_model, source_view
        with open(STRATEGY, encoding="utf-8") as f:
            data = json.load(f)
        view = source_view(data, default_model(data, ["ridge", "rnn", "lstm"]))
        advised = [s for s in view["stints"] if "summary" in s]
        self.assertEqual(cards["Stops in window"], f"{sum(s['hit'] for s in advised)} of {len(advised)}")

    def test_scoreboard_shows_the_saved_results(self):
        cards = {m.label: m.value for m in self.at.metric}
        f1 = []
        for f in sorted((DATA_LAKE_DIR / "mistake_data").glob("*_mistakes_meta.json")):
            with open(f, encoding="utf-8") as fh:
                f1.append(json.load(fh)["comparison"]["planted"]["lstm_autoencoder"]["f1"])
        self.assertEqual(cards["LSTM autoencoder (DL)"], f"F1 {sum(f1) / len(f1):.2f}")
        with open(PROJECT_ROOT / "models" / "tyre_rnn_config.json", encoding="utf-8") as f:
            rnn = json.load(f)["metrics"]["test"]["mae_s"]
        self.assertEqual(cards["RNN (DL)"], f"{rnn:.3f} s")
        self.assertNotIn("Z-score (baseline)", cards)                   # trained models only

    def test_qualifying_has_no_pit_windows(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=300)              # own app: tests run in any order
        at.session_state["sel_session_type"] = "Q"
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        infos = " ".join(i.value for i in at.info)
        self.assertIn("No pit windows for this driver here (races only)", infos)
        self.assertIn("No mistake results for this driver here", infos)


if __name__ == "__main__":
    unittest.main()
