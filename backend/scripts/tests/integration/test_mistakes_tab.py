"""Integration test: the Driver Mistakes tab (ROADMAP §3 step 3) on the real 2023 Bahrain race output
for Verstappen (detect_mistakes.py --compare). No mocks: the whole app runs in Streamlit's AppTest.

Checks that every detector shows its own saved events, that the saved Z-score/autoencoder events
match the counts in the meta file's comparison, and that the cards show measured values (#4).
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
STEM = DATA_LAKE_DIR / "mistake_data" / "2023_1_R_1_mistakes"
LABELS = {"isolation_forest": "Isolation Forest", "lstm_autoencoder": "LSTM autoencoder", "z_score": "Z-score"}


@unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")
@unittest.skipUnless(Path(f"{STEM}_events.json").exists(), "Needs the 2023 R1 #1 mistake output")
class TestMistakesTab(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        from streamlit.testing.v1 import AppTest
        with open(f"{STEM}_events.json", encoding="utf-8") as f:
            cls.events = json.load(f)
        with open(f"{STEM}_meta.json", encoding="utf-8") as f:
            cls.meta = json.load(f)
        cls.df = pd.read_parquet(f"{STEM}.parquet")
        sys.path.insert(0, str(FRONTEND_DIR))
        cls._cwd = os.getcwd()
        os.chdir(FRONTEND_DIR)
        cls.at = AppTest.from_file("app.py", default_timeout=300)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)

    def _open(self, detector: str):
        at = self.at
        at.session_state["main_tab"] = "Driver Mistakes"      # AppTest does not keep the open tab
        at.session_state["mis_detector"] = detector
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return {m.label: m.value for m in at.metric}

    def test_saved_outputs_hold_every_detectors_events(self):
        self.assertIn("detectors", self.events, "re-run detect_mistakes.py --compare --force")
        real = self.meta["comparison"]["real_laps"]
        for name, short in (("z_score", "z"), ("lstm_autoencoder", "ae")):
            self.assertEqual(len(self.events["detectors"][name]), real[f"{name}_events"])
            self.assertEqual(int(self.df[f"{short}_flag"].sum()), real[f"{name}_flagged_samples"])
            ids = self.df[f"{short}_event_id"]
            self.assertEqual(ids[ids >= 0].nunique(), real[f"{name}_events"])

    def test_each_detector_shows_its_own_events_and_measured_rate(self):
        for name in LABELS:
            events = self.events["events"] if name == "isolation_forest" else self.events["detectors"][name]
            cards = self._open(name)
            self.assertEqual(cards["Mistakes found"], str(len(events)), name)
            prefix = {"isolation_forest": "is_mistake", "z_score": "z_flag", "lstm_autoencoder": "ae_flag"}[name]
            flagged = int(self.df[prefix].sum())
            self.assertEqual(cards["Readings flagged"], f"{flagged:,}")
            table = self.at.dataframe[0].value
            self.assertEqual(len(table), len(events))
            self.assertEqual(list(table["#"]), sorted(e["rank"] for e in events))

    def test_going_back_to_all_laps_shows_the_top_mistake_again(self):
        self._open("isolation_forest")
        top = min(self.events["events"], key=lambda e: e["rank"])
        other_lap = next(e["lap"] for e in sorted(self.events["events"], key=lambda e: e["rank"])
                         if e["lap"] != top["lap"])
        at = self.at
        at.selectbox(key="mis_lap_2023_1_R_1").set_value(f"Lap {other_lap}")
        at.session_state["main_tab"] = "Driver Mistakes"
        at.run()
        self.assertTrue(any(h.value.startswith(f"#### Lap {other_lap} ·") for h in at.markdown))
        at.selectbox(key="mis_lap_2023_1_R_1").set_value("All laps")
        at.session_state["main_tab"] = "Driver Mistakes"
        at.run()
        self.assertTrue(any(h.value.startswith(f"#### Lap {top['lap']} · {top['location']}") for h in at.markdown))

    def test_comparison_charts_show_the_meta_numbers(self):
        self._open("isolation_forest")
        figs = [json.loads(c.proto.spec) for c in self.at.get("plotly_chart")]
        bars = next(f for f in figs if f["data"] and f["data"][0].get("type") == "bar" and len(f["data"]) == 3)
        planted = self.meta["comparison"]["planted"]
        for trace in bars["data"]:
            name = next(n for n, label in LABELS.items() if label == trace["name"])
            self.assertEqual(trace["text"][2], f"{planted[name]['f1']:.2f}")        # F1 bar label


if __name__ == "__main__":
    unittest.main()
