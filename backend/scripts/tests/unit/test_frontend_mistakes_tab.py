"""Tests for frontend/components/mistakes_tab.py: which detector's flags and events the Driver
Mistakes tab shows, and the flagged-readings figure (#4: the measured rate, not the setting).
The full tab is tested on real outputs in integration/test_mistakes_tab.py."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

FRONTEND_DIR = Path(__file__).resolve().parents[4] / "frontend"
sys.path.insert(0, str(FRONTEND_DIR))

from components.mistakes_tab import costly_shares, detector_data, mistake_rate_pct


def _parquet(with_compare: bool) -> pd.DataFrame:
    df = pd.DataFrame({"is_mistake": [True, False, False, False], "event_id": [0, -1, -1, -1]})
    if with_compare:
        df["z_flag"] = [False, True, True, False]
        df["z_event_id"] = [-1, 0, 0, -1]
        df["ae_flag"] = [False, False, False, True]
        df["ae_event_id"] = [-1, -1, -1, -1]
    return df


EVENTS = {"events": [{"event_id": 0, "rank": 1}],
          "detectors": {"z_score": [{"event_id": 0, "rank": 1, "reason": "z"}], "lstm_autoencoder": []}}


class TestDetectorData(unittest.TestCase):

    def test_each_detector_gets_its_own_flags_and_events(self) -> None:
        df = _parquet(with_compare=True)
        flags, ids, events = detector_data(df, EVENTS, "isolation_forest")
        self.assertEqual(list(flags), [True, False, False, False])
        self.assertEqual(events, EVENTS["events"])
        flags, ids, events = detector_data(df, EVENTS, "z_score")
        self.assertEqual(list(flags), [False, True, True, False])
        self.assertEqual(list(ids), [-1, 0, 0, -1])
        self.assertEqual(events[0]["reason"], "z")
        self.assertEqual(detector_data(df, EVENTS, "lstm_autoencoder")[2], [])

    def test_outputs_without_compare_only_have_the_isolation_forest(self) -> None:
        df = _parquet(with_compare=False)
        self.assertIsNotNone(detector_data(df, {"events": []}, "isolation_forest"))
        self.assertIsNone(detector_data(df, {"events": []}, "z_score"))
        self.assertIsNone(detector_data(df, {"events": []}, "lstm_autoencoder"))

    def test_flagged_share_is_measured_in_percent(self) -> None:
        self.assertAlmostEqual(mistake_rate_pct(np.array([True, False, False, False])), 25.0)
        self.assertEqual(mistake_rate_pct(np.array([], dtype=bool)), 0.0)

    def test_real_lap_check_shares_come_from_each_detectors_entry(self) -> None:
        meta = {"time_lost_check": {"isolation_forest": {"costly_share": 0.15}, "random_spots": {"costly_share": 0.1}},
                "comparison": {"z_score_time_lost_check": {"costly_share": 0.176},
                               "lstm_autoencoder_time_lost_check": {"costly_share": 0.091}}}
        self.assertEqual(costly_shares(meta), {"isolation_forest": 0.15, "lstm_autoencoder": 0.091,
                                               "z_score": 0.176, "random_spots": 0.1})


if __name__ == "__main__":
    unittest.main()
