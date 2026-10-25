"""Real-data check of FastF1-derived turn windows against the hand-annotated Bahrain 2024 pole lap.

The windows below were typed in by hand from the onboard video (2024 Bahrain, Verstappen) and
were used by the CV pipeline before turn windows were computed from FastF1 circuit data. They are
kept here only as ground truth. Requires the FastF1 data for 2024 round 1 qualifying (cache/).
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

RUN_SLOW = os.environ.get("APEXHUNTER_RUN_SLOW", "0") == "1"
VIDEO = Path("data_lake/edited_videos/2024/01_bahrain_ver_pole - Trim.mp4")
VIDEO_OFFSET_S = 4.9

# (start_s, end_s, direction) per corner T1..T15, hand-annotated from the video
HAND_WINDOWS = [
    (13.0, 16.0, "Right"), (17.0, 19.0, "Left"), (19.5, 20.5, "Right"), (27.0, 29.0, "Right"),
    (33.0, 34.0, "Left"), (34.0, 36.0, "Right"), (36.0, 38.0, "Left"), (41.0, 43.0, "Right"),
    (48.0, 50.0, "Left"), (51.0, 53.0, "Left"), (62.0, 67.0, "Left"), (68.0, 71.0, "Right"),
    (72.0, 77.0, "Right"), (85.0, 87.0, "Right"), (88.0, 90.0, "Right"),
]


@unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")
class TestBahrainTurnWindows(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from cv.extract_turns import build_turn_windows
        cls.windows = build_turn_windows(VIDEO_OFFSET_S, VIDEO)

    def test_all_fifteen_corners_found(self) -> None:
        self.assertEqual([w.turn for w in self.windows], [f"T{i}" for i in range(1, 16)])

    def test_apex_times_match_hand_annotation(self) -> None:
        hand_mid = np.array([(s + e) / 2 for s, e, _ in HAND_WINDOWS])
        err = np.abs(np.array([w.apex_s for w in self.windows]) - hand_mid)
        self.assertLess(err.mean(), 0.6, f"per-corner errors: {np.round(err, 2)}")
        self.assertLess(err.max(), 1.5)

    def test_window_edges_match_hand_annotation(self) -> None:
        starts = np.abs(np.array([w.start_s for w in self.windows]) - [s for s, _, _ in HAND_WINDOWS])
        ends = np.abs(np.array([w.end_s for w in self.windows]) - [e for _, e, _ in HAND_WINDOWS])
        self.assertLess(starts.mean(), 1.0)
        self.assertLess(ends.mean(), 1.0)

    def test_directions_match_hand_annotation(self) -> None:
        # T5 is a ~10° kink between T4 and the T6/T7 esses; telemetry reads it as a slight right
        mismatches = [w.turn for w, (_, _, d) in zip(self.windows, HAND_WINDOWS) if w.direction != d]
        self.assertLessEqual(len(mismatches), 1, mismatches)


if __name__ == "__main__":
    unittest.main()
