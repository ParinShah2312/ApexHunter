"""Unit tests for cv/extract_turns.py: turn windows from circuit corners + pole-lap telemetry."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from cv.extract_turns import (
    NO_TURN, WINDOW_AFTER_M, WINDOW_BEFORE_M, compute_turn_windows, load_turn_windows,
    parse_video_path, save_turn_windows, turn_context_at,
)

OFFSET_S = 4.0


def _synthetic_lap():
    """1 m-resolution lap: 500 m straight east, slow LEFT 90° arc (r=50 m, 90 km/h),
    400 m straight north, fast RIGHT 90° arc (r=50 m, 230 km/h), 300 m straight east."""
    xs, ys, speeds = [], [], []
    x, y, heading = 0.0, 0.0, 0.0

    def straight(length, speed):
        nonlocal x, y
        for _ in range(int(length)):
            x += np.cos(heading); y += np.sin(heading)
            xs.append(x); ys.append(y); speeds.append(speed)

    def arc(radius, turn_deg, speed):
        nonlocal heading
        steps = int(abs(np.radians(turn_deg)) * radius)
        for _ in range(steps):
            heading += np.radians(turn_deg) / steps
            straight(1, speed)

    straight(500, 300.0)
    c1_start = len(xs); arc(50, +90, 90.0); c1_end = len(xs)
    straight(400, 300.0)
    c2_start = len(xs); arc(50, -90, 230.0); c2_end = len(xs)
    straight(300, 300.0)

    dist = np.arange(1, len(xs) + 1, dtype=float)
    speed = np.array(speeds)
    time_s = np.concatenate([[0.0], np.cumsum(1.0 / (speed[1:] / 3.6))])
    telemetry = pd.DataFrame({
        "Time": pd.to_timedelta(time_s, unit="s"), "Distance": dist,
        "X": np.array(xs), "Y": np.array(ys), "Speed": speed,
    })
    corners = pd.DataFrame({
        "Number": [2, 1], "Letter": ["", ""],   # deliberately unsorted
        "Distance": [dist[(c2_start + c2_end) // 2], dist[(c1_start + c1_end) // 2]],
    })
    return corners, telemetry


class TestComputeTurnWindows(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.corners, cls.tel = _synthetic_lap()
        cls.windows = compute_turn_windows(cls.corners, cls.tel, OFFSET_S)

    def test_one_window_per_corner_in_lap_order(self) -> None:
        self.assertEqual([w.turn for w in self.windows], ["T1", "T2"])

    def test_apex_time_is_lap_time_at_corner_plus_offset(self) -> None:
        t = self.tel["Time"].dt.total_seconds().to_numpy()
        d = self.tel["Distance"].to_numpy()
        for w, apex_d in zip(self.windows, sorted(self.corners["Distance"])):
            self.assertAlmostEqual(w.apex_s, np.interp(apex_d, d, t) + OFFSET_S, places=6)

    def test_window_spans_configured_distance_around_apex(self) -> None:
        t = self.tel["Time"].dt.total_seconds().to_numpy()
        d = self.tel["Distance"].to_numpy()
        w, apex_d = self.windows[0], sorted(self.corners["Distance"])[0]
        self.assertAlmostEqual(w.start_s, np.interp(apex_d - WINDOW_BEFORE_M, d, t) + OFFSET_S, places=6)
        self.assertAlmostEqual(w.end_s, np.interp(apex_d + WINDOW_AFTER_M, d, t) + OFFSET_S, places=6)
        self.assertLess(w.start_s, w.apex_s)
        self.assertLess(w.apex_s, w.end_s)

    def test_direction_from_heading(self) -> None:
        self.assertEqual([w.direction for w in self.windows], ["Left", "Right"])

    def test_speed_category_from_minimum_corner_speed(self) -> None:
        self.assertEqual(self.windows[0].speed_category, "Slow")
        self.assertEqual(self.windows[0].min_speed_kmh, 90.0)
        self.assertEqual(self.windows[1].speed_category, "Fast")

    def test_neighbouring_windows_never_overlap(self) -> None:
        # Put the two corners 20 m apart so their ±50 m windows would overlap
        corners = pd.DataFrame({"Number": [1, 2], "Letter": ["", ""], "Distance": [520.0, 540.0]})
        w1, w2 = compute_turn_windows(corners, self.tel, OFFSET_S)
        self.assertLessEqual(w1.end_s, w2.start_s)
        self.assertAlmostEqual(w1.end_s, (w1.apex_s + w2.apex_s) / 2)

    def test_letter_suffix_in_label(self) -> None:
        corners = pd.DataFrame({"Number": [6], "Letter": ["A"], "Distance": [540.0]})
        self.assertEqual(compute_turn_windows(corners, self.tel, 0.0)[0].turn, "T6A")


class TestTurnContext(unittest.TestCase):

    def test_inside_and_outside_windows(self) -> None:
        corners, tel = _synthetic_lap()
        windows = compute_turn_windows(corners, tel, OFFSET_S)
        self.assertEqual(turn_context_at(windows[0].apex_s, windows), ("Left", "Slow"))
        self.assertEqual(turn_context_at(windows[1].apex_s, windows), ("Right", "Fast"))
        self.assertEqual(turn_context_at(0.0, windows), NO_TURN)          # before the lap starts
        gap = (windows[0].end_s + windows[1].start_s) / 2
        self.assertEqual(turn_context_at(gap, windows), NO_TURN)          # on the straight


class TestVideoPathAndPersistence(unittest.TestCase):

    def test_parse_video_path(self) -> None:
        p = Path("data_lake/edited_videos/2024/01_bahrain_ver_pole - Trim.mp4")
        self.assertEqual(parse_video_path(p), (2024, 1, "VER"))
        p = Path("data_lake/edited_videos/2023/18_unitedstates_lec_pole - Trim.mp4")
        self.assertEqual(parse_video_path(p), (2023, 18, "LEC"))

    def test_parse_video_path_rejects_unknown_names(self) -> None:
        with self.assertRaises(ValueError):
            parse_video_path(Path("videos/some_clip.mp4"))

    def test_save_and_load_round_trip(self) -> None:
        corners, tel = _synthetic_lap()
        windows = compute_turn_windows(corners, tel, OFFSET_S)
        tmp = Path(tempfile.mkdtemp())
        try:
            save_turn_windows(windows, tmp / "x_turns.json", OFFSET_S)
            self.assertEqual(load_turn_windows(tmp / "x_turns.json"), windows)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
