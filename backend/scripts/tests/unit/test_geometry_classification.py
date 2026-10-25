"""Unit tests for classify_apex_status."""

import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from cv.inference_geometry import classify_apex_status, HIT_THRESHOLD, NEAR_THRESHOLD


class TestGeometryClassification(unittest.TestCase):

    def test_hitting_apex(self):
        status, color = classify_apex_status(50.0, True, "Left", "Medium")
        self.assertEqual(status, "Hitting Apex")
        self.assertEqual(color, (0, 255, 0))

    def test_near_apex(self):
        status, color = classify_apex_status(HIT_THRESHOLD + 10, True, "Left", "Medium")
        self.assertEqual(status, "Near Apex")
        self.assertEqual(color, (0, 255, 255))

    def test_missing_apex(self):
        status, color = classify_apex_status(NEAR_THRESHOLD + 10, True, "Right", "Medium")
        self.assertEqual(status, "Missing Apex")
        self.assertEqual(color, (0, 0, 255))

    def test_straight_outside_turn_window(self):
        status, color = classify_apex_status(50.0, True, "-", "Medium")
        self.assertEqual(status, "Straight")
        self.assertEqual(color, (200, 200, 200))

    def test_tracking_when_no_curb_in_turn(self):
        status, _ = classify_apex_status(50.0, False, "Left", "Medium")
        self.assertEqual(status, "Tracking")
        status, _ = classify_apex_status(float("inf"), True, "Left", "Medium")
        self.assertEqual(status, "Tracking")

    def test_far_distance_treated_as_exit_curb(self):
        status, _ = classify_apex_status(950.0, True, "Left", "Medium")
        self.assertEqual(status, "Tracking")

    def test_boundary_hit(self):
        status, _ = classify_apex_status(HIT_THRESHOLD - 1, True, "Left", "Medium")
        self.assertEqual(status, "Hitting Apex")

    def test_boundary_near(self):
        status, _ = classify_apex_status(HIT_THRESHOLD, True, "Left", "Medium")
        self.assertEqual(status, "Near Apex")

    def test_slow_corner_is_stricter(self):
        d = HIT_THRESHOLD * 0.9   # inside the medium threshold, outside the slow (×0.75) one
        self.assertEqual(classify_apex_status(d, True, "Left", "Medium")[0], "Hitting Apex")
        self.assertEqual(classify_apex_status(d, True, "Left", "Slow")[0], "Near Apex")

    def test_fast_corner_is_more_forgiving_and_reports_wide_line(self):
        d = HIT_THRESHOLD * 1.2   # outside medium hit threshold, inside fast (×1.35) one
        self.assertEqual(classify_apex_status(d, True, "Right", "Medium")[0], "Near Apex")
        self.assertEqual(classify_apex_status(d, True, "Right", "Fast")[0], "Hitting Apex")
        self.assertEqual(classify_apex_status(NEAR_THRESHOLD * 1.5, True, "Right", "Fast")[0], "Wide Line")


if __name__ == "__main__":
    unittest.main()
