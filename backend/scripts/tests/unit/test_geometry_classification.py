"""Unit tests for classify_apex_status."""

import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from inference_geometry import classify_apex_status, HIT_THRESHOLD, NEAR_THRESHOLD


class TestGeometryClassification(unittest.TestCase):

    def test_hitting_apex(self):
        status, color = classify_apex_status(50.0, True)
        self.assertEqual(status, "Hitting Apex")
        self.assertEqual(color, (0, 255, 0))

    def test_near_apex(self):
        status, color = classify_apex_status(150.0, True)
        self.assertEqual(status, "Near Apex")
        self.assertEqual(color, (0, 255, 255))

    def test_missing_apex(self):
        status, color = classify_apex_status(300.0, True)
        self.assertEqual(status, "Missing Apex")
        self.assertEqual(color, (0, 0, 255))

    def test_no_curb(self):
        status, color = classify_apex_status(50.0, False)
        self.assertEqual(status, "Straight")
        self.assertEqual(color, (200, 200, 200))

    def test_inf_distance(self):
        status, _ = classify_apex_status(float('inf'), True)
        self.assertEqual(status, "Straight")

    def test_boundary_hit(self):
        status, _ = classify_apex_status(HIT_THRESHOLD - 1, True)
        self.assertEqual(status, "Hitting Apex")

    def test_boundary_near(self):
        status, _ = classify_apex_status(HIT_THRESHOLD, True)
        self.assertEqual(status, "Near Apex")


if __name__ == "__main__":
    unittest.main()
