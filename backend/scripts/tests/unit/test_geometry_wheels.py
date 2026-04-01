"""Unit tests for compute_wheel_positions."""

import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from inference_geometry import (
    compute_wheel_positions, LEFT_WHEEL_X_PCT, RIGHT_WHEEL_X_PCT, WHEEL_Y_PCT,
)


class TestGeometryWheels(unittest.TestCase):

    def test_proportions(self):
        left, right = compute_wheel_positions(1920, 1080)
        self.assertEqual(left[0], int(1920 * LEFT_WHEEL_X_PCT))
        self.assertEqual(right[0], int(1920 * RIGHT_WHEEL_X_PCT))
        self.assertEqual(left[1], int(1080 * WHEEL_Y_PCT))
        self.assertEqual(right[1], left[1])

    def test_returns_integers(self):
        left, right = compute_wheel_positions(1920, 1080)
        for val in [left[0], left[1], right[0], right[1]]:
            self.assertIsInstance(val, int)


if __name__ == "__main__":
    unittest.main()
