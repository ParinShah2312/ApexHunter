"""Unit tests for get_closest_distance."""

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from inference_geometry import get_closest_distance


class TestGeometryDistance(unittest.TestCase):

    def test_single_point_at_ref(self):
        contour = np.array([[10, 10]])
        dist, pt = get_closest_distance((10, 10), contour)
        self.assertAlmostEqual(dist, 0.0)
        self.assertEqual(pt, (10, 10))

    def test_known_geometry(self):
        contour = np.array([[0, 0], [3, 0], [0, 4]])
        dist, pt = get_closest_distance((0, 0), contour)
        self.assertAlmostEqual(dist, 0.0)
        self.assertEqual(pt, (0, 0))

    def test_3d_contour(self):
        pts = np.array([[[10, 10]], [[20, 20]], [[30, 30]], [[5, 0]], [[0, 5]]])
        dist, pt = get_closest_distance((0, 0), pts)
        self.assertAlmostEqual(dist, 5.0, places=1)

    def test_empty_contour_none(self):
        dist, pt = get_closest_distance((10, 10), None)
        self.assertEqual(dist, float('inf'))
        self.assertIsNone(pt)

    def test_empty_contour_array(self):
        dist, pt = get_closest_distance((10, 10), np.array([]))
        self.assertEqual(dist, float('inf'))
        self.assertIsNone(pt)


if __name__ == "__main__":
    unittest.main()
