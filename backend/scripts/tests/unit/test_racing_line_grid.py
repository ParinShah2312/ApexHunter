"""Unit tests for racing_line/racing_line_grid.py: grid cells, node cost, adjacency,
nearest-node lookup and lap length."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from racing_line.racing_line_grid import (
    DIAGONAL_COST_FACTOR, GridNode, build_adjacency, build_grid, compute_node_weight,
    get_nearest_node, lap_length_km,
)


def _node(i: int, j: int, weight: float = 1.0) -> GridNode:
    return GridNode(i, j, i * 50.0, j * 50.0, 200.0, 0.0, weight, 1)


class TestBuildGrid(unittest.TestCase):

    def test_cells_aggregate_their_samples(self) -> None:
        df = pd.DataFrame({
            "X": [0.0, 10.0, 60.0, 70.0, 80.0], "Y": [0.0, 20.0, 0.0, 0.0, 49.0],
            "Speed": [100.0, 200.0, 300.0, 300.0, 300.0], "Brake": [1.0, 0.0, 0.0, 0.0, 1.0],
        })
        grid = build_grid(df, resolution=50.0)
        self.assertEqual(set(grid), {(0, 0), (1, 0)})
        a, b = grid[(0, 0)], grid[(1, 0)]
        self.assertEqual((a.point_count, b.point_count), (2, 3))
        self.assertAlmostEqual(a.mean_speed, 150.0)
        self.assertAlmostEqual(a.mean_brake, 0.5)                 # 1 of 2 samples braking
        self.assertAlmostEqual(b.center_x, 70.0)
        self.assertAlmostEqual(b.center_y, 49.0 / 3)
        self.assertAlmostEqual(a.weight, compute_node_weight(150.0, 0.5))


class TestNodeWeight(unittest.TestCase):

    def test_formula(self) -> None:
        # 1 - speed/380 + 2 * brake_fraction
        self.assertAlmostEqual(compute_node_weight(190.0, 0.0), 0.5)
        self.assertAlmostEqual(compute_node_weight(190.0, 0.5), 1.5)
        self.assertAlmostEqual(compute_node_weight(0.0, 1.0), 3.0)

    def test_floor_and_brake_clip(self) -> None:
        self.assertEqual(compute_node_weight(380.0, 0.0), 0.01)  # never zero/negative
        self.assertEqual(compute_node_weight(200.0, 5.0), compute_node_weight(200.0, 1.0))


class TestAdjacency(unittest.TestCase):

    def test_neighbours_and_edge_costs(self) -> None:
        grid = {(0, 0): _node(0, 0, 1.0), (1, 0): _node(1, 0, 3.0), (1, 1): _node(1, 1, 2.0),
                (5, 5): _node(5, 5)}                               # isolated cell
        adj = build_adjacency(grid)
        self.assertEqual(dict(adj[(0, 0)]), {(1, 0): 2.0, (1, 1): 1.5 * DIAGONAL_COST_FACTOR})
        self.assertEqual(dict(adj[(1, 0)]), {(0, 0): 2.0, (1, 1): 2.5})
        self.assertNotIn((5, 5), adj)


class TestNearestNode(unittest.TestCase):

    def test_off_grid_target_snaps_to_closest_cell(self) -> None:
        grid = {(0, 0): _node(0, 0), (10, 0): _node(10, 0), (10, 10): _node(10, 10)}
        self.assertEqual(get_nearest_node(grid, 9, 2), (10, 0))
        self.assertEqual(get_nearest_node(grid, 2, 1), (0, 0))

    def test_empty_grid_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_nearest_node({}, 0, 0)


class TestLapLength(unittest.TestCase):

    def test_known_circle(self) -> None:
        """A 50,000-unit circle is 5,000 m at FastF1's 1/10 m per unit → 5.0 km."""
        t = np.linspace(0, 2 * math.pi, 5001)
        r = 50_000 / (2 * math.pi)
        df = pd.DataFrame({"X": r * np.cos(t), "Y": r * np.sin(t)})
        self.assertAlmostEqual(lap_length_km(df), 5.0, places=3)


if __name__ == "__main__":
    unittest.main()
