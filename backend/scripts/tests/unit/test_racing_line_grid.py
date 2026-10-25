"""Unit tests for racing_line/racing_line_grid.py: grid cells, node cost, adjacency,
nearest-node lookup, coordinate scale and start/end selection."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from racing_line.racing_line_grid import (
    DIAGONAL_COST_FACTOR, GridNode, build_adjacency, build_grid, compute_coordinate_scale,
    compute_node_weight, find_start_end_nodes, get_nearest_node,
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


class TestCoordinateScale(unittest.TestCase):

    def test_known_circle(self) -> None:
        """One 90 s lap around a 10,000-unit circle for a 5 km circuit → 0.5 m per unit."""
        t = np.linspace(0, 2 * math.pi, 5001)
        r = 10_000 / (2 * math.pi)
        df = pd.DataFrame({"X": r * np.cos(t), "Y": r * np.sin(t),
                           "SessionTime": pd.to_timedelta(np.linspace(0, 90, len(t)), unit="s")})
        self.assertAlmostEqual(compute_coordinate_scale(df, 5.0), 0.5, places=3)


class TestStartEndNodes(unittest.TestCase):

    def test_start_uses_same_grid_origin_as_build_grid(self) -> None:
        """Regression for #17: a slow pit-lane stretch to the left (connected to the track) sets
        the grid origin; the start must be the cell of the first fast sample under that origin."""
        pit_x = np.arange(-500.0, 0.0, 10.0)
        pit = pd.DataFrame({"X": pit_x, "Y": np.zeros_like(pit_x), "Speed": 5.0, "Brake": 0.0})
        lap_x = np.arange(0.0, 1000.0, 10.0)
        lap = pd.DataFrame({"X": lap_x, "Y": np.zeros_like(lap_x), "Speed": 200.0, "Brake": 0.0})
        df = pd.concat([pit, lap], ignore_index=True)
        df["SessionTime"] = pd.to_timedelta(np.arange(len(df)), unit="s")

        grid = build_grid(df, 50.0)
        start, end = find_start_end_nodes(grid, df, 50.0, build_adjacency(grid))
        self.assertEqual(start, (10, 0))      # x = 0 is 500 units = 10 cells from origin -500
        self.assertEqual(end, (29, 0))        # furthest cell of the connected track


if __name__ == "__main__":
    unittest.main()
