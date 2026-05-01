import unittest
import numpy as np
import pandas as pd
from typing import Dict, Tuple

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from racing_line_grid import (
    build_grid,
    compute_node_weight,
    build_adjacency,
    get_nearest_node,
    compute_coordinate_scale,
    find_start_end_nodes,
    GridNode
)


class TestRacingLineGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a synthetic DataFrame with 500 rows
        n_rows = 500
        cls.df = pd.DataFrame({
            "X": np.linspace(0, 5000, n_rows, dtype=np.float32),
            "Y": np.sin(np.linspace(0, 4 * np.pi, n_rows)) * 500,
            "Speed": np.random.uniform(100, 300, n_rows).astype(np.float32),
            "Brake": np.random.uniform(0, 30, n_rows).astype(np.float32),
            "SessionTime": pd.to_timedelta(np.linspace(0, 90, n_rows), unit="s"),
            "Driver": ["44"] * n_rows
        })
        cls.df["Y"] = cls.df["Y"].astype(np.float32)

    def test_build_grid_returns_dict(self):
        grid = build_grid(self.df)
        self.assertIsInstance(grid, dict)
        self.assertGreater(len(grid), 0)

    def test_build_grid_all_nodes_have_weight(self):
        grid = build_grid(self.df)
        for node in grid.values():
            self.assertGreater(node.weight, 0.0)

    def test_build_grid_resolution_affects_size(self):
        grid_coarse = build_grid(self.df, resolution=500.0)
        grid_fine = build_grid(self.df, resolution=50.0)
        self.assertGreater(len(grid_fine), len(grid_coarse))

    def test_compute_node_weight_high_speed_low_cost(self):
        w1 = compute_node_weight(350.0, 0.0)
        w2 = compute_node_weight(100.0, 80.0)
        self.assertLess(w1, w2)

    def test_compute_node_weight_never_zero(self):
        w = compute_node_weight(380.0, 0.0)
        self.assertGreater(w, 0.0)

    def test_build_adjacency_returns_dict(self):
        grid = build_grid(self.df)
        adj = build_adjacency(grid)
        self.assertIsInstance(adj, dict)
        self.assertGreater(len(adj), 0)

    def test_build_adjacency_max_8_neighbors(self):
        grid = build_grid(self.df)
        adj = build_adjacency(grid)
        for neighbors in adj.values():
            self.assertLessEqual(len(neighbors), 8)

    def test_build_adjacency_edge_costs_positive(self):
        grid = build_grid(self.df)
        adj = build_adjacency(grid)
        for neighbors in adj.values():
            for _, cost in neighbors:
                self.assertGreater(cost, 0.0)

    def test_get_nearest_node_finds_closest(self):
        grid = build_grid(self.df)
        keys = list(grid.keys())
        target = keys[len(keys) // 2]
        result = get_nearest_node(grid, target[0], target[1])
        self.assertEqual(result, target)

    def test_compute_coordinate_scale_returns_float(self):
        scale = compute_coordinate_scale(self.df, 5.412)
        self.assertIsInstance(scale, float)
        self.assertGreaterEqual(scale, 0.001)
        self.assertLessEqual(scale, 1.0)

    def test_find_start_end_different(self):
        grid = build_grid(self.df)
        adj = build_adjacency(grid)
        start, end = find_start_end_nodes(grid, self.df, 50.0, adj)
        self.assertNotEqual(start, end)


if __name__ == '__main__':
    unittest.main()
