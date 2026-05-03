"""Unit tests for racing_line_search — A*, Dijkstra, BFS, and deviation."""

import logging
import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from racing_line_grid import GridNode
from racing_line_search import (
    SearchResult,
    astar,
    bfs,
    compute_deviation_per_corner,
    compute_path_cost_weighted,
    dijkstra,
)


class TestRacingLineSearch(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.CRITICAL)  # Silence logger for tests

        self.grid = {}
        for i in range(5):
            for j in range(5):
                self.grid[(i, j)] = GridNode(
                    grid_i=i,
                    grid_j=j,
                    center_x=i * 50.0,
                    center_y=j * 50.0,
                    mean_speed=100.0,
                    mean_brake=0.0,
                    weight=1.0,
                    point_count=1
                )

        self.adj = {}
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for i in range(5):
            for j in range(5):
                neighbors = []
                for di, dj in offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 5 and 0 <= nj < 5:
                        edge_cost = 1.0 if (di == 0 or dj == 0) else 1.414
                        neighbors.append(((ni, nj), edge_cost))
                self.adj[(i, j)] = neighbors

    def test_astar_finds_path(self):
        result = astar(self.grid, self.adj, (0, 0), (4, 4), self.logger)
        self.assertTrue(result.found)
        self.assertGreater(len(result.path_keys), 0)
        self.assertEqual(result.path_keys[0], (0, 0))
        self.assertEqual(result.path_keys[-1], (4, 4))

    def test_astar_path_is_connected(self):
        result = astar(self.grid, self.adj, (0, 0), (4, 4), self.logger)
        for i in range(len(result.path_keys) - 1):
            k1 = result.path_keys[i]
            k2 = result.path_keys[i+1]
            self.assertLessEqual(abs(k1[0] - k2[0]), 1)
            self.assertLessEqual(abs(k1[1] - k2[1]), 1)

    def test_dijkstra_finds_same_cost_as_astar(self):
        astar_r = astar(self.grid, self.adj, (0, 0), (4, 4), self.logger)
        dijkstra_r = dijkstra(self.grid, self.adj, (0, 0), (4, 4), self.logger)
        self.assertTrue(dijkstra_r.found)
        self.assertLess(abs(astar_r.total_cost - dijkstra_r.total_cost), 0.001)

    def test_bfs_finds_path(self):
        result = bfs(self.grid, self.adj, (0, 0), (4, 4), self.logger)
        self.assertTrue(result.found)
        self.assertEqual(result.path_keys[0], (0, 0))
        self.assertEqual(result.path_keys[-1], (4, 4))

    def test_bfs_cost_is_zero(self):
        result = bfs(self.grid, self.adj, (0, 0), (4, 4), self.logger)
        self.assertEqual(result.total_cost, 0.0)

    def test_astar_nodes_expanded_less_than_dijkstra(self):
        astar_r = astar(self.grid, self.adj, (0, 0), (4, 4), self.logger)
        dijkstra_r = dijkstra(self.grid, self.adj, (0, 0), (4, 4), self.logger)
        self.assertLessEqual(astar_r.nodes_expanded, dijkstra_r.nodes_expanded)

    def test_no_path_returns_found_false(self):
        adj_blocked = {
            k: [(n, c) for (n, c) in v if n != (4, 4)]
            for k, v in self.adj.items()
        }
        result = astar(self.grid, adj_blocked, (0, 0), (4, 4), self.logger)
        self.assertFalse(result.found)
        self.assertEqual(result.path_keys, [])

    def test_compute_path_cost_weighted(self):
        path = [(0, 0), (1, 0), (2, 0)]
        expected = 0.0
        for i in range(2):
            u, v = path[i], path[i+1]
            for n, c in self.adj[u]:
                if n == v:
                    expected += c
                    break
        result = compute_path_cost_weighted(path, self.grid, self.adj)
        self.assertLess(abs(result - expected), 0.001)

    def test_compute_deviation_per_corner_length(self):
        sr = SearchResult(
            algorithm="astar",
            path_keys=[],
            path_coords=[(float(i), 0.0) for i in range(100)],
            total_cost=0.0,
            nodes_expanded=0,
            compute_time_s=0.0,
            found=True
        )
        driver_path = [(float(i), 0.0) for i in range(100)]
        result = compute_deviation_per_corner(sr, driver_path, scale=0.01, n_corners=8)
        self.assertEqual(len(result), 8)
        for item in result:
            self.assertIn("corner", item)
            self.assertIn("deviation_m", item)

    def test_compute_deviation_zero_for_identical_paths(self):
        path_coords = [(float(i), 0.0) for i in range(100)]
        sr = SearchResult(
            algorithm="astar",
            path_keys=[],
            path_coords=path_coords,
            total_cost=0.0,
            nodes_expanded=0,
            compute_time_s=0.0,
            found=True
        )
        result = compute_deviation_per_corner(sr, path_coords, 0.01, 8)
        for item in result:
            dev = item["deviation_m"]
            if dev is not None:
                self.assertLess(dev, 0.01)


if __name__ == '__main__':
    unittest.main()
