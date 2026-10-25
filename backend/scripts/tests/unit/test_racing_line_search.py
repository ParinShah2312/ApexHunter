"""Unit tests for racing_line/racing_line_search.py: A*, Dijkstra, BFS, path cost, full-lap
stitching and per-segment deviation."""

import logging
import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from racing_line.racing_line_grid import GridNode, build_adjacency
from racing_line.racing_line_search import (
    SearchResult, astar, bfs, compute_deviation_per_corner, compute_path_cost_weighted,
    dijkstra, run_full_lap,
)

LOG = logging.getLogger("test_racing_line_search")
LOG.addHandler(logging.NullHandler())
LOG.propagate = False


def _grid(size: int = 5, expensive=()):
    """size×size grid of nodes, weight 1.0 except the given expensive cells (weight 10)."""
    grid = {(i, j): GridNode(i, j, i * 50.0, j * 50.0, 200.0, 0.0, 10.0 if (i, j) in expensive else 1.0, 1)
            for i in range(size) for j in range(size)}
    return grid, build_adjacency(grid)


def _result(coords, found=True) -> SearchResult:
    return SearchResult("astar", [], coords, 0.0, 0, 0.0, found)


class TestCostAwareSearch(unittest.TestCase):
    """A braking-zone wall at column i=2 (rows 0-3) with a cheap gap at row 4."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.grid, cls.adj = _grid(5, expensive={(2, 0), (2, 1), (2, 2), (2, 3)})
        cls.start, cls.end = (0, 0), (4, 0)
        cls.a = astar(cls.grid, cls.adj, cls.start, cls.end, LOG)
        cls.d = dijkstra(cls.grid, cls.adj, cls.start, cls.end, LOG)
        cls.b = bfs(cls.grid, cls.adj, cls.start, cls.end, LOG)

    def test_all_find_a_connected_path_between_endpoints(self) -> None:
        for r in (self.a, self.d, self.b):
            with self.subTest(algorithm=r.algorithm):
                self.assertTrue(r.found)
                self.assertEqual((r.path_keys[0], r.path_keys[-1]), (self.start, self.end))
                for (i1, j1), (i2, j2) in zip(r.path_keys, r.path_keys[1:]):
                    self.assertLessEqual(max(abs(i1 - i2), abs(j1 - j2)), 1)
                self.assertEqual(r.path_coords[0], (0.0, 0.0))

    def test_cost_aware_search_detours_around_braking_zone(self) -> None:
        for r in (self.a, self.d):
            with self.subTest(algorithm=r.algorithm):
                self.assertNotIn((2, 0), r.path_keys)
                self.assertIn((2, 4), r.path_keys)           # goes through the cheap gap

    def test_bfs_ignores_cost_and_goes_straight_through(self) -> None:
        self.assertEqual(len(self.b.path_keys), 5)            # fewest hops: straight along j=0
        self.assertTrue(any(k[0] == 2 and k[1] < 4 for k in self.b.path_keys))

    def test_astar_equals_dijkstra_and_beats_bfs(self) -> None:
        self.assertAlmostEqual(self.a.total_cost, self.d.total_cost, places=6)
        self.assertAlmostEqual(self.a.total_cost, compute_path_cost_weighted(self.a.path_keys, self.grid, self.adj))
        bfs_cost = compute_path_cost_weighted(self.b.path_keys, self.grid, self.adj)
        self.assertGreater(bfs_cost, self.a.total_cost)

    def test_astar_heuristic_prunes(self) -> None:
        self.assertLess(self.a.nodes_expanded, self.d.nodes_expanded)


class TestSearchEdgeCases(unittest.TestCase):

    def test_unreachable_goal(self) -> None:
        grid, adj = _grid(3)
        adj = {k: [(n, c) for n, c in v if n != (2, 2)] for k, v in adj.items()}
        for fn in (astar, dijkstra, bfs):
            with self.subTest(algorithm=fn.__name__):
                r = fn(grid, adj, (0, 0), (2, 2), LOG)
                self.assertFalse(r.found)
                self.assertEqual(r.path_keys, [])

    def test_start_not_in_grid(self) -> None:
        grid, adj = _grid(3)
        for fn in (astar, dijkstra, bfs):
            self.assertFalse(fn(grid, adj, (9, 9), (0, 0), LOG).found)

    def test_path_cost_of_known_path(self) -> None:
        grid, adj = _grid(3, expensive={(1, 0)})
        # edges: (0,0)->(1,0) avg(1,10)=5.5 ; (1,0)->(2,0) avg(10,1)=5.5
        self.assertAlmostEqual(compute_path_cost_weighted([(0, 0), (1, 0), (2, 0)], grid, adj), 11.0)


class TestRunFullLap(unittest.TestCase):

    def test_segments_are_stitched_and_totals_summed(self) -> None:
        grid, adj = _grid(5)
        waypoints = [(0, 0), (4, 0), (4, 4)]
        lap = run_full_lap(dijkstra, waypoints, grid, adj, LOG)
        seg1 = dijkstra(grid, adj, (0, 0), (4, 0), LOG)
        seg2 = dijkstra(grid, adj, (4, 0), (4, 4), LOG)
        self.assertTrue(lap.found)
        self.assertEqual(lap.path_keys, seg1.path_keys + seg2.path_keys[1:])   # joint not duplicated
        self.assertAlmostEqual(lap.total_cost, seg1.total_cost + seg2.total_cost)
        self.assertEqual(lap.nodes_expanded, seg1.nodes_expanded + seg2.nodes_expanded)

    def test_failed_segment_fails_the_lap(self) -> None:
        grid, adj = _grid(3)
        adj = {k: [(n, c) for n, c in v if n != (2, 2)] for k, v in adj.items()}
        self.assertFalse(run_full_lap(astar, [(0, 0), (1, 1), (2, 2)], grid, adj, LOG).found)


class TestDeviationPerCorner(unittest.TestCase):

    def test_parallel_offset_paths(self) -> None:
        optimal = [(float(x), 0.0) for x in range(100)]
        driver = [(float(x), 10.0) for x in range(100)]       # 10 units to the side
        devs = compute_deviation_per_corner(_result(optimal), driver, scale=0.5, n_corners=5)
        self.assertEqual([d["corner"] for d in devs], ["T1", "T2", "T3", "T4", "T5"])
        for d in devs:
            self.assertAlmostEqual(d["deviation_m"], 5.0, places=2)   # 10 units × 0.5 m/unit

    def test_identical_paths_zero_within_resampling_resolution(self) -> None:
        # The driver path is resampled to 1,000 points and matched to the nearest one, so the
        # error is bounded by half the resampling spacing (99 units / 999 / 2 ≈ 0.05 units).
        path = [(float(x), 0.0) for x in range(100)]
        for d in compute_deviation_per_corner(_result(path), path, scale=1.0, n_corners=4):
            self.assertLess(d["deviation_m"], 0.05)

    def test_no_path_gives_none_per_corner(self) -> None:
        devs = compute_deviation_per_corner(_result([], found=False), [(0.0, 0.0)], 1.0, n_corners=3)
        self.assertEqual(devs, [{"corner": f"T{i}", "deviation_m": None} for i in (1, 2, 3)])


if __name__ == "__main__":
    unittest.main()
