"""Integration tests: end-to-end optimal racing line pipeline."""

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

RUN_SLOW = os.environ.get("APEXHUNTER_RUN_SLOW", "0") == "1"
slow_test = unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")

from optimal_line import run_pipeline

_FIXTURE_CACHE = {}


def run_pipeline_once():
    """Run the racing line pipeline once and cache the result."""
    if "data" in _FIXTURE_CACHE:
        return _FIXTURE_CACHE["data"]

    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)

    n_rows = 1000
    t = np.linspace(0, 6 * np.pi, n_rows)  # 3 laps
    x = np.cos(t) * 3000.0
    y = np.sin(t) * 1500.0

    df = pd.DataFrame({
        "Driver": ["44"] * n_rows,
        "X": x.astype(np.float32),
        "Y": y.astype(np.float32),
        "Speed": np.random.uniform(100, 300, n_rows).astype(np.float32),
        "Brake": np.random.uniform(0, 30, n_rows).astype(np.float32),
        "SessionTime": pd.to_timedelta(np.linspace(0, 270, n_rows), unit="s")
    })
    session_file = temp_path / "synthetic_session.parquet"
    df.to_parquet(session_file)

    output_dir = temp_path / "output"
    args = argparse.Namespace(
        session=str(session_file),
        driver="44",
        resolution=100.0,
        circuit_length=5.0,
        n_corners=16,
        output_dir=str(output_dir),
        force=False
    )
    
    run_pipeline(args)
    
    expected_output_path = output_dir / "synthetic_session_44_racing_line.json"
    
    with open(expected_output_path, "r") as f:
        data = json.load(f)

    _FIXTURE_CACHE["data"] = {
        "temp_dir": temp_dir,
        "temp_path": temp_path,
        "session_file": session_file,
        "output_dir": output_dir,
        "output_path": expected_output_path,
        "json_data": data,
        "args": args
    }
    
    return _FIXTURE_CACHE["data"]


@slow_test
class TestRacingLinePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = run_pipeline_once()
        cls.output_path = cls.fixture["output_path"]
        cls.data = cls.fixture["json_data"]
        cls.args = cls.fixture["args"]

    @classmethod
    def tearDownClass(cls):
        if "data" in _FIXTURE_CACHE:
            _FIXTURE_CACHE["data"]["temp_dir"].cleanup()
            _FIXTURE_CACHE.clear()

    def test_output_file_created(self):
        self.assertTrue(self.output_path.exists())

    def test_output_json_valid(self):
        self.assertIsInstance(self.data, dict)
        self.assertIn("algorithms", self.data)

    def test_all_algorithms_ran(self):
        self.assertTrue(self.data["algorithms"]["astar"]["found"])
        self.assertTrue(self.data["algorithms"]["dijkstra"]["found"])
        self.assertTrue(self.data["algorithms"]["bfs"]["found"])

    def test_paths_start_at_same_node(self):
        astar_start = self.data["algorithms"]["astar"]["path"][0]
        dijkstra_start = self.data["algorithms"]["dijkstra"]["path"][0]
        self.assertEqual(astar_start, dijkstra_start)

    def test_astar_cost_le_dijkstra_cost(self):
        a_cost = self.data["algorithms"]["astar"]["cost"]
        d_cost = self.data["algorithms"]["dijkstra"]["cost"]
        self.assertLess(abs(a_cost - d_cost), 0.01)

    def test_bfs_nodes_expanded_gte_astar(self):
        bfs_nodes = self.data["algorithms"]["bfs"]["nodes_expanded"]
        astar_nodes = self.data["algorithms"]["astar"]["nodes_expanded"]
        self.assertGreaterEqual(bfs_nodes, astar_nodes)

    def test_deviation_per_corner_correct_length(self):
        self.assertEqual(len(self.data["deviation_per_corner"]), self.data["n_corners"])

    def test_driver_path_is_subsampled(self):
        self.assertLess(len(self.data["driver_path"]), 1000)

    def test_skip_if_exists(self):
        mtime_before = self.output_path.stat().st_mtime
        run_pipeline(self.args)
        mtime_after = self.output_path.stat().st_mtime
        self.assertEqual(mtime_before, mtime_after)

    def test_force_overwrites(self):
        mtime_before = self.output_path.stat().st_mtime
        import time
        time.sleep(0.01)  # Ensure time ticks forward for mtime resolution in some OS
        
        args_force = argparse.Namespace(**vars(self.args))
        args_force.force = True
        run_pipeline(args_force)
        
        mtime_after = self.output_path.stat().st_mtime
        self.assertNotEqual(mtime_before, mtime_after)


if __name__ == '__main__':
    unittest.main()
