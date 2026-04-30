import json
import logging
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sys

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from racing_line_grid import GridNode
from racing_line_io import (
    build_output,
    compute_time_saved,
    load_and_validate,
    save_output,
)
from racing_line_search import SearchResult


class TestRacingLineIO(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.CRITICAL)
        
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        self.df = pd.DataFrame({
            "Driver": ["44", "44", "1", "1"],
            "X": [0.0, 10.0, 0.0, 10.0],
            "Y": [0.0, 10.0, 0.0, 10.0],
            "Speed": [100.0, 200.0, 150.0, 250.0],
            "Brake": [0.0, 10.0, 0.0, 20.0],
            "SessionTime": pd.to_timedelta([1, 2, 1, 2], unit="s")
        })
        self.valid_parquet = self.temp_path / "valid.parquet"
        self.df.to_parquet(self.valid_parquet)
        
        self.dummy_sr = SearchResult(
            algorithm="astar",
            path_keys=[(0, 0), (1, 1)],
            path_coords=[(0.0, 0.0), (10.0, 10.0)],
            total_cost=10.5,
            nodes_expanded=5,
            compute_time_s=0.1,
            found=True
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_and_validate_returns_df(self):
        df_res = load_and_validate(self.valid_parquet, "44", self.logger)
        self.assertIsInstance(df_res, pd.DataFrame)
        self.assertGreater(len(df_res), 0)

    def test_load_and_validate_filters_driver(self):
        df_res = load_and_validate(self.valid_parquet, "44", self.logger)
        self.assertTrue((df_res["Driver"] == "44").all())

    def test_load_and_validate_missing_column_raises(self):
        df_bad = self.df.drop(columns=["Brake"])
        bad_parquet = self.temp_path / "bad.parquet"
        df_bad.to_parquet(bad_parquet)
        with self.assertRaises(ValueError):
            load_and_validate(bad_parquet, "44", self.logger)

    def test_load_and_validate_missing_driver_raises(self):
        with self.assertRaises(ValueError):
            load_and_validate(self.valid_parquet, "99", self.logger)

    def test_build_output_has_all_keys(self):
        data = build_output(
            session_path="test.parquet",
            driver="44",
            grid_resolution=50.0,
            circuit_length_km=5.0,
            scale=0.1,
            astar_result=self.dummy_sr,
            dijkstra_result=self.dummy_sr,
            bfs_result=self.dummy_sr,
            astar_cost_bfs=15.0,
            driver_path_coords=[(0.0, 0.0), (10.0, 10.0)],
            deviation_per_corner=[],
            n_corners=16
        )
        
        required_keys = [
            "session_file", "driver", "grid_resolution", "circuit_length_km",
            "coordinate_scale", "n_corners", "timestamp", "driver_path",
            "algorithms", "deviation_per_corner"
        ]
        for key in required_keys:
            self.assertIn(key, data)
            
        self.assertIn("astar", data["algorithms"])
        self.assertIn("dijkstra", data["algorithms"])
        self.assertIn("bfs", data["algorithms"])
        
        algo_keys = ["path", "cost", "nodes_expanded", "compute_time_s", "time_saved_s", "found"]
        for algo in ["astar", "dijkstra", "bfs"]:
            for k in algo_keys:
                self.assertIn(k, data["algorithms"][algo])

    def test_save_output_creates_file(self):
        out_path = self.temp_path / "test.json"
        save_output({"test": 1}, out_path, self.logger)
        self.assertTrue(out_path.exists())

    def test_save_output_is_valid_json(self):
        out_path = self.temp_path / "test2.json"
        save_output({"test": 1}, out_path, self.logger)
        with open(out_path, "r") as f:
            loaded = json.load(f)
        self.assertEqual(loaded, {"test": 1})

    def test_compute_time_saved_returns_float_or_none(self):
        grid = {
            (0, 0): GridNode(0, 0, 0.0, 0.0, 100.0, 0.0, 1.0, 1),
            (1, 1): GridNode(1, 1, 10.0, 10.0, 100.0, 0.0, 1.0, 1)
        }
        driver_path = [(0.0, 0.0), (10.0, 10.0)]
        result = compute_time_saved(self.dummy_sr, driver_path, grid, scale=0.1)
        self.assertTrue(isinstance(result, float) or result is None)

    def test_compute_time_saved_none_if_not_found(self):
        grid = {}
        driver_path = [(0.0, 0.0), (10.0, 10.0)]
        sr_not_found = SearchResult("astar", [], [], 0.0, 0, 0.0, False)
        result = compute_time_saved(sr_not_found, driver_path, grid, scale=0.1)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
