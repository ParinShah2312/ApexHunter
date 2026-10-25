"""Unit tests for racing_line/racing_line_io.py: loading, fastest-lap slicing, output building,
time-saved estimate and saving."""

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from racing_line.racing_line_grid import GridNode
from racing_line.racing_line_io import (
    build_output, compute_time_saved, fetch_fastest_lap_bounds, load_and_validate, save_output,
)
from racing_line.racing_line_search import SearchResult

LOG = logging.getLogger("test_racing_line_io")
LOG.addHandler(logging.NullHandler())
LOG.propagate = False


def _sr(coords, cost=1.0, found=True, name="astar") -> SearchResult:
    return SearchResult(name, [(0, 0)] * len(coords), coords, cost, 7, 0.123456, found)


class TestLoadAndValidate(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "s.parquet"
        pd.DataFrame({
            "Driver": ["44", "44", "1", "44"], "X": [0.0, 1.0, 2.0, 3.0], "Y": [0.0] * 4,
            "Speed": [100.0] * 4, "Brake": [0.0] * 4,
            "SessionTime": pd.to_timedelta([3, 1, 2, 2], unit="s"),
        }).to_parquet(self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_filters_driver_and_sorts_by_time(self) -> None:
        df = load_and_validate(self.path, "44", LOG)
        self.assertEqual(df["X"].tolist(), [1.0, 3.0, 0.0])          # ordered by SessionTime
        self.assertEqual(list(df.index), [0, 1, 2])

    def test_missing_column_raises(self) -> None:
        pd.read_parquet(self.path).drop(columns=["Brake"]).to_parquet(self.path)
        with self.assertRaisesRegex(ValueError, "Brake"):
            load_and_validate(self.path, "44", LOG)

    def test_unknown_driver_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "99"):
            load_and_validate(self.path, "99", LOG)


class TestFetchFastestLapBounds(unittest.TestCase):

    def setUp(self) -> None:
        self.df = pd.DataFrame({"Year": [2023] * 6, "Round": [1] * 6, "Session": ["Q"] * 6,
                                "SessionTime": pd.to_timedelta(range(0, 60, 10), unit="s"), "X": range(6)})

    def test_slices_to_fastest_lap(self) -> None:
        session = mock.MagicMock()
        session.laps.pick_drivers.return_value.pick_fastest.return_value = {
            "LapStartTime": pd.Timedelta(seconds=10), "Time": pd.Timedelta(seconds=30)}
        with mock.patch("fastf1.get_session", return_value=session) as get_session, \
             mock.patch("racing_line.racing_line_io.enable_fastf1_cache"):
            lap = fetch_fastest_lap_bounds(self.df, "1", LOG)
        get_session.assert_called_once_with(2023, 1, "Q")
        session.laps.pick_drivers.assert_called_once_with("1")
        self.assertEqual(lap["X"].tolist(), [1, 2, 3])

    def test_falls_back_to_full_session_on_failure(self) -> None:
        with mock.patch("fastf1.get_session", side_effect=RuntimeError("offline")), \
             mock.patch("racing_line.racing_line_io.enable_fastf1_cache"):
            self.assertIs(fetch_fastest_lap_bounds(self.df, "1", LOG), self.df)


class TestComputeTimeSaved(unittest.TestCase):
    """All nodes run at 180 km/h (50 m/s), so time = path length (m) / 50."""

    def setUp(self) -> None:
        self.grid = {(i, 0): GridNode(i, 0, i * 50.0, 0.0, 180.0, 0.0, 1.0, 1) for i in range(30)}

    def test_shorter_optimal_path_saves_time(self) -> None:
        optimal = [(0.0, 0.0), (100.0, 0.0)]                       # 100 m → 2.0 s
        driver = [(0.0, 0.0), (60.0, 0.0), (120.0, 0.0)]           # 120 m → 2.4 s
        self.assertAlmostEqual(compute_time_saved(_sr(optimal), driver, self.grid, scale=1.0), 0.4)

    def test_clamped_to_five_seconds(self) -> None:
        optimal = [(0.0, 0.0), (100.0, 0.0)]                       # 2 s
        driver = [(0.0, 0.0), (1000.0, 0.0)]                       # 20 s → 18 s, clamped
        self.assertEqual(compute_time_saved(_sr(optimal), driver, self.grid, scale=1.0), 5.0)

    def test_none_when_no_path(self) -> None:
        self.assertIsNone(compute_time_saved(_sr([], found=False), [(0.0, 0.0)], self.grid, 1.0))


class TestOutputs(unittest.TestCase):

    def test_build_output_values(self) -> None:
        a, d, b = _sr([(0.0, 0.0), (1.0, 1.0)], 3.33333), _sr([(0.0, 0.0)], 3.33333, name="dijkstra"), \
            _sr([(0.0, 0.0)], 0.0, name="bfs")
        data = build_output("s.parquet", "44", 50.0, 5.4, 0.3, a, d, b, 4.444449, [(1.0, 2.0)], [], 15)
        self.assertEqual(data["driver_path"], [[1.0, 2.0]])
        self.assertEqual(data["algorithms"]["astar"]["path"], [[0.0, 0.0], [1.0, 1.0]])
        self.assertEqual(data["algorithms"]["astar"]["cost"], 3.3333)
        self.assertEqual(data["algorithms"]["bfs"]["cost"], 4.4444)       # weighted cost, not BFS's 0
        self.assertEqual(data["algorithms"]["dijkstra"]["compute_time_s"], 0.1235)
        self.assertIsNone(data["algorithms"]["astar"]["time_saved_s"])    # filled in by the pipeline
        self.assertEqual((data["n_corners"], data["coordinate_scale"]), (15, 0.3))

    def test_save_output_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            save_output({"a": [1, 2]}, path, LOG)
            with open(path) as f:
                self.assertEqual(json.load(f), {"a": [1, 2]})


if __name__ == "__main__":
    unittest.main()
