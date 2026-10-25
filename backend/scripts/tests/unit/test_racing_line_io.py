"""Unit tests for racing_line/racing_line_io.py: loading, fastest-lap slicing, output building,
time-saved estimate and saving."""

import logging
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np

from racing_line.racing_line_io import (
    build_output, compute_time_saved, estimate_path_time, load_and_validate,
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



class TestTimeEstimate(unittest.TestCase):
    """Fastest lap: a straight line along X, 0-1000 units (100 m at 0.1 m/unit), sampled every
    10 units; 180 km/h (50 m/s) up to X = 500 and 90 km/h (25 m/s) after."""

    def setUp(self) -> None:
        x = np.arange(0.0, 1001.0, 10.0)
        self.lap = pd.DataFrame({"X": x, "Y": 0.0, "Speed": np.where(x <= 500, 180.0, 90.0)})

    def test_driver_lap_uses_nearest_lap_speeds(self) -> None:
        path = [(0.0, 0.0), (500.0, 0.0), (1000.0, 0.0)]
        # 50 m at 50 m/s + 50 m at 25 m/s
        self.assertAlmostEqual(estimate_path_time(path, self.lap, scale=0.1), 1.0 + 2.0)

    def test_speed_comes_from_nearest_sample_not_a_session_average(self) -> None:
        """Regression for #39: speeds were cell averages over the whole session (out-laps
        included) and a path's time was its length / mean speed. A path running 30 units beside
        the slow half is timed at the lap's slow-half speed."""
        path = [(500.0, 30.0), (1000.0, 30.0)]
        self.assertAlmostEqual(estimate_path_time(path, self.lap, scale=0.1), 2.0)

    def test_time_saved_positive_when_faster_and_not_clamped(self) -> None:
        shortcut = _sr([(0.0, 0.0), (500.0, 0.0)])                 # 1 s
        self.assertAlmostEqual(compute_time_saved(shortcut, 3.0, self.lap, 0.1), 2.0)
        # A path 10x longer than the lap at the fast-half speed: 1000 m / 50 m/s = 20 s, so
        # -17 s. The old code clamped every result to +/-5 s, which hid the broken estimate.
        detour = _sr([(0.0, 0.0), (0.0, 10000.0)])
        self.assertAlmostEqual(compute_time_saved(detour, 3.0, self.lap, 0.1), -17.0)

    def test_none_when_no_path(self) -> None:
        self.assertIsNone(compute_time_saved(_sr([], found=False), 3.0, self.lap, 0.1))


class TestOutputs(unittest.TestCase):

    def test_build_output_values(self) -> None:
        a, d, b = _sr([(0.0, 0.0), (1.0, 1.0)], 3.33333), _sr([(0.0, 0.0)], 3.33333, name="dijkstra"), \
            _sr([(0.0, 0.0)], 0.0, name="bfs")
        lap = {"lap_number": 7, "lap_time_s": 89.708, "length_km": 5.36}
        data = build_output("s.parquet", "44", 50.0, lap, 0.1, a, d, b, 4.444449, [(1.0, 2.0)], [], 15)
        self.assertEqual(data["driver_path"], [[1.0, 2.0]])
        self.assertEqual(data["algorithms"]["astar"]["path"], [[0.0, 0.0], [1.0, 1.0]])
        self.assertEqual(data["algorithms"]["astar"]["cost"], 3.3333)
        self.assertEqual(data["algorithms"]["bfs"]["cost"], 4.4444)       # weighted cost, not BFS's 0
        self.assertEqual(data["algorithms"]["dijkstra"]["compute_time_s"], 0.1235)
        self.assertIsNone(data["algorithms"]["astar"]["time_saved_s"])    # filled in by the pipeline
        self.assertEqual((data["n_corners"], data["coordinate_scale"]), (15, 0.1))
        self.assertEqual(data["fastest_lap"], lap)


if __name__ == "__main__":
    unittest.main()
