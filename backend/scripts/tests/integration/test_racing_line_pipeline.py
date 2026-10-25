"""Integration test: the real racing-line pipeline on real data (2023 Bahrain qualifying, car #1).

No mocks: reads data_lake/clean_data/2023_1_Q.parquet and the FastF1 cache (cache/). Expected values
are the real ones documented in PROJECT_DOCUMENTATION §5.3 (Verstappen's pole lap: lap 14, 89.708 s).
Skipped when the local data is not present.
"""

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from optimal_line import run_pipeline
from utils import DATA_LAKE_DIR

RUN_SLOW = os.environ.get("APEXHUNTER_RUN_SLOW", "0") == "1"
SESSION = DATA_LAKE_DIR / "clean_data" / "2023_1_Q.parquet"


@unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")
@unittest.skipUnless(SESSION.exists(), f"Needs local data: {SESSION}")
class TestRacingLineOnRealData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        args = argparse.Namespace(session=str(SESSION), driver="1", resolution=50.0, n_corners=16,
                                  output_dir=cls.tmp.name, force=True)
        run_pipeline(args)
        with open(Path(cls.tmp.name) / "2023_1_Q_1_racing_line.json", encoding="utf-8") as f:
            cls.data = json.load(f)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_uses_the_fastf1_fastest_lap_only(self):
        """#19: the path must come from the fastest lap, never the whole session (3,071 points)."""
        lap = self.data["fastest_lap"]
        self.assertEqual((lap["lap_number"], lap["lap_time_s"]), (14, 89.708))
        self.assertEqual(len(self.data["driver_path"]), 66)          # every 10th of 652 lap samples

    def test_scale_and_lap_length(self):
        """#18: FastF1 X/Y are 1/10 m, so Bahrain's driven lap is ~5.34 km (circuit 5.412 km)."""
        self.assertEqual(self.data["coordinate_scale"], 0.1)
        self.assertAlmostEqual(self.data["fastest_lap"]["length_km"], 5.34, delta=0.05)

    def test_time_estimate_reproduces_the_real_lap(self):
        """#39: the time model applied to the driver's own lap is within 2 % of the real time."""
        lap = self.data["fastest_lap"]
        self.assertAlmostEqual(lap["estimated_time_s"], lap["lap_time_s"], delta=0.02 * lap["lap_time_s"])

    def test_astar_is_optimal_and_bfs_is_not_cheaper(self):
        algos = self.data["algorithms"]
        self.assertTrue(all(a["found"] for a in algos.values()))
        self.assertAlmostEqual(algos["astar"]["cost"], algos["dijkstra"]["cost"], places=3)
        self.assertGreaterEqual(algos["bfs"]["cost"], algos["astar"]["cost"])
        self.assertLess(algos["astar"]["nodes_expanded"], algos["dijkstra"]["nodes_expanded"])


if __name__ == "__main__":
    unittest.main()
