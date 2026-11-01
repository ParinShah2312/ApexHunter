"""Integration tests: detect_mistakes rejects bad input cleanly (missing driver, missing column)."""

import os
import shutil
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

from detect_mistakes import run_pipeline


def _make_df(n, driver, speed=250.0):
    return pd.DataFrame({
        "Driver": [driver] * n,
        "Speed": np.full(n, speed, dtype="float32"),
        "Throttle": np.full(n, 80.0, dtype="float32"),
        "Brake": np.zeros(n, dtype="float32"),
        "X": np.zeros(n, dtype="float32"),
        "Y": np.zeros(n, dtype="float32"),
        "nGear": np.full(n, 6, dtype="int64"),
        "SessionTime": pd.to_timedelta(range(n), unit="s"),
        "Year": 2023, "Round": 1, "Session": "R",
    })


@slow_test
class TestPipelineEdgeCases(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.output_dir = Path(self.tmp) / "output"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_driver_not_in_session(self):
        df = _make_df(100, "1")
        path = Path(self.tmp) / "test.parquet"
        df.to_parquet(path)
        with self.assertRaises(SystemExit):
            run_pipeline(str(path), "99", self.output_dir, True)

    def test_missing_required_column(self):
        df = _make_df(100, "44").drop(columns=["Brake"])
        path = Path(self.tmp) / "test.parquet"
        df.to_parquet(path)
        with self.assertRaises(SystemExit):
            run_pipeline(str(path), "44", self.output_dir, True)


if __name__ == "__main__":
    unittest.main()
