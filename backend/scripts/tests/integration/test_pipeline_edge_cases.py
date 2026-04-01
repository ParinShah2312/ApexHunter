"""Integration tests: edge cases — missing driver, empty data, bad columns."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
TESTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(TESTS_DIR))

RUN_SLOW = os.environ.get("APEXHUNTER_RUN_SLOW", "0") == "1"
slow_test = unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")

from detect_mistakes import run_pipeline


def _make_df(n, driver, speed=250.0, rpm=10000.0):
    return pd.DataFrame({
        "Driver": [driver] * n,
        "Speed": np.full(n, speed, dtype="float32"),
        "RPM": np.full(n, rpm, dtype="float32"),
        "Throttle": np.full(n, 80.0, dtype="float32"),
        "Brake": np.zeros(n, dtype="float32"),
        "X": np.zeros(n, dtype="float32"),
        "Y": np.zeros(n, dtype="float32"),
        "nGear": np.full(n, 6.0, dtype="float32"),
        "Time": pd.to_timedelta(range(n), unit="s"),
        "SessionTime": pd.to_timedelta(range(n), unit="s"),
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
            run_pipeline(str(path), "99", None, self.output_dir, True)

    def test_empty_driver_data_after_filter(self):
        df = _make_df(100, "1")
        path = Path(self.tmp) / "test.parquet"
        df.to_parquet(path)
        with self.assertRaises(SystemExit):
            run_pipeline(str(path), "44", None, self.output_dir, True)

    def test_single_row_driver(self):
        df_single = _make_df(1, "44")
        df_ref = _make_df(50, "1", speed=280.0)
        combined = pd.concat([df_single, df_ref], ignore_index=True)
        path = Path(self.tmp) / "test.parquet"
        combined.to_parquet(path)
        try:
            run_pipeline(str(path), "44", None, self.output_dir, True)
        except SystemExit:
            pass

    def test_missing_required_column(self):
        df = _make_df(100, "44").drop(columns=["Brake"])
        path = Path(self.tmp) / "test.parquet"
        df.to_parquet(path)
        with self.assertRaises(SystemExit):
            run_pipeline(str(path), "44", None, self.output_dir, True)


if __name__ == "__main__":
    unittest.main()
