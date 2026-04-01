"""Integration tests: skip-if-exists and --force flag behavior.

This file builds its OWN fresh temp dir (cannot use shared fixture since
it tests output file control). Also handles cleanup_pipeline_fixture.
"""

import os
import shutil
import sys
import tempfile
import time
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


def _make_small_session(tmp, n=50):
    """Build a small 50-row session with 2 drivers for quick pipeline runs."""
    target = pd.DataFrame({
        "Driver": ["44"] * n,
        "Speed": np.full(n, 250.0, dtype="float32"),
        "RPM": np.full(n, 10000.0, dtype="float32"),
        "Throttle": np.full(n, 80.0, dtype="float32"),
        "Brake": np.zeros(n, dtype="float32"),
        "X": np.linspace(0, 100, n).astype("float32"),
        "Y": np.linspace(0, 100, n).astype("float32"),
        "nGear": np.full(n, 6.0, dtype="float32"),
        "Time": pd.to_timedelta(range(n), unit="s"),
        "SessionTime": pd.to_timedelta(range(n), unit="s"),
    })
    ref = pd.DataFrame({
        "Driver": ["1"] * n,
        "Speed": np.full(n, 280.0, dtype="float32"),
        "RPM": np.full(n, 12000.0, dtype="float32"),
        "Throttle": np.full(n, 90.0, dtype="float32"),
        "Brake": np.zeros(n, dtype="float32"),
        "X": np.linspace(0, 100, n).astype("float32"),
        "Y": np.linspace(0, 100, n).astype("float32"),
        "nGear": np.full(n, 6.0, dtype="float32"),
        "Time": pd.to_timedelta(range(n), unit="s"),
        "SessionTime": pd.to_timedelta(range(n), unit="s"),
    })
    combined = pd.concat([target, ref], ignore_index=True)
    path = Path(tmp) / "small_session.parquet"
    combined.to_parquet(path, compression="snappy")
    return path


@slow_test
class TestPipelineSkipForce(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.session_path = _make_small_session(self.tmp)
        self.output_dir = Path(self.tmp) / "output"
        self.expected_parquet = self.output_dir / "small_session_44_mistakes.parquet"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_skip_if_exists_no_force(self):
        run_pipeline(str(self.session_path), "44", None, self.output_dir, force=True)
        mtime_1 = os.path.getmtime(self.expected_parquet)
        time.sleep(1.1)
        run_pipeline(str(self.session_path), "44", None, self.output_dir, force=False)
        mtime_2 = os.path.getmtime(self.expected_parquet)
        self.assertEqual(mtime_1, mtime_2)

    def test_force_flag_overwrites(self):
        run_pipeline(str(self.session_path), "44", None, self.output_dir, force=True)
        mtime_1 = os.path.getmtime(self.expected_parquet)
        time.sleep(1.1)
        run_pipeline(str(self.session_path), "44", None, self.output_dir, force=True)
        mtime_2 = os.path.getmtime(self.expected_parquet)
        self.assertNotEqual(mtime_1, mtime_2)

    def test_output_dir_created_automatically(self):
        nested = Path(self.tmp) / "deeply" / "nested" / "output_dir"
        run_pipeline(str(self.session_path), "44", None, nested, force=True)
        self.assertTrue(nested.exists())
        self.assertGreater(len(list(nested.glob("*"))), 0)

    @classmethod
    def tearDownClass(cls):
        from integration.conftest import cleanup_pipeline_fixture
        cleanup_pipeline_fixture()


if __name__ == "__main__":
    unittest.main()
