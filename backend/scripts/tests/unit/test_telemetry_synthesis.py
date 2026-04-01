"""Unit tests for missing column synthesis in clean_telemetry_file."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from clean_telemetry import clean_telemetry_file, get_directory_size


def _make_base_df(n=80):
    return pd.DataFrame({
        "Driver": ["1"] * n,
        "Speed": np.full(n, 200.0, dtype="float64"),
        "RPM": np.full(n, 8000.0, dtype="float64"),
        "Throttle": np.full(n, 50.0, dtype="float64"),
        "Brake": np.zeros(n, dtype="float64"),
        "X": np.full(n, 100.0, dtype="float64"),
        "Y": np.full(n, 200.0, dtype="float64"),
        "nGear": np.full(n, 5, dtype="int64"),
        "Time": pd.to_timedelta(np.arange(n), unit="s"),
        "SessionTime": pd.to_timedelta(np.arange(n), unit="s"),
    })


class TestTelemetrySynthesis(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.input_path = Path(self.tmp) / "dirty.parquet"
        self.output_path = Path(self.tmp) / "clean.parquet"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_with_missing_col(self, col_name):
        df = _make_base_df().drop(columns=[col_name])
        df.to_parquet(self.input_path, compression="snappy")
        clean_telemetry_file(self.input_path, self.output_path)
        return pd.read_parquet(self.output_path)

    def test_missing_driver_synthesized(self):
        result = self._run_with_missing_col("Driver")
        self.assertIn("Driver", result.columns)
        self.assertTrue((result["Driver"] == "UNKNOWN").all())

    def test_missing_ngear_synthesized(self):
        result = self._run_with_missing_col("nGear")
        self.assertIn("nGear", result.columns)
        self.assertTrue((result["nGear"] == 8).all())

    def test_missing_time_synthesized(self):
        result = self._run_with_missing_col("Time")
        self.assertIn("Time", result.columns)

    def test_empty_file_skipped(self):
        empty = pd.DataFrame(columns=["Driver", "Speed", "RPM", "Throttle", "Brake",
                                       "X", "Y", "nGear", "Time", "SessionTime"])
        empty.to_parquet(self.input_path, compression="snappy")
        clean_telemetry_file(self.input_path, self.output_path)
        self.assertFalse(self.output_path.exists())

    def test_output_is_snappy_compressed(self):
        import pyarrow.parquet as pq
        df = _make_base_df()
        df.to_parquet(self.input_path, compression="snappy")
        clean_telemetry_file(self.input_path, self.output_path)
        meta = pq.read_metadata(self.output_path)
        compression = meta.row_group(0).column(0).compression
        self.assertEqual(compression.upper(), "SNAPPY")

    def test_get_directory_size_returns_string(self):
        df = _make_base_df()
        df.to_parquet(self.input_path, compression="snappy")
        result = get_directory_size(Path(self.tmp))
        self.assertIsInstance(result, str)
        self.assertTrue(result.endswith(" MB"))

    def test_get_directory_size_nonzero(self):
        df = _make_base_df()
        df.to_parquet(self.input_path, compression="snappy")
        result = get_directory_size(Path(self.tmp))
        self.assertNotEqual(result, "0.00 MB")


if __name__ == "__main__":
    unittest.main()
