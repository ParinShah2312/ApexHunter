"""Unit tests for telemetry dtype assertions only. Read-only, no re-clean."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from clean_telemetry import clean_telemetry_file


class TestTelemetryDtypes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        inp = Path(cls.tmp) / "input.parquet"
        cls.output_path = Path(cls.tmp) / "output.parquet"
        n = 50
        df = pd.DataFrame({
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
        df.to_parquet(inp, compression="snappy")
        clean_telemetry_file(inp, cls.output_path)
        cls.df = pd.read_parquet(cls.output_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_speed_dtype_is_float32(self):
        self.assertEqual(self.df["Speed"].dtype, np.dtype("float32"))

    def test_rpm_dtype_is_float32(self):
        self.assertEqual(self.df["RPM"].dtype, np.dtype("float32"))

    def test_throttle_dtype_is_float32(self):
        self.assertEqual(self.df["Throttle"].dtype, np.dtype("float32"))

    def test_brake_dtype_is_float32(self):
        self.assertEqual(self.df["Brake"].dtype, np.dtype("float32"))

    def test_x_dtype_is_float32(self):
        self.assertEqual(self.df["X"].dtype, np.dtype("float32"))

    def test_y_dtype_is_float32(self):
        self.assertEqual(self.df["Y"].dtype, np.dtype("float32"))


if __name__ == "__main__":
    unittest.main()
