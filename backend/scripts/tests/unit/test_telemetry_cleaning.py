"""Unit tests for clean_telemetry_file: row dropping, clipping, and ffill.

Uses setUpClass to run clean_telemetry_file ONCE for all read-only tests.
"""

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


class TestTelemetryCleaning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.input_path = Path(cls.tmp) / "dirty.parquet"
        cls.output_path = Path(cls.tmp) / "clean.parquet"

        n = 100
        df = pd.DataFrame({
            "Driver": ["1"] * n,
            "Speed": [200.0] * 80 + [500.0] * 5 + [200.0] * 5 + [float('nan')] * 10,
            "RPM": [8000.0] * 80 + [8000.0] * 5 + [20000.0] * 5 + [float('nan')] * 10,
            "Throttle": [50.0] * 80 + [150.0] * 5 + [50.0] * 10 + [float('nan')] * 5,
            "Brake": [0.0] * 80 + [-10.0] * 3 + [0.0] * 12 + [float('nan')] * 5,
            "X": [100.0] * 90 + [float('nan')] * 10,
            "Y": [200.0] * 90 + [float('nan')] * 10,
            "nGear": [5] * n,
            "Time": pd.to_timedelta(range(n), unit='s'),
            "SessionTime": pd.to_timedelta(range(n), unit='s'),
        })
        df.to_parquet(cls.input_path, compression='snappy')
        clean_telemetry_file(cls.input_path, cls.output_path)
        cls.output_df = pd.read_parquet(cls.output_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_null_rows_dropped(self):
        self.assertEqual(len(self.output_df), 90)

    def test_speed_clipped_upper(self):
        self.assertLessEqual(self.output_df["Speed"].max(), 380.0)

    def test_speed_clipped_lower(self):
        self.assertGreaterEqual(self.output_df["Speed"].min(), 0.0)

    def test_rpm_clipped(self):
        self.assertLessEqual(self.output_df["RPM"].max(), 15000.0)

    def test_throttle_clipped_upper(self):
        self.assertLessEqual(self.output_df["Throttle"].max(), 100.0)

    def test_brake_clipped_lower(self):
        self.assertGreaterEqual(self.output_df["Brake"].min(), 0.0)

    def test_ffill_fills_partial_nulls(self):
        """Builds a separate DataFrame with partial nulls (not all-core NaN)."""
        tmp2 = tempfile.mkdtemp()
        try:
            n = 50
            df = pd.DataFrame({
                "Driver": ["1"] * n,
                "Speed": [200.0] * 10 + [float('nan')] * 5 + [200.0] * 35,
                "RPM": [8000.0] * n,
                "Throttle": [50.0] * n,
                "Brake": [0.0] * n,
                "X": [100.0] * n,
                "Y": [200.0] * n,
                "nGear": [5] * n,
                "Time": pd.to_timedelta(range(n), unit='s'),
                "SessionTime": pd.to_timedelta(range(n), unit='s'),
            })
            inp = Path(tmp2) / "dirty.parquet"
            out = Path(tmp2) / "clean.parquet"
            df.to_parquet(inp, compression='snappy')
            clean_telemetry_file(inp, out)
            result = pd.read_parquet(out)
            self.assertEqual(len(result), n)
            self.assertFalse(result["Speed"].isna().any())
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
