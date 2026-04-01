"""Unit tests for load_and_validate and select_reference_driver."""

import logging
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from mistakes_io import load_and_validate, select_reference_driver, REQUIRED_COLUMNS


def _make_silent_logger():
    lg = logging.getLogger("test_io_load")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.DEBUG)
    return lg


def _make_session_df(n=100, drivers=None):
    if drivers is None:
        drivers = ["44"] * n
    return pd.DataFrame({
        "Driver": drivers,
        "Speed": np.full(n, 200.0, dtype="float32"),
        "RPM": np.full(n, 8000.0, dtype="float32"),
        "Throttle": np.full(n, 50.0, dtype="float32"),
        "Brake": np.zeros(n, dtype="float32"),
        "X": np.zeros(n, dtype="float32"),
        "Y": np.zeros(n, dtype="float32"),
        "nGear": np.full(n, 5.0, dtype="float32"),
        "Time": pd.to_timedelta(np.arange(n), unit="s"),
        "SessionTime": pd.to_timedelta(np.arange(n), unit="s"),
    })


class TestIOLoading(unittest.TestCase):

    def setUp(self):
        self.logger = _make_silent_logger()
        self.tmp = tempfile.mkdtemp()
        self.session_path = Path(self.tmp) / "session.parquet"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_valid_session(self):
        df = _make_session_df(100)
        df.to_parquet(self.session_path)
        df_full, df_driver = load_and_validate(self.session_path, "44", self.logger)
        self.assertEqual(len(df_driver), 100)

    def test_load_missing_column_raises(self):
        df = _make_session_df(100).drop(columns=["Brake"])
        df.to_parquet(self.session_path)
        with self.assertRaises(ValueError):
            load_and_validate(self.session_path, "44", self.logger)

    def test_load_missing_driver_raises(self):
        df = _make_session_df(100, drivers=["1"] * 100)
        df.to_parquet(self.session_path)
        with self.assertRaises(ValueError):
            load_and_validate(self.session_path, "44", self.logger)

    def test_load_nonexistent_file_raises(self):
        with self.assertRaises(ValueError):
            load_and_validate(Path(self.tmp) / "nofile.parquet", "44", self.logger)

    def test_select_reference_fastest(self):
        n_each = 50
        drivers = ["44"] * n_each + ["1"] * n_each
        speeds = [200.0] * n_each + [300.0] * n_each
        df = pd.DataFrame({
            "Driver": drivers,
            "Speed": np.array(speeds, dtype="float32"),
        })
        ref = select_reference_driver(df, "44", self.logger)
        self.assertEqual(ref, "1")

    def test_select_reference_same_driver_warning(self):
        df = pd.DataFrame({
            "Driver": ["44"] * 50,
            "Speed": np.full(50, 200.0, dtype="float32"),
        })
        ref = select_reference_driver(df, "44", self.logger)
        self.assertEqual(ref, "44")


if __name__ == "__main__":
    unittest.main()
