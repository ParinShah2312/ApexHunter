"""Unit tests for isolation_forest/mistakes_io.py: loading/validation, reference-driver choice,
metadata contents and output round-trip."""

import json
import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from isolation_forest.mistakes_features import FEATURE_COLUMNS
from isolation_forest.mistakes_io import build_meta, load_and_validate, save_outputs, select_reference_driver
from isolation_forest.mistakes_model import N_ESTIMATORS


def _logger() -> logging.Logger:
    lg = logging.getLogger("test_mistakes_io")
    lg.handlers = [logging.NullHandler()]
    lg.propagate = False
    return lg


def _session(drivers) -> pd.DataFrame:
    n = len(drivers)
    return pd.DataFrame({
        "Driver": drivers, "Speed": np.full(n, 200.0, dtype="float32"),
        "RPM": np.full(n, 8000.0, dtype="float32"), "Throttle": np.full(n, 50.0, dtype="float32"),
        "Brake": np.zeros(n, dtype="float32"), "X": np.arange(n, dtype="float32"),
        "Y": np.zeros(n, dtype="float32"), "nGear": np.full(n, 5.0, dtype="float32"),
        "SessionTime": pd.to_timedelta(np.arange(n), unit="s"),
    })


class TestLoadAndValidate(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "session.parquet"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_full_session_and_target_driver_rows(self) -> None:
        _session(["44"] * 30 + ["1"] * 20).to_parquet(self.path)
        full, drv = load_and_validate(self.path, "44", _logger())
        self.assertEqual(len(full), 50)
        self.assertEqual(len(drv), 30)
        self.assertTrue((drv["Driver"] == "44").all())

    def test_missing_required_column_raises(self) -> None:
        _session(["44"] * 10).drop(columns=["Brake"]).to_parquet(self.path)
        with self.assertRaisesRegex(ValueError, "Brake"):
            load_and_validate(self.path, "44", _logger())

    def test_missing_driver_raises(self) -> None:
        _session(["1"] * 10).to_parquet(self.path)
        with self.assertRaisesRegex(ValueError, "44"):
            load_and_validate(self.path, "44", _logger())

    def test_unreadable_file_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_and_validate(self.tmp / "missing.parquet", "44", _logger())


class TestSelectReferenceDriver(unittest.TestCase):

    def test_fastest_driver_by_mean_speed(self) -> None:
        df = pd.DataFrame({"Driver": ["44", "44", "1", "1", "16"],
                           "Speed": [200.0, 220.0, 300.0, 250.0, 260.0]})
        self.assertEqual(select_reference_driver(df, "44", _logger()), "1")

    def test_target_can_be_its_own_reference(self) -> None:
        # Current behaviour (see roadmap §B: the reference should never be the target)
        df = pd.DataFrame({"Driver": ["44", "1"], "Speed": [300.0, 200.0]})
        self.assertEqual(select_reference_driver(df, "44", _logger()), "44")


class TestMetaAndOutputs(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        n = 50
        self.df = pd.DataFrame({
            "Driver": ["44"] * n, "Speed": np.linspace(100, 300, n).astype("float32"),
            "is_mistake": np.array([False] * 45 + [True] * 5),
            "anomaly_score": np.linspace(-0.3, 0.3, n).astype("float32"),
        })

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_meta_values(self) -> None:
        meta = build_meta("s.parquet", "44", "1", "r.parquet", 0.01, {"0.01": 0.2}, 0.2, self.df)
        expected = {
            "session_file": "s.parquet", "driver": "44", "reference_driver": "1",
            "reference_file": "r.parquet", "best_contamination": 0.01, "cv_scores": {"0.01": 0.2},
            "best_cv_score": 0.2, "total_rows": 50, "total_mistakes": 5, "mistake_rate_pct": 10.0,
            "n_estimators": N_ESTIMATORS, "feature_columns": FEATURE_COLUMNS,
        }
        self.assertEqual({k: meta[k] for k in expected}, expected)
        self.assertIn("+05:30", meta["timestamp"])   # IST timestamps

    def test_build_meta_empty_frame_rate_zero(self) -> None:
        meta = build_meta("s", "44", "1", "r", 0.01, {}, 0.0, self.df.iloc[0:0])
        self.assertEqual(meta["mistake_rate_pct"], 0.0)

    def test_save_outputs_round_trip(self) -> None:
        meta = build_meta("s.parquet", "44", "1", "r.parquet", 0.01, {}, 0.2, self.df)
        pq_path, js_path = save_outputs(self.df, meta, self.tmp, "2024_1_Q_44", _logger())
        self.assertEqual(pq_path.name, "2024_1_Q_44_mistakes.parquet")
        self.assertEqual(js_path.name, "2024_1_Q_44_mistakes_meta.json")
        pd.testing.assert_frame_equal(pd.read_parquet(pq_path), self.df)
        with open(js_path) as f:
            self.assertEqual(json.load(f), meta)


if __name__ == "__main__":
    unittest.main()
