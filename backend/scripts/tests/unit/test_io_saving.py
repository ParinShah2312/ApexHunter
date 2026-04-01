"""Unit tests for save_outputs and build_meta."""

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from mistakes_io import save_outputs, build_meta
from mistakes_features import FEATURE_COLUMNS


def _make_silent_logger():
    lg = logging.getLogger("test_io_save")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.DEBUG)
    return lg


class TestIOSaving(unittest.TestCase):

    def setUp(self):
        self.logger = _make_silent_logger()
        self.tmp = tempfile.mkdtemp()
        self.output_dir = Path(self.tmp) / "output"
        self.output_dir.mkdir()
        n = 50
        self.df = pd.DataFrame({
            "Driver": ["44"] * n,
            "Speed": np.full(n, 200.0, dtype="float32"),
            "is_mistake": np.array([False] * 45 + [True] * 5),
            "anomaly_score": np.random.uniform(-0.3, 0.3, n).astype("float32"),
        })

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_creates_parquet(self):
        meta = {"test": True}
        pq, js = save_outputs(self.df, meta, self.output_dir, "test", self.logger)
        self.assertTrue(pq.exists())
        self.assertTrue(str(pq).endswith(".parquet"))

    def test_save_creates_json(self):
        meta = {"test": True}
        pq, js = save_outputs(self.df, meta, self.output_dir, "test", self.logger)
        self.assertTrue(js.exists())
        with open(js) as f:
            loaded = json.load(f)
        self.assertEqual(loaded["test"], True)

    def test_build_meta_keys(self):
        meta = build_meta("session.parquet", "44", "1", "ref.parquet",
                          0.10, {"0.1": 0.5}, 0.5, self.df)
        required_keys = [
            "session_file", "driver", "reference_driver", "reference_file",
            "best_contamination", "cv_scores", "best_cv_score", "total_rows",
            "total_mistakes", "mistake_rate_pct", "n_estimators",
            "feature_columns", "timestamp",
        ]
        for key in required_keys:
            self.assertIn(key, meta)

    def test_build_meta_mistake_count(self):
        meta = build_meta("s.parquet", "44", "1", "r.parquet",
                          0.10, {}, 0.5, self.df)
        self.assertEqual(meta["total_mistakes"], 5)
        self.assertEqual(meta["total_rows"], 50)

    def test_build_meta_feature_columns(self):
        meta = build_meta("s.parquet", "44", "1", "r.parquet",
                          0.10, {}, 0.5, self.df)
        self.assertEqual(meta["feature_columns"], FEATURE_COLUMNS)

    def test_build_meta_contamination_is_float(self):
        meta = build_meta("s.parquet", "44", "1", "r.parquet",
                          0.10, {}, 0.5, self.df)
        self.assertIsInstance(meta["best_contamination"], float)


if __name__ == "__main__":
    unittest.main()
