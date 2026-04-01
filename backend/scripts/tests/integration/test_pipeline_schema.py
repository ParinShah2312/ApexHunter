"""Integration tests: output parquet and JSON schema validation."""

import json
import os
import sys
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

from mistakes_features import FEATURE_COLUMNS


@slow_test
class TestPipelineSchema(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from integration.conftest import run_pipeline_once
        cls.fixture = run_pipeline_once()
        cls.df = pd.read_parquet(cls.fixture["parquet_path"])
        with open(cls.fixture["meta_path"]) as f:
            cls.meta = json.load(f)
        cls.input_df = pd.read_parquet(cls.fixture["input_path"])

    def test_anomaly_score_dtype_is_float32(self):
        self.assertEqual(self.df["anomaly_score"].dtype, np.dtype("float32"))

    def test_is_mistake_dtype_is_bool(self):
        self.assertEqual(self.df["is_mistake"].dtype, np.dtype("bool"))

    def test_all_feature_columns_are_float32(self):
        for col in FEATURE_COLUMNS:
            self.assertEqual(self.df[col].dtype, np.dtype("float32"), f"{col}")

    def test_xy_columns_unchanged(self):
        orig = self.input_df[self.input_df["Driver"] == "44"].sort_values("SessionTime").reset_index(drop=True)
        np.testing.assert_array_equal(self.df["X"].values, orig["X"].values)
        np.testing.assert_array_equal(self.df["Y"].values, orig["Y"].values)

    def test_driver_column_all_target_driver(self):
        self.assertTrue((self.df["Driver"] == "44").all())

    def test_row_count_unchanged(self):
        self.assertEqual(len(self.df), self.fixture["n_rows"])

    def test_meta_json_all_keys_present(self):
        for key in ["session_file", "driver", "reference_driver", "reference_file",
                     "best_contamination", "cv_scores", "best_cv_score", "total_rows",
                     "total_mistakes", "mistake_rate_pct", "n_estimators",
                     "feature_columns", "timestamp"]:
            self.assertIn(key, self.meta, f"Missing: {key}")

    def test_meta_contamination_is_float_in_list(self):
        self.assertIsInstance(self.meta["best_contamination"], float)
        self.assertIn(self.meta["best_contamination"], [0.05, 0.08, 0.10, 0.12, 0.15, 0.20])

    def test_meta_cv_scores_has_six_keys(self):
        self.assertEqual(len(self.meta["cv_scores"]), 6)

    def test_meta_feature_columns_length_seven(self):
        self.assertIsInstance(self.meta["feature_columns"], list)
        self.assertEqual(len(self.meta["feature_columns"]), 7)

    def test_meta_mistake_rate_matches_parquet(self):
        self.assertEqual(self.meta["total_mistakes"], int(self.df["is_mistake"].sum()))


if __name__ == "__main__":
    unittest.main()
