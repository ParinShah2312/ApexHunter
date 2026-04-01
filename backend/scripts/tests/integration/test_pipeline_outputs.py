"""Integration tests: pipeline output file existence and column presence."""

import os
import sys
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
TESTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(TESTS_DIR))

RUN_SLOW = os.environ.get("APEXHUNTER_RUN_SLOW", "0") == "1"
slow_test = unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")

from mistakes_features import FEATURE_COLUMNS


@slow_test
class TestPipelineOutputs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from integration.conftest import run_pipeline_once
        cls.fixture = run_pipeline_once()
        cls.df = pd.read_parquet(cls.fixture["parquet_path"])

    def test_parquet_file_exists(self):
        self.assertTrue(self.fixture["parquet_path"].exists())

    def test_meta_json_file_exists(self):
        self.assertTrue(self.fixture["meta_path"].exists())

    def test_parquet_has_all_original_columns(self):
        for col in ["Driver", "Speed", "RPM", "Throttle", "Brake", "X", "Y",
                     "SessionTime", "nGear"]:
            self.assertIn(col, self.df.columns, f"Missing: {col}")

    def test_parquet_has_anomaly_score_column(self):
        self.assertIn("anomaly_score", self.df.columns)

    def test_parquet_has_is_mistake_column(self):
        self.assertIn("is_mistake", self.df.columns)

    def test_parquet_has_all_feature_columns(self):
        for col in FEATURE_COLUMNS:
            self.assertIn(col, self.df.columns, f"Missing feature: {col}")


if __name__ == "__main__":
    unittest.main()
