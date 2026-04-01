"""Integration tests: anomaly detection correctness on injected data."""

import json
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
class TestPipelineCorrectness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from integration.conftest import run_pipeline_once
        cls.fixture = run_pipeline_once()
        cls.df = pd.read_parquet(cls.fixture["parquet_path"])
        with open(cls.fixture["meta_path"]) as f:
            cls.meta = json.load(f)

    def test_anomalies_detected_in_injected_region(self):
        anomaly_region = self.df.iloc[200:211]
        n_flagged = int(anomaly_region["is_mistake"].sum())
        self.assertGreaterEqual(n_flagged, 5,
                                f"Only {n_flagged}/11 anomaly rows flagged")

    def test_mistake_rate_in_sane_range(self):
        rate = self.meta["mistake_rate_pct"]
        self.assertGreaterEqual(rate, 0.5)
        self.assertLessEqual(rate, 30.0)

    def test_annotated_features_not_all_zero(self):
        for col in FEATURE_COLUMNS:
            self.assertFalse((self.df[col] == 0.0).all(),
                             f"Feature {col} is all zeros")

    def test_anomaly_scores_are_not_all_identical(self):
        scores = self.df["anomaly_score"]
        self.assertGreater(scores.std(), 0.0)


if __name__ == "__main__":
    unittest.main()
