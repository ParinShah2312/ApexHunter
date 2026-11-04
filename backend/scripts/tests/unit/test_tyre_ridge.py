"""Unit test for the shared tyre model metrics in lstm/tyre_ridge.py (ROADMAP §C step 5)."""

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from lstm.tyre_ridge import regression_metrics


class TestRegressionMetrics(unittest.TestCase):

    def test_metrics_are_on_the_change_except_mape_on_the_lap_time(self) -> None:
        change_true = np.array([0.2, -0.2, 0.6, -0.6])
        change_pred = np.array([0.0, 0.0, 0.4, -0.4])        # errors -0.2, 0.2, -0.2, 0.2
        last_lap = np.array([90.0, 90.4, 90.0, 90.8])        # actual lap times 90.2, 90.2, 90.6, 90.2
        m = regression_metrics(change_true, change_pred, last_lap)
        self.assertAlmostEqual(m["mae_s"], 0.2)
        self.assertAlmostEqual(m["rmse_s"], 0.2)
        self.assertAlmostEqual(m["r2"], 1 - 0.16 / 0.80)    # SS_res 4 x 0.04, SS_tot 2 x 0.04 + 2 x 0.36
        self.assertAlmostEqual(m["mape_pct"], 100 * np.mean(0.2 / np.array([90.2, 90.2, 90.6, 90.2])))


if __name__ == "__main__":
    unittest.main()
