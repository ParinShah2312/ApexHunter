"""Tests for frontend/components/telemetry_charts.py: the Brake Overlap card (#33)."""

import sys
import unittest
from pathlib import Path

import pandas as pd

FRONTEND_DIR = Path(__file__).resolve().parents[4] / "frontend"
sys.path.insert(0, str(FRONTEND_DIR))

from components.telemetry_charts import _compute_brake_overlap


def _samples(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Speed", "Brake", "Throttle"])


class TestBrakeOverlap(unittest.TestCase):

    def test_stationary_samples_are_ignored(self) -> None:
        """Regression for #33: a car parked in the garage with brake held and throttle blipped
        used to dominate the figure (here it would have been 6 of 10 = 60 %)."""
        garage = [(0.0, 1.0, 30.0)] * 5 + [(10.0, 1.0, 20.0)]          # below 50 km/h
        on_track = [(250.0, 1.0, 40.0),                                  # overlap
                    (250.0, 1.0, 0.0), (300.0, 0.0, 100.0), (120.0, 0.0, 3.0)]
        self.assertAlmostEqual(_compute_brake_overlap(_samples(garage + on_track)), 25.0)  # 1 of 4

    def test_threshold_is_inclusive_and_pedal_rules_hold(self) -> None:
        df = _samples([(50.0, 1.0, 6.0),     # exactly 50 km/h counts; throttle > 5 → overlap
                       (50.0, 1.0, 5.0),     # throttle 5 % is noise, not overlap
                       (49.9, 1.0, 50.0),    # below the threshold, ignored
                       (200.0, 0.0, 50.0)])  # no brake
        self.assertAlmostEqual(_compute_brake_overlap(df), 100 / 3)


if __name__ == "__main__":
    unittest.main()
