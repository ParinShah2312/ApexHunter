"""Unit tests for isolation_forest/mistakes_features.engineer_features: exact feature values,
clipping, the brake/throttle overlap rule, and the output contract."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from isolation_forest.mistakes_features import (
    FEATURE_COLUMNS, GEAR_CHANGE_CLIP, SPEED_DELTA_CLIP, engineer_features,
)


def _df(**cols) -> pd.DataFrame:
    """Telemetry rows; unspecified channels get neutral constant values."""
    n = len(next(iter(cols.values())))
    base = {
        "Speed": [200.0] * n, "Throttle": [50.0] * n, "Brake": [0.0] * n,
        "RPM": [10000.0] * n, "nGear": [5.0] * n,
        "X": [0.0] * n, "Y": [0.0] * n, "SessionTime": pd.to_timedelta(range(n), unit="s"),
    }
    base.update(cols)
    return pd.DataFrame(base)


class TestFeatureValues(unittest.TestCase):

    def test_exact_values_for_known_rows(self) -> None:
        df = _df(Speed=[190.0, 380.0, 0.0], Throttle=[25.0, 100.0, 0.0], Brake=[0.0, 1.0, 1.0],
                 RPM=[7500.0, 15000.0, 0.0], nGear=[4.0, 6.0, 5.0])
        f = engineer_features(df)
        np.testing.assert_allclose(f["speed_normalized"], [0.5, 1.0, 0.0])          # Speed / 380
        np.testing.assert_allclose(f["throttle_intensity"], [0.25, 1.0, 0.0])       # Throttle / 100
        np.testing.assert_allclose(f["brake_intensity"], [0.0, 1.0, 1.0])           # brake on/off flag
        np.testing.assert_allclose(f["rpm_normalized"], [0.5, 1.0, 0.0])            # RPM / 15000
        np.testing.assert_allclose(f["brake_throttle_overlap"], [0.0, 1.0, 0.0])    # brake on AND throttle > 5
        np.testing.assert_allclose(f["speed_delta"], [0.0, SPEED_DELTA_CLIP, -SPEED_DELTA_CLIP])
        np.testing.assert_allclose(f["gear_change"], [0.0, 2.0, 1.0])               # |Δ gear|

    def test_speed_delta_unclipped_inside_limits(self) -> None:
        f = engineer_features(_df(Speed=[200.0, 230.0, 190.0]))
        np.testing.assert_allclose(f["speed_delta"], [0.0, 30.0, -40.0])

    def test_gear_change_clipped(self) -> None:
        f = engineer_features(_df(nGear=[1.0, 8.0, 1.0]))
        np.testing.assert_allclose(f["gear_change"], [0.0, GEAR_CHANGE_CLIP, GEAR_CHANGE_CLIP])

    def test_overlap_truth_table(self) -> None:
        cases = [  # (brake, throttle, expected overlap)
            (0.0, 80.0, 0.0), (1.0, 0.0, 0.0), (1.0, 5.0, 0.0), (1.0, 6.0, 1.0), (1.0, 80.0, 1.0),
        ]
        f = engineer_features(_df(Brake=[c[0] for c in cases], Throttle=[c[1] for c in cases]))
        for (brake, throttle, expected), got in zip(cases, f["brake_throttle_overlap"]):
            with self.subTest(brake=brake, throttle=throttle):
                self.assertEqual(got, expected)

    def test_raw_boolean_brake_column_accepted(self) -> None:
        """FastF1 raw data stores Brake as bool."""
        f = engineer_features(_df(Brake=[True, False], Throttle=[20.0, 20.0]))
        np.testing.assert_allclose(f["brake_intensity"], [1.0, 0.0])
        np.testing.assert_allclose(f["brake_throttle_overlap"], [1.0, 0.0])

    def test_position_does_not_affect_features(self) -> None:
        a = engineer_features(_df(Speed=[100.0, 150.0], X=[0.0, 0.0], Y=[0.0, 0.0]))
        b = engineer_features(_df(Speed=[100.0, 150.0], X=[999.0, 5.0], Y=[-3.0, 7.0]))
        pd.testing.assert_frame_equal(a, b)


class TestFeatureContract(unittest.TestCase):

    def setUp(self) -> None:
        self.df = _df(Speed=list(np.linspace(100, 300, 20)), Brake=[0.0, 1.0] * 10)

    def test_columns_order_and_dtype(self) -> None:
        f = engineer_features(self.df)
        self.assertEqual(list(f.columns), FEATURE_COLUMNS)
        self.assertTrue((f.dtypes == np.float32).all())

    def test_rows_and_index_preserved(self) -> None:
        df = self.df.set_index(pd.Index(range(100, 120)))
        f = engineer_features(df)
        self.assertTrue(f.index.equals(df.index))

    def test_input_not_mutated(self) -> None:
        before = self.df.copy()
        engineer_features(self.df)
        pd.testing.assert_frame_equal(self.df, before)

    def test_single_row_has_no_nans(self) -> None:
        f = engineer_features(_df(Speed=[250.0]))
        self.assertEqual(len(f), 1)
        self.assertFalse(f.isna().any().any())


if __name__ == "__main__":
    unittest.main()
