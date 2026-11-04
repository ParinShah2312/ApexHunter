"""Unit test for the planted-mistake test set (isolation_forest/mistakes_eval.inject_mistakes)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from isolation_forest.mistakes_eval import inject_mistakes
from isolation_forest.mistakes_features import ACCEL_WINDOW_S, compute_acceleration


def _laps_with_a_traffic_gap(n_laps=3, lap_s=60.0, dt=0.25):
    """Laps of straights and braking zones; acceleration is computed on the full telemetry and then
    a 3 s stretch of every lap is removed, as the race traffic filter does."""
    rows = []
    for lap in range(1, n_laps + 1):
        for k in range(int(lap_s / dt)):
            ts = k * dt
            phase = ts % 15.0                       # 10 s accelerating, 5 s braking, repeated
            braking = phase >= 10.0
            speed = 100 + 20 * phase if not braking else 300 - 40 * (phase - 10.0)
            rows.append({"LapNumber": lap, "SessionTime": pd.Timedelta(seconds=(lap - 1) * lap_s + ts),
                         "Speed": speed, "Throttle": 0.0 if braking else 100.0, "Brake": 1.0 if braking else 0.0,
                         "nGear": int(min(8, 1 + speed // 40)), "t_in_lap": ts})
    df = pd.DataFrame(rows)
    df["accel"] = compute_acceleration(df)
    keep = ~df["t_in_lap"].between(30.0, 33.0)        # traffic samples removed after accel
    return df[keep].drop(columns="t_in_lap").reset_index(drop=True)


class TestInjectMistakes(unittest.TestCase):

    def test_unplanted_samples_keep_their_acceleration_across_a_traffic_gap(self) -> None:
        """Regression for #48: acceleration was recomputed over the whole lap after planting, which
        interpolated across the removed traffic samples and gave normal samples a fake value."""
        df = _laps_with_a_traffic_gap()
        out, mistake_ids, _ = inject_mistakes(df, np.random.default_rng(0))
        self.assertGreater(len(np.unique(mistake_ids[mistake_ids >= 0])), 0)
        t = df["SessionTime"].dt.total_seconds().to_numpy()
        planted_t = t[mistake_ids >= 0]
        near = np.array([np.any((planted_t <= x) & (planted_t >= x - ACCEL_WINDOW_S - 1e-9)) for x in t])
        changed = ~np.isclose(out["accel"].to_numpy(), df["accel"].to_numpy(), atol=1e-4)
        self.assertFalse((changed & ~near).any(), np.flatnonzero(changed & ~near)[:10])
        self.assertTrue(changed[mistake_ids >= 0].any())      # planted speed changes did update it


if __name__ == "__main__":
    unittest.main()
