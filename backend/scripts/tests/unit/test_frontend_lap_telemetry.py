"""Tests for frontend/components/lap_telemetry.py: distance, time delta and corners on hand-built
laps with known answers (Telemetry tab, ROADMAP §3 step 2)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

FRONTEND_DIR = Path(__file__).resolve().parents[4] / "frontend"
sys.path.insert(0, str(FRONTEND_DIR))

from components.lap_telemetry import (corner_distances, full_throttle_pct, lap_samples, location_text,
                                      time_delta)


def _telemetry(times_s, speed_kmh, throttle=100.0) -> pd.DataFrame:
    """Readings at the given session times; X runs along the track (1 unit per reading)."""
    n = len(times_s)
    return pd.DataFrame({"SessionTime": pd.to_timedelta(times_s, unit="s"),
                         "Speed": np.full(n, speed_kmh, dtype=float), "Throttle": throttle,
                         "Brake": 0.0, "nGear": 7, "RPM": 11000.0,
                         "X": np.arange(n, dtype=float), "Y": 0.0})


class TestLapSamples(unittest.TestCase):

    def test_distance_covers_the_whole_lap_and_ignores_other_laps(self) -> None:
        # 36 km/h = 10 m/s; readings every 0.5 s from 95 s to 125 s; the lap is 100.25-110.25 s
        df = _telemetry(np.arange(95.0, 125.0, 0.5), 36.0)
        lap = lap_samples(df, start_s=100.25, lap_s=10.0)
        self.assertEqual(lap["t"].iloc[0], 0.0)                 # pinned to the official start...
        self.assertEqual(lap["t"].iloc[-1], 10.0)               # ...and end
        self.assertAlmostEqual(lap["dist"].iloc[-1], 100.0)     # 10 m/s for 10 s
        self.assertTrue((np.diff(lap["dist"]) >= 0).all())

    def test_too_few_readings_give_an_empty_lap(self) -> None:
        df = _telemetry([0.0, 50.0], 36.0)
        self.assertTrue(lap_samples(df, start_s=10.0, lap_s=10.0).empty)


class TestTimeDelta(unittest.TestCase):

    def test_slower_lap_loses_time_evenly_and_ends_at_the_lap_time_gap(self) -> None:
        # Same 100 m: 10 s at 10 m/s vs 5 s at 20 m/s (reference). Halfway the slow lap is 2.5 s down.
        slow = lap_samples(_telemetry(np.arange(0.0, 10.01, 0.5), 36.0), 0.0, 10.0)
        fast = lap_samples(_telemetry(np.arange(0.0, 5.01, 0.25), 72.0), 0.0, 5.0)
        delta = time_delta(slow, fast)
        self.assertAlmostEqual(delta[0], 0.0)
        self.assertAlmostEqual(delta[-1], 5.0)
        half = int(np.argmin(np.abs(slow["dist"].to_numpy() - 50.0)))
        self.assertAlmostEqual(delta[half], 2.5)
        self.assertAlmostEqual(time_delta(fast, slow)[-1], -5.0)


class TestCornersAndThrottle(unittest.TestCase):

    def test_corner_is_placed_at_the_nearest_reading(self) -> None:
        lap = lap_samples(_telemetry(np.arange(0.0, 10.01, 1.0), 36.0), 0.0, 10.0)   # X = reading no.
        corners = pd.DataFrame({"label": ["T2", "T1"], "X": [7.2, 2.9], "Y": [0.0, 0.0]})
        out = corner_distances(lap, corners)
        self.assertEqual(list(out["label"]), ["T1", "T2"])                            # sorted by distance
        x_to_dist = dict(zip(lap["X"], lap["dist"]))
        self.assertAlmostEqual(out["dist"].iloc[0], x_to_dist[3.0])
        self.assertAlmostEqual(out["dist"].iloc[1], x_to_dist[7.0])

    def test_place_names_along_the_lap(self) -> None:
        # Corners at 700, 800 (close pair) and 1500 m on a 2000 m lap
        corners = pd.DataFrame({"label": ["T1", "T2", "T3"], "dist": [700.0, 800.0, 1500.0]})
        cases = {100: "Start straight → Turn 1", 600: "Approaching Turn 1", 720: "Turn 1",
                 900: "Exit of Turn 2", 1200: "Straight: Turn 2 → Turn 3", 1460: "Turn 3",
                 1900: "Turn 3 → finish line"}
        for dist, expected in cases.items():
            self.assertEqual(location_text(dist, corners), expected, dist)

    def test_full_throttle_share_is_weighted_by_time(self) -> None:
        df = _telemetry([0.0, 1.0, 4.0, 5.0], 36.0)
        df["Throttle"] = [100.0, 40.0, 100.0, 100.0]          # 0-1 s full, 1-4 s partial, 4-5 s full
        self.assertAlmostEqual(full_throttle_pct(lap_samples(df, 0.0, 5.0)), 40.0)


if __name__ == "__main__":
    unittest.main()
