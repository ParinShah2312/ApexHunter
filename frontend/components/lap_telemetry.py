"""
ApexHunter Frontend - Lap telemetry maths (no Streamlit)
Cuts one lap out of a driver's telemetry and puts it on a distance axis, so two laps can be
compared corner by corner.

- Distance: speed × time between readings, added up from the start line (trapezoids).
- The lap is pinned to its official start and end: a point is added at time 0 and at the lap time,
  so the time-delta line ends exactly at the lap-time gap.
- Time delta: for each point of the chosen lap, the time the other lap needed to reach the same
  fraction of the lap. Fractions, not metres, so two laps whose summed distances differ by a few
  metres (different lines, ~4 readings a second) still line up at the finish.
"""

from typing import Optional

import numpy as np
import pandas as pd

CHANNELS = ["Speed", "Throttle", "Brake", "nGear", "RPM", "X", "Y"]
FULL_THROTTLE_PCT = 98.0


def lap_samples(df_driver: pd.DataFrame, start_s: float, lap_s: float) -> pd.DataFrame:
    """One lap of telemetry with columns t (s from the lap start), dist (m) and CHANNELS.

    Args:
        df_driver: the driver's session telemetry (SessionTime as timedelta, Speed km/h, ...).
        start_s: lap start, seconds of session time.
        lap_s: lap time, seconds.

    Returns an empty frame if fewer than two readings fall inside the lap.
    """
    t = df_driver["SessionTime"].dt.total_seconds().to_numpy() - start_s
    inside = (t >= 0) & (t <= lap_s)
    if inside.sum() < 2:
        return pd.DataFrame(columns=["t", "dist"] + CHANNELS)
    lap = df_driver.loc[inside, CHANNELS].reset_index(drop=True)
    lap.insert(0, "t", t[inside])
    # Pin the lap to its official start and end (copies of the nearest reading)
    first, last = lap.iloc[[0]].copy(), lap.iloc[[-1]].copy()
    first["t"], last["t"] = 0.0, float(lap_s)
    lap = pd.concat([first, lap, last], ignore_index=True)
    lap = lap.drop_duplicates(subset="t", keep="first").sort_values("t").reset_index(drop=True)
    v = lap["Speed"].to_numpy(dtype=float) / 3.6
    dt = np.diff(lap["t"].to_numpy())
    lap.insert(1, "dist", np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * dt)]))
    return lap


def time_delta(lap: pd.DataFrame, ref: pd.DataFrame) -> np.ndarray:
    """Seconds the chosen lap is behind the reference lap at each of its points (+ = slower)."""
    frac = lap["dist"].to_numpy() / lap["dist"].iloc[-1]
    ref_frac = ref["dist"].to_numpy() / ref["dist"].iloc[-1]
    return lap["t"].to_numpy() - np.interp(frac, ref_frac, ref["t"].to_numpy())


def to_lap_distance(ref: pd.DataFrame, lap: pd.DataFrame) -> np.ndarray:
    """The reference lap's points placed on the chosen lap's distance axis (same fraction of lap)."""
    return ref["dist"].to_numpy() / ref["dist"].iloc[-1] * lap["dist"].iloc[-1]


def corner_distances(lap: pd.DataFrame, corners: pd.DataFrame) -> pd.DataFrame:
    """Corners (label, X, Y) with dist = distance along this lap of the lap point nearest to each."""
    if corners is None or corners.empty or lap.empty:
        return pd.DataFrame(columns=["label", "dist"])
    xy = lap[["X", "Y"]].to_numpy(dtype=float)
    nearest = [int(np.argmin(((xy - p) ** 2).sum(axis=1))) for p in corners[["X", "Y"]].to_numpy(dtype=float)]
    return pd.DataFrame({"label": corners["label"].to_numpy(),
                         "dist": lap["dist"].to_numpy()[nearest]}).sort_values("dist").reset_index(drop=True)


# Zones around each corner's marker (FastF1 puts it near the apex), in metres along the lap
IN_CORNER_BEFORE_M, IN_CORNER_AFTER_M = 40.0, 60.0
APPROACH_M = 150.0          # up to this far before the corner = approaching it (braking zone)
EXIT_M = 120.0              # up to this far after it = exit of the corner


def _turn(label: str) -> str:
    return "Turn " + label[1:] if label.startswith("T") else label


def location_text(dist_m: float, corners: pd.DataFrame) -> str:
    """Plain-language place on the lap, from the nearest corner:
    'Turn 4', 'Approaching Turn 4', 'Exit of Turn 4', 'Straight: Turn 4 → Turn 5',
    'Start straight → Turn 1' or 'Turn 15 → finish line'.

    corners: label + dist, sorted by dist (corner_distances).
    """
    if corners is None or corners.empty:
        return f"{dist_m:,.0f} m"
    d = corners["dist"].to_numpy()
    labels = corners["label"].to_numpy()
    nearest = int(np.argmin(np.abs(d - dist_m)))
    offset = dist_m - d[nearest]
    name = _turn(labels[nearest])
    if -IN_CORNER_BEFORE_M <= offset <= IN_CORNER_AFTER_M:
        return name
    if -APPROACH_M <= offset < 0:
        return f"Approaching {name}"
    if 0 < offset <= EXIT_M:
        return f"Exit of {name}"
    after = int(np.searchsorted(d, dist_m))          # index of the next corner
    if after == 0:
        return f"Start straight → {_turn(labels[0])}"
    if after == len(d):
        return f"{_turn(labels[-1])} → finish line"
    return f"Straight: {_turn(labels[after - 1])} → {_turn(labels[after])}"


def full_throttle_pct(lap: pd.DataFrame) -> float:
    """Share of the lap's time spent at full throttle (>= 98 %), in percent."""
    dt = np.diff(lap["t"].to_numpy())
    if dt.sum() <= 0:
        return 0.0
    full = lap["Throttle"].to_numpy()[:-1] >= FULL_THROTTLE_PCT
    return float(dt[full].sum() / dt.sum() * 100)


def value_at(lap: pd.DataFrame, dist_m: float, column: str) -> Optional[float]:
    """The value of a channel at the reading nearest to a distance."""
    if lap.empty:
        return None
    i = int(np.argmin(np.abs(lap["dist"].to_numpy() - dist_m)))
    return float(lap[column].iloc[i])
