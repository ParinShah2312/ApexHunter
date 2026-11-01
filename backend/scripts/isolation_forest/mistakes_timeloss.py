"""Time lost by each mistake event, measured against the driver's own laps.

For each event, the time the driver took over a short stretch of track around it (from
WINDOW_BEFORE_M before the event to WINDOW_AFTER_M after it) is compared with the median time over
the same stretch on nearby laps of the same stint (similar tyres and fuel; in qualifying, all the
driver's other kept laps, since every run is its own stint). An event is **costly**
when it lost clearly more than the normal lap-to-lap variation there. Events are ranked by time
lost. Coasting that the driver repeats at the same spot on most laps of the stint is labelled as
likely lift-and-coast or tyre saving (a choice, not a mistake) and ranked last.

The same measurement on random spots of the same laps is the real-data check: a detector whose
events are costly much more often than random spots points at moments that really cost time.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

WINDOW_BEFORE_M: float = 100.0
WINDOW_AFTER_M: float = 200.0
BASELINE_LAPS: int = 5                  # laps either side, same stint
MIN_BASELINE_LAPS: int = 3
COSTLY_SPREAD_FACTOR: float = 2.0       # costly = more than 2x the normal spread ...
COSTLY_MIN_S: float = 0.05              # ... and at least 0.05 s
IQR_TO_STD: float = 1.349
COAST_THROTTLE_PCT: float = 10.0        # coasting = throttle below 10 % with the brake off ...
COAST_MIN_KMH: float = 150.0            # ... at speed (slow corners have short no-pedal phases) ...
COAST_MIN_S: float = 0.3                # ... for at least 0.3 s
LINE_MARGIN_M: float = 20.0             # stretches are cut to stay inside the lap
SAVING_RADIUS_M: float = 50.0
SAVING_LAP_SHARE: float = 0.5           # coasting here on at least half the stint's laps
N_RANDOM_SPOTS: int = 200
RANDOM_SEED: int = 0


def build_lap_timing(df: pd.DataFrame, lap_length_m: float) -> Dict[int, dict]:
    """Per lap: position along the lap (unwrapped across the start/finish line and made
    non-decreasing), time, speed and coasting flag. `df` holds the driver's kept-lap samples with
    LapNumber, SessionTime, track_pos, Speed, Throttle and Brake."""
    timing = {}
    for lap, g in df.sort_values("SessionTime").groupby("LapNumber"):
        pos = g["track_pos"].to_numpy(dtype=float)
        if len(pos) < 10:
            continue
        # A sample just before the line projects to ~lap_length: unwrap jumps of more than half a lap
        jumps = np.diff(pos)
        offset = np.concatenate([[0.0], np.cumsum(np.where(jumps < -lap_length_m / 2, lap_length_m,
                                                           np.where(jumps > lap_length_m / 2, -lap_length_m, 0.0)))])
        pos = pos + offset
        pos -= np.round((np.median(pos) - lap_length_m / 2) / lap_length_m) * lap_length_m   # most of the lap in [0, L]
        pos = np.maximum.accumulate(pos)
        timing[int(lap)] = {
            "pos": pos, "t": g["SessionTime"].dt.total_seconds().to_numpy(),
            "speed": g["Speed"].to_numpy(dtype=float),
            "coast": coasting(g),
        }
    return timing


def coasting(df: pd.DataFrame) -> np.ndarray:
    """Coasting per sample: throttle below COAST_THROTTLE_PCT, brake off, at COAST_MIN_KMH or more."""
    return ((df["Throttle"] < COAST_THROTTLE_PCT) & (df["Brake"] < 0.5) & (df["Speed"] >= COAST_MIN_KMH)).to_numpy()


def _coast_time(t: np.ndarray, coast: np.ndarray) -> float:
    """Seconds between the first and last coasting sample (0 if none)."""
    return float(t[coast][-1] - t[coast][0]) if coast.any() else 0.0


def _stretch(lap: dict, a: float, b: float) -> Optional[Tuple[float, float]]:
    """(seconds to drive from a to b, minimum speed in between), or None if the lap's samples do
    not cover the stretch."""
    pos = lap["pos"]
    if a < pos[0] or b > pos[-1]:
        return None
    t_a, t_b = np.interp([a, b], pos, lap["t"])
    inside = (pos >= a) & (pos <= b)
    return float(t_b - t_a), float(lap["speed"][inside].min()) if inside.any() else float("nan")


def time_lost(timing: Dict[int, dict], stint_of: Dict[int, int], lap: int,
              start_pos: float, end_pos: float, lap_length_m: float,
              nearby_laps: Optional[int] = BASELINE_LAPS) -> Optional[dict]:
    """Time lost over the stretch around [start_pos, end_pos] on `lap` vs the median of the
    laps of the same stint within `nearby_laps` laps (None = any distance; the stretch is cut at
    the start/finish line). None if there are too few comparable laps."""
    if lap not in timing:
        return None
    a = max(start_pos - WINDOW_BEFORE_M, LINE_MARGIN_M)
    b = min(end_pos + WINDOW_AFTER_M, lap_length_m - LINE_MARGIN_M)
    if b <= a:
        return None
    own = _stretch(timing[lap], a, b)
    others = [_stretch(timing[l], a, b) for l in timing
              if l != lap and stint_of.get(l) == stint_of.get(lap)
              and (nearby_laps is None or abs(l - lap) <= nearby_laps)]
    others = [o for o in others if o is not None]
    if own is None or len(others) < MIN_BASELINE_LAPS:
        return None
    times = np.array([o[0] for o in others])
    lost = own[0] - float(np.median(times))
    spread = float(np.subtract(*np.percentile(times, [75, 25])) / IQR_TO_STD)
    return {
        "time_lost_s": round(lost, 3),
        "costly": bool(lost > max(COSTLY_SPREAD_FACTOR * spread, COSTLY_MIN_S)),
        "min_speed_vs_own_kmh": round(own[1] - float(np.median([o[1] for o in others])), 1),
    }


def saving_share(timing: Dict[int, dict], stint_of: Dict[int, int], lap: int, pos: float) -> Tuple[int, int]:
    """(laps of the stint that coast for COAST_MIN_S within SAVING_RADIUS_M of pos, laps of the
    stint checked), the event's own lap excluded."""
    laps = [l for l in timing if l != lap and stint_of.get(l) == stint_of.get(lap)]
    near = {l: np.abs(timing[l]["pos"] - pos) <= SAVING_RADIUS_M for l in laps}
    count = sum(_coast_time(timing[l]["t"][near[l]], timing[l]["coast"][near[l]]) >= COAST_MIN_S for l in laps)
    return int(count), len(laps)


def add_time_lost(events: List[dict], df_events: pd.DataFrame, timing: Dict[int, dict],
                  stint_of: Dict[int, int], lap_length_m: float,
                  nearby_laps: Optional[int] = BASELINE_LAPS) -> List[dict]:
    """Add time lost, costly, the own-laps comparison and the saving label to every event, then
    rank: likely saving last, otherwise by time lost (most first; unknown time lost after).

    Args:
        events: mistakes_events.build_events output (with start_pos_m / end_pos_m).
        df_events: The driver's scored samples with event_id, SessionTime, Speed, Throttle and Brake.
    """
    ids = df_events["event_id"].to_numpy()
    for e in events:
        tl = time_lost(timing, stint_of, e["lap"], e["start_pos_m"], e["end_pos_m"], lap_length_m, nearby_laps)
        e.update(tl or {"time_lost_s": None, "costly": None, "min_speed_vs_own_kmh": None})
        ev = df_events[ids == e["event_id"]]
        coasts = _coast_time(ev["SessionTime"].dt.total_seconds().to_numpy(), coasting(ev)) >= COAST_MIN_S
        k, n = saving_share(timing, stint_of, e["lap"], e["track_pos_m"]) if coasts else (0, 0)
        e["likely_saving"] = bool(n and k / n >= SAVING_LAP_SHARE)
        e["saving_note"] = (f"Coasts here on {k} of {n} other laps of the stint: likely lift-and-coast "
                            f"or tyre saving" if e["likely_saving"] else "")
    events.sort(key=lambda e: (e["likely_saving"], e["time_lost_s"] is None, -(e["time_lost_s"] or 0.0)))
    for rank, e in enumerate(events, start=1):
        e["rank"] = rank
    return events


def event_windows(df: pd.DataFrame, event_ids: np.ndarray) -> List[Tuple[int, float, float]]:
    """(lap, first track_pos, last track_pos) of every event (event_ids from assign_events)."""
    out = []
    for event_id in np.unique(event_ids[event_ids >= 0]):
        ev = df[event_ids == event_id]
        out.append((int(ev["LapNumber"].iloc[0]), float(ev["track_pos"].iloc[0]), float(ev["track_pos"].iloc[-1])))
    return out


def costly_share(windows: List[Tuple[int, float, float]], timing: Dict[int, dict],
                 stint_of: Dict[int, int], lap_length_m: float,
                 nearby_laps: Optional[int] = BASELINE_LAPS) -> dict:
    """Share of windows whose stretch lost clearly more time than usual (windows without enough
    comparable laps are skipped)."""
    results = [time_lost(timing, stint_of, *w, lap_length_m, nearby_laps) for w in windows]
    results = [r for r in results if r is not None]
    return {"windows": len(windows), "measured": len(results),
            "costly_share": round(float(np.mean([r["costly"] for r in results])), 3) if results else None,
            "median_time_lost_s": round(float(np.median([r["time_lost_s"] for r in results])), 3) if results else None}


def random_windows(timing: Dict[int, dict], lap_length_m: float) -> List[Tuple[int, float, float]]:
    """N_RANDOM_SPOTS random (lap, position) spots on the analysed laps, for the chance level."""
    rng = np.random.default_rng(RANDOM_SEED)
    laps = sorted(timing)
    spots = rng.uniform(WINDOW_BEFORE_M, lap_length_m - WINDOW_AFTER_M, N_RANDOM_SPOTS)
    return [(int(laps[rng.integers(len(laps))]), float(p), float(p)) for p in spots]
