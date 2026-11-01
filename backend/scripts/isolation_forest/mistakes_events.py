"""Turn per-sample Isolation Forest flags into mistake events with a plain-language reason.

Consecutive flagged samples of the same lap less than EVENT_GAP_S apart form one event; runs
shorter than MIN_EVENT_S are not events (single noisy samples; their flags stay in the parquet).
Each event gets a location (nearest corner) from FastF1's circuit info. build_events ranks events
by severity = how unusual x how long (the sum over the event of the anomaly excess times the
sample spacing); the pipeline then re-ranks them by time lost (mistakes_timeloss).
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from isolation_forest.mistakes_features import FEATURE_COLUMNS

EVENT_GAP_S: float = 1.0                # flags < 1 s apart = one event (0.5 -> 1.0: planted-mistake
                                        # recall 0.22 -> 0.27, F1 0.29 -> 0.34 on 4 sessions)
MIN_EVENT_S: float = 0.5                # shorter flag runs are not events
CORNER_RADIUS_M: float = 50.0           # within 50 m of a corner marker = "Turn N"
CORNER_APPROACH_M: float = 200.0        # up to 200 m before it = "Approaching Turn N"
MAX_SAMPLE_SPACING_S: float = 0.3       # caps a sample's time weight across telemetry gaps
OVERLAP_REASON_SHARE: float = 0.5       # pedal overlap is the reason if it is on for half the event


def assign_events(df: pd.DataFrame) -> np.ndarray:
    """Event number per sample (-1 where not flagged or in a flag run shorter than MIN_EVENT_S).
    `df` is sorted by SessionTime and has LapNumber, SessionTime and is_mistake."""
    flagged = df["is_mistake"].to_numpy()
    t = df["SessionTime"].dt.total_seconds().to_numpy()[flagged]
    lap = df["LapNumber"].to_numpy()[flagged]
    new_event = np.ones(len(t), dtype=bool)
    new_event[1:] = (np.diff(t) >= EVENT_GAP_S) | (lap[1:] != lap[:-1])
    run = np.cumsum(new_event) - 1
    duration = pd.Series(t).groupby(run).agg(lambda x: x.iloc[-1] - x.iloc[0]).to_numpy() if len(t) else t
    long_runs = np.flatnonzero(duration >= MIN_EVENT_S - 1e-9)
    new_id = np.full(len(duration), -1, dtype=np.int32)
    new_id[long_runs] = np.arange(len(long_runs))
    ids = np.full(len(df), -1, dtype=np.int32)
    ids[flagged] = new_id[run]
    return ids


def _reason(event: pd.DataFrame, feature_std: pd.Series) -> (str, str):
    """(main feature, plain-language reason) for one event.

    The main feature is the one with the largest mean |deviation| in units of its own spread over
    the driver's samples, so features on different scales compare fairly.
    """
    if event["brake_throttle_overlap"].mean() >= OVERLAP_REASON_SHARE:
        return "brake_throttle_overlap", "Brake and throttle at the same time"
    z = (event[FEATURE_COLUMNS].abs().mean() / feature_std).fillna(0)
    feature = str(z.idxmax())
    sign = np.sign(event[feature].mean())
    speed = event["speed_vs_field_kmh"].mean()
    throttle = event["throttle_vs_field_pct"].mean()
    gear = event["gear_vs_field"].mean()
    field_brake = event["field_brake_share"].mean() * 100
    text = {
        ("speed_deficit", 1): f"{-speed:.0f} km/h slower than the field here",
        ("speed_deficit", -1): f"{speed:.0f} km/h faster than the field here",
        ("throttle_diff", -1): f"Off the throttle: {-throttle:.0f} % less than the field",
        ("throttle_diff", 1): f"On the throttle: {throttle:.0f} % more than the field",
        ("brake_mismatch", 1): f"Braking where the field mostly doesn't ({field_brake:.0f} % of field samples brake here)",
        ("brake_mismatch", -1): f"Not braking where the field does ({field_brake:.0f} % of field samples brake here)",
        ("gear_diff", 1): f"{gear:.1f} gear(s) higher than the field",
        ("gear_diff", -1): f"{-gear:.1f} gear(s) lower than the field",
        ("accel_diff", -1): "Slowing down harder than the field",
        ("accel_diff", 1): "Accelerating harder than the field",
    }.get((feature, int(sign)), "Unusual compared with the field")
    return feature, text


def build_events(df: pd.DataFrame) -> List[Dict[str, object]]:
    """Mistake events, most severe first.

    Args:
        df: The scored driver samples sorted by SessionTime, with event_id (assign_events),
            anomaly_score (decision_function; negative = anomalous), the FEATURE_COLUMNS and the
            INFO_COLUMNS, LapNumber, SessionTime, track_pos, X, Y and Speed.

    Returns:
        One dict per event with rank, lap, times, track position, peak location and speed,
        peak score, severity, main feature and reason.
    """
    t = df["SessionTime"].dt.total_seconds().to_numpy()
    spacing = np.clip(np.diff(t, append=t[-1]), 0, MAX_SAMPLE_SPACING_S) if len(t) else t
    excess = np.clip(-df["anomaly_score"].to_numpy(dtype=float), 0, None)
    feature_std = df[FEATURE_COLUMNS].std()
    ids = df["event_id"].to_numpy()
    events = []
    for event_id in np.unique(ids[ids >= 0]):
        rows = np.flatnonzero(ids == event_id)
        ev = df.iloc[rows]
        peak = ev.iloc[int(np.argmax(excess[rows]))]
        feature, reason = _reason(ev, feature_std)
        events.append({
            "event_id": int(event_id),
            "lap": int(ev["LapNumber"].iloc[0]),
            "start_s": round(float(t[rows[0]]), 3),
            "end_s": round(float(t[rows[-1]]), 3),
            "duration_s": round(float(t[rows[-1]] - t[rows[0]]), 3),
            "n_samples": int(len(rows)),
            "track_pos_m": round(float(peak["track_pos"]), 1),
            "start_pos_m": round(float(ev["track_pos"].iloc[0]), 1),
            "end_pos_m": round(float(ev["track_pos"].iloc[-1]), 1),
            "x": float(peak["X"]), "y": float(peak["Y"]),
            "speed_kmh": round(float(peak["Speed"]), 1),
            "peak_score": round(float(excess[rows].max()), 4),
            "severity": round(float((excess[rows] * spacing[rows]).sum()), 5),
            "main_feature": feature,
            "reason": reason,
        })
    events.sort(key=lambda e: e["severity"], reverse=True)
    for rank, e in enumerate(events, start=1):
        e["rank"] = rank
    return events


def location_label(track_pos: float, corners: pd.DataFrame, lap_length_m: float) -> str:
    """Readable location of a track position: "Turn N" within CORNER_RADIUS_M of a corner marker,
    "Approaching Turn N" up to CORNER_APPROACH_M before it, otherwise "After Turn N".

    Args:
        corners: label + track_pos (metres along the track line), as mistakes_io.corner_positions.
        lap_length_m: Track line length (distances wrap around the start/finish line).
    """
    if corners.empty:
        return f"{track_pos:.0f} m into the lap"
    ahead = (corners["track_pos"].to_numpy() - track_pos) % lap_length_m     # distance to each corner, driving forwards
    behind = lap_length_m - ahead
    labels = corners["label"].to_numpy()
    nearest = int(np.argmin(np.minimum(ahead, behind)))
    if min(ahead[nearest], behind[nearest]) <= CORNER_RADIUS_M:
        return str(labels[nearest])
    if ahead.min() <= CORNER_APPROACH_M:
        return f"Approaching {labels[int(np.argmin(ahead))]}"
    return f"After {labels[int(np.argmin(behind))]}"


def event_overlap(event_ids: np.ndarray, other_flags: np.ndarray) -> float:
    """Share of events (event_ids from assign_events) that contain at least one sample flagged
    by another detector; 0.0 when there are no events."""
    ids = event_ids[event_ids >= 0]
    if len(ids) == 0:
        return 0.0
    hit = pd.Series(other_flags[event_ids >= 0]).groupby(ids).any()
    return float(hit.mean())
