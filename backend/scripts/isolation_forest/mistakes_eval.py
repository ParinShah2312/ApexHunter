"""Planted-mistake evaluation for the ApexHunter mistake detectors (ROADMAP §B, rework §2.4).

F1 telemetry has no labelled mistakes, so we make a test we can score: plant fake mistakes at
known places in real held-out laps and count how many of them the detector's events find, per
mistake (not per sample): a planted mistake is found when any of its samples is inside an event,
and an event is correct when it touches a planted mistake. Any
detector with fit(X, laps) and score(X, laps) (higher = more anomalous) can be evaluated the same way, so the
Z-score baseline and the LSTM autoencoder reuse this module.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from isolation_forest.mistakes_events import assign_events

from isolation_forest.mistakes_features import (
    ACCEL_WINDOW_S, BRAKE_ON_THRESHOLD, FEATURE_COLUMNS, engineer_features,
)

# ── Injection constants ───────────────────────────────────────────────────────
MISTAKE_TYPES: Tuple[str, ...] = ("lockup", "early_lift", "late_throttle", "pedal_overlap", "missed_downshift")
MISTAKES_PER_LAP: int = 3               # ~1-3 % of a lap's samples, like real mistakes
LAP_EDGE_MARGIN_S: float = 1.0          # never plant within 1 s of the lap start/end
MIN_BRAKE_ZONE_S: float = 0.6
SLOW_CORNER_KMH: float = 150.0
MAX_TRIES: int = 30


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """(first, last) positions of each run of True values."""
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    return list(zip(starts, ends))


def _window(t: np.ndarray, start_t: float, duration: float) -> np.ndarray:
    return np.flatnonzero((t >= start_t) & (t < start_t + duration))


def _plant_one(lap: pd.DataFrame, kind: str, rng: np.random.Generator) -> Optional[Tuple[np.ndarray, dict]]:
    """Choose a window for one mistake of `kind` in this lap and return (positions, changes).

    `changes` maps a column to the new values for those positions. Returns None if the lap has
    no suitable spot. Sizes are moderate on purpose so the test is not trivially easy.
    """
    t = lap["t"].to_numpy()
    v = lap["Speed"].to_numpy(dtype=float)
    thr = lap["Throttle"].to_numpy(dtype=float)
    brake = lap["Brake"].to_numpy() >= BRAKE_ON_THRESHOLD
    zones = [(a, b) for a, b in _runs(brake) if t[b] - t[a] >= MIN_BRAKE_ZONE_S]

    if kind == "lockup":                    # extra 15-30 km/h speed drop inside a braking zone, then recovery
        if not zones:
            return None
        a, b = zones[rng.integers(len(zones))]
        dur = rng.uniform(0.3, 1.0)
        pos = _window(t, rng.uniform(t[a], (t[a] + t[b]) / 2), dur)
        if len(pos) == 0:
            return None
        drop = rng.uniform(15, 30) * np.sin(np.pi * (t[pos] - t[pos[0]]) / dur)
        return pos, {"Speed": np.maximum(v[pos] - drop, 0)}

    if kind == "early_lift":                # throttle off 0.5-1 s before the driver's own lift, on a full-throttle stretch
        if not zones:
            return None
        a, _ = zones[rng.integers(len(zones))]
        full = np.flatnonzero(thr[:a] >= 90)
        if len(full) == 0:
            return None
        lift = full[-1] + 1                 # first sample after the full-throttle stretch
        dur = rng.uniform(0.5, 1.0)
        pos = _window(t, t[lift] - dur, dur)
        if len(pos) == 0 or thr[pos].min() < 90:
            return None
        loss = rng.uniform(3, 8) * (t[pos] - t[pos[0]]) / dur
        return pos, {"Throttle": np.zeros(len(pos)), "Speed": v[pos] - loss}

    if kind == "late_throttle":             # throttle kept at 0 for 0.5-1 s after the driver re-applied it
        reapply = np.flatnonzero((thr[:-1] < 10) & (thr[1:] >= 10)) + 1
        if len(reapply) == 0:
            return None
        i = reapply[rng.integers(len(reapply))]
        dur = rng.uniform(0.5, 1.0)
        pos = _window(t, t[i], dur)
        if len(pos) == 0:
            return None
        loss = rng.uniform(5, 10) * (t[pos] - t[pos[0]]) / dur
        return pos, {"Throttle": np.zeros(len(pos)), "Speed": v[pos] - loss}

    if kind == "pedal_overlap":             # throttle held at 20-40 % while braking
        if not zones:
            return None
        a, b = zones[rng.integers(len(zones))]
        pos = _window(t, rng.uniform(t[a], t[b]), rng.uniform(0.3, 0.8))
        pos = pos[brake[pos]]
        if len(pos) == 0:
            return None
        return pos, {"Throttle": np.full(len(pos), rng.uniform(20, 40))}

    if kind == "missed_downshift":          # one gear too high through a slow corner
        slow = _runs(v < SLOW_CORNER_KMH)
        if not slow:
            return None
        a, b = slow[rng.integers(len(slow))]
        pos = _window(t, rng.uniform(t[a], t[b]), rng.uniform(0.5, 1.0))
        pos = pos[pos <= b]
        if len(pos) == 0:
            return None
        return pos, {"nGear": np.minimum(lap["nGear"].to_numpy()[pos] + 1, 8)}

    raise ValueError(f"Unknown mistake type: {kind}")


def inject_mistakes(df: pd.DataFrame, rng: np.random.Generator) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Plant MISTAKES_PER_LAP non-overlapping fake mistakes of random types in every lap.

    Args:
        df: Kept-lap samples (LapNumber, SessionTime, Speed, Throttle, Brake, nGear, accel, ...),
            sorted by SessionTime.
        rng: Random generator (seeded by the caller for reproducibility).

    Returns:
        (modified copy with acceleration updated by the planted speed change, planted-mistake number per sample (-1 where
        nothing was planted), mistake type per sample ("" where nothing was planted)).
    """
    out = df.copy()
    out["t"] = out["SessionTime"].dt.total_seconds()
    mistake_ids = np.full(len(out), -1, dtype=np.int32)
    kinds = np.full(len(out), "", dtype=object)
    for _, idx in out.groupby("LapNumber").indices.items():
        lap = out.iloc[idx]
        t = lap["t"].to_numpy()
        allowed = (t - t[0] >= LAP_EDGE_MARGIN_S) & (t[-1] - t >= LAP_EDGE_MARGIN_S)
        planted = 0
        for _ in range(MAX_TRIES):
            if planted == MISTAKES_PER_LAP:
                break
            kind = MISTAKE_TYPES[rng.integers(len(MISTAKE_TYPES))]
            found = _plant_one(lap, kind, rng)
            if found is None:
                continue
            pos, changes = found
            rows = idx[pos]
            if not allowed[pos].all() or (mistake_ids[rows] >= 0).any():
                continue
            for col, values in changes.items():
                out.iloc[rows, out.columns.get_loc(col)] = np.asarray(values).astype(out[col].dtype)
            mistake_ids[rows] = mistake_ids.max() + 1
            kinds[rows] = kind
            planted += 1
        # Speed changed -> update acceleration by the change alone (#48). accel was computed on the
        # full telemetry before the traffic filter; recomputing it on this lap would interpolate
        # across the removed traffic samples and give unplanted samples a fake acceleration. The
        # planted speed change dv is 0 outside the mistakes, so only samples within 0.5 s after a
        # planted one change, and a gap in the lap cannot leak into the result.
        dv = out["Speed"].to_numpy(dtype=float)[idx] - df["Speed"].to_numpy(dtype=float)[idx]
        if np.any(dv):
            dv_before = np.interp(t - ACCEL_WINDOW_S, t, dv, left=0.0)
            col = out.columns.get_loc("accel")
            new_accel = out.iloc[idx, col].to_numpy(dtype=float) + (dv - dv_before) / ACCEL_WINDOW_S
            out.iloc[idx, col] = new_accel.astype(out["accel"].dtype)
    return out.drop(columns=["t"]), mistake_ids, kinds


def mistake_metrics(flagged: np.ndarray, frame: pd.DataFrame, mistake_ids: np.ndarray,
                    kinds: np.ndarray) -> Dict[str, object]:
    """Per-mistake precision / recall / F1 of one detector's flags on a planted test set.

    Flags become events with the same rule as the real output (mistakes_events.assign_events,
    so runs shorter than 0.5 s are not events). Recall = planted mistakes with at least one
    sample inside an event; precision = events that touch a planted mistake.

    Args:
        flagged: Flag per sample. frame: SessionTime and LapNumber per sample.
        mistake_ids: Planted-mistake number per sample (-1 = none). kinds: mistake type per sample.
    """
    event_ids = assign_events(frame.assign(is_mistake=flagged))
    in_event = event_ids >= 0
    planted = np.unique(mistake_ids[mistake_ids >= 0])
    found = np.unique(mistake_ids[in_event & (mistake_ids >= 0)])
    events = np.unique(event_ids[in_event])
    correct = np.unique(event_ids[in_event & (mistake_ids >= 0)])
    precision = len(correct) / len(events) if len(events) else 0.0
    recall = len(found) / len(planted) if len(planted) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    kind_of = {int(m): kinds[np.flatnonzero(mistake_ids == m)[0]] for m in planted}
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "recall_by_type": {k: float(np.mean([m in found for m, kk in kind_of.items() if kk == k]))
                           for k in MISTAKE_TYPES if k in kind_of.values()},
        "counts": {"planted": int(len(planted)), "found": int(len(found)),
                   "events": int(len(events)), "correct_events": int(len(correct))},
    }


def planted_test_set(df_test: pd.DataFrame, profile: pd.DataFrame, seed: int):
    """Inject mistakes into held-out laps and compute their features.

    Returns:
        (feature matrix, planted-mistake number per sample, mistake type per sample, frame with
        SessionTime and LapNumber for building events).
    """
    planted, mistake_ids, kinds = inject_mistakes(df_test, np.random.default_rng(seed))
    X = engineer_features(planted, profile)[FEATURE_COLUMNS].to_numpy()
    return X, mistake_ids, kinds, planted[["SessionTime", "LapNumber"]].reset_index(drop=True)


def build_folds(df: pd.DataFrame, profile: pd.DataFrame, n_folds: int, seeds: Tuple[int, ...]):
    """Lap-grouped folds with planted test sets, shared by every detector being evaluated.

    Returns:
        List of (clean training-lap features, [planted_test_set output per seed], lap number of each
        training row). The lap numbers let sequence detectors (the LSTM autoencoder) keep their
        windows inside one lap.

    Raises:
        ValueError: If there are fewer than 2 laps.
    """
    laps = df["LapNumber"].to_numpy()
    n_folds = min(n_folds, len(np.unique(laps)))
    if n_folds < 2:
        raise ValueError("At least 2 laps are needed for the planted-mistake evaluation.")
    X_all = engineer_features(df, profile)[FEATURE_COLUMNS].to_numpy()
    folds = []
    for train_idx, test_idx in GroupKFold(n_splits=n_folds).split(X_all, groups=laps):
        tests = [planted_test_set(df.iloc[test_idx], profile, seed) for seed in seeds]
        folds.append((X_all[train_idx], tests, laps[train_idx]))
    return folds


def fold_metrics(train_scores: np.ndarray, test_scores: np.ndarray, test, contamination: float) -> Dict[str, object]:
    """Per-mistake metrics for one planted test set (planted_test_set output); the flag threshold
    is the (1 - contamination) quantile of the training-lap scores, so every detector flags the
    same share of normal driving."""
    _, mistake_ids, kinds, frame = test
    threshold = np.quantile(train_scores, 1 - contamination)
    return mistake_metrics(test_scores > threshold, frame, mistake_ids, kinds)


def summarise(per_test: List[dict]) -> Dict[str, object]:
    """Mean metrics over folds and seeds; counts are summed."""
    return {
        **{k: float(np.mean([m[k] for m in per_test])) for k in ("precision", "recall", "f1")},
        "recall_by_type": {k: float(np.mean(v)) if v else None for k, v in (
            (k, [m["recall_by_type"][k] for m in per_test if k in m["recall_by_type"]]) for k in MISTAKE_TYPES)},
        "counts": {k: int(sum(m["counts"][k] for m in per_test)) for k in ("planted", "found", "events", "correct_events")},
    }


# ── Detectors (fit on clean laps, score: higher = more anomalous) ─────────────
class ZScoreDetector:
    """Baseline without machine learning: how many standard deviations each feature is from the
    driver's own mean (learned on the training laps); the score is the largest |z| over the
    features, so a sample is flagged when any one feature is extreme."""

    def fit(self, X: np.ndarray, laps=None) -> "ZScoreDetector":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def score(self, X: np.ndarray, laps=None) -> np.ndarray:
        return np.abs((X - self.mean_) / self.std_).max(axis=1)


def compare_detectors(folds, detectors: Dict[str, Callable[[], object]], contamination: float) -> Dict[str, dict]:
    """Evaluate each detector on the same folds and planted mistakes at the same flag rate.

    Args:
        folds: Output of build_folds.
        detectors: name -> factory returning an unfitted detector with fit(X, laps) and score(X, laps)
            (laps = lap number per row; only sequence detectors use it).
        contamination: Share of training-lap samples flagged (the same for every detector).

    Returns:
        name -> summarise() metrics.
    """
    results = {}
    for name, make in detectors.items():
        per_test = []
        for X_train, tests, train_laps in folds:
            det = make().fit(X_train, train_laps)
            train_scores = det.score(X_train, train_laps)
            per_test += [fold_metrics(train_scores, det.score(test[0], test[3]["LapNumber"].to_numpy()), test,
                                      contamination) for test in tests]
        results[name] = summarise(per_test)
    return results
