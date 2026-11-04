"""Stint detection and lap aggregation using FastF1 lap markers.
Converts raw telemetry into per-lap feature sequences."""

import gc
import logging
from pathlib import Path
from typing import List, Tuple

import fastf1
import numpy as np
import pandas as pd

from utils import enable_fastf1_cache, setup_logger

logger = setup_logger(__name__)

MIN_STINT_LAPS: int = 3
SEQUENCE_LENGTH: int = 5
LAP_FEATURES: List[str] = [
    "lap_time_seconds",
    "mean_throttle",
    "mean_brake",
    "mean_rpm",
    "tyre_age",
    "is_soft",
    "is_medium",
    "is_hard",
    "race_lap_number",
]
# Lap time and per-lap telemetry means given to the model as differences from the window's last
# lap, so a window shows how pace is changing rather than which track it is (#45).
RELATIVE_FEATURES: List[str] = ["lap_time_seconds", "mean_throttle", "mean_brake", "mean_rpm"]
# What the model predicts: the next lap time minus the last input lap's (s, positive = slower).
# Absolute values mostly encode the track (#45); lap time replaced mean speed in §C step 1 because
# the pit strategy adds up lap times.
TARGET: str = "next_lap_time_change_s"

# Dry tyres only: intermediate/wet laps (2.5k of 44.8k representative race laps, 2023-24) behave
# differently and have no compound column; 35 laps with an unknown compound also lack TyreLife.
SLICK_COMPOUNDS: Tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")

# FastF1 TrackStatus codes: 1 green, 2 yellow, 4 safety car, 5 red flag, 6 VSC, 7 VSC ending.
# A lap's TrackStatus string lists every status seen during it (e.g. "126").
# Yellow flags are local and barely affect pace, so only SC / red / VSC laps are excluded.
NEUTRALISED_TRACK_STATUS: Tuple[str, ...] = ("4", "5", "6", "7")


def representative_lap_mask(laps: pd.DataFrame) -> pd.Series:
    """Return a boolean mask of laps that reflect genuine racing pace on the current tyres.

    Excluded: lap 1 (standing start), out-laps (PitOutTime set), in-laps (PitInTime set),
    laps run under safety car / VSC / red flag, deleted or FastF1-generated laps, laps without
    a lap time, and laps FastF1 marks as inaccurately timed. Columns that are absent are
    ignored, so plain DataFrames with a subset of FastF1's columns are accepted.
    """
    mask = pd.Series(True, index=laps.index)
    if "LapNumber" in laps.columns:
        mask &= laps["LapNumber"] > 1
    for col in ("PitOutTime", "PitInTime"):
        if col in laps.columns:
            mask &= laps[col].isna()
    if "TrackStatus" in laps.columns:
        status = laps["TrackStatus"].fillna("").astype(str)
        for code in NEUTRALISED_TRACK_STATUS:
            mask &= ~status.str.contains(code, regex=False)
    for col in ("Deleted", "FastF1Generated"):
        if col in laps.columns:
            mask &= ~laps[col].astype("boolean").fillna(False).astype(bool)
    if "IsAccurate" in laps.columns:
        mask &= laps["IsAccurate"].astype("boolean").fillna(False).astype(bool)
    if "LapTime" in laps.columns:
        mask &= laps["LapTime"].notna()
    return mask


def slick_lap_mask(laps: pd.DataFrame) -> pd.Series:
    """Laps on a dry compound with a known TyreLife (the only laps the tyre model uses)."""
    compound = laps["Compound"].fillna("").astype(str).str.upper()
    return compound.isin(SLICK_COMPOUNDS) & laps["TyreLife"].notna()


def lap_means(lap_tel: pd.DataFrame) -> dict:
    """Time-weighted means of Speed, Throttle, Brake and RPM over one lap.

    Telemetry samples are irregularly spaced (0.014-0.76 s), so a plain mean over samples
    over-weights stretches with dense samples; on real laps that added ~1 km/h of noise to the
    mean speed. Each pair of neighbouring samples is weighted by the time between them.
    """
    t = lap_tel["SessionTime"].dt.total_seconds().to_numpy()
    dt = np.diff(t)
    out = {}
    for col, name in (("Speed", "mean_speed"), ("Throttle", "mean_throttle"),
                      ("Brake", "mean_brake"), ("RPM", "mean_rpm")):
        v = lap_tel[col].to_numpy(dtype=float)
        out[name] = float(np.sum(0.5 * (v[1:] + v[:-1]) * dt) / np.sum(dt))
    return out


def _consecutive_runs(lap_numbers: pd.Series) -> List[np.ndarray]:
    """Split positional indices into runs where lap numbers increase by exactly 1."""
    values = lap_numbers.to_numpy()
    breaks = np.where(np.diff(values) != 1)[0] + 1
    return np.split(np.arange(len(values)), breaks)


def make_window(lap_features: pd.DataFrame, j: int, seq_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Model input for kept lap j: laps j-seq_len .. j-1 (LAP_FEATURES columns), with the
    RELATIVE_FEATURES given as differences from the last of those laps."""
    window = lap_features[LAP_FEATURES].iloc[j - seq_len : j].to_numpy(dtype=float).copy()
    cols = [LAP_FEATURES.index(c) for c in RELATIVE_FEATURES]
    window[:, cols] -= window[-1, cols]
    return window


def pace_change(lap_features: pd.DataFrame, j: int) -> float:
    """Training target for kept lap j: its lap time minus the previous kept lap's (s).
    Positive = slower. Only used where window_positions allows j, so lap j-1 is the race lap before."""
    lap_time = lap_features["lap_time_seconds"]
    return float(lap_time.iloc[j] - lap_time.iloc[j - 1])


def window_positions(race_lap_numbers: pd.Series, seq_len: int = SEQUENCE_LENGTH) -> List[int]:
    """Positions (0-based, in stint order) of laps that have a full input window: the seq_len
    laps immediately before them are kept laps with consecutive race lap numbers.

    Training windows and predictions both use this rule, so the model is only ever asked about
    the kind of window it was trained on (never padding, never a window across a removed lap).
    """
    positions: List[int] = []
    for run in _consecutive_runs(race_lap_numbers):
        positions.extend(int(j) for j in run[seq_len:])
    return positions


def extract_stints(
    driver_laps: pd.DataFrame, telemetry: pd.DataFrame, logger: logging.Logger
) -> List[dict]:
    """Segment a driver's session into individual tyre stints and aggregate lap-level features.

    Stints come from FastF1's Stint column. Only representative laps on dry tyres are used (see
    representative_lap_mask and slick_lap_mask); training windows are built only from runs of consecutive
    representative laps, so a window never bridges a removed lap (e.g. a safety-car period).

    Args:
        driver_laps: FastF1 laps DataFrame for a single driver.
        telemetry: Cleaned telemetry for that driver.
        logger: Logger instance.

    Returns:
        List of dicts, one per stint with at least MIN_STINT_LAPS kept laps. Stints too short or
        too broken up for a training window are still returned (with empty sequences) so that
        prediction can show them:
        - stint_index: int, 0-based stint index
        - n_laps: int, number of representative laps kept
        - lap_features: dict of lists, per-lap aggregated features (incl. race_lap_number)
        - sequences / targets: training windows (make_window) and next-lap pace changes (pace_change)
        - last_lap_times: lap time (s) of each window's last lap, to turn a change into a lap time
    """
    stint_results = []

    # Sort laps just in case
    driver_laps = driver_laps.sort_values("LapNumber")
    keep = representative_lap_mask(driver_laps) & slick_lap_mask(driver_laps)
    if (~keep).any():
        logger.debug(f"Excluded {int((~keep).sum())}/{len(driver_laps)} non-representative laps")
    driver_laps = driver_laps[keep]

    for stint_num, stint_laps in driver_laps.groupby("Stint"):
        lap_feature_list = []

        for _, lap in stint_laps.iterrows():
            start_time = lap["LapStartTime"]
            end_time = lap["Time"]

            # Slice telemetry for this lap
            mask = (telemetry["SessionTime"] >= start_time) & (telemetry["SessionTime"] <= end_time)
            lap_tel = telemetry[mask]

            if len(lap_tel) < 5:
                continue

            compound = str(lap["Compound"]).upper()
            lap_time_secs = lap["LapTime"].total_seconds() if not pd.isna(lap["LapTime"]) else float('nan')

            features = {
                **lap_means(lap_tel),
                "tyre_age": float(lap["TyreLife"]),
                "is_soft": 1.0 if compound == "SOFT" else 0.0,
                "is_medium": 1.0 if compound == "MEDIUM" else 0.0,
                "is_hard": 1.0 if compound == "HARD" else 0.0,
                "race_lap_number": float(lap["LapNumber"]),
                "lap_time_seconds": float(lap_time_secs),
            }
            lap_feature_list.append(features)

        if len(lap_feature_list) < MIN_STINT_LAPS:
            continue

        lap_features_df = pd.DataFrame(lap_feature_list)

        # Sliding windows: 5 consecutive input laps -> how much slower/faster the next lap is (s)
        positions = window_positions(lap_features_df["race_lap_number"])
        sequences = [make_window(lap_features_df, j) for j in positions]
        targets = [pace_change(lap_features_df, j) for j in positions]
        last_lap_times = [float(lap_features_df["lap_time_seconds"].iloc[j - 1]) for j in positions]

        # Kept even without windows (#20): training simply gets nothing from it, while
        # prediction still reports the stint (actual laps, no predictions)
        stint_results.append({
            "stint_index": int(stint_num) - 1, # Make 0-indexed to match old logic
            "n_laps": len(lap_features_df),
            "lap_features": lap_features_df.to_dict(orient="list"),
            "sequences": sequences,
            "targets": targets,
            "last_lap_times": last_lap_times,
        })

    return stint_results


def _process_single_file(
    file_path: Path, logger: logging.Logger
) -> Tuple[List[np.ndarray], List[float], List[float]]:
    """Process a single parquet file and extract sequences, targets and last-lap times."""
    X_list = []
    y_list = []
    base_list = []

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        logger.warning(f"Failed to read parquet {file_path.name}: {e}")
        return [], [], []

    drivers = df["Driver"].unique()
    if len(drivers) == 0:
        return [], [], []

    stem = file_path.stem
    try:
        parts = stem.split("_")
        year, round_num, session_type = int(parts[0]), int(parts[1]), parts[2]
        session_f1 = fastf1.get_session(year, round_num, session_type)
        session_f1.load(telemetry=False, weather=False)
        f1_laps = session_f1.laps
    except Exception as e:
        logger.warning(f"Failed to load fastf1 laps for {stem}: {e}")
        return [], [], []

    for driver in drivers:
        df_driver = df[df["Driver"] == driver].copy()
        df_driver.sort_values("SessionTime", inplace=True)
        df_driver.reset_index(drop=True, inplace=True)

        try:
            driver_laps = f1_laps.pick_drivers(driver)
            if len(driver_laps) == 0:
                continue
        except Exception:
            continue

        stints = extract_stints(driver_laps, df_driver, logger)
        for stint_data in stints:
            X_list.extend(stint_data["sequences"])
            y_list.extend(stint_data["targets"])
            base_list.extend(stint_data["last_lap_times"])

    return X_list, y_list, base_list


def session_sort_key(file_path: Path) -> Tuple[int, int]:
    """Return (year, round) parsed from a '{year}_{round}_{session}.parquet' filename.

    Sorting by this key gives true chronological order; plain string sorting would put
    round 10 before round 2.
    """
    parts = file_path.stem.split("_")
    return int(parts[0]), int(parts[1])


def race_files(sessions_dir: Path, seasons: List[str]) -> List[Path]:
    """Race parquet files of the given seasons in chronological order; groups[i] from
    build_training_dataset indexes this list."""
    files = []
    for year in seasons:
        files += list(sessions_dir.glob(f"{year}_*_R.parquet"))
    return sorted(files, key=session_sort_key)


def build_training_dataset(
    sessions_dir: Path,
    seasons: List[str],
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate sequences and targets across all Race sessions in the given seasons.

    Sessions are processed in chronological order, so later rows of X come from later races.

    Returns:
        Tuple of (X, y, groups, last_lap_time). groups[i] is the chronological index of the
        session that sequence i came from; use it to split data by whole sessions, because
        neighbouring sliding windows share laps and would leak between train and test.
        last_lap_time[i] is the window's last lap time (s): predicted lap time = it + change.
    """
    files = race_files(sessions_dir, seasons)

    logger.info(f"Found {len(files)} Race parquet files for training.")
    if len(files) == 0:
        raise ValueError("No Race parquet files found.")

    all_sequences = []
    all_targets = []
    all_groups = []
    all_base = []

    enable_fastf1_cache()

    for session_idx, filepath in enumerate(files):
        X_file, y_file, base_file = _process_single_file(filepath, logger)
        all_sequences.extend(X_file)
        all_targets.extend(y_file)
        all_base.extend(base_file)
        all_groups.extend([session_idx] * len(X_file))
        gc.collect()

    if len(all_sequences) == 0:
        raise ValueError("No valid sequences built from Race parquet files.")

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    groups = np.array(all_groups, dtype=np.int64)
    last_lap_time = np.array(all_base, dtype=np.float64)
    logger.info(f"Training dataset: {len(X)} sequences from {len(files)} Race sessions.")
    return (X, y, groups, last_lap_time)
