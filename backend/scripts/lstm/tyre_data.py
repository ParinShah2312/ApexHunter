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
    "mean_speed",
    "mean_throttle",
    "mean_brake",
    "mean_rpm",
    "tyre_age",
    "is_soft",
    "is_medium",
    "is_hard",
    "race_lap_number",
]
CLIFF_SPEED_DROP_KMH: float = 2.0

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


def _consecutive_runs(lap_numbers: pd.Series) -> List[np.ndarray]:
    """Split positional indices into runs where lap numbers increase by exactly 1."""
    values = lap_numbers.to_numpy()
    breaks = np.where(np.diff(values) != 1)[0] + 1
    return np.split(np.arange(len(values)), breaks)


def extract_stints(
    driver_laps: pd.DataFrame, telemetry: pd.DataFrame, logger: logging.Logger
) -> List[dict]:
    """Segment a driver's session into individual tyre stints and aggregate lap-level features.

    Stints come from FastF1's Stint column. Only representative laps are used (see
    representative_lap_mask); training windows are built only from runs of consecutive
    representative laps, so a window never bridges a removed lap (e.g. a safety-car period).

    Args:
        driver_laps: FastF1 laps DataFrame for a single driver.
        telemetry: Cleaned telemetry for that driver.
        logger: Logger instance.

    Returns:
        List of dicts, one per stint with at least one training window:
        - stint_index: int, 0-based stint index
        - n_laps: int, number of representative laps kept
        - cliff_lap: int or None, first kept lap whose speed dropped CLIFF_SPEED_DROP_KMH below the first
        - lap_features: dict of lists, per-lap aggregated features (incl. race_lap_number)
        - sequences / targets: training windows and next-lap mean-speed targets
    """
    stint_results = []

    # Sort laps just in case
    driver_laps = driver_laps.sort_values("LapNumber")
    keep = representative_lap_mask(driver_laps)
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

            # Note: fastf1 TyreLife is a float, handle missing if any
            tyre_life = float(lap["TyreLife"]) if not pd.isna(lap["TyreLife"]) else float(lap["LapNumber"])

            lap_time_secs = lap["LapTime"].total_seconds() if not pd.isna(lap["LapTime"]) else float('nan')

            features = {
                "mean_speed": float(lap_tel["Speed"].mean()),
                "mean_throttle": float(lap_tel["Throttle"].mean()),
                "mean_brake": float(lap_tel["Brake"].mean()),
                "mean_rpm": float(lap_tel["RPM"].mean()),
                "tyre_age": tyre_life,
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

        # Detect cliff lap
        first_lap_speed = lap_features_df["mean_speed"].iloc[0]
        cliff_lap = None
        for i in range(1, len(lap_features_df)):
            if lap_features_df["mean_speed"].iloc[i] < first_lap_speed - CLIFF_SPEED_DROP_KMH:
                cliff_lap = i
                break

        # Build sliding windows (5 input laps + 1 target lap) within each run of consecutive laps
        sequences = []
        targets = []
        for run in _consecutive_runs(lap_features_df["race_lap_number"]):
            run_df = lap_features_df.iloc[run]
            for i in range(len(run_df) - SEQUENCE_LENGTH):
                window = run_df.iloc[i : i + SEQUENCE_LENGTH][LAP_FEATURES].values
                target = run_df["mean_speed"].iloc[i + SEQUENCE_LENGTH]
                sequences.append(window)
                targets.append(float(target))

        if len(sequences) > 0:
            stint_results.append({
                "stint_index": int(stint_num) - 1, # Make 0-indexed to match old logic
                "n_laps": len(lap_features_df),
                "cliff_lap": cliff_lap,
                "lap_features": lap_features_df.to_dict(orient="list"),
                "sequences": sequences,
                "targets": targets,
            })

    return stint_results


def _process_single_file(
    file_path: Path, logger: logging.Logger
) -> Tuple[List[np.ndarray], List[float]]:
    """Process a single parquet file and extract sequences and targets."""
    X_list = []
    y_list = []

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        logger.warning(f"Failed to read parquet {file_path.name}: {e}")
        return [], []

    drivers = df["Driver"].unique()
    if len(drivers) == 0:
        return [], []

    stem = file_path.stem
    try:
        parts = stem.split("_")
        year, round_num, session_type = int(parts[0]), int(parts[1]), parts[2]
        session_f1 = fastf1.get_session(year, round_num, session_type)
        session_f1.load(telemetry=False, weather=False)
        f1_laps = session_f1.laps
    except Exception as e:
        logger.warning(f"Failed to load fastf1 laps for {stem}: {e}")
        return [], []

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

    return X_list, y_list


def session_sort_key(file_path: Path) -> Tuple[int, int]:
    """Return (year, round) parsed from a '{year}_{round}_{session}.parquet' filename.

    Sorting by this key gives true chronological order; plain string sorting would put
    round 10 before round 2.
    """
    parts = file_path.stem.split("_")
    return int(parts[0]), int(parts[1])


def build_training_dataset(
    sessions_dir: Path,
    seasons: List[str],
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate sequences and targets across all Race sessions in the given seasons.

    Sessions are processed in chronological order, so later rows of X come from later races.

    Returns:
        Tuple of (X, y, groups). groups[i] is the chronological index of the session that
        sequence i came from; use it to split data by whole sessions, because neighbouring
        sliding windows share laps and would leak between train and test.
    """
    files = []
    for year in seasons:
        files += list(sessions_dir.glob(f"{year}_*_R.parquet"))
    files.sort(key=session_sort_key)

    logger.info(f"Found {len(files)} Race parquet files for training.")
    if len(files) == 0:
        raise ValueError("No Race parquet files found.")

    all_sequences = []
    all_targets = []
    all_groups = []

    enable_fastf1_cache()

    for session_idx, filepath in enumerate(files):
        X_file, y_file = _process_single_file(filepath, logger)
        all_sequences.extend(X_file)
        all_targets.extend(y_file)
        all_groups.extend([session_idx] * len(X_file))
        gc.collect()

    if len(all_sequences) == 0:
        raise ValueError("No valid sequences built from Race parquet files.")

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    groups = np.array(all_groups, dtype=np.int64)
    logger.info(f"Training dataset: {len(X)} sequences from {len(files)} Race sessions.")
    return (X, y, groups)
