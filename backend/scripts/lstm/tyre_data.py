"""Stint detection and lap aggregation using FastF1 lap markers.
Converts raw telemetry into per-lap feature sequences."""

import gc
import logging
from pathlib import Path
from typing import List, Tuple

import fastf1
import numpy as np
import pandas as pd

from utils import setup_logger

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


def extract_stints(
    driver_laps: pd.DataFrame, telemetry: pd.DataFrame, logger: logging.Logger
) -> List[dict]:
    """Segment a driver's session into individual tyre stints and aggregate lap-level features.

    A new stint is defined when the Compound or Stint number changes.

    Args:
        driver_laps: FastF1 laps DataFrame for a single driver.
        telemetry: Cleaned 10Hz telemetry for that driver.
        logger: Logger instance.

    Returns:
        List of dicts, where each dict represents a stint and contains:
        - stint_index: int, index of the stint
        - compound: str, tyre compound
        - lap_features: List[dict], lap-level aggregated features
    """
    stint_results = []

    # Sort laps just in case
    driver_laps = driver_laps.sort_values("LapNumber")

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

        # Build sliding window sequences
        sequences = []
        targets = []
        for i in range(len(lap_features_df) - SEQUENCE_LENGTH):
            window = lap_features_df.iloc[i : i + SEQUENCE_LENGTH][LAP_FEATURES].values
            target = lap_features_df["mean_speed"].iloc[i + SEQUENCE_LENGTH]
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


def build_training_dataset(
    sessions_dir: Path,
    seasons: List[str],
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate sequences and targets across all sessions in the given seasons."""
    files = []
    for year in seasons:
        files += list(sessions_dir.glob(f"{year}_*_R.parquet"))

    logger.info(f"Found {len(files)} Race parquet files for training.")
    if len(files) == 0:
        raise ValueError("No Race parquet files found.")

    all_sequences = []
    all_targets = []

    cache_dir = sessions_dir.parent.parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    for filepath in files:
        X_file, y_file = _process_single_file(filepath, logger)
        all_sequences.extend(X_file)
        all_targets.extend(y_file)
        gc.collect()

    if len(all_sequences) == 0:
        raise ValueError("No valid sequences built from Race parquet files.")

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    logger.info(f"Training dataset: {len(X)} sequences from {len(files)} Race sessions.")
    return (X, y)
