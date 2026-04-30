"""Stint detection and lap aggregation using FastF1 lap markers.
Converts raw telemetry into per-lap feature sequences."""

import gc
import logging
from pathlib import Path
from typing import List, Optional, Tuple

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
    driver_laps: pd.DataFrame,
    telemetry_df: pd.DataFrame,
    logger: logging.Logger,
) -> List[dict]:
    """Extract sequences of laps for each stint using FastF1 lap markers."""
    stint_results = []
    
    # Sort laps just in case
    driver_laps = driver_laps.sort_values("LapNumber")
    
    for stint_num, stint_laps in driver_laps.groupby("Stint"):
        lap_feature_list = []
        
        for _, lap in stint_laps.iterrows():
            start_time = lap["LapStartTime"]
            end_time = lap["Time"]
            
            # Slice telemetry for this lap
            mask = (telemetry_df["SessionTime"] >= start_time) & (telemetry_df["SessionTime"] <= end_time)
            lap_tel = telemetry_df[mask]
            
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


def build_training_dataset(
    sessions_dir: Path,
    seasons: List[str],
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray]:
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
        try:
            logger.info(f"Processing {filepath.name}...")
            parts = filepath.stem.split("_")
            year, round_num, session_type = int(parts[0]), int(parts[1]), parts[2]
            
            session_f1 = fastf1.get_session(year, round_num, session_type)
            session_f1.load(telemetry=False, weather=False)
            f1_laps = session_f1.laps
            
            df = pd.read_parquet(filepath)

            for driver in df["Driver"].unique():
                df_driver = df[df["Driver"] == driver].copy()
                try:
                    driver_laps = f1_laps.pick_drivers(driver)
                    if len(driver_laps) == 0:
                        continue
                except Exception:
                    continue
                    
                stints = extract_stints(driver_laps, df_driver, logger)
                
                for stint_data in stints:
                    all_sequences.extend(stint_data["sequences"])
                    all_targets.extend(stint_data["targets"])

        except Exception as e:
            logger.error(f"Failed to process {filepath.name}: {e}")
            continue

        gc.collect()

    if len(all_sequences) == 0:
        raise ValueError("No valid sequences built from Race parquet files.")

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    logger.info(f"Training dataset: {len(X)} sequences from {len(files)} Race sessions.")
    return (X, y)
