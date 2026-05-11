"""Script to automatically extract track turning points from telemetry X, Y coordinates."""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from utils import DATA_LAKE_DIR, setup_logger

logger = setup_logger(__name__)


def extract_turn_windows(
    parquet_path: Path,
    driver_num: str = "1",
    speed_threshold: int = 100,
    curvature_threshold: float = 0.005,
    min_turn_duration: float = 0.5,
    merge_gap: float = 0.5,
) -> List[Tuple[float, float]]:
    """Extract turn windows (start_sec, end_sec) from telemetry for a single lap.
    
    Args:
        parquet_path: Path to the session telemetry parquet file.
        driver_num: Driver number to use for tracing the track shape (e.g., "1" for Verstappen).
        speed_threshold: Minimum speed to consider a valid lap segment.
        curvature_threshold: Minimum curvature to trigger a turn state.
        min_turn_duration: Minimum duration (seconds) for a valid turn.
        merge_gap: Gap (seconds) between turns to merge them into a single complex corner.
        
    Returns:
        List of tuples representing (start_time, end_time) of turns.
    """
    logger.info(f"Loading telemetry from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Isolate the requested driver
    df_driver = df[df["Driver"] == driver_num].copy()
    if df_driver.empty:
        raise ValueError(f"Driver {driver_num} not found in telemetry.")
        
    # Sort by time
    time_col = "SessionTime" if "SessionTime" in df_driver.columns else "Time"
    df_driver = df_driver.sort_values(time_col).reset_index(drop=True)
    
    # Ensure time is in seconds
    if pd.api.types.is_timedelta64_dtype(df_driver[time_col]):
        df_driver["TimeSec"] = df_driver[time_col].dt.total_seconds()
    else:
        df_driver["TimeSec"] = df_driver[time_col].astype(float)
        
    # To perfectly isolate a single hot lap, we find the highest speed point
    # which is almost always the end of the main straight just before Turn 1
    # and use its (X, Y) coordinate as the Start/Finish line reference.
    df_driver["Speed"] = df_driver["Speed"].astype(float)
    
    # We want a high speed point that isn't an anomaly, so we apply a slight rolling max
    peak_idx = df_driver["Speed"].rolling(10, center=True).mean().idxmax()
    start_time = df_driver.loc[peak_idx, "TimeSec"]
    start_x = df_driver.loc[peak_idx, "X"]
    start_y = df_driver.loc[peak_idx, "Y"]
    
    # We know a Bahrain lap is around 90 seconds. We look for when the car returns to this exact X,Y point
    # Search window: 80 to 110 seconds after the peak
    search_window = df_driver[
        (df_driver["TimeSec"] > start_time + 80) & 
        (df_driver["TimeSec"] < start_time + 110)
    ]
    
    if search_window.empty:
        # Fallback if the telemetry doesn't cover a full lap
        lap_start_idx = df_driver[df_driver["TimeSec"] <= start_time - 4.5].index[-1]
        lap_end_idx = df_driver[df_driver["TimeSec"] <= start_time + 95].index[-1]
    else:
        # Find the point in the search window closest to (start_x, start_y)
        dists = np.sqrt((search_window["X"] - start_x)**2 + (search_window["Y"] - start_y)**2)
        end_idx = dists.idxmin()
        lap_start_idx = df_driver[df_driver["TimeSec"] <= start_time - 4.5].index[-1]
        lap_end_idx = end_idx

    df_lap = df_driver.loc[lap_start_idx:lap_end_idx].copy().reset_index(drop=True)
    
    from scipy.ndimage import gaussian_filter1d
    
    # Normalize time to start at 0
    df_lap["LapTime"] = df_lap["TimeSec"] - df_lap["TimeSec"].min()
    
    x = df_lap["X"].astype(float).values
    y = df_lap["Y"].astype(float).values
    t = df_lap["LapTime"].values
    
    # Smooth coordinates to remove telemetry jitter
    x = gaussian_filter1d(x, sigma=5)
    y = gaussian_filter1d(y, sigma=5)
    
    # Calculate heading angle (theta)
    dx = np.gradient(x)
    dy = np.gradient(y)
    heading = np.unwrap(np.arctan2(dy, dx))
    
    # Curvature (change in heading over distance or time)
    dt = np.gradient(t)
    dt[dt == 0] = np.nan
    d_heading = np.gradient(heading) / dt
    
    # Smooth curvature to avoid noise
    window = 10
    curvature = pd.Series(np.abs(d_heading)).rolling(window, center=True, min_periods=1).mean().values
    
    # Auto-calculate threshold: lowering to 65% to catch slight kinks
    dynamic_threshold = np.nanpercentile(curvature, 65)
    logger.info(f"Using dynamic curvature threshold: {dynamic_threshold:.4f}")
    
    # Identify turns based on threshold
    is_turn = curvature > dynamic_threshold
    
    # Extract continuous segments
    turn_windows = []
    in_turn = False
    start_t = 0.0
    
    for i in range(len(t)):
        if is_turn[i] and not in_turn:
            in_turn = True
            start_t = t[i]
        elif not is_turn[i] and in_turn:
            in_turn = False
            end_t = t[i]
            if end_t - start_t >= min_turn_duration:
                turn_windows.append([start_t, end_t])
                
    # Handle case where turn continues to the end
    if in_turn:
        end_t = t[-1]
        if end_t - start_t >= min_turn_duration:
            turn_windows.append([start_t, end_t])
            
    # Merge turns that are very close to each other
    merged_windows = []
    for current in turn_windows:
        if not merged_windows:
            merged_windows.append(current)
        else:
            prev = merged_windows[-1]
            if current[0] - prev[1] <= merge_gap:
                prev[1] = current[1]  # Extend the previous turn
            else:
                merged_windows.append(current)
                
    # Format output and calculate exact apex (max curvature in window)
    final_windows = []
    for w in merged_windows:
        mask = (t >= w[0]) & (t <= w[1])
        if mask.any():
            apex_t = t[mask][np.argmax(curvature[mask])]
        else:
            apex_t = (w[0] + w[1]) / 2.0
        final_windows.append((round(w[0], 2), round(apex_t, 2), round(w[1], 2)))
    
    logger.info(f"Extracted {len(final_windows)} turn windows based on telemetry curvature.")
    return final_windows


def main():
    parser = argparse.ArgumentParser(description="Extract track turning points from telemetry.")
    parser.add_argument("--input", type=str, default="2023_1_Q.parquet", help="Parquet file name in season_data.")
    parser.add_argument("--driver", type=str, default="1", help="Driver number.")
    args = parser.parse_args()
    
    parquet_path = DATA_LAKE_DIR / "season_data" / args.input
    if not parquet_path.exists():
        logger.error(f"File not found: {parquet_path}")
        return
        
    windows = extract_turn_windows(parquet_path, driver_num=args.driver)
    
    print("\n--- EXTRACTED TURN WINDOWS ---")
    print(f"File: {args.input}")
    print("[\n" + ",\n".join([f"    ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f})" for w in windows]) + "\n]")
    print("------------------------------\n")
    

if __name__ == "__main__":
    main()
