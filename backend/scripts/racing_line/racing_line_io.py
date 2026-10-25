"""Data loading and output writing for the ApexHunter racing line
pipeline. Isolates all file I/O operations from the grid and search logic."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from racing_line.racing_line_search import SearchResult
from utils import IST, enable_fastf1_cache

REQUIRED_COLUMNS: List[str] = ["Driver", "X", "Y", "Speed", "Brake", "SessionTime"]


def load_and_validate(
    session_path: Path,
    driver: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Load the parquet. Validate REQUIRED_COLUMNS are present.
    Filter to the specified driver.

    Args:
        session_path: Path to the clean parquet file.
        driver: The driver code string.
        logger: The logger instance.

    Returns:
        The filtered DataFrame.

    Raises:
        ValueError: If a required column is missing or driver not found.
    """
    logger.info(f"Loading session file: {session_path}")
    try:
        df = pd.read_parquet(session_path)
    except Exception as e:
        raise ValueError(f"Failed to load session file: {e}") from e

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        for col in missing_cols:
            logger.error(f"Missing required column: {col}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    df_driver = df[df["Driver"] == driver].copy()
    if df_driver.empty:
        logger.error(f"Driver '{driver}' not found in session file.")
        raise ValueError(f"Driver '{driver}' not found in session file.")

    row_count = len(df_driver)
    logger.info(f"Loaded {row_count} rows for driver {driver} from {session_path.name}")

    df_driver.sort_values("SessionTime", inplace=True)
    df_driver.reset_index(drop=True, inplace=True)

    return df_driver


def build_output(
    session_path: str,
    driver: str,
    grid_resolution: float,
    fastest_lap: dict,
    scale: float,
    astar_result: SearchResult,
    dijkstra_result: SearchResult,
    bfs_result: SearchResult,
    astar_cost_bfs: float,
    driver_path_coords: List[Tuple[float, float]],
    deviation_per_corner: List[dict],
    n_corners: int
) -> dict:
    """Construct the complete output JSON dictionary.

    Args:
        session_path: The session file path.
        driver: The driver code string.
        grid_resolution: The grid cell size.
        fastest_lap: The FastF1 lap the line was built from (lap_number, lap_time_s, length_km,
            estimated_time_s = the time model applied to that lap, a check on its accuracy).
        scale: Metres per coordinate unit.
        astar_result: Result from A*.
        dijkstra_result: Result from Dijkstra.
        bfs_result: Result from BFS.
        astar_cost_bfs: The weighted cost of the BFS path.
        driver_path_coords: Subsampled telemetry path for the driver.
        deviation_per_corner: Computed per-corner deviation.
        n_corners: Number of corners for deviation.

    Returns:
        The complete dictionary matching the expected output schema.
    """
    return {
        "session_file": session_path,
        "driver": driver,
        "grid_resolution": grid_resolution,
        "fastest_lap": fastest_lap,
        "coordinate_scale": scale,
        "n_corners": n_corners,
        "timestamp": datetime.now(IST).isoformat(),
        "driver_path": [[x, y] for x, y in driver_path_coords],
        "algorithms": {
            "astar": {
                "path": [[x, y] for x, y in astar_result.path_coords],
                "cost": round(astar_result.total_cost, 4),
                "nodes_expanded": astar_result.nodes_expanded,
                "compute_time_s": round(astar_result.compute_time_s, 4),
                "time_saved_s": None,
                "found": astar_result.found
            },
            "dijkstra": {
                "path": [[x, y] for x, y in dijkstra_result.path_coords],
                "cost": round(dijkstra_result.total_cost, 4),
                "nodes_expanded": dijkstra_result.nodes_expanded,
                "compute_time_s": round(dijkstra_result.compute_time_s, 4),
                "time_saved_s": None,
                "found": dijkstra_result.found
            },
            "bfs": {
                "path": [[x, y] for x, y in bfs_result.path_coords],
                "cost": round(astar_cost_bfs, 4),
                "nodes_expanded": bfs_result.nodes_expanded,
                "compute_time_s": round(bfs_result.compute_time_s, 4),
                "time_saved_s": None,
                "found": bfs_result.found
            }
        },
        "deviation_per_corner": deviation_per_corner
    }


# Floor for the speed used in time estimates, so a stationary sample can't make a segment infinite
MIN_SPEED_KMH: float = 1.0


def estimate_path_time(
    path_coords: List[Tuple[float, float]], df_lap: pd.DataFrame, scale: float
) -> float:
    """Estimate how long a path takes, driven at the speeds of the driver's fastest lap.

    Each segment's length (coordinate units x scale = metres) is divided by the speed of the
    fastest-lap sample nearest to the segment's midpoint, and the segment times are summed.
    On the driver's own lap this reproduces the real lap time to about 1 % (2023 R1 Q #1:
    88.76 s estimated vs 89.708 s), which is the accuracy of any time comparison below.
    """
    pts = np.asarray(path_coords, dtype=float)
    if len(pts) < 2:
        return 0.0
    mids = (pts[:-1] + pts[1:]) / 2.0
    lengths_m = np.linalg.norm(np.diff(pts, axis=0), axis=1) * scale
    _, nearest = cKDTree(df_lap[["X", "Y"]].to_numpy(dtype=float)).query(mids)
    speeds_ms = np.maximum(df_lap["Speed"].to_numpy(dtype=float)[nearest], MIN_SPEED_KMH) / 3.6
    return float((lengths_m / speeds_ms).sum())


def compute_time_saved(
    result: SearchResult,
    driver_time_s: float,
    df_lap: pd.DataFrame,
    scale: float
) -> Optional[float]:
    """Seconds saved by a search path compared with the driver's fastest lap.

    Both times come from estimate_path_time with the same fastest-lap speeds, so the comparison
    is like-for-like. Positive = the path is estimated to be faster. Differences smaller than
    about 1 % of a lap are within the estimate's error.

    Args:
        result: The search result (A*, Dijkstra, BFS).
        driver_time_s: estimate_path_time of the driver's full-resolution fastest lap.
        df_lap: The fastest lap's telemetry (X, Y, Speed).
        scale: Metres per coordinate unit.

    Returns:
        The estimated time saved in seconds, or None if the search found no path.
    """
    if not result.found or len(result.path_coords) < 2:
        return None
    return round(driver_time_s - estimate_path_time(result.path_coords, df_lap, scale), 3)


def save_output(data: dict, output_path: Path, logger: logging.Logger) -> None:
    """Write data to output_path as JSON.

    Args:
        data: The dictionary to save.
        output_path: The file path to save to.
        logger: The logger instance.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved racing line JSON: {output_path}")


def fetch_fastest_lap_bounds(
    df: pd.DataFrame, driver: str, logger: logging.Logger
) -> Optional[Tuple[pd.DataFrame, dict]]:
    """Slice the driver's telemetry to their fastest lap, as reported by FastF1.

    Returns:
        (lap telemetry, {"lap_number", "lap_time_s"}), or None if FastF1 has no usable
        fastest lap for this driver or the lap's time window holds no telemetry. Callers
        must not fall back to the whole session: that mixes pit lanes, out-laps and every
        lap into one "driver path".
    """
    import fastf1
    try:
        enable_fastf1_cache()
        year = int(df["Year"].iloc[0])
        round_val = df["Round"].iloc[0]
        try:
            round_val = int(round_val)
        except ValueError:
            pass # Keep as string if it's a name
        session_name = df["Session"].iloc[0]

        session = fastf1.get_session(year, round_val, session_name)
        session.load(telemetry=False, weather=False, messages=False)
        lap = session.laps.pick_drivers(driver).pick_fastest()
    except Exception as e:
        logger.error(f"Could not fetch the fastest lap from FastF1: {e}")
        return None

    # pick_fastest() returns None (or an empty lap) when the driver set no valid timed lap
    if lap is None or len(lap) == 0 or pd.isna(lap["LapStartTime"]) or pd.isna(lap["Time"]):
        logger.error(f"FastF1 has no valid fastest lap for driver {driver}.")
        return None

    df_lap = df[(df["SessionTime"] >= lap["LapStartTime"]) & (df["SessionTime"] <= lap["Time"])]
    if df_lap.empty:
        logger.error("The fastest lap's time window contains no telemetry rows.")
        return None

    lap_time = lap.get("LapTime")
    info = {
        "lap_number": int(lap["LapNumber"]) if pd.notna(lap.get("LapNumber")) else None,
        "lap_time_s": round(lap_time.total_seconds(), 3) if pd.notna(lap_time) else None,
    }
    logger.info(f"Fastest lap: lap {info['lap_number']}, {info['lap_time_s']} s, {len(df_lap)} rows")
    return df_lap, info


def log_racing_line_complete(args, stem, grid, scale, astar_r, astar_ts, dijkstra_r, dijkstra_ts, bfs_r, astar_cost_bfs, bfs_ts, output_path, logger):
    logger.info(f"""======================================================
   ApexHunter - Racing Line - Run Complete
======================================================
   Driver         : {args.driver}
   Session        : {stem}
   Grid nodes     : {len(grid):,}
   Resolution     : {args.resolution}
   Scale          : {scale:.6f} m/unit
------------------------------------------------------
   A*        cost={astar_r.total_cost:.3f}  nodes={astar_r.nodes_expanded:,}  time={astar_r.compute_time_s:.3f}s  saved={astar_ts}s
   Dijkstra  cost={dijkstra_r.total_cost:.3f}  nodes={dijkstra_r.nodes_expanded:,}  time={dijkstra_r.compute_time_s:.3f}s  saved={dijkstra_ts}s
   BFS       cost={astar_cost_bfs:.3f}  nodes={bfs_r.nodes_expanded:,}  time={bfs_r.compute_time_s:.3f}s  saved={bfs_ts}s
======================================================
   Output: {output_path}
======================================================""")
