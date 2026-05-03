"""Data loading and output writing for the ApexHunter racing line
pipeline. Isolates all file I/O operations from the grid and search logic."""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from racing_line_grid import GridNode
from racing_line_search import SearchResult
from utils import IST

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
    circuit_length_km: float,
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
        circuit_length_km: The circuit length in km.
        scale: The coordinate scale factor.
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
        "circuit_length_km": circuit_length_km,
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


def _path_length_meters(path_coords: List[Tuple[float, float]], scale: float) -> float:
    """Compute the path length in meters using Euclidean distance."""
    length_units = 0.0
    for i in range(len(path_coords) - 1):
        x1, y1 = path_coords[i]
        x2, y2 = path_coords[i+1]
        length_units += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return length_units * scale


def compute_time_saved(
    result: SearchResult,
    driver_path_coords: List[Tuple[float, float]],
    grid: Dict[Tuple[int, int], GridNode],
    scale: float
) -> Optional[float]:
    """Estimate seconds saved by following the optimal path vs the driver's
    actual path. Computed by comparing mean node speeds along each path, then
    converting path length difference to time using mean speed.

    Args:
        result: The search result (A*, Dijkstra, BFS).
        driver_path_coords: The actual path coordinates of the driver.
        grid: The node dictionary.
        scale: The coordinate scale factor.

    Returns:
        The estimated time saved in seconds, or None if invalid.
    """
    if not result.found or len(result.path_coords) < 2:
        return None

    def get_nearest_node_speed(pt: Tuple[float, float]) -> float:
        min_dist = float('inf')
        nearest_speed = 0.0
        for node in grid.values():
            dist = abs(node.center_x - pt[0]) + abs(node.center_y - pt[1])
            if dist < min_dist:
                min_dist = dist
                nearest_speed = node.mean_speed
        return nearest_speed

    optimal_speeds = [get_nearest_node_speed(pt) for pt in result.path_coords]
    optimal_mean_speed_kmh = sum(optimal_speeds) / len(optimal_speeds) if optimal_speeds else 0.0

    driver_speeds = [get_nearest_node_speed(pt) for pt in driver_path_coords]
    driver_mean_speed_kmh = sum(driver_speeds) / len(driver_speeds) if driver_speeds else 0.0

    optimal_length = _path_length_meters(result.path_coords, scale)
    driver_length = _path_length_meters(driver_path_coords, scale)

    if optimal_mean_speed_kmh <= 0 or driver_mean_speed_kmh <= 0:
        return None

    optimal_time = optimal_length / (optimal_mean_speed_kmh / 3.6)
    driver_time = driver_length / (driver_mean_speed_kmh / 3.6)
    time_saved = driver_time - optimal_time

    if time_saved > 5.0 or time_saved < -5.0:
        logging.getLogger(__name__).warning(
            f"Suspicious time_saved={time_saved:.2f}s - clamping to ±5s"
        )
        time_saved = max(-5.0, min(5.0, time_saved))

    return round(float(time_saved), 3)


def save_output(data: dict, output_path: Path, logger: logging.Logger) -> None:
    """Write data to output_path as JSON.

    Args:
        data: The dictionary to save.
        output_path: The file path to save to.
        logger: The logger instance.
    """
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved racing line JSON: {output_path}")


def fetch_fastest_lap_bounds(df: pd.DataFrame, driver: str, logger: logging.Logger) -> pd.DataFrame:
    """Fetch the fastest lap boundaries for the given driver."""
    import fastf1
    try:
        year = int(df["Year"].iloc[0])
        round_val = df["Round"].iloc[0]
        try:
            round_val = int(round_val)
        except ValueError:
            pass # Keep as string if it's a name
        session_name = df["Session"].iloc[0]

        session = fastf1.get_session(year, round_val, session_name)
        session.load(telemetry=False, weather=False, messages=False)
        lap = session.laps.pick_driver(driver).pick_fastest()
        start_t = lap["LapStartTime"]
        end_t = lap["Time"]

        df_lap = df[(df["SessionTime"] >= start_t) & (df["SessionTime"] <= end_t)]
        if df_lap.empty:
            logger.warning("Fastest lap empty in telemetry, falling back to full session.")
            return df
        return df_lap
    except Exception as e:
        logger.warning(f"Could not fetch fastest lap bounds: {e}. Falling back to full session.")
        return df


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
