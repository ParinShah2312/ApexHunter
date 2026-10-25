"""
================================================================================
  ApexHunter - Racing Line Search Pipeline
  Script: optimal_line.py
--------------------------------------------------------------------------------
  Purpose : Builds a weighted grid from telemetry, runs A*, Dijkstra, and BFS,
            and saves results to a JSON file for frontend rendering.

  Usage   : python backend/scripts/optimal_line.py --session <path> --driver <code>
================================================================================
"""

import argparse
import gc
import sys
from pathlib import Path

from racing_line.racing_line_grid import DEFAULT_GRID_RESOLUTION, FASTF1_POSITION_SCALE, build_adjacency, build_grid, get_node_for_row, lap_length_km
from racing_line.racing_line_io import build_output, compute_time_saved, estimate_path_time, fetch_fastest_lap_bounds, load_and_validate, log_racing_line_complete, save_output
from racing_line.racing_line_search import astar, bfs, compute_deviation_per_corner, compute_path_cost_weighted, dijkstra, run_full_lap
from utils import DATA_LAKE_DIR, setup_logger

logger = setup_logger(__name__)


def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else DATA_LAKE_DIR / "racing_lines"
    stem = Path(args.session).stem
    output_stem = f"{stem}_{args.driver}"
    output_path = output_dir / f"{output_stem}_racing_line.json"

    if output_path.exists() and not args.force:
        logger.info("Already processed. Use --force to re-run.")
        return

    try:
        df = load_and_validate(Path(args.session), args.driver, logger)
    except ValueError:
        sys.exit(1)

    lap_lookup = fetch_fastest_lap_bounds(df, args.driver, logger)
    if lap_lookup is None:
        logger.error("No fastest lap for this driver - not building a line from the whole session.")
        sys.exit(1)
    df_lap, fastest_lap = lap_lookup
    fastest_lap["length_km"] = round(lap_length_km(df_lap), 3)
    scale = FASTF1_POSITION_SCALE

    # The grid uses the whole session (more samples per cell); the path is the fastest lap
    grid = build_grid(df, args.resolution)
    if len(grid) < 10:
        logger.error("Grid has fewer than 10 nodes - check resolution and data.")
        sys.exit(1)

    adjacency = build_adjacency(grid)

    # Subsample the fastest lap for the driver_path field
    df_sub = df_lap.sort_values("SessionTime").iloc[::10]
    driver_path = [(float(row.X), float(row.Y)) for row in df_sub.itertuples()]

    x_min = df["X"].min()
    y_min = df["Y"].min()

    # Waypoints at the start, quarters and end of the fastest lap keep the search on one lap
    n = len(df_lap)
    nodes = [get_node_for_row(df_lap.iloc[i], x_min, y_min, args.resolution, grid)
             for i in (0, n // 4, n // 2, 3 * n // 4, n - 1)]

    astar_r = run_full_lap(astar, nodes, grid, adjacency, logger)
    dijkstra_r = run_full_lap(dijkstra, nodes, grid, adjacency, logger)
    bfs_r = run_full_lap(bfs, nodes, grid, adjacency, logger)
    gc.collect()

    astar_cost_bfs = compute_path_cost_weighted(bfs_r.path_keys, grid, adjacency) if bfs_r.found else 0.0

    # Driver and search paths are timed the same way (fastest-lap speeds), so the error cancels
    lap_coords = list(zip(df_lap["X"].astype(float), df_lap["Y"].astype(float)))
    driver_time = estimate_path_time(lap_coords, df_lap, scale)
    fastest_lap["estimated_time_s"] = round(driver_time, 3)
    astar_ts = compute_time_saved(astar_r, driver_time, df_lap, scale)
    dijkstra_ts = compute_time_saved(dijkstra_r, driver_time, df_lap, scale)
    bfs_ts = compute_time_saved(bfs_r, driver_time, df_lap, scale)

    deviation = compute_deviation_per_corner(astar_r, driver_path, scale, args.n_corners)

    data = build_output(
        str(args.session), args.driver, args.resolution, fastest_lap, scale,
        astar_r, dijkstra_r, bfs_r, astar_cost_bfs, driver_path, deviation, args.n_corners
    )
    data["algorithms"]["astar"]["time_saved_s"] = astar_ts
    data["algorithms"]["dijkstra"]["time_saved_s"] = dijkstra_ts
    data["algorithms"]["bfs"]["time_saved_s"] = bfs_ts

    save_output(data, output_path, logger)
    gc.collect()

    log_racing_line_complete(args, stem, grid, scale, astar_r, astar_ts, dijkstra_r, dijkstra_ts, bfs_r, astar_cost_bfs, bfs_ts, output_path, logger)


def main() -> None:
    p = argparse.ArgumentParser(description="Build optimal racing lines.")
    p.add_argument("--session", type=str, required=True, help="Cleaned parquet session file.")
    p.add_argument("--driver", type=str, required=True, help="Driver code string.")
    p.add_argument("--resolution", type=float, default=DEFAULT_GRID_RESOLUTION, help="Grid cell size.")
    p.add_argument("--n-corners", type=int, default=16, help="Number of corners.")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    p.add_argument("--force", action="store_true", help="Overwrite existing output.")

    try:
        run_pipeline(p.parse_args())
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
