"""Orchestrator for the ApexHunter racing line search pipeline.
Builds a weighted grid from telemetry, runs A*, Dijkstra, and BFS, and saves
results to a JSON file for frontend rendering."""

import argparse
import gc
import sys
from pathlib import Path

from racing_line_grid import DEFAULT_GRID_RESOLUTION, build_adjacency, build_grid, compute_coordinate_scale, find_start_end_nodes, get_nearest_node
from racing_line_io import build_output, compute_time_saved, load_and_validate, save_output
from racing_line_search import astar, bfs, compute_deviation_per_corner, compute_path_cost_weighted, dijkstra, SearchResult
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

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_and_validate(Path(args.session), args.driver, logger)
    except ValueError:
        sys.exit(1)

    scale = compute_coordinate_scale(df, args.circuit_length)
    grid = build_grid(df, args.resolution)
    if len(grid) < 10:
        logger.error("Grid has fewer than 10 nodes - check resolution and data.")
        sys.exit(1)

    adjacency = build_adjacency(grid)
    start, end = find_start_end_nodes(grid, df, args.resolution, adjacency)

    import fastf1
    # Find the fastest lap time bounds using fastf1
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
        lap = session.laps.pick_driver(args.driver).pick_fastest()
        start_t = lap["LapStartTime"]
        end_t = lap["Time"]
        
        df_lap = df[(df["SessionTime"] >= start_t) & (df["SessionTime"] <= end_t)]
        if df_lap.empty:
            logger.warning("Fastest lap empty in telemetry, falling back to full session.")
            df_lap = df
    except Exception as e:
        logger.warning(f"Could not fetch fastest lap bounds: {e}. Falling back to full session.")
        df_lap = df

    # Subsample the telemetry for the driver_path field using only the fastest lap
    df_sub = df_lap.sort_values("SessionTime").iloc[::10]
    driver_path = [(float(row.X), float(row.Y)) for row in df_sub.itertuples()]

    x_min = df["X"].min()
    y_min = df["Y"].min()
    def get_node_for_row(row):
        i = int((row.X - x_min) / args.resolution)
        j = int((row.Y - y_min) / args.resolution)
        if (i, j) in grid: return (i, j)
        return get_nearest_node(grid, i, j)

    if not df_lap.empty:
        n = len(df_lap)
        nodes = [
            get_node_for_row(df_lap.iloc[0]),
            get_node_for_row(df_lap.iloc[n // 4]),
            get_node_for_row(df_lap.iloc[n // 2]),
            get_node_for_row(df_lap.iloc[3 * n // 4]),
            get_node_for_row(df_lap.iloc[-1])
        ]
    else:
        start, end = find_start_end_nodes(grid, df, args.resolution, adjacency)
        nodes = [start, end]

    def run_full_lap(search_func):
        if len(nodes) == 2:
            return search_func(grid, adjacency, nodes[0], nodes[1], logger)
            
        results = []
        for i in range(len(nodes) - 1):
            res = search_func(grid, adjacency, nodes[i], nodes[i+1], logger)
            if not res.found:
                return res # Failed segment
            results.append(res)
            
        # Stitch all results together
        final_coords = results[0].path_coords
        final_keys = results[0].path_keys
        total_cost = results[0].total_cost
        total_expanded = results[0].nodes_expanded
        total_time = results[0].compute_time_s
        
        for r in results[1:]:
            final_coords += r.path_coords[1:]
            final_keys += r.path_keys[1:]
            total_cost += r.total_cost
            total_expanded += r.nodes_expanded
            total_time += r.compute_time_s
            
        return SearchResult(
            algorithm=results[0].algorithm,
            path_coords=final_coords,
            path_keys=final_keys,
            total_cost=total_cost,
            nodes_expanded=total_expanded,
            compute_time_s=total_time,
            found=True
        )

    astar_r = run_full_lap(astar)
    dijkstra_r = run_full_lap(dijkstra)
    bfs_r = run_full_lap(bfs)
    gc.collect()

    astar_cost_bfs = compute_path_cost_weighted(bfs_r.path_keys, grid, adjacency) if bfs_r.found else 0.0

    astar_ts = compute_time_saved(astar_r, driver_path, grid, scale)
    dijkstra_ts = compute_time_saved(dijkstra_r, driver_path, grid, scale)
    bfs_ts = compute_time_saved(bfs_r, driver_path, grid, scale)

    deviation = compute_deviation_per_corner(astar_r, driver_path, scale, args.n_corners)

    data = build_output(
        str(args.session), args.driver, args.resolution, args.circuit_length, scale,
        astar_r, dijkstra_r, bfs_r, astar_cost_bfs, driver_path, deviation, args.n_corners
    )
    data["algorithms"]["astar"]["time_saved_s"] = astar_ts
    data["algorithms"]["dijkstra"]["time_saved_s"] = dijkstra_ts
    data["algorithms"]["bfs"]["time_saved_s"] = bfs_ts

    save_output(data, output_path, logger)
    gc.collect()

    print(f"""======================================================
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


def main() -> None:
    p = argparse.ArgumentParser(description="Build optimal racing lines.")
    p.add_argument("--session", type=str, required=True, help="Cleaned parquet session file.")
    p.add_argument("--driver", type=str, required=True, help="Driver code string.")
    p.add_argument("--resolution", type=float, default=DEFAULT_GRID_RESOLUTION, help="Grid cell size.")
    # Common lengths: Bahrain 5.412, Monaco 3.337, Silverstone 5.891, Monza 5.793,
    # Spa 7.004, Suzuka 5.807, Abu Dhabi 5.281, Melbourne 5.278
    p.add_argument("--circuit-length", type=float, default=5.412, help="Circuit length in km.")
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
