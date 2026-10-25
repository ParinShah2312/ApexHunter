"""Grid construction and cost weighting for the ApexHunter racing
line search pipeline. Converts raw telemetry X/Y coordinates into a weighted graph
of on-track nodes suitable for pathfinding algorithms."""

import gc
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import setup_logger

logger = setup_logger(__name__)

# ── Module-level constants ────────────────────────────────────────────────────
DEFAULT_GRID_RESOLUTION: float = 50.0
SPEED_WEIGHT_FACTOR: float = 1.0
BRAKE_WEIGHT_FACTOR: float = 2.0
MIN_POINTS_PER_CELL: int = 1
DIAGONAL_COST_FACTOR: float = 1.4142135623730951
# FastF1 telemetry X/Y are in 1/10 m (fastf1.core.Telemetry docs); checked by integrating
# speed over fastest laps at Bahrain, Monaco, Silverstone and Spa: 0.0996–0.1006 m/unit.
FASTF1_POSITION_SCALE: float = 0.1
MAX_SPEED_FOR_WEIGHT: float = 380.0
MAX_BRAKE_FOR_WEIGHT: float = 1.0       # FastF1 Brake is binary, so a cell's mean brake is a 0–1 fraction

@dataclass


class GridNode:
    grid_i: int
    grid_j: int
    center_x: float
    center_y: float
    mean_speed: float
    mean_brake: float
    weight: float
    point_count: int


def compute_node_weight(mean_speed: float, mean_brake: float) -> float:
    """Compute the traversal cost of a grid node from speed and brake values.

    Lower cost = better for the racing line. High speed = low cost.
    High braking = high cost.

    Args:
        mean_speed: The mean speed of the node.
        mean_brake: Fraction of the node's samples with the brake on (0–1).

    Returns:
        The computed traversal cost.
    """
    speed_reward = (mean_speed / MAX_SPEED_FOR_WEIGHT) * SPEED_WEIGHT_FACTOR
    brake_fraction = min(max(mean_brake / MAX_BRAKE_FOR_WEIGHT, 0.0), 1.0)
    brake_penalty = brake_fraction * BRAKE_WEIGHT_FACTOR
    weight = max(0.01, 1.0 - speed_reward + brake_penalty)
    return weight


def _compute_grid_indices(df: pd.DataFrame, x_min: float, y_min: float, resolution: float) -> pd.DataFrame:
    """Compute grid indices for telemetry coordinates."""
    df_grid = df.copy()
    df_grid["grid_i"] = ((df_grid["X"] - x_min) / resolution).astype(int)
    df_grid["grid_j"] = ((df_grid["Y"] - y_min) / resolution).astype(int)
    return df_grid


def _aggregate_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate points inside each grid cell."""
    grouped = df.groupby(["grid_i", "grid_j"]).agg(
        mean_speed=("Speed", "mean"),
        mean_brake=("Brake", "mean"),
        point_count=("X", "count"),
        center_x=("X", "mean"),
        center_y=("Y", "mean")
    ).reset_index()
    return grouped[grouped["point_count"] >= MIN_POINTS_PER_CELL]


def _create_grid_nodes(grouped: pd.DataFrame) -> Dict[Tuple[int, int], GridNode]:
    """Create GridNode objects from aggregated cell data."""
    grid = {}
    for row in grouped.itertuples(index=False):
        i = int(row.grid_i)
        j = int(row.grid_j)
        weight = compute_node_weight(row.mean_speed, row.mean_brake)
        node = GridNode(
            grid_i=i,
            grid_j=j,
            center_x=row.center_x,
            center_y=row.center_y,
            mean_speed=row.mean_speed,
            mean_brake=row.mean_brake,
            weight=weight,
            point_count=int(row.point_count)
        )
        grid[(i, j)] = node
    return grid


def build_grid(
    df: pd.DataFrame,
    resolution: float = DEFAULT_GRID_RESOLUTION
) -> Dict[Tuple[int, int], GridNode]:
    """Discretize telemetry X/Y coordinates into a 2D grid of weighted nodes.

    Args:
        df: The telemetry DataFrame.
        resolution: Grid cell size in coordinate units.

    Returns:
        A dictionary mapping (i, j) coordinates to GridNode objects.
    """
    x_min = df["X"].min()
    y_min = df["Y"].min()

    df_grid = _compute_grid_indices(df, x_min, y_min, resolution)
    grouped = _aggregate_cells(df_grid)
    grid = _create_grid_nodes(grouped)

    logger.info(f"Grid built: {len(grid)} valid nodes at resolution={resolution}")
    gc.collect()
    return grid


def build_adjacency(
    grid: Dict[Tuple[int, int], GridNode]
) -> Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]:
    """Build the 8-connected adjacency list for the grid.

    Each node connects to its up to 8 neighbors (cardinal + diagonal) if they exist.
    Edge cost is the average of the two nodes' weights multiplied by the Euclidean
    distance between their centers.

    Args:
        grid: The node dictionary built by build_grid.

    Returns:
        Adjacency dictionary mapping node coordinates to lists of (neighbor_coord, cost).
    """
    adjacency = {}
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for (i, j), node in grid.items():
        neighbors = []
        for di, dj in offsets:
            neighbor_key = (i + di, j + dj)
            if neighbor_key in grid:
                neighbor = grid[neighbor_key]
                avg_weight = (node.weight + neighbor.weight) / 2.0
                is_diagonal = (di != 0 and dj != 0)
                dist_factor = DIAGONAL_COST_FACTOR if is_diagonal else 1.0
                edge_cost = avg_weight * dist_factor
                neighbors.append((neighbor_key, edge_cost))
        if neighbors:
            adjacency[(i, j)] = neighbors

    logger.info(f"Adjacency built: {sum(len(v) for v in adjacency.values())} total edges")
    gc.collect()
    return adjacency


def get_nearest_node(
    grid: Dict[Tuple[int, int], GridNode],
    target_i: int,
    target_j: int
) -> Tuple[int, int]:
    """Find the grid node key closest to the given (i, j) indices.

    Used when the exact target cell is not a valid node (off-track).

    Args:
        grid: The valid node dictionary.
        target_i: The target column index.
        target_j: The target row index.

    Returns:
        The (i, j) tuple of the nearest valid grid node.
    """
    if not grid:
        raise ValueError("Grid is empty - cannot find nearest node.")

    nearest_key = None
    min_dist = float('inf')

    for key in grid:
        dist = abs(key[0] - target_i) + abs(key[1] - target_j)
        if dist < min_dist:
            min_dist = dist
            nearest_key = key

    return nearest_key


def lap_length_km(df_lap: pd.DataFrame) -> float:
    """Length of one lap of telemetry (driven path, X/Y in FastF1 units), in km."""
    dx = df_lap["X"].diff().fillna(0.0)
    dy = df_lap["Y"].diff().fillna(0.0)
    return float(np.sqrt(dx**2 + dy**2).sum()) * FASTF1_POSITION_SCALE / 1000.0


def get_node_for_row(row, x_min: float, y_min: float, resolution: float, grid: dict) -> tuple:
    """Get the appropriate grid node for a given telemetry row."""
    i = int((row.X - x_min) / resolution)
    j = int((row.Y - y_min) / resolution)
    if (i, j) in grid:
        return (i, j)
    return get_nearest_node(grid, i, j)
