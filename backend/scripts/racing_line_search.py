"""A*, Dijkstra, and BFS pathfinding algorithms for the ApexHunter
racing line optimization pipeline. All three algorithms operate on the same weighted
grid and adjacency structure from racing_line_grid.py."""

import gc
import heapq
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from racing_line_grid import GridNode


@dataclass


class SearchResult:
    algorithm: str
    path_keys: List[Tuple[int, int]]
    path_coords: List[Tuple[float, float]]
    total_cost: float
    nodes_expanded: int
    compute_time_s: float
    found: bool


def _reconstruct_path(came_from: dict, end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Reconstruct path from came_from dict backwards."""
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


def _astar_heuristic_full(
    current_node: GridNode, end_node: GridNode, min_weight: float
) -> float:
    """Compute the heuristic cost from current node to end."""
    grid_dist = math.sqrt(
        (current_node.grid_i - end_node.grid_i)**2 +
        (current_node.grid_j - end_node.grid_j)**2
    )
    return grid_dist * min_weight


def astar(
    grid: Dict[Tuple[int, int], GridNode],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    start: Tuple[int, int],
    end: Tuple[int, int],
    logger: logging.Logger
) -> SearchResult:
    """A* search using Euclidean distance heuristic scaled by the minimum
    edge cost. Guaranteed optimal when the heuristic is admissible.

    Args:
        grid: The node dictionary.
        adjacency: The adjacency dictionary.
        start: The start node coordinate.
        end: The end node coordinate.
        logger: The logger instance.

    Returns:
        A SearchResult containing the path and metadata.
    """
    t_start = time.time()

    if start not in grid or end not in grid:
        logger.warning("A*: start or end node not in grid.")
        return SearchResult("astar", [], [], 0.0, 0, time.time() - t_start, False)

    end_node = grid[end]
    min_weight = min(n.weight for n in grid.values())


    open_set = []
    initial_h = _astar_heuristic_full(grid[start], end_node, min_weight)
    heapq.heappush(open_set, (0.0 + initial_h, start))
    g_scores: Dict[Tuple[int, int], float] = {start: 0.0}
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    closed_set = set()
    nodes_expanded = 0

    while open_set:
        f, current = heapq.heappop(open_set)
        if current in closed_set:
            continue

        closed_set.add(current)
        nodes_expanded += 1

        if current == end:
            path_keys = _reconstruct_path(came_from, end)

            path_coords = [(grid[k].center_x, grid[k].center_y) for k in path_keys]
            total_cost = g_scores[end]
            compute_time = time.time() - t_start

            logger.info(
                f"A*: path found, {nodes_expanded} nodes expanded, "
                f"cost={total_cost:.3f}, time={compute_time:.3f}s"
            )
            gc.collect()
            return SearchResult(
                "astar", path_keys, path_coords, total_cost, nodes_expanded, compute_time, True
            )

        for neighbor, edge_cost in adjacency.get(current, []):
            if neighbor in closed_set:
                continue

            tentative_g = g_scores[current] + edge_cost
            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                came_from[neighbor] = current
                f_new = tentative_g + _astar_heuristic_full(grid[neighbor], end_node, min_weight)
                heapq.heappush(open_set, (f_new, neighbor))

    logger.warning("A*: no path found from start to end.")
    gc.collect()
    return SearchResult("astar", [], [], 0.0, nodes_expanded, time.time() - t_start, False)


def dijkstra(
    grid: Dict[Tuple[int, int], GridNode],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    start: Tuple[int, int],
    end: Tuple[int, int],
    logger: logging.Logger
) -> SearchResult:
    """Dijkstra's algorithm — A* with h(n) = 0. Explores more nodes than A*
    but provides ground-truth verification of the globally optimal path cost.

    Args:
        grid: The node dictionary.
        adjacency: The adjacency dictionary.
        start: The start node coordinate.
        end: The end node coordinate.
        logger: The logger instance.

    Returns:
        A SearchResult containing the path and metadata.
    """
    t_start = time.time()

    if start not in grid or end not in grid:
        logger.warning("Dijkstra: start or end node not in grid.")
        return SearchResult("dijkstra", [], [], 0.0, 0, time.time() - t_start, False)

    open_set = []
    heapq.heappush(open_set, (0.0, start))
    g_scores: Dict[Tuple[int, int], float] = {start: 0.0}
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    closed_set = set()
    nodes_expanded = 0

    while open_set:
        g, current = heapq.heappop(open_set)
        if current in closed_set:
            continue

        closed_set.add(current)
        nodes_expanded += 1

        if current == end:
            path_keys = _reconstruct_path(came_from, end)

            path_coords = [(grid[k].center_x, grid[k].center_y) for k in path_keys]
            total_cost = g_scores[end]
            compute_time = time.time() - t_start

            logger.info(
                f"Dijkstra: path found, {nodes_expanded} nodes expanded, "
                f"cost={total_cost:.3f}, time={compute_time:.3f}s"
            )
            gc.collect()
            return SearchResult(
                "dijkstra", path_keys, path_coords, total_cost, nodes_expanded, compute_time, True
            )

        for neighbor, edge_cost in adjacency.get(current, []):
            if neighbor in closed_set:
                continue

            tentative_g = g_scores[current] + edge_cost
            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                came_from[neighbor] = current
                heapq.heappush(open_set, (tentative_g, neighbor))

    logger.warning("Dijkstra: no path found from start to end.")
    gc.collect()
    return SearchResult("dijkstra", [], [], 0.0, nodes_expanded, time.time() - t_start, False)


def bfs(
    grid: Dict[Tuple[int, int], GridNode],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    start: Tuple[int, int],
    end: Tuple[int, int],
    logger: logging.Logger
) -> SearchResult:
    """Breadth-first search — ignores edge weights entirely. Finds the path
    with the fewest hops. Used as a geometric baseline comparison against cost-weighted
    paths from A* and Dijkstra.

    Args:
        grid: The node dictionary.
        adjacency: The adjacency dictionary.
        start: The start node coordinate.
        end: The end node coordinate.
        logger: The logger instance.

    Returns:
        A SearchResult containing the path and metadata.
    """
    t_start = time.time()

    if start not in grid or end not in grid:
        logger.warning("BFS: start or end node not in grid.")
        return SearchResult("bfs", [], [], 0.0, 0, time.time() - t_start, False)

    queue = deque([(start, [start])])
    visited = {start}
    nodes_expanded = 0

    while queue:
        current, path = queue.popleft()
        nodes_expanded += 1

        if current == end:
            path_coords = [(grid[k].center_x, grid[k].center_y) for k in path]
            compute_time = time.time() - t_start

            logger.info(
                f"BFS: path found, {nodes_expanded} nodes expanded, "
                f"hops={len(path)}, time={compute_time:.3f}s"
            )
            gc.collect()
            return SearchResult(
                "bfs", path, path_coords, 0.0, nodes_expanded, compute_time, True
            )

        for neighbor, _ in adjacency.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    logger.warning("BFS: no path found.")
    gc.collect()
    return SearchResult("bfs", [], [], 0.0, nodes_expanded, time.time() - t_start, False)


def compute_path_cost_weighted(
    path_keys: List[Tuple[int, int]],
    grid: Dict[Tuple[int, int], GridNode],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]
) -> float:
    """Compute the true weighted cost of any path through the grid.

    Used to give BFS a comparable cost even though BFS ignores weights during search.

    Args:
        path_keys: The list of grid keys forming the path.
        grid: The node dictionary.
        adjacency: The adjacency dictionary.

    Returns:
        The total path cost.
    """
    total = 0.0
    for i in range(len(path_keys) - 1):
        u = path_keys[i]
        v = path_keys[i + 1]

        edge_cost = None
        for neighbor, cost in adjacency.get(u, []):
            if neighbor == v:
                edge_cost = cost
                break

        if edge_cost is not None:
            total += edge_cost
        else:
            logging.getLogger(__name__).warning(f"Edge {u} -> {v} not found in adjacency.")

    return total


def compute_deviation_per_corner(
    astar_result: SearchResult,
    driver_path_coords: List[Tuple[float, float]],
    scale: float,
    n_corners: int = 15
) -> List[dict]:
    """Compute per-corner deviation between the A* optimal path and the
    driver's actual GPS path. Divides both paths into N equal arc-length segments
    and computes mean nearest-point distance for each segment.

    Args:
        astar_result: The SearchResult from A*.
        driver_path_coords: The actual path coordinates of the driver.
        scale: The coordinate scale factor.
        n_corners: The number of corners to divide into.

    Returns:
        A list of dictionaries containing corner name and deviation in meters.
    """
    if not astar_result.found or not astar_result.path_coords or not driver_path_coords:
        return [{"corner": f"T{i+1}", "deviation_m": None} for i in range(n_corners)]

    astar_arr = np.array(astar_result.path_coords)
    driver_arr = np.array(driver_path_coords)

    # Interpolate driver path to much higher resolution to get accurate distance
    from scipy.interpolate import interp1d
    dist = np.zeros(len(driver_arr))
    dist[1:] = np.linalg.norm(driver_arr[1:] - driver_arr[:-1], axis=1)
    cum_dist = np.cumsum(dist)
    if cum_dist[-1] > 0:
        f_x = interp1d(cum_dist, driver_arr[:, 0], kind='linear')
        f_y = interp1d(cum_dist, driver_arr[:, 1], kind='linear')
        fine_dist = np.linspace(0, cum_dist[-1], 1000)
        driver_arr = np.column_stack((f_x(fine_dist), f_y(fine_dist)))

    deviations = []
    segment_size = max(1, len(astar_arr) // n_corners)

    for s in range(n_corners):
        segment_start = s * segment_size
        segment_end = min((s + 1) * segment_size, len(astar_arr))
        segment_pts = astar_arr[segment_start:segment_end]

        if len(segment_pts) == 0:
            deviations.append({"corner": f"T{s+1}", "deviation_m": None})
            continue

        min_dists = []
        for pt in segment_pts:
            dists = np.linalg.norm(driver_arr - pt, axis=1)
            min_dists.append(dists.min())

        mean_deviation_units = np.mean(min_dists)
        deviation_m = mean_deviation_units * scale

        deviations.append({"corner": f"T{s+1}", "deviation_m": round(float(deviation_m), 3)})

    return deviations


def run_full_lap(search_func, nodes: list, grid: dict, adjacency: dict, logger) -> SearchResult:
    """Run search function across multiple nodes to form a complete lap."""
    if len(nodes) == 2:
        return search_func(grid, adjacency, nodes[0], nodes[1], logger)

    results = []
    for i in range(len(nodes) - 1):
        res = search_func(grid, adjacency, nodes[i], nodes[i+1], logger)
        if not res.found:
            return res # Failed segment
        results.append(res)

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
