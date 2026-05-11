"""Geometric calculations for the ApexHunter CV inference pipeline.
Provides distance computation between wheel reference points and detected curb contours,
apex status classification, and wheel position calculation.
"""

from typing import Any, Optional, Tuple

import numpy as np

from utils import CONFIG

# ── Configuration ─────────────────────────────────────────────────────────────
_inf_config = CONFIG.get("inference", {})

HIT_THRESHOLD: int = _inf_config.get("hit_threshold", 300)
NEAR_THRESHOLD: int = _inf_config.get("near_threshold", 550)
LEFT_WHEEL_X_PCT: float = _inf_config.get("left_wheel_x_pct", 0.18)
RIGHT_WHEEL_X_PCT: float = _inf_config.get("right_wheel_x_pct", 0.85)
WHEEL_Y_PCT: float = _inf_config.get("wheel_y_pct", 0.60)


STRAIGHT_THRESHOLD: int = _inf_config.get("straight_threshold", 400)

def get_closest_distance(
    ref_point: Tuple[int, int], contour: Any
) -> Tuple[float, Optional[Tuple[int, int]]]:
    """Calculate the shortest Euclidean distance from a reference point to a contour.

    Args:
        ref_point: (x, y) coordinate of the wheel reference point.
        contour: OpenCV contour array, or None.

    Returns:
        Tuple of (min_distance, closest_point). Returns (inf, None) if contour
        is None or empty.
    """
    if contour is None or len(contour) == 0:
        return float('inf'), None

    pts = np.squeeze(contour, axis=1) if contour.ndim == 3 else contour
    distances = np.linalg.norm(pts - np.array(ref_point), axis=1)
    min_dist = np.min(distances)
    closest_pt = pts[np.argmin(distances)]

    return min_dist, tuple(closest_pt)

def extract_inner_edge(contour: Any, turn_direction: str) -> Any:
    """Filter a curb contour to only include the track-side inner edge.

    If it's a Right turn (curb on the right), the inner edge is the leftmost boundary (min X).
    If it's a Left turn (curb on the left), the inner edge is the rightmost boundary (max X).

    Args:
        contour: The raw OpenCV contour.
        turn_direction: 'Left', 'Right', or '-'.

    Returns:
        Filtered contour containing only the inner edge points.
    """
    if contour is None or len(contour) == 0 or turn_direction == "-":
        return contour

    pts = np.squeeze(contour, axis=1) if contour.ndim == 3 else contour
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)

    # Group points by Y coordinate
    y_coords = np.unique(pts[:, 1])
    inner_edge_pts = []

    for y in y_coords:
        x_vals = pts[pts[:, 1] == y][:, 0]
        if turn_direction == "Right":
            # Curb is on the right, track is on the left -> get min X
            inner_edge_pts.append([np.min(x_vals), y])
        elif turn_direction == "Left":
            # Curb is on the left, track is on the right -> get max X
            inner_edge_pts.append([np.max(x_vals), y])

    filtered_contour = np.array(inner_edge_pts, dtype=np.int32).reshape((-1, 1, 2))
    return filtered_contour


# Exact video timestamp windows for Bahrain 2023 corners (provided by user)
# Format: (start_time, end_time, direction, speed_category)
BAHRAIN_TURN_WINDOWS = [
    (13.0, 16.0, "Right", "Slow"),    # Turn 1: Tight hairpin after straight
    (17.0, 19.0, "Left", "Medium"),   # Turn 2
    (19.5, 20.5, "Right", "Medium"),  # Turn 3
    (27.0, 29.0, "Right", "Medium"),  # Turn 4
    (33.0, 34.0, "Left", "Fast"),     # Turn 5: Fast sweeper
    (34.0, 36.0, "Right", "Fast"),    # Turn 6: Fast sweeper
    (36.0, 38.0, "Left", "Fast"),     # Turn 7: Fast sweeper
    (41.0, 43.0, "Right", "Slow"),    # Turn 8: Hairpin
    (48.0, 50.0, "Left", "Medium"),   # Turn 9
    (51.0, 53.0, "Left", "Slow"),     # Turn 10: Tight, heavy braking
    (62.0, 67.0, "Left", "Medium"),   # Turn 11
    (68.0, 71.0, "Right", "Fast"),    # Turn 12: Fast flat-out
    (72.0, 77.0, "Right", "Medium"),  # Turn 13
    (85.0, 87.0, "Right", "Slow"),    # Turn 14: Tight entry
    (88.0, 90.0, "Right", "Medium")   # Turn 15
]

def get_turn_context(timestamp_sec: float) -> Tuple[str, str]:
    """Return the hardcoded turn direction and speed category for the current timestamp.

    Returns:
        Tuple of (turn_direction, speed_category). Defaults to ('-', 'Medium') on straights.
    """
    for start_t, end_t, direction, speed_cat in BAHRAIN_TURN_WINDOWS:
        if start_t <= timestamp_sec <= end_t:
            return direction, speed_cat
    return "-", "Medium"

def classify_apex_status(
    distance_px: float, has_curb: bool, timestamp_sec: float
) -> Tuple[str, Tuple[int, int, int]]:
    """Determine the apex hitting status based on distance and dynamic speed windows.

    Args:
        distance_px: Pixel distance from front wheel to the optimal racing line or curb.
        has_curb: True if an apex curb mask was detected in the frame.
        timestamp_sec: Current timestamp of the video frame in seconds.

    Returns:
        Tuple of (Status String, BGR Color Tuple).
    """
    turn_dir, speed_cat = get_turn_context(timestamp_sec)
    in_turn = (turn_dir != "-")

    if not in_turn:
        return "Straight", (200, 200, 200)

    # If the distance is physically larger than half the screen, we have latched
    # onto the outer exit curb. Treat the apex curb as missing.
    if distance_px > 900:
        distance_px = float('inf')

    if not has_curb or distance_px == float('inf'):
        # If the valid apex curb is physically off-screen, we cannot definitively judge distance.
        # We reserve 'Missing Apex' for when the curb is visible but the car is far away.
        return "Tracking", (200, 200, 200)

    # Scale distance thresholds dynamically based on corner speed
    multiplier = 1.0
    if speed_cat == "Slow":
        multiplier = 0.75  # Stricter: car is moving slower, must hit tighter
    elif speed_cat == "Fast":
        multiplier = 1.35  # Forgiving: aerodynamic wide lines are acceptable

    dynamic_hit_threshold = int(HIT_THRESHOLD * multiplier)
    dynamic_near_threshold = int(NEAR_THRESHOLD * multiplier)

    if distance_px < dynamic_hit_threshold:
        return "Hitting Apex", (0, 255, 0)
    elif distance_px < dynamic_near_threshold:
        return "Near Apex", (0, 255, 255)
    else:
        if speed_cat == "Fast":
            return "Wide Line", (0, 165, 255)  # Orange for fast sweeping lines
        return "Missing Apex", (0, 0, 255)


def compute_wheel_positions(
    width: int, height: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Compute the left and right front wheel reference positions.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Tuple of (left_wheel, right_wheel) as (x, y) integer tuples.
    """
    left_wheel = (int(width * LEFT_WHEEL_X_PCT), int(height * WHEEL_Y_PCT))
    right_wheel = (int(width * RIGHT_WHEEL_X_PCT), int(height * WHEEL_Y_PCT))
    return left_wheel, right_wheel
