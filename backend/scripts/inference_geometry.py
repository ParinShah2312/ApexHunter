"""Geometric calculations for the ApexHunter CV inference pipeline.
Provides distance computation between wheel reference points and detected curb contours,
apex status classification, and wheel position calculation.
"""

from typing import Any, Optional, Tuple

import numpy as np

from utils import CONFIG

# ── Configuration ─────────────────────────────────────────────────────────────
_inf_config = CONFIG.get("inference", {})

HIT_THRESHOLD: int = _inf_config.get("hit_threshold", 130)
NEAR_THRESHOLD: int = _inf_config.get("near_threshold", 250)
LEFT_WHEEL_X_PCT: float = _inf_config.get("left_wheel_x_pct", 0.18)
RIGHT_WHEEL_X_PCT: float = _inf_config.get("right_wheel_x_pct", 0.85)
WHEEL_Y_PCT: float = _inf_config.get("wheel_y_pct", 0.60)


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


def classify_apex_status(
    distance: float, has_curb: bool
) -> Tuple[str, Tuple[int, int, int]]:
    """Classify the apex status based on distance and curb presence.

    Args:
        distance: Euclidean distance in pixels from the wheel to the curb.
        has_curb: Whether a curb was detected in this frame.

    Returns:
        Tuple of (status_string, bgr_color_tuple).
    """
    if not has_curb or distance == float('inf'):
        return "Straight", (200, 200, 200)
    elif distance < HIT_THRESHOLD:
        return "Hitting Apex", (0, 255, 0)
    elif distance < NEAR_THRESHOLD:
        return "Near Apex", (0, 255, 255)
    else:
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
