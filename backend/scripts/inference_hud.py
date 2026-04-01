"""HUD drawing functions for the ApexHunter CV inference pipeline.
All cv2 drawing calls that render the augmented reality overlay onto each frame.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


def draw_hud(
    frame: np.ndarray,
    hud_layer: np.ndarray,
    left_wheel: Tuple[int, int],
    right_wheel: Tuple[int, int],
    closest_point: Optional[Tuple[int, int]],
    detect_wheel: Optional[Tuple[int, int]],
    has_curb: bool,
    status: str,
    color: Tuple[int, int, int],
    distance_str: str,
    turn_direction: str,
    alpha: float,
) -> np.ndarray:
    """Blend the HUD overlay and draw all AR elements onto a frame.

    Args:
        frame: The original BGR frame.
        hud_layer: The colour overlay layer for segmentation masks.
        left_wheel: (x, y) of the left wheel reference point.
        right_wheel: (x, y) of the right wheel reference point.
        closest_point: (x, y) of the closest curb point, or None.
        detect_wheel: (x, y) of the active wheel used for distance, or None.
        has_curb: Whether a curb was detected in this frame.
        status: Status string (e.g. "Hitting Apex").
        color: BGR colour tuple for the status text.
        distance_str: Formatted distance string (e.g. "123px" or "N/A").
        turn_direction: "Left", "Right", or "Straight".
        alpha: Transparency for the HUD blend.

    Returns:
        The final blended frame ready for writing.
    """
    # Blend HUD and original frame
    blended_frame = cv2.addWeighted(frame, 1.0, hud_layer, alpha, 0)

    # Draw wheel reference circles
    cv2.circle(blended_frame, left_wheel, 5, (255, 0, 0), -1)
    cv2.circle(blended_frame, right_wheel, 5, (255, 0, 0), -1)

    # Draw line to closest point on the active wheel
    if closest_point is not None and detect_wheel is not None and has_curb:
        cv2.line(blended_frame, detect_wheel, closest_point, (255, 255, 0), 2)
        cv2.circle(blended_frame, closest_point, 5, (0, 255, 255), -1)

    # Draw status box (top left)
    cv2.rectangle(blended_frame, (10, 10), (450, 150), (0, 0, 0), -1)
    cv2.putText(blended_frame, "ApexHunter CV Pipeline", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(blended_frame, f"Turn:      {turn_direction}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(blended_frame, f"Status:    {status}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(blended_frame, f"Deviation: {distance_str}", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return blended_frame
