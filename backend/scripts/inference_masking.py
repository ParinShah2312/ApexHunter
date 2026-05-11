"""YOLO mask processing for the ApexHunter CV inference pipeline.
Handles mask resizing, contour extraction, turn direction detection,
and HUD layer building.
"""

from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from inference_geometry import extract_inner_edge, get_closest_distance


def process_masks(
    results: Any,
    frame: np.ndarray,
    left_wheel: Tuple[int, int],
    right_wheel: Tuple[int, int],
    center_x: int,
    expected_turn: str = "-",
    speed_cat: str = "Medium",
) -> Tuple[np.ndarray, float, Optional[Tuple[int, int]], Optional[Tuple[int, int]], bool, str]:
    """Process all YOLO segmentation masks for a single frame.

    Builds the HUD colour overlay, computes the closest curb distance,
    and determines the turn direction.

    Args:
        results: YOLO prediction result object for one frame.
        frame: The original BGR frame as a numpy array.
        left_wheel: (x, y) position of the left wheel reference point.
        right_wheel: (x, y) position of the right wheel reference point.
        center_x: Horizontal centre of the frame in pixels.
        expected_turn: Hardcoded expected turn direction ('Left', 'Right', or '-').
        speed_cat: Speed classification for the current turn.

    Returns:
        Tuple of (hud_layer, distance, closest_point, detect_wheel,
        has_curb, turn_direction).
    """
    height, width = frame.shape[:2]
    distance = float('inf')
    closest_point: Optional[Tuple[int, int]] = None
    detect_wheel: Optional[Tuple[int, int]] = None
    has_curb = False
    turn_direction = "-"

    hud_layer = np.zeros_like(frame, dtype=np.uint8)

    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for i, mask in enumerate(masks):
            cls_id = int(classes[i])

            mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_resized * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            color_mask = np.zeros_like(frame)

            if cls_id == 0:  # Curb
                has_curb = True
                color_mask[mask_binary == 255] = [0, 0, 255]
                cv2.addWeighted(hud_layer, 1.0, color_mask, 1.0, 0, hud_layer)

                if contours:
                    valid_contours = []
                    for c in contours:
                        # Re-apply spatial filtering for tight/medium corners to ignore exit curbs.
                        # Disable for "Fast" corners to prevent tracking loss in aggressive S-curves.
                        if speed_cat != "Fast":
                            M_temp = cv2.moments(c)
                            if M_temp['m00'] != 0:
                                cx_temp = int(M_temp['m10'] / M_temp['m00'])
                                if expected_turn == "Right" and cx_temp < center_x - 100:
                                    continue
                                if expected_turn == "Left" and cx_temp > center_x + 100:
                                    continue
                        valid_contours.append(c)
                        
                    if valid_contours:
                        largest_contour = max(valid_contours, key=cv2.contourArea)
                        M = cv2.moments(largest_contour)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
        
                            # Use hardcoded expected_turn if available, otherwise fallback to dynamic
                            is_right = (cx > center_x) if expected_turn == "-" else (expected_turn == "Right")
                            
                            # Apply Track Limit Extraction (Isolate the asphalt-side inner edge)
                            turn_direction_local = "Right" if is_right else "Left"
                            active_turn_dir = turn_direction_local if expected_turn != "-" else "-"
                            
                            inner_edge_contour = extract_inner_edge(largest_contour, turn_direction_local)
                            
                            if is_right:
                                turn_direction = active_turn_dir
                                dist, pt = get_closest_distance(right_wheel, inner_edge_contour)
                                if dist < distance:
                                    distance = dist
                                    closest_point = pt
                                    detect_wheel = right_wheel
                            else:
                                turn_direction = active_turn_dir
                                dist, pt = get_closest_distance(left_wheel, inner_edge_contour)
                                if dist < distance:
                                    distance = dist
                                    closest_point = pt
                                    detect_wheel = left_wheel

            elif cls_id == 1:  # Road
                color_mask[mask_binary == 255] = [0, 255, 0]
                cv2.addWeighted(hud_layer, 1.0, color_mask, 1.0, 0, hud_layer)

    return hud_layer, distance, closest_point, detect_wheel, has_curb, turn_direction
