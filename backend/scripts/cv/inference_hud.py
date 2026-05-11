"""HUD drawing functions for the ApexHunter CV inference pipeline.
All cv2 drawing calls that render the augmented reality overlay onto each frame.
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np


class HUDStateTracker:
    """Tracks state across frames to apply Temporal Smoothing (EMA) to HUD elements.

    This eliminates frame-to-frame jitter when plotting AR lines or text metrics.
    """
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.smoothed_cp: Optional[Tuple[float, float]] = None
        self.smoothed_dist: Optional[float] = None

    def update(
        self,
        cp: Optional[Tuple[int, int]],
        dist: float,
        has_curb: bool
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        """Update tracker with new frame data and return smoothed values."""
        if not has_curb or cp is None or dist == float('inf'):
            self.smoothed_cp = None
            self.smoothed_dist = None
            return None, float('inf')

        # Initialize if first valid frame or after losing track
        if self.smoothed_cp is None:
            self.smoothed_cp = (float(cp[0]), float(cp[1]))
            self.smoothed_dist = dist
        else:
            # Apply EMA
            self.smoothed_cp = (
                self.smoothed_cp[0] * (1 - self.alpha) + cp[0] * self.alpha,
                self.smoothed_cp[1] * (1 - self.alpha) + cp[1] * self.alpha
            )
            self.smoothed_dist = self.smoothed_dist * (1 - self.alpha) + dist * self.alpha

        final_cp = (int(self.smoothed_cp[0]), int(self.smoothed_cp[1]))
        return final_cp, self.smoothed_dist


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
    frame_idx: int = 0,
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
        distance_str: Formatted distance string (e.g. "123cm" or "N/A").
        turn_direction: "Left", "Right", or "Straight".
        alpha: Transparency for the HUD blend.

    Returns:
        The final blended frame ready for writing.
    """
    # Calculate phase-based pulse for 'Hitting Apex'
    pulse = 1.0
    if status == "Hitting Apex":
        # Sine wave oscillating between 0.4 and 1.0 for a breathing effect
        pulse = 0.7 + 0.3 * math.sin(frame_idx * 0.2)

    pulse_alpha = alpha * pulse

    # Blend HUD and original frame with pulsing alpha
    blended_frame = cv2.addWeighted(frame, 1.0, hud_layer, pulse_alpha, 0)

    # Draw wheel reference circles
    cv2.circle(blended_frame, left_wheel, 5, (255, 0, 0), -1)
    cv2.circle(blended_frame, right_wheel, 5, (255, 0, 0), -1)

    # Phase 3: Gradient AR Tether
    if closest_point is not None and detect_wheel is not None and has_curb and turn_direction != "-":
        p1 = np.array(detect_wheel, dtype=float)
        p2 = np.array(closest_point, dtype=float)
        dist = np.linalg.norm(p2 - p1)

        # Draw dot-matrix gradient line fading from wheel (Yellow) to curb (Neon Green)
        if dist > 0:
            steps = max(2, int(dist / 8))  # Space out the dots
            for i in range(steps):
                t = i / steps
                pt = tuple((p1 * (1 - t) + p2 * t).astype(int))
                b_color = 0
                g_color = int(255)
                r_color = int(255 * (1 - t))  # Yellow fades to green
                cv2.circle(blended_frame, pt, 2, (b_color, g_color, r_color), -1)

        # Draw pulsing target ring on the curb
        ring_radius = int(6 + 4 * pulse)
        cv2.circle(blended_frame, closest_point, ring_radius, (0, 255, 100), max(1, int(2 * pulse)))

    # Phase 2: High-Fidelity Broadcast UI (Glassmorphism + PIL Typography)
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        pass  # Fallback gracefully if PIL is not installed, though it's standard

    # 1. Glassmorphism Base
    x, y, w, h = 20, 20, 420, 120  # Shrunk height since Deviation is moved
    roi = blended_frame[y:y+h, x:x+w]

    blurred_roi = cv2.GaussianBlur(roi, (31, 31), 0)
    dark_overlay = np.zeros_like(blurred_roi)
    glass_roi = cv2.addWeighted(blurred_roi, 0.4, dark_overlay, 0.6, 0)

    # Create mask for rounded corners
    mask = np.zeros((h, w), dtype=np.uint8)
    radius = 16
    cv2.rectangle(mask, (radius, 0), (w-radius, h), 255, -1)
    cv2.rectangle(mask, (0, radius), (w, h-radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (w-radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, h-radius), radius, 255, -1)
    cv2.circle(mask, (w-radius, h-radius), radius, 255, -1)

    # Apply mask
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    roi_blended = (glass_roi * mask_3d + roi * (1 - mask_3d)).astype(np.uint8)
    blended_frame[y:y+h, x:x+w] = roi_blended

    # Phase 5: Predictive Trajectory Mapping
    # Project a Bezier curve representing the car's intended path based on turn direction
    height_px, width_px = frame.shape[:2]
    cx_bottom = width_px // 2
    cy_bottom = height_px

    offset_x = 0
    if turn_direction == "Right":
        offset_x = 350
    elif turn_direction == "Left":
        offset_x = -350

    p0 = np.array([cx_bottom, cy_bottom])
    p1 = np.array([cx_bottom, int(height_px * 0.85)])
    p2 = np.array([cx_bottom + int(offset_x * 0.4), int(height_px * 0.65)])
    p3 = np.array([cx_bottom + offset_x, int(height_px * 0.55)])

    curve_pts = []
    for t in np.linspace(0, 1, 25):
        pt = (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3
        curve_pts.append([int(pt[0]), int(pt[1])])

    curve_pts = np.array(curve_pts, np.int32).reshape((-1, 1, 2))

    # Draw trajectory on a transparent layer for blending
    traj_layer = np.zeros_like(blended_frame)
    cv2.polylines(traj_layer, [curve_pts], isClosed=False, color=(255, 100, 0), thickness=6, lineType=cv2.LINE_AA)

    # Add distance markers along the trajectory curve
    for i in range(5, 25, 5):
        pt = curve_pts[i][0]
        cv2.circle(traj_layer, tuple(pt), 4, (255, 200, 0), -1)

    blended_frame = cv2.addWeighted(blended_frame, 1.0, traj_layer, 0.4, 0)

    # Phase 2 & 4: Typography and Spatial Pinned Telemetry using PIL
    # Convert to RGBA to allow transparent drawings like the pill background
    pil_img = Image.fromarray(cv2.cvtColor(blended_frame, cv2.COLOR_BGR2RGBA))
    draw = ImageDraw.Draw(pil_img, "RGBA")

    try:
        font_title = ImageFont.truetype("arialbd.ttf", 26)
        font_text = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()

    rgb_color = (color[2], color[1], color[0])

    draw.text((x + 25, y + 20), "ApexHunter CV Pipeline", font=font_title, fill=(255, 255, 255, 255))
    draw.text((x + 25, y + 60), f"Turn:      {turn_direction}", font=font_text, fill=(220, 220, 220, 255))
    draw.text((x + 25, y + 90), f"Status:    {status}", font=font_text, fill=(rgb_color[0], rgb_color[1], rgb_color[2], 255))

    # Phase 4: Spatial Pinned Telemetry
    if closest_point is not None and has_curb and distance_str != "N/A" and turn_direction != "-":
        height_px, width_px = frame.shape[:2]
        cx, cy = closest_point

        # Depth scaling: Objects lower on screen (larger Y) are physically closer
        depth_ratio = max(0.2, min(1.0, cy / height_px))
        font_size = int(14 + 30 * depth_ratio)

        try:
            spatial_font = ImageFont.truetype("arialbd.ttf", font_size)
        except Exception:
            spatial_font = ImageFont.load_default()

        # Widget positioning
        text = f"{distance_str}"
        # Approximate text dimensions
        left, top, right, bottom = draw.textbbox((0, 0), text, font=spatial_font)
        text_w = right - left
        text_h = bottom - top

        pill_x = cx + 20
        pill_y = cy - int(40 * depth_ratio)

        # Draw translucent background pill
        draw.rounded_rectangle(
            [pill_x - 10, pill_y - 5, pill_x + text_w + 10, pill_y + text_h + 10],
            radius=8,
            fill=(0, 0, 0, 180)
        )

        # Draw the depth-scaled distance text (Neon Green)
        draw.text((pill_x, pill_y), text, font=spatial_font, fill=(0, 255, 100, 255))

    blended_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)

    return blended_frame
