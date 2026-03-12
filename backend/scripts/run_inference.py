"""
================================================================================
  ApexHunter - F1 Computer Vision Project
  Script: run_inference.py
--------------------------------------------------------------------------------
  Purpose : Runs the trained YOLOv11-Seg model on an onboard video, calculates
            the Apex Deviation Metric, and outputs a CSV and an AR/HUD video.
            
            Phases 4 & 5 are combined here:
            - Extracts segmentation mask
            - Calculates Car Nose distance to Road Edge (Deviation Metric)
            - Blends green/red overlays dynamically onto the video
            - Embeds the apex status directly into the output video
================================================================================
"""

import os
import sys
import cv2
import csv
import torch
import numpy as np
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics is not installed. Run: pip install ultralytics")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join("models", "best.pt")
INPUT_VIDEO = os.path.join("data_lake", "edited_videos", "2024", "01_bahrain_ver_pole - Trim.mp4")
OUTPUT_DIR = os.path.join("data_lake", "processed_video")
OUTPUT_CSV_DIR = os.path.join("data_lake", "processed_csv")

# Thresholds (in pixels) for Apex Status categorization
HIT_THRESHOLD = 130    # Distance < 130px = Hitting apex
NEAR_THRESHOLD = 250   # Distance < 250px = Near apex

# Screen coordinates for the front wheels (approximate based on F1 T-cam)
# These may need slight tuning depending on the exact video resolution
LEFT_WHEEL_X = 250
RIGHT_WHEEL_X = 1030
WHEEL_Y = 600

# Transparency for HUD blending
ALPHA = 0.5

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)


# ── Core Logic ────────────────────────────────────────────────────────────────

def get_closest_distance(ref_point, contour):
    """
    Calculates the shortest Euclidean distance from a reference point (wheel)
    to any point on the provided contour. Also returns the closest point's coords.
    """
    if contour is None or len(contour) == 0:
        return float('inf'), None
    
    # squeeze removes redundant dimensions from OpenCV contours
    pts = np.squeeze(contour, axis=1) if contour.ndim == 3 else contour
    
    # Calculate Euclidean distance to all points
    distances = np.linalg.norm(pts - np.array(ref_point), axis=1)
    min_dist = np.min(distances)
    
    # Find the exact point that is closest
    closest_pt = pts[np.argmin(distances)]
    
    return min_dist, tuple(closest_pt)


def process_video():
    print("======================================================")
    print("   ApexHunter - Phase 4 & 5: Inference Pipeline")
    print("======================================================\n")

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Trained model not found at: {MODEL_PATH}")
        sys.exit(1)
        
    if not os.path.exists(INPUT_VIDEO):
        print(f"[ERROR] Input video not found at: {INPUT_VIDEO}")
        sys.exit(1)

    print(f"[INFO] Loading model: {MODEL_PATH}")
    # Run on GPU if available (User probably has CPU, but model runs fine on CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH).to(device)
    
    # Extract file names
    basename = os.path.basename(INPUT_VIDEO)
    name_no_ext = os.path.splitext(basename)[0]
    out_video_path = os.path.join(OUTPUT_DIR, f"{name_no_ext}_HUD.mp4")
    out_csv_path = os.path.join(OUTPUT_CSV_DIR, f"{name_no_ext}_metrics.csv")
    
    # Open Video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    
    # Setup CSV Writer
    csv_file = open(out_csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame_number', 'timestamp_sec', 'distance_px', 'status', 'has_curb'])
    
    # The reference points for the left and right wheels
    # Adjusted dynamically based on video resolution (assuming 1280x720 standard)
    left_wheel = (int(width * 0.20), int(height * 0.85))
    right_wheel = (int(width * 0.80), int(height * 0.85))
    center_x = width // 2
    
    print(f"[INFO] Starting inference on {total_frames} frames...")
    
    for frame_idx in tqdm(range(total_frames), desc="Processing", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = frame_idx / fps
        
        # 1. Run inference (conf=0.25 to catch kerbs)
        results = model.predict(frame, conf=0.25, verbose=False)[0]
        
        distance = float('inf')
        status = "Straight"
        color = (200, 200, 200) # Grey text for straight
        closest_point = None
        detect_wheel = None
        has_curb = False
        turn_direction = "Straight"
        
        hud_layer = np.zeros_like(frame, dtype=np.uint8)
        
        # 2. Extract Masks
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()  # Extract masks
            classes = results.boxes.cls.cpu().numpy() # Extract classes for the masks
            
            for i, mask in enumerate(masks):
                cls_id = int(classes[i])
                
                # Resize mask back to original frame size
                mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_binary = (mask_resized * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                color_mask = np.zeros_like(frame)
                
                if cls_id == 0:  # Class 0 = Curb
                    has_curb = True
                    color_mask[mask_binary == 255] = [0, 0, 255] # Red BGR
                    cv2.addWeighted(hud_layer, 1.0, color_mask, 1.0, 0, hud_layer)
                    
                    if contours:
                        # Find the center/centroid of the curb to determine if it's a Left or Right turn
                        # We use the largest contour for direction
                        largest_contour = max(contours, key=cv2.contourArea)
                        M = cv2.moments(largest_contour)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            
                            # If curb is on the right side of the screen -> Right turn
                            if cx > center_x:
                                turn_direction = "Right"
                                dist, pt = get_closest_distance(right_wheel, largest_contour)
                                if dist < distance:
                                    distance = dist
                                    closest_point = pt
                                    detect_wheel = right_wheel
                                    
                            # If curb is on the left side of the screen -> Left turn
                            else:
                                turn_direction = "Left"
                                dist, pt = get_closest_distance(left_wheel, largest_contour)
                                if dist < distance:
                                    distance = dist
                                    closest_point = pt
                                    detect_wheel = left_wheel
                                    
                elif cls_id == 1:  # Class 1 = Road
                    color_mask[mask_binary == 255] = [0, 255, 0] # Green BGR
                    cv2.addWeighted(hud_layer, 1.0, color_mask, 1.0, 0, hud_layer)
                    
        # 3. Categorize Status
        if not has_curb or distance == float('inf'):
            status = "Straight"
            distance_str = "N/A"
            color = (200, 200, 200)
        else:
            distance_str = f"{int(distance)}px"
            
            if distance < HIT_THRESHOLD:
                status = "Hitting Apex"
                color = (0, 255, 0)
            elif distance < NEAR_THRESHOLD:
                status = "Near Apex"
                color = (0, 255, 255)
            else:
                status = "Missing Apex"
                color = (0, 0, 255)
        
        # 4. Write to CSV
        csv_writer.writerow([frame_idx, round(timestamp, 2), distance_str, status, has_curb])
        
        # 5. Blend HUD and Original Frame
        blended_frame = cv2.addWeighted(frame, 1.0, hud_layer, ALPHA, 0)
        
        # 6. Draw HUD Elements
        # Draw the target wheels
        cv2.circle(blended_frame, left_wheel, 5, (255, 0, 0), -1) 
        cv2.circle(blended_frame, right_wheel, 5, (255, 0, 0), -1) 
        
        # Draw line to closest point on the active wheel
        if closest_point is not None and detect_wheel is not None and has_curb:
            cv2.line(blended_frame, detect_wheel, closest_point, (255, 255, 0), 2)
            cv2.circle(blended_frame, closest_point, 5, (0, 255, 255), -1)
        
        # Draw Status Box (Top Left)
        cv2.rectangle(blended_frame, (10, 10), (450, 150), (0, 0, 0), -1)
        cv2.putText(blended_frame, "ApexHunter CV Pipeline", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(blended_frame, f"Turn:      {turn_direction}", (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(blended_frame, f"Status:    {status}", (20, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(blended_frame, f"Deviation: {distance_str}", (20, 135), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(blended_frame)

    cap.release()
    out.release()
    csv_file.close()
    
    print(f"\n======================================================")
    print(f"  [DONE] Processing Complete!")
    print(f"         Video saved: {os.path.abspath(out_video_path)}")
    print(f"         CSV saved:   {os.path.abspath(out_csv_path)}")
    print(f"======================================================")

if __name__ == "__main__":
    process_video()
