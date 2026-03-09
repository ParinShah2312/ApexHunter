"""
================================================================================
  ApexHunter - F1 Computer Vision Project
  Script: select_training_frames.py
--------------------------------------------------------------------------------
  Purpose : Selects ~500 diverse, representative frames from the full 21,282
            extracted frames for annotation in Roboflow.
            
            Strategy:
            - Picks frames evenly across ALL 46 tracks (diversity of scenery)
            - Within each track, picks frames spread across the full lap
              (straights, corners, braking zones, pit entries, etc.)
            - Copies them into a single flat folder for easy bulk upload.
================================================================================
"""

import os
import sys
import glob
import shutil
import random

# ── Configuration ─────────────────────────────────────────────────────────────

FRAMES_ROOT = os.path.join("data_lake", "cv_frames")
OUTPUT_DIR = os.path.join("data_lake", "cv_dataset", "upload_to_roboflow")
FRAMES_PER_VIDEO = 11  # ~11 frames × 46 videos ≈ 506 frames total

random.seed(42)  # Reproducible selection

# ── Main ──────────────────────────────────────────────────────────────────────

def select_frames():
    print("======================================================")
    print("   ApexHunter - Training Frame Selector")
    print("======================================================\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_selected = 0
    
    for year in ["2023", "2024"]:
        year_dir = os.path.join(FRAMES_ROOT, year)
        if not os.path.isdir(year_dir):
            print(f"[WARN] {year_dir} not found. Skipping.")
            continue
        
        video_dirs = sorted([
            d for d in os.listdir(year_dir) 
            if os.path.isdir(os.path.join(year_dir, d))
        ])
        
        print(f"[INFO] Year {year}: {len(video_dirs)} track folders found.\n")
        
        for video_name in video_dirs:
            video_frame_dir = os.path.join(year_dir, video_name)
            all_frames = sorted(glob.glob(os.path.join(video_frame_dir, "*.jpg")))
            
            if len(all_frames) == 0:
                print(f"  [SKIP] {video_name}: No frames found.")
                continue
            
            # Strategy: Pick frames evenly spread across the entire video
            # This ensures we get straights, corners, braking zones, tunnels, etc.
            step = max(1, len(all_frames) // FRAMES_PER_VIDEO)
            selected = all_frames[::step][:FRAMES_PER_VIDEO]
            
            print(f"  {video_name}: {len(all_frames)} total → selected {len(selected)} frames")
            
            for frame_path in selected:
                frame_basename = os.path.basename(frame_path)
                # Prefix with year and track to make filenames unique
                new_name = f"{year}_{video_name}_{frame_basename}"
                dest = os.path.join(OUTPUT_DIR, new_name)
                shutil.copy2(frame_path, dest)
                total_selected += 1
    
    print(f"\n======================================================")
    print(f"  [DONE] Selected {total_selected} frames for annotation.")
    print(f"         Upload folder: {os.path.abspath(OUTPUT_DIR)}")
    print(f"======================================================")
    print(f"\n  Next step: Upload this entire folder to Roboflow!")

if __name__ == "__main__":
    select_frames()
