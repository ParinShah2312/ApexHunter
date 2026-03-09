"""
================================================================================
  ApexHunter - F1 Computer Vision Project
  Script: extract_frames.py
--------------------------------------------------------------------------------
  Purpose : Extracts frames at 5fps from all pole lap videos in the
            edited_videos directory for YOLO segmentation training.
================================================================================
"""

import os
import sys
import glob

try:
    import cv2
except ImportError:
    print("[ERROR] opencv-python is not installed.")
    print("        Run:  pip install opencv-python")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────

VIDEO_ROOT = os.path.join("data_lake", "edited_videos")
OUTPUT_ROOT = os.path.join("data_lake", "cv_frames")
TARGET_FPS = 5  # Extract 5 frames per second

# ── Main ──────────────────────────────────────────────────────────────────────

def extract_frames_from_video(video_path, output_dir):
    """Extract frames at TARGET_FPS from a single video."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Could not open: {video_path}")
        return 0
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps if video_fps > 0 else 0
    
    # Calculate frame interval: skip this many frames between captures
    frame_interval = int(video_fps / TARGET_FPS) if video_fps > 0 else 1
    
    print(f"  Video FPS: {video_fps:.1f} | Duration: {duration_sec:.1f}s | Interval: every {frame_interval} frames")
    
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            saved_count += 1
            filename = f"frame_{saved_count:05d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        frame_count += 1
    
    cap.release()
    return saved_count


def main():
    print("======================================================")
    print("   ApexHunter - Frame Extraction (5fps)")
    print("======================================================\n")
    
    total_extracted = 0
    
    for year in ["2023", "2024"]:
        year_video_dir = os.path.join(VIDEO_ROOT, year)
        
        if not os.path.isdir(year_video_dir):
            print(f"[WARN] Directory not found: {year_video_dir}. Skipping.")
            continue
        
        videos = sorted(glob.glob(os.path.join(year_video_dir, "*.mp4")))
        print(f"[INFO] Year {year}: Found {len(videos)} videos in {os.path.abspath(year_video_dir)}\n")
        
        for video_path in videos:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(OUTPUT_ROOT, year, video_name)
            
            # Skip if already extracted
            if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
                existing = len(os.listdir(output_dir))
                print(f"  [SKIP] {video_name} ({existing} frames already exist)")
                total_extracted += existing
                continue
            
            print(f"\n  >>> Extracting: {video_name}")
            count = extract_frames_from_video(video_path, output_dir)
            total_extracted += count
            print(f"      Saved {count} frames -> {os.path.abspath(output_dir)}")
    
    print("\n======================================================")
    print(f"  [DONE] Total frames extracted: {total_extracted}")
    print(f"         Output: {os.path.abspath(OUTPUT_ROOT)}")
    print("======================================================")


if __name__ == "__main__":
    main()
