"""
================================================================================
  ApexHunter - F1 Computer Vision Project
  Script: extract_frames.py
--------------------------------------------------------------------------------
  Purpose : Extracts frames at 5fps from all pole lap videos in the
            edited_videos directory for YOLO segmentation training.
================================================================================
"""

import sys
import os
import argparse
import concurrent.futures
from pathlib import Path
from utils import setup_logger, DATA_LAKE_DIR, CONFIG

logger = setup_logger(__name__)

try:
    import cv2
except ImportError:
    logger.error("opencv-python is not installed.")
    logger.error("Run:  pip install opencv-python")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────

VIDEO_ROOT = DATA_LAKE_DIR / "edited_videos"
OUTPUT_ROOT = DATA_LAKE_DIR / "cv_frames"
TARGET_FPS = CONFIG.get("cv_frames", {}).get("target_fps", 5)  # Extract 5 frames per second
SEASONS = CONFIG.get("seasons", ["2023", "2024"])

# ── Main ──────────────────────────────────────────────────────────────────────

def extract_frames_from_video(video_path: Path, output_dir: Path) -> int:
    """Extract frames at TARGET_FPS from a single video."""
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open: {video_path.name}")
        return 0
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps if video_fps > 0 else 0
    
    # Calculate frame interval: skip this many frames between captures
    frame_interval = int(video_fps / TARGET_FPS) if video_fps > 0 else 1
    
    logger.info(f"{video_path.name} | FPS: {video_fps:.1f} | Duration: {duration_sec:.1f}s | Interval: every {frame_interval} frames")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            saved_count += 1
            filename = f"frame_{saved_count:05d}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        frame_count += 1
    
    cap.release()
    return saved_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from videos for CV training.")
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Number of concurrent processes.")
    args = parser.parse_args()

    logger.info("======================================================")
    logger.info("   ApexHunter - Frame Extraction (5fps)")
    logger.info(f"   Using Process Pool with {args.workers} workers")
    logger.info("======================================================")
    
    total_extracted = 0
    tasks = []
    
    for year in SEASONS:
        year_str = str(year)
        year_video_dir = VIDEO_ROOT / year_str
        
        if not year_video_dir.is_dir():
            logger.warning(f"Directory not found: {year_video_dir}. Skipping.")
            continue
        
        videos = sorted(year_video_dir.glob("*.mp4"))
        logger.info(f"Year {year_str}: Found {len(videos)} videos in {year_video_dir.resolve()}")
        
        for video_path in videos:
            video_name = video_path.stem
            output_dir = OUTPUT_ROOT / year_str / video_name
            
            # Skip if already extracted
            if output_dir.is_dir() and any(output_dir.iterdir()):
                existing = sum(1 for _ in output_dir.iterdir())
                logger.info(f"SKIP - {video_name} ({existing} frames already exist)")
                total_extracted += existing
                continue
            
            tasks.append((video_path, output_dir))

    if tasks:
        logger.info(f"\nStarting extraction for {len(tasks)} new videos in parallel...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Map the function over the argument tuples
            futures = [executor.submit(extract_frames_from_video, vp, od) for vp, od in tasks]
            
            for future in concurrent.futures.as_completed(futures):
                count = future.result()
                total_extracted += count
                
    logger.info("======================================================")
    logger.info(f"  [DONE] Total frames extracted: {total_extracted}")
    logger.info(f"         Output: {OUTPUT_ROOT.resolve()}")
    logger.info("======================================================")


if __name__ == "__main__":
    main()
