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

import shutil
import random
from utils import setup_logger, DATA_LAKE_DIR, CONFIG

logger = setup_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

FRAMES_ROOT = DATA_LAKE_DIR / "cv_frames"
OUTPUT_DIR = DATA_LAKE_DIR / "cv_dataset" / "upload_to_roboflow"
FRAMES_PER_VIDEO = CONFIG.get("cv_frames", {}).get("frames_per_video", 11)  # ~11 frames × 46 videos ≈ 506 frames total
SEASONS = CONFIG.get("seasons", ["2023", "2024"])

random.seed(42)  # Reproducible selection

# ── Main ──────────────────────────────────────────────────────────────────────

def select_frames() -> None:
    logger.info("======================================================")
    logger.info("   ApexHunter - Training Frame Selector")
    logger.info("======================================================")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_selected = 0
    
    for year in SEASONS:
        year_str = str(year)
        year_dir = FRAMES_ROOT / year_str
        if not year_dir.is_dir():
            logger.warning(f"{year_dir} not found. Skipping.")
            continue
        
        video_dirs = sorted([d for d in year_dir.iterdir() if d.is_dir()])
        
        logger.info(f"Year {year_str}: {len(video_dirs)} track folders found.")
        
        for video_dir in video_dirs:
            video_name = video_dir.name
            all_frames = sorted(video_dir.glob("*.jpg"))
            
            if len(all_frames) == 0:
                logger.info(f"SKIP - {video_name}: No frames found.")
                continue
            
            # Strategy: Pick frames evenly spread across the entire video
            # This ensures we get straights, corners, braking zones, tunnels, etc.
            step = max(1, len(all_frames) // FRAMES_PER_VIDEO)
            selected = all_frames[::step][:FRAMES_PER_VIDEO]
            
            logger.info(f"{video_name}: {len(all_frames)} total → selected {len(selected)} frames")
            
            for frame_path in selected:
                frame_basename = frame_path.name
                # Prefix with year and track to make filenames unique
                new_name = f"{year_str}_{video_name}_{frame_basename}"
                dest = OUTPUT_DIR / new_name
                shutil.copy2(str(frame_path), str(dest))
                total_selected += 1
    
    logger.info("======================================================")
    logger.info(f"  [DONE] Selected {total_selected} frames for annotation.")
    logger.info(f"         Upload folder: {OUTPUT_DIR.resolve()}")
    logger.info("======================================================")
    logger.info("  Next step: Upload this entire folder to Roboflow!")

if __name__ == "__main__":
    select_frames()
