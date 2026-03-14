"""
================================================================================
  ApexHunter - F1 Computer Vision Project
  Script: download_manual_videos.py
--------------------------------------------------------------------------------
  Purpose : Overrides the dynamic yt-dlp search and forces the download of 
            specific YouTube links provided by the User.
================================================================================
"""

import sys
from typing import Dict, Any
from utils import setup_logger, DATA_LAKE_DIR, CONFIG

logger = setup_logger(__name__)

try:
    import yt_dlp
except ImportError:
    logger.error("yt-dlp is not installed. Please install it using `pip install yt-dlp`.")
    sys.exit(1)

DATA_LAKE_ROOT = DATA_LAKE_DIR / "raw_video"

# The precise YouTube links provided by the user mapping to their correct filenames
MANUAL_DOWNLOADS: Dict[str, str] = CONFIG.get("manual_downloads", {})

base_ydl_opts: Dict[str, Any] = {
    "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]",
    "merge_output_format": "mp4",
    "quiet": False,
    "progress": True,
    "retries": 3,
}

def download_manual() -> None:
    """Forces the download of specific YouTube links provided by the user."""
    logger.info("======================================================")
    logger.info("   ApexHunter - MANUAL Video Correction")
    logger.info("======================================================")
    
    year_dir = DATA_LAKE_ROOT / "2024"
    year_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, url in MANUAL_DOWNLOADS.items():
        logger.info(f"Forcing Download: {filename}")
        logger.info(f"Source: {url}")
        
        file_template = str(year_dir / f"{filename}.%(ext)s")
        final_mp4_path = year_dir / f"{filename}.mp4"
        
        # We explicitly WANT to overwrite the bad files here, so we delete them first
        if final_mp4_path.exists():
            logger.warning(f"Deleting existing incorrect file: {final_mp4_path.name}")
            final_mp4_path.unlink()
            
        ydl_opts = base_ydl_opts.copy()
        ydl_opts['outtmpl'] = file_template
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            logger.info(f"Saved to {final_mp4_path.name}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            
if __name__ == "__main__":
    download_manual()
