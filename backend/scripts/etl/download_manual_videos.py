"""
================================================================================
  ApexHunter - F1 Computer Vision Project
  Script: download_manual_videos.py
--------------------------------------------------------------------------------
  Purpose : Downloads one batch of onboard videos with yt-dlp. The batch (a year
            plus name -> YouTube URL pairs) is set under "manual_downloads" in
            config.json; videos were collected in batches of six, and the config
            holds the last batch (2024 rounds 19-24).

  Usage   : python backend/scripts/etl/download_manual_videos.py
================================================================================
"""

import sys
from typing import Any, Dict

from pathlib import Path

# Make backend/scripts importable so this file runs both by path and as `python -m <pkg>.<module>`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import CONFIG, DATA_LAKE_DIR, setup_logger

logger = setup_logger(__name__)

try:
    import yt_dlp
except ImportError:
    logger.error("yt-dlp is not installed. Please install it using `pip install yt-dlp`.")
    sys.exit(1)

# Same folder the rest of the CV pipeline uses (it was data_lake/raw_video/, which nothing reads)
DATA_LAKE_ROOT = DATA_LAKE_DIR / "downloaded_videos"
_BATCH: Dict[str, Any] = CONFIG.get("manual_downloads", {})
MANUAL_DOWNLOADS_YEAR: str = str(_BATCH.get("year", ""))
# File name (without .mp4) -> YouTube URL
MANUAL_DOWNLOADS: Dict[str, str] = _BATCH.get("videos", {})

base_ydl_opts: Dict[str, Any] = {
    "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]",
    "merge_output_format": "mp4",
    "quiet": False,
    "progress": True,
    "retries": 3,
}


def download_manual() -> None:
    """Download every video in the configured batch into downloaded_videos/{year}/."""
    if not MANUAL_DOWNLOADS_YEAR or not MANUAL_DOWNLOADS:
        logger.error('config.json needs "manual_downloads": {"year": ..., "videos": {name: url}}')
        sys.exit(1)
    logger.info("======================================================")
    logger.info("   ApexHunter - MANUAL Video Correction")
    logger.info("======================================================")

    year_dir = DATA_LAKE_ROOT / MANUAL_DOWNLOADS_YEAR
    year_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in MANUAL_DOWNLOADS.items():
        logger.info(f"Forcing Download: {filename}")
        logger.info(f"Source: {url}")

        final_mp4_path = year_dir / f"{filename}.mp4"
        # Download under a temporary name and replace the existing (incorrect) file only once the
        # download has succeeded; deleting it first lost the file whenever a download failed
        temp_mp4_path = year_dir / f"{filename}.download.mp4"

        ydl_opts = base_ydl_opts.copy()
        ydl_opts['outtmpl'] = str(year_dir / f"{filename}.download.%(ext)s")

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            if not temp_mp4_path.exists():
                raise FileNotFoundError(f"yt-dlp produced no {temp_mp4_path.name}")
            if final_mp4_path.exists():
                logger.warning(f"Replacing existing file: {final_mp4_path.name}")
            temp_mp4_path.replace(final_mp4_path)
            logger.info(f"Saved to {final_mp4_path}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e} (existing file left unchanged)")


if __name__ == "__main__":
    download_manual()
