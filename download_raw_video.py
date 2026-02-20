"""
================================================================================
  ApexHunter - F1 Computer Vision Project
  Script: download_raw_video.py
--------------------------------------------------------------------------------
  Purpose : Downloads the unstructured video data (Verstappen Monaco 2023
            Pole Lap) from YouTube using yt-dlp and stores it in the
            data lake's raw_video layer.
================================================================================
"""

import os
import sys

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import yt_dlp
except ImportError:
    print("[ERROR] yt-dlp is not installed.")
    print("        Run:  pip install yt-dlp")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────

# Target YouTube URL — Verstappen Monaco 2023 Qualifying Pole Lap (official F1)
VIDEO_URL = "https://www.youtube.com/watch?v=lZ0bHr8UW7k"   # Official F1 channel

# Data lake directory structure
RAW_VIDEO_DIR = os.path.join("data_lake", "raw_video")
OUTPUT_FILENAME = "verstappen_monaco_2023"          # yt-dlp appends the ext

# ── Create directory ──────────────────────────────────────────────────────────
os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
print(f"[INFO] Output directory ready : {os.path.abspath(RAW_VIDEO_DIR)}")

# ── yt-dlp options ────────────────────────────────────────────────────────────
#
#   Format selection strategy:
#     'bestvideo[height=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height=1080]+bestaudio/best[height=1080]'
#
#   This tries (in order):
#     1. A native 1080p mp4 video stream merged with m4a audio  (ideal)
#     2. Any 1080p video stream merged with the best audio      (fallback)
#     3. A single 1080p stream that already has audio           (last resort)
#
#   merge_output_format='mp4' ensures the final container is always .mp4.

ydl_opts = {
    "format": (
        "bestvideo[height=1080][ext=mp4]+bestaudio[ext=m4a]"
        "/bestvideo[height=1080]+bestaudio"
        "/best[height<=1080]"
    ),
    "merge_output_format": "mp4",

    # Output path — yt-dlp will write:  data_lake/raw_video/verstappen_monaco_2023.mp4
    "outtmpl": os.path.join(RAW_VIDEO_DIR, f"{OUTPUT_FILENAME}.%(ext)s"),

    # Show a clean progress bar in the terminal
    "quiet"            : False,
    "no_warnings"      : False,
    "progress"         : True,

    # Embed metadata for provenance tracking
    "addmetadata"      : True,

    # Retry settings for reliability
    "retries"          : 5,
    "fragment_retries" : 5,
}

# ── Download ──────────────────────────────────────────────────────────────────
print(f"\n[INFO] Target URL     : {VIDEO_URL}")
print(f"[INFO] Target quality : 1080p MP4")
print(f"[INFO] Destination    : {os.path.abspath(RAW_VIDEO_DIR)}")
print("[INFO] Starting download ...\n")

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([VIDEO_URL])

except yt_dlp.utils.DownloadError as exc:
    print(f"\n[ERROR] Download failed: {exc}")
    sys.exit(1)

# ── Post-download verification ────────────────────────────────────────────────
final_path = os.path.join(RAW_VIDEO_DIR, f"{OUTPUT_FILENAME}.mp4")

if os.path.exists(final_path):
    size_bytes = os.path.getsize(final_path)
    size_mb    = size_bytes / (1024 ** 2)
    size_gb    = size_bytes / (1024 ** 3)

    print("\n" + "="*60)
    print("  [DONE] Download complete!")
    print("="*60)
    print(f"  File  : {os.path.abspath(final_path)}")
    if size_gb >= 1:
        print(f"  Size  : {size_gb:.2f} GB  ({size_bytes:,} bytes)")
    else:
        print(f"  Size  : {size_mb:.1f} MB  ({size_bytes:,} bytes)")
    print("="*60)
    print("\n[INFO] The raw video is now stored in the data lake.")
    print("       Path for HDFS ingestion:")
    print(f"         hdfs dfs -put \"{final_path}\" /user/apexhunter/raw_video/")
else:
    print("\n[WARN] Expected output file not found at the default path.")
    print("       yt-dlp may have saved it with a different name.")
    print(f"       Check the folder: {os.path.abspath(RAW_VIDEO_DIR)}")
