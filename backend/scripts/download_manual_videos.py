"""
================================================================================
  ApexHunter - F1 Computer Vision Project
  Script: download_manual_videos.py
--------------------------------------------------------------------------------
  Purpose : Overrides the dynamic yt-dlp search and forces the download of 
            specific YouTube links provided by the User.
================================================================================
"""

import os
import sys

try:
    import yt_dlp
except ImportError:
    print("[ERROR] yt-dlp is not installed.")
    sys.exit(1)

DATA_LAKE_ROOT = os.path.join("data_lake", "raw_video")

# The precise YouTube links provided by the user mapping to their correct filenames
MANUAL_DOWNLOADS = {
    # 2024 Season - Batch 4 (Final)
    "19_unitedstates_nor_pole": "https://youtu.be/sC8pW4HsOs4",
    "20_mexico_sai_pole": "https://youtu.be/wVFJP4ECnAc",
    "21_brazil_nor_pole": "https://youtu.be/ZhERj8n4HEQ",
    "22_lasvegas_rus_pole": "https://youtu.be/7N_GVP_CUqA",
    "23_qatar_ver_pole": "https://youtu.be/aMEoQSPRRh4",
    "24_abudhabi_nor_pole": "https://youtu.be/ul4wmQwxrwA",
}

base_ydl_opts = {
    "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]",
    "merge_output_format": "mp4",
    "quiet": False,
    "progress": True,
    "retries": 3,
}

def download_manual():
    print("======================================================")
    print("   ApexHunter - MANUAL Video Correction")
    print("======================================================\n")
    
    year_dir = os.path.join(DATA_LAKE_ROOT, "2024")
    os.makedirs(year_dir, exist_ok=True)
    
    for filename, url in MANUAL_DOWNLOADS.items():
        print(f"\n[INFO] Forcing Download: {filename}")
        print(f"       Source: {url}")
        
        file_template = os.path.join(year_dir, f"{filename}.%(ext)s")
        final_mp4_path = os.path.join(year_dir, f"{filename}.mp4")
        
        # We explicitly WANT to overwrite the bad files here, so we delete them first
        if os.path.exists(final_mp4_path):
            print(f"       [WARN] Deleting existing incorrect file...")
            os.remove(final_mp4_path)
            
        ydl_opts = base_ydl_opts.copy()
        ydl_opts['outtmpl'] = file_template
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"       [SUCCESS] Saved to {final_mp4_path}")
        except Exception as e:
            print(f"       [ERROR] Failed to download {filename}: {e}")
            
if __name__ == "__main__":
    download_manual()
