"""
================================================================================
  ApexHunter - F1 Computer Vision Project
  Script: download_raw_video.py
--------------------------------------------------------------------------------
  Purpose : Downloads unstructured video data (pole laps from 2023 and 2024)
            from YouTube using yt-dlp. Stores them systematically in the
            data lake's raw_video layer (by year). 
            Filenames are formatted as: racenumber_trackname_driver_pole.mp4
================================================================================
"""

import os
import sys

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import yt_dlp
    import fastf1
except ImportError:
    print("[ERROR] yt-dlp or fastf1 is not installed.")
    print("        Run:  pip install yt-dlp fastf1")
    sys.exit(1)

# Enable cache for FastF1 to speed up repeated queries
fastf1.Cache.enable_cache('cache/')

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_LAKE_ROOT = os.path.join("data_lake", "raw_video")

SEASONS = [2023, 2024]

# ── yt-dlp Base Options ───────────────────────────────────────────────────────
def filter_official_f1(info_dict):
    """
    Ensures that we only download onboard pole lap videos uploaded by the official F1 channel.
    Returns None if it matches (proceeds with download), or a string error message if it fails.
    """
    uploader = info_dict.get('uploader', '').lower()
    channel = info_dict.get('channel', '').lower()
    title = info_dict.get('title', '').lower()
    
    # 1. Must be from official F1 channel
    if 'formula 1' not in uploader and 'formula 1' not in channel:
        return "Not the official FORMULA 1 channel"
        
    # 2. Must not be a highlights package
    if 'highlights' in title:
        return "Rejected: Title implies this is a highlights video."
        
    # 3. Must be an onboard pole lap
    if 'onboard' not in title and 'pole' not in title:
        return "Rejected: Title does not signify an onboard pole lap."
        
    return None

base_ydl_opts = {
    # Prefer 1080p native mp4 + m4a
    "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]",
    "merge_output_format": "mp4",
    "quiet": False,
    "no_warnings": False,
    "progress": True,
    "addmetadata": True,
    "retries": 3,
    "fragment_retries": 3,
    "ignoreerrors": True, # skip to next video on error
    "match_filter": filter_official_f1,
}

def download_pole_laps():
    print("======================================================")
    print("   ApexHunter Video Ingestion (2023 & 2024)")
    print("======================================================\n")
    
    total_downloaded = 0
    
    for year in SEASONS:
        year_dir = os.path.join(DATA_LAKE_ROOT, str(year))
        os.makedirs(year_dir, exist_ok=True)
        
        # Load the official schedule for the year
        schedule = fastf1.get_event_schedule(year)
        races = schedule[schedule['EventFormat'] != 'testing']
        
        print(f"\n[INFO] Processing Year: {year} ({len(races)} races) -> {os.path.abspath(year_dir)}")
        
        for index, event in races.iterrows():
            race_num = event['RoundNumber']
            race_name = event['EventName']
            country = event['Country']
            
            # 1. Fetch the Qualifying session to find out who got Pole Position
            print(f"\n  >>> Querying FastF1 for {year} Round {race_num} ({country})...")
            try:
                session = fastf1.get_session(year, race_num, 'Q')
                session.load(telemetry=False, weather=False, messages=False) # Fast load
                
                # The winner of Q is the pole sitter
                pole_sitter = session.results.iloc[0]['Abbreviation'] # e.g., 'VER', 'LEC'
                
            except Exception as e:
                print(f"  [WARN] Could not retrieve Q session for {year} Round {race_num}. Skipping.")
                continue
                
            # 2. Construct the exact filename format needed: "racenumber_tracknumber_driver_pole"
            # Note: We use the country or event name as "trackname"
            track_clean = country.lower().replace(" ", "")
            filename = f"{race_num:02d}_{track_clean}_{pole_sitter.lower()}_pole"
            file_template = os.path.join(year_dir, f"{filename}.%(ext)s")
            final_mp4_path = os.path.join(year_dir, f"{filename}.mp4")
            
            # Check if file already exists
            if os.path.exists(final_mp4_path):
                print(f"  [SKIP] {filename}.mp4 already exists.")
                continue

            # 3. Construct YouTube Search String (Highly specific)
            search_query = f"ytsearch10: F1 {year} {race_name} {pole_sitter} pole lap onboard -sim"
            print(f"  >>> Searching YouTube: '{search_query}'")
            print(f"  >>> Saving as: {filename}.mp4")
            
            ydl_opts = base_ydl_opts.copy()
            ydl_opts['outtmpl'] = file_template
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([search_query])
                total_downloaded += 1
            except Exception as e:
                print(f"  [ERROR] Failed to download {filename}: {e}")
                
    print("\n======================================================")
    print(f"  [DONE] Ingestion Complete! Downloaded {total_downloaded} new files.")
    print(f"         Location: {os.path.abspath(DATA_LAKE_ROOT)}")
    print("======================================================")
    
if __name__ == "__main__":
    download_pole_laps()

