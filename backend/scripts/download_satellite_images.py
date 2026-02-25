"""
================================================================================
  ApexHunter - F1 Computer Vision Project
  Script: download_satellite_images.py
--------------------------------------------------------------------------------
  Purpose : Connects to the fastf1 API to get circuit coordinates (Lat/Lon) 
            for the 2024 season, and provides a framework to download 
            Static Satellite Maps for the UI background.

  Note    : To download high-res satellite images, an API key (like Google Maps
            Static API or Mapbox) is usually required. This script uses a free
            fallback/placeholder approach but is designed to take a real API key.
================================================================================
"""

import os
import sys
import requests

try:
    import fastf1
except ImportError:
    print("[ERROR] fastf1 is not installed. Run: pip install fastf1")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_LAKE_ROOT = os.path.join("data_lake", "raw_images", "satellite_maps")

# Google Maps Static API (Requires Key)
# e.g., "AIzaSy..."
GOOGLE_MAPS_API_KEY = "YOUR_API_KEY_HERE"  

def download_satellite_images():
    print("======================================================")
    print("   ApexHunter Track Satellite Image Ingestion")
    print("======================================================\n")
    
    os.makedirs(DATA_LAKE_ROOT, exist_ok=True)
    print(f"[INFO] Output Directory: {os.path.abspath(DATA_LAKE_ROOT)}")
    
    # We use the 2024 season to get the most recent track layouts
    schedule = fastf1.get_event_schedule(2024)
    
    # Filter out testing events, keeping only real Grands Prix
    races = schedule[schedule['EventFormat'] != 'testing']
    
    total_downloaded = 0
    
    for _, event in races.iterrows():
        race_name = event['EventName']
        country = event['Country']
        location = event['Location']  # e.g., "Sakhir", "Monza"
        
        # FastF1 doesn't guarantee exact Lat/Lon in the schedule object,
        # but we can query standard locations or use the event location string for a Map API search.
        
        filename_clean = race_name.lower().replace(" ", "_").replace("/", "_")
        final_img_path = os.path.join(DATA_LAKE_ROOT, f"{filename_clean}.jpg")
        
        if os.path.exists(final_img_path):
            print(f"  [SKIP] Satellite image for {race_name} already exists.")
            continue

        print(f"\n  >>> Fetching: {race_name} ({location}, {country})")
        
        # --- STATIC MAP API URL ---
        # If the user provides a Google Maps API Key:
        if GOOGLE_MAPS_API_KEY != "YOUR_API_KEY_HERE":
            search_query = f"Formula 1 Circuit, {location}, {country}"
            zoom = "14"  # Zoom level to capture the whole track
            size = "1280x1280" # High-Res 
            maptype = "satellite"
            
            url = f"https://maps.googleapis.com/maps/api/staticmap?center={search_query}&zoom={zoom}&size={size}&maptype={maptype}&key={GOOGLE_MAPS_API_KEY}"
            
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(final_img_path, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                    print(f"      Mapped and saved to {final_img_path}")
                    total_downloaded += 1
                else:
                    print(f"  [ERROR] API request failed (Code: {response.status_code}).")
            except Exception as e:
                print(f"  [ERROR] Failed to download {race_name}: {e}")
                
        else:
            # Fallback warning if no API key is present
            print("  [WARN] No Google Maps API Key found. Skipping actual download.")
            print("         Please insert your GOOGLE_MAPS_API_KEY in the script.")
            # We break after first warning to avoid spamming the terminal
            print("\n[INFO] Please update the script with your API key to download actual satellite imagery.")
            break
            
    print("\n======================================================")
    print(f"  [DONE] Ingestion Complete! Downloaded {total_downloaded} new images.")
    print("======================================================")
    print("NOTE: You can also use Mapbox Static Images API if preferred.")

if __name__ == "__main__":
    download_satellite_images()
