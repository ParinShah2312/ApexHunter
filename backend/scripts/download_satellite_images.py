"""
================================================================================
  ApexHunter - F1 Computer Vision Project
  Script: download_satellite_images.py
--------------------------------------------------------------------------------
  Purpose : Downloads satellite imagery for all 2024 F1 circuits using the
            staticmap library with Esri World Imagery tiles (FREE, no API key).
            Images are saved to data_lake/raw_images/satellite_maps/

  Usage   : pip install staticmap
            python backend/scripts/download_satellite_images.py
================================================================================
"""

import os
import sys

try:
    from staticmap import StaticMap
except ImportError:
    print("[ERROR] staticmap is not installed. Run: pip install staticmap")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_LAKE_ROOT = os.path.join("data_lake", "raw_images", "satellite_maps")

# Esri World Imagery - Free satellite tile server (no API key required)
ESRI_SATELLITE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 1280
ZOOM_LEVEL = 15  # Good zoom for seeing the full circuit layout

# ── F1 2024 Circuit Coordinates ───────────────────────────────────────────────
# Hard-coded lat/lon for precise centering on each circuit

CIRCUITS_2024 = [
    {"name": "Bahrain Grand Prix",           "lat": 26.0325, "lon": 50.5106},
    {"name": "Saudi Arabian Grand Prix",     "lat": 21.6319, "lon": 39.1044},
    {"name": "Australian Grand Prix",        "lat": -37.8497, "lon": 144.9680},
    {"name": "Japanese Grand Prix",          "lat": 34.8431, "lon": 136.5407},
    {"name": "Chinese Grand Prix",           "lat": 31.3389, "lon": 121.2197},
    {"name": "Miami Grand Prix",             "lat": 25.9581, "lon": -80.2389},
    {"name": "Emilia Romagna Grand Prix",    "lat": 44.3439, "lon": 11.7167},
    {"name": "Monaco Grand Prix",            "lat": 43.7347, "lon": 7.4206},
    {"name": "Canadian Grand Prix",          "lat": 45.5000, "lon": -73.5228},
    {"name": "Spanish Grand Prix",           "lat": 41.5700, "lon": 2.2611},
    {"name": "Austrian Grand Prix",          "lat": 47.2197, "lon": 14.7647},
    {"name": "British Grand Prix",           "lat": 52.0786, "lon": -1.0169},
    {"name": "Hungarian Grand Prix",         "lat": 47.5789, "lon": 19.2486},
    {"name": "Belgian Grand Prix",           "lat": 50.4372, "lon": 5.9714},
    {"name": "Dutch Grand Prix",             "lat": 52.3888, "lon": 4.5409},
    {"name": "Italian Grand Prix",           "lat": 45.6156, "lon": 9.2811},
    {"name": "Azerbaijan Grand Prix",        "lat": 40.3725, "lon": 49.8533},
    {"name": "Singapore Grand Prix",         "lat": 1.2914, "lon": 103.8636},
    {"name": "United States Grand Prix",     "lat": 30.1328, "lon": -97.6411},
    {"name": "Mexico City Grand Prix",       "lat": 19.4042, "lon": -99.0907},
    {"name": "São Paulo Grand Prix",         "lat": -23.7036, "lon": -46.6997},
    {"name": "Las Vegas Grand Prix",         "lat": 36.1147, "lon": -115.1728},
    {"name": "Qatar Grand Prix",             "lat": 25.4900, "lon": 51.4542},
    {"name": "Abu Dhabi Grand Prix",         "lat": 24.4672, "lon": 54.6031},
]


def download_satellite_images():
    print("=" * 60)
    print("   ApexHunter Track Satellite Image Ingestion")
    print("   Using: Esri World Imagery (Free, No API Key)")
    print("=" * 60 + "\n")

    os.makedirs(DATA_LAKE_ROOT, exist_ok=True)
    print(f"[INFO] Output Directory: {os.path.abspath(DATA_LAKE_ROOT)}")
    print(f"[INFO] Image Size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"[INFO] Zoom Level: {ZOOM_LEVEL}\n")

    total_downloaded = 0
    total_skipped = 0

    for circuit in CIRCUITS_2024:
        race_name = circuit["name"]
        lat = circuit["lat"]
        lon = circuit["lon"]

        filename_clean = race_name.lower().replace(" ", "_").replace("/", "_")
        final_img_path = os.path.join(DATA_LAKE_ROOT, f"{filename_clean}.png")

        if os.path.exists(final_img_path):
            print(f"  [SKIP] {race_name} — already exists.")
            total_skipped += 1
            continue

        print(f"  >>> Downloading: {race_name} ({lat}, {lon})")

        try:
            m = StaticMap(IMAGE_WIDTH, IMAGE_HEIGHT, url_template=ESRI_SATELLITE_URL)
            image = m.render(zoom=ZOOM_LEVEL, center=[lon, lat])
            image.save(final_img_path)
            print(f"      ✓ Saved to {final_img_path}")
            total_downloaded += 1
        except Exception as e:
            print(f"  [ERROR] Failed to download {race_name}: {e}")

    print("\n" + "=" * 60)
    print(f"  [DONE] Downloaded: {total_downloaded} | Skipped: {total_skipped}")
    print("=" * 60)


if __name__ == "__main__":
    download_satellite_images()
