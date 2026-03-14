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

import sys
import concurrent.futures
from typing import Dict, Union
from utils import setup_logger, DATA_LAKE_DIR, CONFIG

logger = setup_logger(__name__)

try:
    from staticmap import StaticMap
except ImportError:
    print("[ERROR] staticmap is not installed. Run: pip install staticmap")
    sys.exit(1)

DATA_LAKE_ROOT = DATA_LAKE_DIR / "raw_images" / "satellite_maps"

# Esri World Imagery - Free satellite tile server (no API key required)
ESRI_SATELLITE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

sat_config = CONFIG.get("satellite", {})
IMAGE_WIDTH = sat_config.get("image_width", 1280)
IMAGE_HEIGHT = sat_config.get("image_height", 1280)
ZOOM_LEVEL = sat_config.get("zoom_level", 15)  # Good zoom for seeing the full circuit layout

# ── F1 2024 Circuit Coordinates ───────────────────────────────────────────────
# Hard-coded lat/lon for precise centering on each circuit loaded from config
CIRCUITS_2024 = sat_config.get("circuits", [])


def download_single_circuit(circuit: Dict[str, Union[str, float]]) -> int:
    """Downloads a single circuit, returns 1 if downloaded, 0 if skipped/failed."""
    race_name = circuit["name"]
    lat = circuit["lat"]
    lon = circuit["lon"]

    filename_clean = str(race_name).lower().replace(" ", "_").replace("/", "_")
    final_img_path = DATA_LAKE_ROOT / f"{filename_clean}.png"

    if final_img_path.exists():
        logger.info(f"SKIP - {race_name} already exists.")
        return 0

    logger.info(f"Downloading: {race_name} ({lat}, {lon})")

    try:
        m = StaticMap(IMAGE_WIDTH, IMAGE_HEIGHT, url_template=ESRI_SATELLITE_URL)
        image = m.render(zoom=ZOOM_LEVEL, center=[lon, lat])
        
        # Using str(final_img_path) because some libraries don't fully support Path objects
        image.save(str(final_img_path))
        logger.info(f"✓ Saved to {final_img_path.name}")
        return 1
    except Exception as e:
        logger.error(f"Failed to download {race_name}: {e}")
        return 0

def download_satellite_images() -> None:
    """Downloads satellite imagery for all 2024 F1 circuits using ThreadPoolExecutor."""
    logger.info("======================================================")
    logger.info("   ApexHunter Track Satellite Image Ingestion")
    logger.info("   Using: Esri World Imagery (Free, No API Key)")
    logger.info("======================================================")

    DATA_LAKE_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output Directory: {DATA_LAKE_ROOT.resolve()}")
    logger.info(f"Image Size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    logger.info(f"Zoom Level: {ZOOM_LEVEL}")

    total_downloaded = 0
    total_skipped = 0

    # Max 10 threads to avoid hammering the free API server too hard
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_single_circuit, c): c for c in CIRCUITS_2024}
        
        for future in concurrent.futures.as_completed(futures):
            downloaded = future.result()
            if downloaded == 1:
                total_downloaded += 1
            else:
                total_skipped += 1

    logger.info("======================================================")
    logger.info(f"  [DONE] Downloaded: {total_downloaded} | Skipped/Failed: {total_skipped}")
    logger.info("======================================================")


if __name__ == "__main__":
    download_satellite_images()
