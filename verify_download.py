import fastf1
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_LAKE_DIR = Path("data_lake/season_2023_test")
CACHE_DIR = Path("cache")

def test_download():
    DATA_LAKE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    
    logger.info("Testing Round 1 Qualifying download...")
    try:
        session = fastf1.get_session(2023, 1, 'Q')
        session.load(telemetry=True, laps=True, weather=False)
        
        # Test with just one driver for speed
        driver = session.drivers[0]
        laps = session.laps.pick_driver(driver)
        telemetry = laps.get_telemetry()
        
        logger.info(f"Downloaded telemetry shape: {telemetry.shape}")
        
        # Save simplified parquet
        file_path = DATA_LAKE_DIR / "test_r1_q.parquet"
        telemetry.to_parquet(file_path, compression='snappy')
        logger.info(f"Saved test file to {file_path}")
        
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_download()
