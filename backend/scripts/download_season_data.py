import fastf1
import pandas as pd
from typing import List
from pathlib import Path
from utils import setup_logger, DATA_LAKE_DIR, CACHE_DIR, CONFIG

# Configure logging
logger = setup_logger(__name__)

# Constants
SEASONS: List[int] = CONFIG.get("seasons", [2023, 2024])
DATA_LAKE_DIR = DATA_LAKE_DIR / "season_data"

def setup_directories() -> None:
    """Creates necessary directories for data storage and caching."""
    DATA_LAKE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    logger.info(f"Data lake directory: {DATA_LAKE_DIR}")
    logger.info(f"Cache directory: {CACHE_DIR}")

def get_directory_size(directory: Path) -> str:
    """Calculates the total size of files in a directory in MB."""
    total_size = 0
    for path in directory.rglob('*'):
        if path.is_file():
            total_size += path.stat().st_size
    
    size_mb = total_size / (1024 * 1024)
    return f"{size_mb:.2f} MB"

def process_session(year: int, round_num: int, session_type: str) -> None:
    """Downloads and processes data for a specific round and session."""
    file_name = f"{year}_{round_num}_{session_type}.parquet"
    file_path = DATA_LAKE_DIR / file_name

    if file_path.exists():
        logger.info(f"Skipping {file_name} (already exists)")
        return

    try:
        session = fastf1.get_session(year, round_num, session_type)
        logger.info(f"Loading Session: {year} Round {round_num} - {session_type}")
        session.load(telemetry=True, laps=True, weather=False) 

        all_drivers_data = []
        
        # Get list of drivers
        drivers = session.drivers
        logger.info(f"Processing {len(drivers)} drivers...")

        for driver in drivers:
            try:
                driver_laps = session.laps.pick_drivers(driver)
                if driver_laps.empty:
                    continue
                
                # Get telemetry
                telemetry = driver_laps.get_telemetry()
                
                # Add identifiers
                telemetry['Driver'] = driver
                telemetry['Round'] = round_num
                telemetry['Session'] = session_type
                telemetry['Year'] = year
                
                # Keep core columns
                columns_to_keep = ['Date', 'SessionTime', 'Speed', 'RPM', 'nGear', 'Throttle', 'Brake', 'X', 'Y', 'Z', 'Driver', 'Round', 'Session', 'Year']
                # Filter columns that exist
                existing_cols = [col for col in columns_to_keep if col in telemetry.columns]
                telemetry = telemetry[existing_cols]

                all_drivers_data.append(telemetry)
            
            except Exception as e:
                logger.warning(f"Failed to load driver {driver} in {year} Round {round_num} {session_type}: {e}")
                continue

        if not all_drivers_data:
            logger.warning(f"No data found for {year} Round {round_num} {session_type}")
            return

        # Concatenate and Save
        session_df = pd.concat(all_drivers_data, ignore_index=True)
        session_df.to_parquet(file_path, compression='snappy')
        
        logger.info(f"Saved {file_path}")

    except Exception as e:
        logger.error(f"Error processing {year} Round {round_num} {session_type}: {e}")

def main() -> None:
    setup_directories()
    
    for year in SEASONS:
        logger.info(f"Processing Season {year}...")
        
        # Determine total rounds for the year
        # fastf1 doesn't have a simple "get_total_rounds" without loading schedule.
        # We can fetch the schedule.
        try:
            schedule = fastf1.get_event_schedule(year)
            # Filter for official events (not testing)
            # Usually we just take the max round number from the schedule
            total_rounds = schedule['RoundNumber'].max()
            logger.info(f"Season {year} has {total_rounds} rounds.")
        except Exception as e:
            logger.error(f"Could not fetch schedule for {year}: {e}. Defaulting to 24.")
            total_rounds = 24 # Fallback

        for round_num in range(1, int(total_rounds) + 1):
            for session_type in ['Q', 'R']:
                process_session(year, round_num, session_type)
                
                # Monitor size
                current_size = get_directory_size(DATA_LAKE_DIR)
                logger.info(f"Current Data Lake Size: {current_size}")

    logger.info("Download complete.")
    final_size = get_directory_size(DATA_LAKE_DIR)
    logger.info(f"Final Data Lake Size: {final_size}")

if __name__ == "__main__":
    main()
