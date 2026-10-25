"""
================================================================================
  ApexHunter - Season Data Downloader
  Script: download_season_data.py
--------------------------------------------------------------------------------
  Purpose : Fetches F1 telemetry via FastF1 for configured seasons and saves
            per-session parquet files to the data lake.

  Usage   : python backend/scripts/etl/download_season_data.py
================================================================================
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import fastf1
import pandas as pd

# Make backend/scripts importable so this file runs both by path and as `python -m <pkg>.<module>`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import CACHE_DIR, CONFIG, DATA_LAKE_DIR, enable_fastf1_cache, setup_logger

# ── Configuration ─────────────────────────────────────────────────────────────
logger = setup_logger(__name__)

# config.json stores years as strings (other scripts use them as folder names); FastF1 needs ints
SEASONS: List[int] = [int(year) for year in CONFIG.get("seasons", [2023, 2024])]
SEASON_DATA_DIR: Path = DATA_LAKE_DIR / "season_data"


def setup_directories() -> None:
    """Creates necessary directories for data storage and caching."""
    SEASON_DATA_DIR.mkdir(parents=True, exist_ok=True)
    enable_fastf1_cache()
    logger.info(f"Data lake directory: {SEASON_DATA_DIR}")
    logger.info(f"Cache directory: {CACHE_DIR}")


def get_directory_size(directory: Path) -> str:
    """Calculates the total size of files in a directory in MB."""
    total_size = 0
    for path in directory.rglob("*"):
        if path.is_file():
            total_size += path.stat().st_size

    size_mb = total_size / (1024 * 1024)
    return f"{size_mb:.2f} MB"


def _process_driver(session: Any, driver: str, round_num: int, session_type: str, year: int) -> Optional[pd.DataFrame]:
    """Telemetry for a single driver, or None if the driver set no laps. Errors propagate."""
    driver_laps = session.laps.pick_drivers(driver)
    if driver_laps.empty:
        return None

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
    return telemetry[existing_cols]


def process_session(year: int, round_num: int, session_type: str) -> str:
    """Downloads and processes data for a specific round and session.

    Returns:
        "skipped" (file exists), "saved", "partial" (saved, but some drivers failed),
        "empty" (no driver had laps) or "failed".
    """
    file_name = f"{year}_{round_num}_{session_type}.parquet"
    file_path = SEASON_DATA_DIR / file_name

    if file_path.exists():
        logger.info(f"Skipping {file_name} (already exists)")
        return "skipped"

    try:
        session = fastf1.get_session(year, round_num, session_type)
        logger.info(f"Loading Session: {year} Round {round_num} - {session_type}")
        session.load(telemetry=True, laps=True, weather=False)


        all_drivers_data = []
        failed_drivers = []

        # Get list of drivers
        drivers = session.drivers
        logger.info(f"Processing {len(drivers)} drivers...")

        for driver in drivers:
            try:
                telemetry = _process_driver(session, driver, round_num, session_type, year)
            except Exception as e:
                logger.warning(f"Failed to load driver {driver} in {year} Round {round_num} {session_type}: {e}")
                failed_drivers.append(driver)
                continue
            if telemetry is not None:
                all_drivers_data.append(telemetry)

        if not all_drivers_data:
            logger.warning(f"No data found for {year} Round {round_num} {session_type}")
            return "failed" if failed_drivers else "empty"

        # Concatenate and Save
        session_df = pd.concat(all_drivers_data, ignore_index=True)
        # Write to a temporary file and rename when complete: a crash mid-write used to leave a
        # partial file that the skip-if-exists check then treated as done
        tmp_file = file_path.with_name(file_path.name + ".tmp")
        session_df.to_parquet(tmp_file, compression="snappy")
        tmp_file.replace(file_path)

        logger.info(f"Saved {file_path}")
        if failed_drivers:
            logger.warning(f"{file_name} is missing drivers {failed_drivers}; delete it and re-run to retry them.")
            return "partial"
        return "saved"

    except Exception as e:
        logger.error(f"Error processing {year} Round {round_num} {session_type}: {e}")
        return "failed"


def main() -> int:
    """Download every configured season. Returns 1 if any season or session failed, else 0."""
    setup_directories()
    results: Dict[str, List[str]] = {}

    for year in SEASONS:
        logger.info(f"Processing Season {year}...")

        # The schedule's highest round number is the season length (pre-season testing is round 0).
        # No fallback guess: a wrong round count requests sessions that don't exist (2023 had 22).
        try:
            schedule = fastf1.get_event_schedule(year)
            total_rounds = int(schedule['RoundNumber'].max())
            logger.info(f"Season {year} has {total_rounds} rounds.")
        except Exception as e:
            logger.error(f"Could not fetch schedule for {year}: {e}. Skipping the season.")
            results.setdefault("failed", []).append(f"{year} (schedule)")
            continue

        for round_num in range(1, total_rounds + 1):
            for session_type in ['Q', 'R']:
                status = process_session(year, round_num, session_type)
                results.setdefault(status, []).append(f"{year}_{round_num}_{session_type}")

                # Monitor size
                current_size = get_directory_size(SEASON_DATA_DIR)
                logger.info(f"Current Data Lake Size: {current_size}")

    summary = ", ".join(f"{status}={len(items)}" for status, items in sorted(results.items()))
    logger.info(f"Download complete: {summary or 'nothing to do'}")
    final_size = get_directory_size(SEASON_DATA_DIR)
    logger.info(f"Final Data Lake Size: {final_size}")

    problems = results.get("failed", []) + results.get("partial", [])
    if problems:
        logger.error(f"Failed or incomplete: {problems}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
