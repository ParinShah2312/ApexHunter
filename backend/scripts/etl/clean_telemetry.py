"""
================================================================================
  ApexHunter - Telemetry Cleaning Pipeline
  Script: clean_telemetry.py
--------------------------------------------------------------------------------
  Purpose : Reads raw parquet files, applies domain-specific cleaning (null
            handling, outlier clipping, type downcasting), and writes cleaned
            parquet output.

  Usage   : python backend/scripts/clean_telemetry.py [--input-dir] [--file]
================================================================================
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from utils import DATA_LAKE_DIR, setup_logger

# ── Configuration ─────────────────────────────────────────────────────────────
logger = setup_logger(__name__)

RAW_DATA_DIR = DATA_LAKE_DIR / "season_data"
CLEAN_DATA_DIR = DATA_LAKE_DIR / "clean_data"


def get_directory_size(directory: Path) -> str:
    """Calculates the total size of files in a directory in MB."""
    total_size = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    return f"{total_size / (1024 * 1024):.2f} MB"


def _synthesize_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: Synthesize missing essential columns."""
    expected_cols = ['Driver', 'Speed', 'RPM', 'Throttle', 'Brake', 'X', 'Y', 'Time', 'SessionTime', 'nGear']
    for c in expected_cols:
        if c not in df.columns:
            if c == 'Driver':
                df['Driver'] = 'UNKNOWN'
            elif c in ['Time', 'SessionTime']:
                df[c] = pd.to_timedelta(np.arange(len(df)), unit='s')
            elif c == 'nGear':
                df['nGear'] = 8
            else:
                df[c] = 0
    return df


def _drop_all_null_core_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Drop rows missing all core telemetry."""
    core_telemetry = ['Speed', 'RPM', 'X', 'Y']
    cols_to_check = [c for c in core_telemetry if c in df.columns]
    if cols_to_check:
        df.dropna(subset=cols_to_check, how='all', inplace=True)
    return df


def _forward_fill_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Forward fill small gaps in numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill()
    return df


def _clip_telemetry_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: Domain specific outlier clipping."""
    if 'Speed' in df.columns:
        df['Speed'] = df['Speed'].clip(lower=0, upper=380) # Modern F1 cars max out ~360 km/h
    if 'RPM' in df.columns:
        df['RPM'] = df['RPM'].clip(lower=0, upper=15000)   # V6 Hybrid limit
    if 'Throttle' in df.columns:
        df['Throttle'] = df['Throttle'].clip(lower=0, upper=100)
    if 'Brake' in df.columns:
        df['Brake'] = df['Brake'].clip(lower=0, upper=100)
    return df


def _downcast_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5: Optimize Memory (Downcast types)."""
    if 'Speed' in df.columns:
        df['Speed'] = df['Speed'].astype('float32')
    if 'RPM' in df.columns:
        df['RPM'] = df['RPM'].astype('float32')
    if 'Throttle' in df.columns:
        df['Throttle'] = df['Throttle'].astype('float32')
    if 'Brake' in df.columns:
        df['Brake'] = df['Brake'].astype('float32')
    if 'X' in df.columns:
        df['X'] = df['X'].astype('float32')
    if 'Y' in df.columns:
        df['Y'] = df['Y'].astype('float32')
    return df


def clean_telemetry_file(input_file: Path, output_file: Path) -> None:
    """Loads, cleans, and saves a Single telemetry file.

    Args:
        input_file: Path to the raw telemetry file.
        output_file: Path to save the cleaned telemetry.
    """
    try:
        df = pd.read_parquet(input_file)
        initial_rows = len(df)
        if initial_rows == 0:
            logger.warning(f"File {input_file.name} is empty. Skipping.")
            return

        # 1. Essential Columns Check
        df = _synthesize_missing_columns(df)

        # 2. Drop Rows Missing Critical Core Telemetry
        df = _drop_all_null_core_rows(df)

        dropped_rows = initial_rows - len(df)

        # 3. Forward Fill Small Gaps (Interpolate missing sensor packets within laps)
        df = _forward_fill_numeric(df)

        # 4. Outlier Clipping (Domain Specific Caps)
        df = _clip_telemetry_outliers(df)

        # 5. Optimize Memory (Downcast types)
        df = _downcast_to_float32(df)

        # Save Cleaned Data to Parquet (Better for big data than CSV)
        df.to_parquet(output_file, compression="snappy")

        logger.info(f"Cleaned {input_file.name}: Dropped {dropped_rows} rows. Saved to clean_data/.")

        # Cleanup memory immediately
        del df
        gc.collect()

    except Exception as e:
        logger.error(f"Error processing {input_file.name}: {e}")


def main() -> None:
    """Parse CLI arguments and run telemetry cleaning batch pipeline."""
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description="Clean raw telemetry parquet files.")
    parser.add_argument('--input-dir', type=str, default=str(RAW_DATA_DIR), help="Directory containing raw parquet files.")
    parser.add_argument('--output-dir', type=str, default=str(CLEAN_DATA_DIR), help="Directory to save cleaned files.")
    parser.add_argument('--file', type=str, help="Process a specific file instead of a directory.")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.file:
        file_to_process = Path(args.file)
        if file_to_process.exists():
            out_file = output_path / file_to_process.name
            logger.info(f"Processing specific file: {file_to_process.name}...")
            clean_telemetry_file(file_to_process, out_file)
        else:
            logger.error(f"File not found: {args.file}")
        return

    logger.info("Starting Batch Telemetry Data Cleaning Pipeline...")
    logger.info(f"Raw Data Lake Size: {get_directory_size(input_path)}")

    # Process all parquets in the raw data lake
    raw_files = list(input_path.glob("*.parquet"))

    for i, file_path in enumerate(raw_files, 1):
        out_file = output_path / file_path.name

        # Skip if already cleaned (saves time on subsequent runs)
        if out_file.exists():
            logger.info(f"[{i}/{len(raw_files)}] Skipping {file_path.name} (already cleaned)")
            continue

        logger.info(f"[{i}/{len(raw_files)}] Processing {file_path.name}...")
        clean_telemetry_file(file_path, out_file)

    logger.info("--- Data Cleaning Complete ---")
    logger.info(f"Clean Data Lake Size: {get_directory_size(output_path)}")


if __name__ == "__main__":
    main()
