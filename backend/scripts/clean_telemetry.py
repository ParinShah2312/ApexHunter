import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import gc
from utils import setup_logger, DATA_LAKE_DIR

# Configure logging
logger = setup_logger(__name__)

# Input / Output Directories
RAW_DATA_DIR = DATA_LAKE_DIR / "season_data"
CLEAN_DATA_DIR = DATA_LAKE_DIR / "clean_data"

# Create clean data directory if it doesn't exist
CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_directory_size(directory: Path) -> str:
    """Calculates the total size of files in a directory in MB."""
    total_size = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
    return f"{total_size / (1024 * 1024):.2f} MB"

def clean_telemetry_file(input_file: Path, output_file: Path) -> None:
    """Loads, cleans, and saves a Single telemetry file."""
    try:
        df = pd.read_parquet(input_file)
        initial_rows = len(df)
        if initial_rows == 0:
            logger.warning(f"File {input_file.name} is empty. Skipping.")
            return

        # 1. Essential Columns Check
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

        # 2. Drop Rows Missing Critical Core Telemetry
        core_telemetry = ['Speed', 'RPM', 'X', 'Y']
        cols_to_check = [c for c in core_telemetry if c in df.columns]
        if cols_to_check:
            df.dropna(subset=cols_to_check, how='all', inplace=True)
            
        dropped_rows = initial_rows - len(df)

        # 3. Forward Fill Small Gaps (Interpolate missing sensor packets within laps)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill()

        # 4. Outlier Clipping (Domain Specific Caps)
        if 'Speed' in df.columns:
            df['Speed'] = df['Speed'].clip(lower=0, upper=380) # Modern F1 cars max out ~360 km/h
        if 'RPM' in df.columns:
            df['RPM'] = df['RPM'].clip(lower=0, upper=15000)   # V6 Hybrid limit
        if 'Throttle' in df.columns:
            df['Throttle'] = df['Throttle'].clip(lower=0, upper=100)
        if 'Brake' in df.columns:
            df['Brake'] = df['Brake'].clip(lower=0, upper=100)

        # 5. Optimize Memory (Downcast types)
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

        # Save Cleaned Data to Parquet (Better for big data than CSV)
        df.to_parquet(output_file, compression='snappy')
        
        logger.info(f"Cleaned {input_file.name}: Dropped {dropped_rows} rows. Saved to clean_data/.")
        
        # Cleanup memory immediately
        del df
        gc.collect()

    except Exception as e:
        logger.error(f"Error processing {input_file.name}: {e}")

def main() -> None:
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
