import pandas as pd
import os
from pathlib import Path
import fastf1

# Constants
DATA_LAKE_DIR = Path("data_lake/season_data")
CACHE_DIR = Path("cache")

def get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def inspect_parquet_file(file_path):
    """Reads and prints summary of a parquet file."""
    try:
        df = pd.read_parquet(file_path)
        print(f"\n--- Inspecting: {file_path.name} ---")
        print(f"Shape: {df.shape} (Rows, Columns)")
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nData Types:")
        print(df.dtypes)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def main():
    if not DATA_LAKE_DIR.exists():
        print(f"Directory {DATA_LAKE_DIR} does not exist. Run the downloader first.")
        return

    # 1. Verification: Check total size and file count
    files = list(DATA_LAKE_DIR.glob("*.parquet"))
    total_size_bytes = get_dir_size(DATA_LAKE_DIR)
    total_size_mb = total_size_bytes / (1024 * 1024)
    total_size_gb = total_size_mb / 1024

    print(f"=== Data Lake Status ===")
    print(f"Location: {DATA_LAKE_DIR.absolute()}")
    print(f"Total Parquet Files: {len(files)}")
    print(f"Total Size: {total_size_mb:.2f} MB ({total_size_gb:.2f} GB)")

    if not files:
        print("No parquet files found.")
    else:
        # 2. Open a Parquet file
        print(f"\nFound {len(files)} files. Inspecting the latest one...")
        # Sort by modification time to get the latest
        latest_file = max(files, key=os.path.getmtime)
        inspect_parquet_file(latest_file)

    # 3. Explain .ff1pkl files
    print("\n=== About .ff1pkl Files ===")
    if CACHE_DIR.exists():
        cache_size = get_dir_size(CACHE_DIR) / (1024 * 1024)
        print(f"Cache Directory: {CACHE_DIR.absolute()}")
        print(f"Cache Size: {cache_size:.2f} MB")
        print("NOTE: .ff1pkl files are internal FastF1 cache files (Python pickles).")
        print("You do not need to open them manually. FastF1 uses them automatically to speed up data loading.")
        print("To 'see' them, just trust that FastF1 is working if your downloads are fast on repeat runs!")
    else:
        print("Cache directory not found.")

if __name__ == "__main__":
    main()
