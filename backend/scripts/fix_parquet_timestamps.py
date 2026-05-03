"""
================================================================================
  ApexHunter - Parquet Timestamp Fixer
  Script: fix_parquet_timestamps.py
--------------------------------------------------------------------------------
  Purpose : Re-writes parquet files in-place with microsecond precision to
            resolve nanosecond timestamp incompatibility with Spark 3.x.

  Usage   : python backend/scripts/fix_parquet_timestamps.py
================================================================================
"""

import sys
from pathlib import Path

# Add parent to path for utils import
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq

from utils import DATA_LAKE_DIR, setup_logger

logger = setup_logger(__name__)


def fix_directory(directory: Path) -> int:
    """Fix all parquet files in a directory. Returns count of files fixed."""
    files = sorted(directory.glob("*.parquet"))
    if not files:
        logger.warning(f"No parquet files found in {directory}")
        return 0

    fixed = 0
    for f in files:
        try:
            table = pq.read_table(f)
            pq.write_table(
                table,
                f,
                coerce_timestamps="us",
                allow_truncated_timestamps=True,
            )
            fixed += 1
            logger.info(f"Fixed: {f.name}")
        except Exception as e:
            logger.error(f"Failed to fix {f.name}: {e}")

    return fixed


def main():
    season_dir = DATA_LAKE_DIR / "season_data"
    logger.info(f"Fixing parquet timestamps in {season_dir}")
    count = fix_directory(season_dir)
    logger.info(f"Done — {count} files converted from nanosecond to microsecond timestamps")


if __name__ == "__main__":
    main()
