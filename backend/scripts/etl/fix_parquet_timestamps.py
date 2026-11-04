"""
================================================================================
  ApexHunter - Parquet Timestamp Fixer
  Script: fix_parquet_timestamps.py
--------------------------------------------------------------------------------
  Purpose : Re-writes parquet files in-place with microsecond precision to
            resolve nanosecond timestamp incompatibility with Spark 3.x.

  Usage   : python backend/scripts/etl/fix_parquet_timestamps.py
================================================================================
"""

import sys
from pathlib import Path

# Make backend/scripts importable so this file runs both by path and as `python -m <pkg>.<module>`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq

from utils import DATA_LAKE_DIR, setup_logger

logger = setup_logger(__name__)


def fix_directory(directory: Path) -> int:
    """Fix all parquet files in a directory. Returns count of files fixed.

    Each file is written to a temporary name and then renamed over the original (#49): these are
    the raw downloads, and an in-place write that crashed half-way would corrupt one."""
    files = sorted(directory.glob("*.parquet"))
    if not files:
        logger.warning(f"No parquet files found in {directory}")
        return 0

    fixed = 0
    for f in files:
        try:
            table = pq.read_table(f)
            tmp = f.with_name(f.name + ".tmp")
            pq.write_table(
                table,
                tmp,
                coerce_timestamps="us",
                allow_truncated_timestamps=True,
            )
            tmp.replace(f)
            fixed += 1
            logger.info(f"Fixed: {f.name}")
        except Exception as e:
            logger.error(f"Failed to fix {f.name}: {e}")

    return fixed


def main():
    season_dir = DATA_LAKE_DIR / "season_data"
    logger.info(f"Fixing parquet timestamps in {season_dir}")
    total = len(list(season_dir.glob("*.parquet")))
    count = fix_directory(season_dir)
    logger.info(f"Done — {count} files converted from nanosecond to microsecond timestamps")
    if count < total:
        sys.exit(1)                                  # some files failed (logged above)


if __name__ == "__main__":
    main()
