"""
================================================================================
  ApexHunter - Big Data Integration
  Script: spark_clean_telemetry.py
--------------------------------------------------------------------------------
  Purpose : Apache Spark ETL pipeline that reads raw season telemetry data,
            applies cleaning transformations (null-drop, forward-fill, clip,
            float-cast), and writes partitioned Parquet output.

  Usage   : python spark_clean_telemetry.py --hdfs --force
            python spark_clean_telemetry.py
================================================================================
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, greatest, last, least, lit
from pyspark.sql.types import FloatType

from utils import DATA_LAKE_DIR, IST, setup_logger

# ── Configuration ─────────────────────────────────────────────────────────────
HADOOP_CMD: str = "hadoop.cmd" if os.name == "nt" else "hadoop"
SPARK_APP_NAME: str = "ApexHunter-ETL"
SPARK_DRIVER_MEMORY: str = "4g"
SPARK_SHUFFLE_PARTITIONS: str = "8"
SPARK_HDFS_URI: str = "hdfs://localhost:9000"
SPARK_RUN_LOG_PATH: Path = DATA_LAKE_DIR / "spark_run_log.json"
BYTES_PER_MB: int = 1024 * 1024

logger = setup_logger(__name__)


# ── ETL Pipeline ──────────────────────────────────────────────────────────────
def _create_spark_session() -> SparkSession:
    """Build and return a configured SparkSession for the ETL pipeline.

    Returns:
        A SparkSession with local[*] master, 4g driver memory,
        8 shuffle partitions, and HDFS default filesystem.
    """
    spark = SparkSession.builder \
        .appName(SPARK_APP_NAME) \
        .master("local[*]") \
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS) \
        .config("spark.hadoop.fs.defaultFS", SPARK_HDFS_URI) \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return spark


# ── Status Log ────────────────────────────────────────────────────────────────
def write_run_log(log_data: Dict[str, Any]) -> None:
    """Write the Spark ETL run log JSON consumed by the Streamlit Big Data tab.

    Args:
        log_data: Complete log dictionary assembled by main().
    """
    try:
        DATA_LAKE_DIR.mkdir(parents=True, exist_ok=True)
        with open(SPARK_RUN_LOG_PATH, "w") as f:
            json.dump(log_data, f, indent=2)
        logger.info(f"Wrote Spark run log to {SPARK_RUN_LOG_PATH}")
    except Exception as e:
        logger.error(f"Failed to write Spark run log: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """Parse CLI arguments and execute the Spark ETL cleaning pipeline."""
    parser = argparse.ArgumentParser(description="Spark ETL for ApexHunter")
    parser.add_argument("--hdfs", action="store_true", help="Use HDFS instead of local paths")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")

    args = parser.parse_args()
    use_hdfs = args.hdfs or os.environ.get("USE_HDFS", "0") == "1"

    # Must be set before SparkSession is created — Windows PySpark requirement
    _venv_python = Path(__file__).resolve().parents[2] / ".venv" / "Scripts" / "python.exe"
    _python = str(_venv_python) if _venv_python.exists() else sys.executable
    os.environ["PYSPARK_PYTHON"] = _python
    os.environ["PYSPARK_DRIVER_PYTHON"] = _python

    # Use pyspark's bundled jars (compatible with Java 8)
    _pyspark_home = str(
        Path(__file__).resolve().parents[2] / ".venv" / "Lib" / "site-packages" / "pyspark"
    )
    if Path(_pyspark_home).exists():
        os.environ["SPARK_HOME"] = _pyspark_home
    else:
        os.environ.pop("SPARK_HOME", None)

    if use_hdfs:
        input_path = f"{SPARK_HDFS_URI}/apexhunter/season_data"
        output_path = f"{SPARK_HDFS_URI}/apexhunter/clean_data"
    else:
        input_path = str(DATA_LAKE_DIR / "season_data")
        output_path = str(DATA_LAKE_DIR / "clean_data")

    start_time = time.time()

    log_data: Dict[str, Any] = {
        "run_timestamp": datetime.now(IST).isoformat(),
        "mode": "hdfs" if use_hdfs else "local",
        "input_path": input_path,
        "output_path": output_path,
        "files_processed": 0,
        "rows_input": 0,
        "rows_output": 0,
        "rows_dropped": 0,
        "drop_rate_pct": 0.0,
        "output_size_mb": 0.0,
        "duration_seconds": 0.0,
        "spark_version": "",
        "status": "failed",
        "error": None,
    }

    spark: Optional[SparkSession] = None
    try:
        spark = _create_spark_session()
        log_data["spark_version"] = spark.version

        logger.info(f"Reading from {input_path}")
        df = spark.read.parquet(input_path)

        rows_input = df.count()
        log_data["rows_input"] = rows_input
        logger.info(f"Rows in: {rows_input:,}")

        # Step 2: Drop rows missing all core telemetry simultaneously
        core = ["Speed", "RPM", "X", "Y"]
        existing_core = [c for c in core if c in df.columns]
        if existing_core:
            condition = None
            for c in existing_core:
                cond = col(c).isNull()
                condition = cond if condition is None else condition & cond
            df = df.filter(~condition)

        # Step 3: Forward fill small gaps using last-observation-carry-forward
        numeric_cols = [
            f.name for f in df.schema.fields
            if str(f.dataType) in ("DoubleType", "FloatType", "LongType", "IntegerType")
            and f.name not in ("Round", "Year")
        ]
        window_spec = Window.partitionBy("Driver").orderBy("SessionTime").rowsBetween(
            Window.unboundedPreceding, Window.currentRow
        )
        for c in numeric_cols:
            if c in df.columns:
                df = df.withColumn(c, last(col(c), ignorenulls=True).over(window_spec))

        # Step 5: Downcast to float32 equivalent
        # (must happen before clipping so BOOLEAN cols like Brake become numeric)
        float_cols = ["Speed", "RPM", "Throttle", "Brake", "X", "Y"]
        for c in float_cols:
            if c in df.columns:
                df = df.withColumn(c, col(c).cast(FloatType()))

        # Step 4: Domain-specific outlier clipping
        clips: Dict[str, Tuple[float, float]] = {
            "Speed":    (0.0, 380.0),
            "RPM":      (0.0, 15000.0),
            "Throttle": (0.0, 100.0),
            "Brake":    (0.0, 100.0),
        }
        for c, (lo, hi) in clips.items():
            if c in df.columns:
                df = df.withColumn(c, greatest(least(col(c), lit(hi)), lit(lo)))

        # Write output
        mode = "overwrite" if args.force else "errorifexists"
        logger.info(f"Writing output to {output_path}")
        df.write.mode(mode).partitionBy("Year", "Round", "Session").parquet(output_path)

        rows_output = spark.read.parquet(output_path).count()
        log_data["rows_output"] = rows_output
        logger.info(f"Rows out: {rows_output:,}")

        rows_dropped = rows_input - rows_output
        log_data["rows_dropped"] = rows_dropped

        drop_rate_pct = round(rows_dropped / rows_input * 100, 2) if rows_input > 0 else 0.0
        log_data["drop_rate_pct"] = drop_rate_pct
        logger.info(f"Dropped: {rows_dropped:,} ({drop_rate_pct}%)")

        # Calculate output_size_mb
        output_size_mb = 0.0
        if use_hdfs:
            # hadoop fs -du -s <output_path>
            du_res = subprocess.run(
                [HADOOP_CMD, "fs", "-du", "-s", output_path],
                capture_output=True, text=True, shell=False,
            )
            if du_res.returncode == 0 and du_res.stdout.strip():
                try:
                    bytes_str = du_res.stdout.split()[0]
                    output_size_mb = round(int(bytes_str) / BYTES_PER_MB, 2)
                except (ValueError, IndexError):
                    pass

        else:
            out_dir = Path(output_path)
            if out_dir.exists():
                total_bytes = sum(
                    f.stat().st_size for f in out_dir.glob("**/*") if f.is_file()
                )
                output_size_mb = round(total_bytes / BYTES_PER_MB, 2)

        log_data["output_size_mb"] = output_size_mb
        log_data["duration_seconds"] = round(time.time() - start_time, 2)
        logger.info(f"Duration: {log_data['duration_seconds']}s")

        log_data["status"] = "success"

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Spark ETL failed: {e}")
        log_data["error"] = str(e)
        log_data["duration_seconds"] = round(time.time() - start_time, 2)

    finally:
        if spark is not None:
            spark.stop()
            # NOTE: Windows-only Spark temp dir cleanup error may appear here — harmless.

        write_run_log(log_data)


if __name__ == "__main__":
    main()
