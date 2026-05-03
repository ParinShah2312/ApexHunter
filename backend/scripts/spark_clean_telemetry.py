import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, last, least, greatest, lit
from pyspark.sql.types import FloatType

from utils import setup_logger, DATA_LAKE_DIR

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Spark ETL for ApexHunter")
    parser.add_argument("--hdfs", action="store_true", help="Use HDFS instead of local paths")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    
    args = parser.parse_args()
    
    use_hdfs = args.hdfs or os.environ.get("USE_HDFS", "0") == "1"
    
    # Resolve the correct python executable — prefer the venv if it exists
    _venv_python = Path(__file__).resolve().parents[2] / ".venv" / "Scripts" / "python.exe"
    _python = str(_venv_python) if _venv_python.exists() else sys.executable
    os.environ["PYSPARK_PYTHON"] = _python
    os.environ["PYSPARK_DRIVER_PYTHON"] = _python
    # Use pyspark's bundled jars (compatible with Java 8) instead of C:\Spark (Java 17)
    _pyspark_home = str(Path(__file__).resolve().parents[2] / ".venv" / "Lib" / "site-packages" / "pyspark")
    if Path(_pyspark_home).exists():
        os.environ["SPARK_HOME"] = _pyspark_home
    else:
        os.environ.pop("SPARK_HOME", None)
    
    if use_hdfs:
        input_path = "hdfs://localhost:9000/apexhunter/season_data"
        output_path = "hdfs://localhost:9000/apexhunter/clean_data"
    else:
        input_path = str(DATA_LAKE_DIR / "season_data")
        output_path = str(DATA_LAKE_DIR / "clean_data")
        
    start_time = time.time()
    
    log_data = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
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
        "error": None
    }
    
    spark = None
    try:
        spark = SparkSession.builder \
            .appName("ApexHunter-ETL") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "8") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
            .getOrCreate()
            
        spark.sparkContext.setLogLevel("WARN")
        log_data["spark_version"] = spark.version
        
        logger.info(f"Reading from {input_path}")
        df = spark.read.parquet(input_path)
        
        # We can't trivially get files_processed from the df in Spark without reading input_file_name,
        # but we can do a rough estimate or leave it as 0. Let's just grab the count.
        # To get files_processed:
        # files = df.select(pyspark.sql.functions.input_file_name()).distinct().count()
        # but that takes an action. We'll skip for performance unless needed.
        
        rows_input = df.count()
        log_data["rows_input"] = rows_input
        logger.info(f"Rows in: {rows_input}")
        
        # a. Drop rows where ALL of Speed, RPM, X, Y are null simultaneously
        core = ["Speed", "RPM", "X", "Y"]
        existing_core = [c for c in core if c in df.columns]
        if existing_core:
            condition = None
            for c in existing_core:
                cond = col(c).isNull()
                condition = cond if condition is None else condition & cond
            df = df.filter(~condition)
            
        # b. Forward fill numeric columns within each Driver partition
        numeric_cols = [f.name for f in df.schema.fields
                        if str(f.dataType) in ("DoubleType", "FloatType", "LongType", "IntegerType")
                        and f.name not in ("Round", "Year")]
        
        window_spec = Window.partitionBy("Driver").orderBy("SessionTime").rowsBetween(
            Window.unboundedPreceding, Window.currentRow
        )
        for c in numeric_cols:
            if c in df.columns:
                df = df.withColumn(c, last(col(c), ignorenulls=True).over(window_spec))
                
        # c. Cast to float (must happen before clipping so BOOLEAN cols like Brake become numeric)
        float_cols = ["Speed", "RPM", "Throttle", "Brake", "X", "Y"]
        for c in float_cols:
            if c in df.columns:
                df = df.withColumn(c, col(c).cast(FloatType()))
                
        # d. Outlier clipping
        clips = {
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
        logger.info(f"Rows out: {rows_output}")
        
        rows_dropped = rows_input - rows_output
        log_data["rows_dropped"] = rows_dropped
        
        drop_rate_pct = round(rows_dropped / rows_input * 100, 2) if rows_input > 0 else 0.0
        log_data["drop_rate_pct"] = drop_rate_pct
        
        logger.info(f"Dropped: {rows_dropped} ({drop_rate_pct}%)")
        
        # Calculate output_size_mb
        output_size_mb = 0.0
        if use_hdfs:
            import subprocess
            hadoop_cmd = "hadoop.cmd" if os.name == "nt" else "hadoop"
            du_res = subprocess.run([hadoop_cmd, "fs", "-du", "-s", output_path], capture_output=True, text=True, shell=False)
            if du_res.returncode == 0 and du_res.stdout.strip():
                try:
                    bytes_str = du_res.stdout.split()[0]
                    output_size_mb = round(int(bytes_str) / (1024 * 1024), 2)
                except (ValueError, IndexError):
                    pass
        else:
            out_dir = Path(output_path)
            if out_dir.exists():
                total_bytes = sum(f.stat().st_size for f in out_dir.glob("**/*") if f.is_file())
                output_size_mb = round(total_bytes / (1024 * 1024), 2)
                
        log_data["output_size_mb"] = output_size_mb
        
        duration = time.time() - start_time
        log_data["duration_seconds"] = round(duration, 2)
        logger.info(f"Duration: {log_data['duration_seconds']}s")
        
        log_data["status"] = "success"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Spark ETL failed: {e}")
        log_data["error"] = str(e)
        duration = time.time() - start_time
        log_data["duration_seconds"] = round(duration, 2)
        
    finally:
        if spark is not None:
            spark.stop()
            
        log_path = DATA_LAKE_DIR / "spark_run_log.json"
        try:
            DATA_LAKE_DIR.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(log_data, f, indent=2)
            logger.info(f"Wrote Spark run log to {log_path}")
        except Exception as e:
            logger.error(f"Failed to write Spark run log: {e}")

if __name__ == "__main__":
    main()
