import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime, timezone

HADOOP_CMD = "hadoop.cmd" if os.name == "nt" else "hadoop"

from utils import setup_logger, DATA_LAKE_DIR

logger = setup_logger(__name__)

def check_hdfs_available() -> bool:
    try:
        result = subprocess.run([HADOOP_CMD, "fs", "-ls", "/"], capture_output=True, text=True, shell=False)
        if result.returncode == 0:
            return True
        logger.error(f"HDFS check failed with return code {result.returncode}: {result.stderr}")
        return False
    except Exception as e:
        logger.error(f"Failed to execute hadoop fs command: {e}")
        return False

def ensure_hdfs_dirs(base_path: str, subdirs: list[str]) -> None:
    for subdir in subdirs:
        hdfs_path = f"{base_path}/{subdir}"
        check_res = subprocess.run([HADOOP_CMD, "fs", "-test", "-d", hdfs_path], capture_output=True, text=True, shell=False)
        if check_res.returncode != 0:
            logger.info(f"Creating HDFS directory: {hdfs_path}")
            subprocess.run([HADOOP_CMD, "fs", "-mkdir", "-p", hdfs_path], capture_output=True, text=True, shell=False)

def upload_directory(local_dir: Path, hdfs_dir: str, pattern: str, force: bool) -> dict:
    stats = {"uploaded": 0, "skipped": 0, "failed": 0}
    if not local_dir.exists():
        logger.warning(f"Local directory {local_dir} does not exist. Skipping.")
        return stats
    
    for local_file in local_dir.glob(pattern):
        if not local_file.is_file():
            continue
        
        hdfs_file_path = f"{hdfs_dir}/{local_file.name}"
        
        # Check if exists
        check_res = subprocess.run([HADOOP_CMD, "fs", "-test", "-e", hdfs_file_path], capture_output=True, text=True, shell=False)
        file_exists = (check_res.returncode == 0)
        
        if file_exists and not force:
            logger.info(f"Skipping {local_file.name}, already exists in HDFS")
            stats["skipped"] += 1
            continue
            
        cmd = [HADOOP_CMD, "fs", "-copyFromLocal"]
        if force:
            cmd.append("-f")
        cmd.append("-d")
        cmd.extend([str(local_file), hdfs_file_path])
        
        logger.info(f"Uploading {local_file.name} to {hdfs_dir}")
        res = subprocess.run(cmd, capture_output=True, text=True, shell=False)
        if res.returncode == 0:
            stats["uploaded"] += 1
        else:
            logger.error(f"Failed to upload {local_file.name}: {res.stderr}")
            stats["failed"] += 1
            
    return stats

def get_hdfs_dir_stats(hdfs_dir: str) -> dict:
    stats = {"exists": False, "file_count": 0, "total_size_mb": 0.0, "files": []}
    
    check_res = subprocess.run([HADOOP_CMD, "fs", "-test", "-d", hdfs_dir], capture_output=True, text=True, shell=False)
    if check_res.returncode != 0:
        return stats
        
    stats["exists"] = True
    
    ls_res = subprocess.run([HADOOP_CMD, "fs", "-ls", hdfs_dir], capture_output=True, text=True, shell=False)
    if ls_res.returncode == 0:
        lines = ls_res.stdout.strip().split("\n")
        for line in lines:
            if line.startswith("Found"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                file_path = parts[-1]
                file_name = file_path.split("/")[-1]
                if file_name:
                    stats["files"].append(file_name)
    
    stats["file_count"] = len(stats["files"])
    
    du_res = subprocess.run([HADOOP_CMD, "fs", "-du", "-s", hdfs_dir], capture_output=True, text=True, shell=False)
    if du_res.returncode == 0 and du_res.stdout.strip():
        try:
            bytes_str = du_res.stdout.split()[0]
            stats["total_size_mb"] = round(int(bytes_str) / (1024 * 1024), 2)
        except (ValueError, IndexError):
            pass
            
    return stats

def get_all_hdfs_stats() -> tuple[dict, dict]:
    dirs = {
        "season_data": "/apexhunter/season_data",
        "clean_data": "/apexhunter/clean_data",
        "mistake_data": "/apexhunter/mistake_data"
    }
    
    dir_stats = {}
    total_files = 0
    total_size_mb = 0.0
    
    for k, v in dirs.items():
        st = get_hdfs_dir_stats(v)
        dir_stats[k] = st
        total_files += st["file_count"]
        total_size_mb += st["total_size_mb"]
        
    summary = {
        "total_files": total_files,
        "total_size_mb": round(total_size_mb, 2),
        "total_size_gb": round(total_size_mb / 1024, 3)
    }
    
    return dir_stats, summary

def write_status_log(hdfs_available: bool, dir_stats: dict, summary: dict, upload_summary: dict) -> None:
    status_path = DATA_LAKE_DIR / "hdfs_status.json"
    
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hdfs_available": hdfs_available,
        "directories": dir_stats or {
            "season_data":  { "exists": False, "file_count": 0, "total_size_mb": 0.0, "files": [] },
            "clean_data":   { "exists": False, "file_count": 0, "total_size_mb": 0.0, "files": [] },
            "mistake_data": { "exists": False, "file_count": 0, "total_size_mb": 0.0, "files": [] }
        },
        "summary": summary or {
            "total_files": 0,
            "total_size_mb": 0.0,
            "total_size_gb": 0.0
        },
        "last_upload": upload_summary or {}
    }
    
    try:
        DATA_LAKE_DIR.mkdir(parents=True, exist_ok=True)
        with open(status_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Wrote status log to {status_path}")
    except Exception as e:
        logger.error(f"Failed to write status log: {e}")

def verify_uploads(local_dir: Path, hdfs_dir: str, pattern: str) -> dict:
    res = {"local_count": 0, "hdfs_count": 0, "missing": [], "in_sync": False}
    
    if not local_dir.exists():
        res["in_sync"] = True
        return res
        
    local_files = [f.name for f in local_dir.glob(pattern) if f.is_file()]
    res["local_count"] = len(local_files)
    
    st = get_hdfs_dir_stats(hdfs_dir)
    hdfs_files = set(st["files"])
    res["hdfs_count"] = len(hdfs_files)
    
    missing = [f for f in local_files if f not in hdfs_files]
    res["missing"] = missing
    res["in_sync"] = (len(missing) == 0 and res["local_count"] == res["hdfs_count"])
    
    return res

def main():
    parser = argparse.ArgumentParser(description="Manage HDFS data for ApexHunter")
    parser.add_argument("--upload", choices=["all", "raw", "clean", "mistakes"], help="Upload data to HDFS")
    parser.add_argument("--stats", action="store_true", help="Generate HDFS stats and JSON log")
    parser.add_argument("--verify", action="store_true", help="Verify local vs HDFS files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files in HDFS")
    
    args = parser.parse_args()
    
    hdfs_available = check_hdfs_available()
    if not hdfs_available:
        logger.error("HDFS is not available. Aborting.")
        # We still write the offline status log if requested
        if args.stats or args.upload:
            write_status_log(False, {}, {}, {})
        sys.exit(1)
        
    base_hdfs_path = "/apexhunter"
    ensure_hdfs_dirs(base_hdfs_path, ["season_data", "clean_data", "mistake_data"])
    
    upload_summary = {}
    
    try:
        if args.upload:
            targets = []
            if args.upload == "all":
                targets = ["raw", "clean", "mistakes"]
            else:
                targets = [args.upload]
                
            if "raw" in targets:
                logger.info("Uploading raw season data...")
                upload_summary["season_data"] = upload_directory(
                    DATA_LAKE_DIR / "season_data", 
                    f"{base_hdfs_path}/season_data", 
                    "*.parquet", 
                    args.force
                )
            if "clean" in targets:
                logger.info("Uploading clean data...")
                upload_summary["clean_data"] = upload_directory(
                    DATA_LAKE_DIR / "clean_data", 
                    f"{base_hdfs_path}/clean_data", 
                    "*.parquet", 
                    args.force
                )
            if "mistakes" in targets:
                logger.info("Uploading mistake data...")
                upload_summary["mistake_data"] = upload_directory(
                    DATA_LAKE_DIR / "mistake_data", 
                    f"{base_hdfs_path}/mistake_data", 
                    "*_mistakes.parquet", 
                    args.force
                )
                
        if args.verify:
            logger.info("Verifying raw season data...")
            v_raw = verify_uploads(DATA_LAKE_DIR / "season_data", f"{base_hdfs_path}/season_data", "*.parquet")
            logger.info(f"Raw: {v_raw}")
            
            logger.info("Verifying clean data...")
            v_clean = verify_uploads(DATA_LAKE_DIR / "clean_data", f"{base_hdfs_path}/clean_data", "*.parquet")
            logger.info(f"Clean: {v_clean}")
            
            logger.info("Verifying mistake data...")
            v_mis = verify_uploads(DATA_LAKE_DIR / "mistake_data", f"{base_hdfs_path}/mistake_data", "*_mistakes.parquet")
            logger.info(f"Mistakes: {v_mis}")
            
    finally:
        if args.upload or args.stats:
            logger.info("Gathering HDFS stats...")
            dir_stats, summary = get_all_hdfs_stats()
            write_status_log(hdfs_available, dir_stats, summary, upload_summary)

if __name__ == "__main__":
    main()
