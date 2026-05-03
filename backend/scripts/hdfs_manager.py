"""
================================================================================
  ApexHunter - Big Data Integration
  Script: hdfs_manager.py
--------------------------------------------------------------------------------
  Purpose : Manage HDFS storage for ApexHunter season, clean, and mistake data.
            Supports uploading, verification, statistics gathering, and status
            log generation consumed by the Streamlit Big Data tab.

  Usage   : python hdfs_manager.py --upload all --force
            python hdfs_manager.py --stats
            python hdfs_manager.py --verify
================================================================================
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils import DATA_LAKE_DIR, IST, setup_logger

# ── Configuration ────────────────────────────────────────────────────────────

HADOOP_CMD: str = "hadoop.cmd" if os.name == "nt" else "hadoop"
HDFS_ROOT: str = "/apexhunter"
HDFS_RAW_DIR: str = f"{HDFS_ROOT}/season_data"
HDFS_CLEAN_DIR: str = f"{HDFS_ROOT}/clean_data"
HDFS_MISTAKE_DIR: str = f"{HDFS_ROOT}/mistake_data"
HDFS_SUBDIRS: List[str] = ["season_data", "clean_data", "mistake_data"]
STATUS_LOG_PATH: Path = DATA_LAKE_DIR / "hdfs_status.json"

logger = setup_logger(__name__)


# ── Connection Helpers ───────────────────────────────────────────────────────


def check_hdfs_available() -> bool:
    """Check whether HDFS is reachable by listing the root directory.

    Returns:
        True if HDFS responds successfully, False otherwise.
    """
    try:
        # hadoop fs -ls /
        result = subprocess.run(
            [HADOOP_CMD, "fs", "-ls", "/"],
            capture_output=True, text=True, shell=False,
        )
        if result.returncode == 0:
            return True
        logger.error(f"HDFS check failed with return code {result.returncode}: {result.stderr}")
        return False
    except Exception as e:
        logger.error(f"Failed to execute hadoop fs command: {e}")
        return False


def _run_hadoop(hadoop_args: List[str]) -> subprocess.CompletedProcess:
    """Execute a hadoop CLI command and return the CompletedProcess result.

    Args:
        hadoop_args: List of arguments to pass after the hadoop binary,
                     e.g. ["fs", "-ls", "/apexhunter"].

    Returns:
        The subprocess.CompletedProcess instance from the execution.
    """
    return subprocess.run(
        [HADOOP_CMD] + hadoop_args,
        capture_output=True, text=True, shell=False,
    )


# ── Directory Management ────────────────────────────────────────────────────


def ensure_hdfs_dirs(base_path: str, subdirs: List[str]) -> None:
    """Create HDFS directories if they do not already exist.

    Args:
        base_path: The HDFS base path (e.g. "/apexhunter").
        subdirs: List of subdirectory names to create under base_path.
    """
    for subdir in subdirs:
        hdfs_path = f"{base_path}/{subdir}"

        # hadoop fs -test -d <path>
        check_res = _run_hadoop(["fs", "-test", "-d", hdfs_path])
        if check_res.returncode != 0:
            logger.info(f"Creating HDFS directory: {hdfs_path}")
            # hadoop fs -mkdir -p <path>
            _run_hadoop(["fs", "-mkdir", "-p", hdfs_path])


# ── Upload Logic ─────────────────────────────────────────────────────────────


def _file_exists_in_hdfs(hdfs_file_path: str) -> bool:
    """Check whether a specific file exists in HDFS.

    Args:
        hdfs_file_path: Full HDFS path to the file.

    Returns:
        True if the file exists, False otherwise.
    """
    # hadoop fs -test -e <file>
    check_res = _run_hadoop(["fs", "-test", "-e", hdfs_file_path])
    return check_res.returncode == 0


def upload_directory(local_dir: Path, hdfs_dir: str, pattern: str, force: bool) -> Dict[str, int]:
    """Upload matching files from a local directory to an HDFS directory.

    Args:
        local_dir: Local filesystem directory to scan for files.
        hdfs_dir: Destination HDFS directory path.
        pattern: Glob pattern to match files (e.g. "*.parquet").
        force: If True, overwrite existing files in HDFS.

    Returns:
        Dict with counts of uploaded, skipped, and failed files.
    """
    stats: Dict[str, int] = {"uploaded": 0, "skipped": 0, "failed": 0}

    if not local_dir.exists():
        logger.warning(f"Local directory {local_dir} does not exist. Skipping.")
        return stats

    for local_file in local_dir.glob(pattern):
        if not local_file.is_file():
            continue

        hdfs_file_path = f"{hdfs_dir}/{local_file.name}"
        file_exists = _file_exists_in_hdfs(hdfs_file_path)

        if file_exists and not force:
            logger.info(f"Skipping {local_file.name}, already exists in HDFS")
            stats["skipped"] += 1
            continue

        cmd = [HADOOP_CMD, "fs", "-copyFromLocal"]
        if force:
            cmd.append("-f")
        cmd.append("-d")
        cmd.extend([str(local_file), hdfs_file_path])

        # hadoop fs -copyFromLocal [-f] -d <local> <hdfs>
        logger.info(f"Uploading {local_file.name} to {hdfs_dir}")
        res = subprocess.run(cmd, capture_output=True, text=True, shell=False)
        if res.returncode == 0:
            stats["uploaded"] += 1
        else:
            logger.error(f"Failed to upload {local_file.name}: {res.stderr}")
            stats["failed"] += 1

    return stats


# ── Storage Statistics ───────────────────────────────────────────────────────


def get_hdfs_dir_stats(hdfs_dir: str) -> Dict[str, object]:
    """Retrieve file listing and total size for an HDFS directory.

    Args:
        hdfs_dir: HDFS directory path to inspect.

    Returns:
        Dict containing exists flag, file_count, total_size_mb, and files list.
    """
    stats: Dict[str, object] = {
        "exists": False,
        "file_count": 0,
        "total_size_mb": 0.0,
        "files": [],
    }

    # hadoop fs -test -d <dir>
    check_res = _run_hadoop(["fs", "-test", "-d", hdfs_dir])
    if check_res.returncode != 0:
        return stats

    stats["exists"] = True

    # hadoop fs -ls <dir>
    ls_res = _run_hadoop(["fs", "-ls", hdfs_dir])
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

    # hadoop fs -du -s <dir>
    du_res = _run_hadoop(["fs", "-du", "-s", hdfs_dir])
    if du_res.returncode == 0 and du_res.stdout.strip():
        try:
            bytes_str = du_res.stdout.split()[0]
            stats["total_size_mb"] = round(int(bytes_str) / (1024 * 1024), 2)
        except (ValueError, IndexError):
            pass

    return stats


def get_all_hdfs_stats() -> Tuple[Dict[str, Dict], Dict[str, object]]:
    """Gather file statistics for all three HDFS directories.

    Returns:
        A tuple of (dir_stats, summary) where dir_stats maps directory names
        to their individual stats dicts, and summary contains aggregate totals.
    """
    dirs = {
        "season_data": HDFS_RAW_DIR,
        "clean_data": HDFS_CLEAN_DIR,
        "mistake_data": HDFS_MISTAKE_DIR,
    }

    dir_stats: Dict[str, Dict] = {}
    total_files = 0
    total_size_mb = 0.0

    for k, v in dirs.items():
        st = get_hdfs_dir_stats(v)
        dir_stats[k] = st
        total_files += st["file_count"]
        total_size_mb += st["total_size_mb"]

    summary: Dict[str, object] = {
        "total_files": total_files,
        "total_size_mb": round(total_size_mb, 2),
        "total_size_gb": round(total_size_mb / 1024, 3),
    }

    return dir_stats, summary


# ── Verification ─────────────────────────────────────────────────────────────


def verify_uploads(local_dir: Path, hdfs_dir: str, pattern: str) -> Dict[str, object]:
    """Compare local files against HDFS to find any missing uploads.

    Args:
        local_dir: Local directory containing source files.
        hdfs_dir: HDFS directory that should mirror the local files.
        pattern: Glob pattern used to match local files.

    Returns:
        Dict with local_count, hdfs_count, missing file list, and in_sync flag.
    """
    res: Dict[str, object] = {
        "local_count": 0,
        "hdfs_count": 0,
        "missing": [],
        "in_sync": False,
    }

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


# ── Status Log ───────────────────────────────────────────────────────────────


def write_status_log(
    hdfs_available: bool,
    dir_stats: Dict[str, Dict],
    summary: Dict[str, object],
    upload_summary: Dict[str, object],
) -> None:
    """Write the HDFS status JSON log consumed by the Streamlit Big Data tab.

    This file is read by ``frontend/components/bigdata_tab.py`` to render live
    HDFS storage metrics in the dashboard.  A failure to write this log is
    logged but never raises to the caller.

    Args:
        hdfs_available: Whether HDFS was reachable at the time of this run.
        dir_stats: Per-directory statistics from get_all_hdfs_stats().
        summary: Aggregate summary dict from get_all_hdfs_stats().
        upload_summary: Results from the most recent upload operation (if any).
    """
    data = {
        "generated_at": datetime.now(IST).isoformat(),
        "hdfs_available": hdfs_available,
        "directories": dir_stats or {
            "season_data":  {"exists": False, "file_count": 0, "total_size_mb": 0.0, "files": []},
            "clean_data":   {"exists": False, "file_count": 0, "total_size_mb": 0.0, "files": []},
            "mistake_data": {"exists": False, "file_count": 0, "total_size_mb": 0.0, "files": []},
        },
        "summary": summary or {
            "total_files": 0,
            "total_size_mb": 0.0,
            "total_size_gb": 0.0,
        },
        "last_upload": upload_summary or {},
    }

    try:
        DATA_LAKE_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATUS_LOG_PATH, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Wrote status log to {STATUS_LOG_PATH}")
    except Exception as e:
        logger.error(f"Failed to write status log: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse CLI arguments and execute the appropriate HDFS pipeline actions."""
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

    ensure_hdfs_dirs(HDFS_ROOT, HDFS_SUBDIRS)

    upload_summary: Dict[str, object] = {}

    try:
        if args.upload:
            targets: List[str] = []
            if args.upload == "all":
                targets = ["raw", "clean", "mistakes"]
            else:
                targets = [args.upload]

            if "raw" in targets:
                logger.info("Uploading raw season data...")
                upload_summary["season_data"] = upload_directory(
                    DATA_LAKE_DIR / "season_data",
                    HDFS_RAW_DIR,
                    "*.parquet",
                    args.force,
                )

            if "clean" in targets:
                logger.info("Uploading clean data...")
                upload_summary["clean_data"] = upload_directory(
                    DATA_LAKE_DIR / "clean_data",
                    HDFS_CLEAN_DIR,
                    "*.parquet",
                    args.force,
                )

            if "mistakes" in targets:
                logger.info("Uploading mistake data...")
                upload_summary["mistake_data"] = upload_directory(
                    DATA_LAKE_DIR / "mistake_data",
                    HDFS_MISTAKE_DIR,
                    "*_mistakes.parquet",
                    args.force,
                )

        if args.verify:
            logger.info("Verifying raw season data...")
            v_raw = verify_uploads(DATA_LAKE_DIR / "season_data", HDFS_RAW_DIR, "*.parquet")
            logger.info(f"Raw: {v_raw}")

            logger.info("Verifying clean data...")
            v_clean = verify_uploads(DATA_LAKE_DIR / "clean_data", HDFS_CLEAN_DIR, "*.parquet")
            logger.info(f"Clean: {v_clean}")

            logger.info("Verifying mistake data...")
            v_mis = verify_uploads(DATA_LAKE_DIR / "mistake_data", HDFS_MISTAKE_DIR, "*_mistakes.parquet")
            logger.info(f"Mistakes: {v_mis}")

    finally:
        if args.upload or args.stats:
            logger.info("Gathering HDFS stats...")
            dir_stats, summary = get_all_hdfs_stats()
            write_status_log(hdfs_available, dir_stats, summary, upload_summary)


if __name__ == "__main__":
    main()
