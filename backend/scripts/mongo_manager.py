"""
================================================================================
  ApexHunter - Big Data Integration
  Script: mongo_manager.py
--------------------------------------------------------------------------------
  Purpose : Manage MongoDB storage for ApexHunter mistake annotations and
            telemetry session metadata.  Supports uploading, querying, and
            status log generation consumed by the Streamlit Big Data tab.

  Usage   : python mongo_manager.py --upload --hdfs --force
            python mongo_manager.py --stats
            python mongo_manager.py --query mistakes
================================================================================
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from utils import DATA_LAKE_DIR, IST, setup_logger

# ── Configuration ────────────────────────────────────────────────────────────

MONGO_URI: str = "mongodb://localhost:27017/"
MONGO_DB_NAME: str = "apexhunter"
MONGO_TIMEOUT_MS: int = 3000
MONGO_STATUS_LOG: Path = DATA_LAKE_DIR / "mongo_status.json"
HADOOP_CMD: str = "hadoop.cmd" if os.name == "nt" else "hadoop"
HDFS_SEASON_PATH: str = "/apexhunter/season_data"
HDFS_CLEAN_PATH: str = "/apexhunter/clean_data"
SESSION_FILE_PATTERN: str = r"(\d{4})_(\d+)_([a-zA-Z0-9]+)\.parquet"

logger = setup_logger(__name__)


# ── Connection Helpers ───────────────────────────────────────────────────────


def get_client() -> MongoClient:
    """Create a MongoClient, ping the server, and return it.

    The client uses serverSelectionTimeoutMS=3000 so that callers fail
    fast when MongoDB is unreachable.

    Returns:
        A connected MongoClient instance.

    Raises:
        ConnectionFailure: If MongoDB does not respond within the timeout.
    """
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=MONGO_TIMEOUT_MS)
    client.admin.command("ping")
    return client


def check_mongo_available() -> bool:
    """Check whether MongoDB is reachable.

    Returns:
        True if a ping succeeds, False otherwise.
    """
    try:
        get_client()
        return True
    except Exception as e:
        logger.error(f"MongoDB not available: {e}")
        return False


def _safe_float(val: Any) -> float:
    """Convert a value to float, returning 0.0 for NaN or non-numeric types.

    Handles pandas NaN values and arbitrary non-numeric inputs safely
    without raising exceptions.

    Args:
        val: Any value to attempt float conversion on.

    Returns:
        The float representation, or 0.0 if conversion fails or result is NaN.
    """
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else f
    except (ValueError, TypeError):
        return 0.0


# ── Upload Logic ─────────────────────────────────────────────────────────────


def upload_mistake_outputs(force: bool = False) -> Dict[str, int]:
    """Upload mistake annotation parquets and their metadata to MongoDB.

    Each parquet file is paired with a ``*_meta.json`` sidecar.  Documents
    are upserted into the ``mistake_annotations`` collection keyed by
    ``{session_file, driver}``.

    Args:
        force: If True, overwrite existing documents; otherwise skip them.

    Returns:
        Dict with counts of inserted, updated, skipped, and failed documents.
    """
    stats: Dict[str, int] = {"inserted": 0, "updated": 0, "skipped": 0, "failed": 0}
    try:
        client = get_client()
        db = client[MONGO_DB_NAME]
        coll = db["mistake_annotations"]

        mistake_dir = DATA_LAKE_DIR / "mistake_data"
        if not mistake_dir.exists():
            return stats

        parquets = list(mistake_dir.glob("*_mistakes.parquet"))
        for p in parquets:
            meta_path = mistake_dir / f"{p.stem}_meta.json"
            if not meta_path.exists():
                continue

            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                filter_key = {
                    "session_file": meta.get("session_file"),
                    "driver": meta.get("driver"),
                }

                if not force:
                    existing = coll.find_one(filter_key)
                    if existing:
                        stats["skipped"] += 1
                        continue

                df = pd.read_parquet(p)

                summary = {
                    "mean_anomaly_score": (
                        _safe_float(df["anomaly_score"].mean())
                        if "anomaly_score" in df.columns and not df.empty else 0.0
                    ),
                    "std_anomaly_score": (
                        _safe_float(df["anomaly_score"].std())
                        if "anomaly_score" in df.columns and not df.empty else 0.0
                    ),
                    "mistake_rows": (
                        int(df["is_mistake"].sum())
                        if "is_mistake" in df.columns else 0
                    ),
                    "total_rows": len(df),
                    "mistake_rate_pct": (
                        _safe_float(round(float(df["is_mistake"].sum()) / len(df) * 100, 2))
                        if "is_mistake" in df.columns and len(df) > 0 else 0.0
                    ),
                    "mean_speed_kmh": (
                        _safe_float(df["Speed"].mean())
                        if "Speed" in df.columns and not df.empty else 0.0
                    ),
                    "max_speed_kmh": (
                        _safe_float(df["Speed"].max())
                        if "Speed" in df.columns and not df.empty else 0.0
                    ),
                    "mean_speed_at_mistake": (
                        _safe_float(df[df["is_mistake"]]["Speed"].mean())
                        if "is_mistake" in df.columns and df["is_mistake"].any() and "Speed" in df.columns
                        else 0.0
                    ),
                }

                doc = {
                    **meta,
                    **summary,
                    "source_parquet": str(p.name),
                    "uploaded_at": datetime.now(IST).isoformat(),
                }

                res = coll.update_one(filter_key, {"$set": doc}, upsert=True)
                if res.upserted_id:
                    stats["inserted"] += 1
                elif res.modified_count > 0:
                    stats["updated"] += 1
                else:
                    stats["skipped"] += 1

            except Exception as e:
                logger.error(f"Failed to process {p.name}: {e}")
                stats["failed"] += 1

        return stats
    except Exception as e:
        logger.error(f"Error in upload_mistake_outputs: {e}")
        return stats


def _list_hdfs_files(hdfs_path: str) -> List[Dict[str, Any]]:
    """List files in an HDFS directory.

    Args:
        hdfs_path: HDFS directory path to list.

    Returns:
        List of dicts, each containing name, size_bytes, and hdfs_path.
    """
    # hadoop fs -ls <hdfs_path>
    result = subprocess.run(
        [HADOOP_CMD, "fs", "-ls", hdfs_path],
        capture_output=True, text=True, shell=False,
    )
    files: List[Dict[str, Any]] = []
    if result.returncode != 0:
        logger.error(f"Failed to list HDFS path {hdfs_path}: {result.stderr.strip()}")
        return files

    for line in result.stdout.strip().split("\n"):
        parts = line.split()
        # HDFS -ls format: perms replication user group size date time path
        if len(parts) >= 8 and parts[0].startswith("-"):
            full_path = parts[-1]
            size_bytes = int(parts[4])
            name = full_path.split("/")[-1]
            files.append({"name": name, "size_bytes": size_bytes, "hdfs_path": full_path})

    return files


def _get_hdfs_dir_size(hdfs_path: str) -> int:
    """Get total size in bytes of an HDFS directory.

    Args:
        hdfs_path: HDFS directory path to measure.

    Returns:
        Total size in bytes, or 0 if the command fails.
    """
    # hadoop fs -du -s <hdfs_path>
    result = subprocess.run(
        [HADOOP_CMD, "fs", "-du", "-s", hdfs_path],
        capture_output=True, text=True, shell=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        try:
            return int(result.stdout.split()[0])
        except (ValueError, IndexError):
            pass
    return 0


def upload_session_metadata(use_hdfs: bool = False) -> Dict[str, int]:
    """Upload telemetry session metadata to the telemetry_sessions collection.

    Parses session filenames using SESSION_FILE_PATTERN to extract year, round,
    and session type. Supports both HDFS and local filesystem sources.

    Args:
        use_hdfs: If True, read session files from HDFS; otherwise use local.

    Returns:
        Dict with counts of inserted, updated, skipped, and failed documents.
    """
    stats: Dict[str, int] = {"inserted": 0, "updated": 0, "skipped": 0, "failed": 0}
    try:
        client = get_client()
        db = client[MONGO_DB_NAME]
        coll = db["telemetry_sessions"]

        clean_data_size_bytes = 0

        if use_hdfs:
            hdfs_files = _list_hdfs_files(HDFS_SEASON_PATH)
            if not hdfs_files:
                logger.warning("No files found in HDFS season_data")
                return stats

            clean_data_size_bytes = _get_hdfs_dir_size(HDFS_CLEAN_PATH)
            logger.info(f"HDFS clean_data total size: {round(clean_data_size_bytes / (1024 * 1024), 2)} MB")
            logger.info(f"Found {len(hdfs_files)} session files in HDFS")

            for hf in hdfs_files:
                try:
                    filename = hf["name"]
                    year, round_num, session_type = 0, 0, ""
                    match = re.match(SESSION_FILE_PATTERN, filename)
                    if match:
                        year = int(match.group(1))
                        round_num = int(match.group(2))
                        session_type = match.group(3)

                    filter_key = {"session_file": filename}
                    doc = {
                        "session_file": filename,
                        "year": year,
                        "round": round_num,
                        "session_type": session_type,
                        "file_size_mb": round(hf["size_bytes"] / (1024 * 1024), 2),
                        "source": "hdfs",
                        "hdfs_raw_path": f"hdfs://localhost:9000{HDFS_SEASON_PATH}/{filename}",
                        "hdfs_clean_path": f"hdfs://localhost:9000{HDFS_CLEAN_PATH}",
                        "uploaded_at": datetime.now(IST).isoformat(),
                    }

                    res = coll.update_one(filter_key, {"$set": doc}, upsert=True)
                    if res.upserted_id:
                        stats["inserted"] += 1
                    elif res.modified_count > 0:
                        stats["updated"] += 1
                    else:
                        stats["skipped"] += 1
                except Exception as e:
                    logger.error(f"Failed to process HDFS file {hf['name']}: {e}")
                    stats["failed"] += 1
        else:
            clean_dir = DATA_LAKE_DIR / "clean_data"
            if not clean_dir.exists():
                return stats

            for p in clean_dir.glob("*.parquet"):
                try:
                    filename = p.name
                    year, round_num, session_type = 0, 0, ""
                    match = re.match(SESSION_FILE_PATTERN, filename)
                    if match:
                        year = int(match.group(1))
                        round_num = int(match.group(2))
                        session_type = match.group(3)

                    filter_key = {"session_file": filename}
                    doc = {
                        "session_file": filename,
                        "year": year,
                        "round": round_num,
                        "session_type": session_type,
                        "file_size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                        "source": "local",
                        "uploaded_at": datetime.now(IST).isoformat(),
                    }

                    res = coll.update_one(filter_key, {"$set": doc}, upsert=True)
                    if res.upserted_id:
                        stats["inserted"] += 1
                    elif res.modified_count > 0:
                        stats["updated"] += 1
                    else:
                        stats["skipped"] += 1
                except Exception as e:
                    logger.error(f"Failed to process session file {p.name}: {e}")
                    stats["failed"] += 1

        return stats
    except Exception as e:
        logger.error(f"Error in upload_session_metadata: {e}")
        return stats


# ── Query Functions ──────────────────────────────────────────────────────────


def get_mistake_leaderboard(db: Any) -> List[Dict[str, Any]]:
    """Return all mistake annotations sorted by mistake rate descending.

    Args:
        db: A pymongo Database instance for the apexhunter database.

    Returns:
        List of dicts, each containing: driver, session_file,
        mistake_rate_pct, mean_anomaly_score, total_rows.
    """
    coll = db["mistake_annotations"]
    cursor = coll.find(
        {},
        {"driver": 1, "session_file": 1, "mistake_rate_pct": 1,
         "mean_anomaly_score": 1, "total_rows": 1, "_id": 0},
    )
    return sorted(list(cursor), key=lambda x: (x.get("mistake_rate_pct") or 0), reverse=True)


def get_anomaly_score_distribution(db: Any) -> Dict[str, List[float]]:
    """Return the distribution of mean anomaly scores across all annotations.

    Args:
        db: A pymongo Database instance for the apexhunter database.

    Returns:
        Dict with a single key "scores" containing a list of float values.
    """
    coll = db["mistake_annotations"]
    cursor = coll.find({}, {"mean_anomaly_score": 1, "_id": 0})
    scores = [
        doc.get("mean_anomaly_score")
        for doc in cursor
        if doc.get("mean_anomaly_score") is not None
    ]
    return {"scores": scores}


def get_session_summary(db: Any) -> Dict[str, Any]:
    """Return aggregate session and mistake statistics.

    Args:
        db: A pymongo Database instance for the apexhunter database.

    Returns:
        Dict containing: total_sessions, total_drivers_analyzed,
        overall_mistake_rate_pct, sessions_by_type (with Q and R counts).
    """
    col_sessions = db["telemetry_sessions"]
    col_mistakes = db["mistake_annotations"]

    total_sessions = col_sessions.count_documents({})
    drivers = col_mistakes.distinct("driver")
    total_drivers = len(drivers)

    pipeline = [
        {"$group": {"_id": None, "avg_mistake_rate": {"$avg": "$mistake_rate_pct"}}}
    ]
    res = list(col_mistakes.aggregate(pipeline))
    overall_mistake_rate_pct = (
        res[0].get("avg_mistake_rate")
        if res and res[0].get("avg_mistake_rate") is not None
        else 0.0
    )

    q_count = col_sessions.count_documents({"session_type": "Q"})
    r_count = col_sessions.count_documents({"session_type": "R"})

    return {
        "total_sessions": total_sessions,
        "total_drivers_analyzed": total_drivers,
        "overall_mistake_rate_pct": round(overall_mistake_rate_pct, 2) if overall_mistake_rate_pct else 0.0,
        "sessions_by_type": {"Q": q_count, "R": r_count},
    }


def get_mistakes_by_driver(db: Any, driver: str) -> List[Dict[str, Any]]:
    """Return all mistake annotations for a specific driver.

    Args:
        db: A pymongo Database instance for the apexhunter database.
        driver: The driver identifier string to filter on.

    Returns:
        List of annotation document dicts (without _id), sorted by session_file.
    """
    coll = db["mistake_annotations"]
    cursor = coll.find({"driver": driver}, {"_id": 0}).sort("session_file", 1)
    return list(cursor)


# ── Status Log ───────────────────────────────────────────────────────────────


def write_status_log(mongo_available: bool, upload_summary: Optional[Dict] = None) -> None:
    """Write the MongoDB status JSON log consumed by the Streamlit Big Data tab.

    A failure to write this log is logged but never raises, so that a log
    write failure cannot crash the pipeline.

    Args:
        mongo_available: Whether MongoDB was reachable at the time of this run.
        upload_summary: Results from the most recent upload operation (if any).
    """
    doc_count_mistakes = 0
    doc_count_sessions = 0
    last_updated_mistakes: Optional[str] = None
    last_updated_sessions: Optional[str] = None

    if mongo_available:
        try:
            client = get_client()
            db = client[MONGO_DB_NAME]

            col_m = db["mistake_annotations"]
            doc_count_mistakes = col_m.count_documents({})
            last_m = col_m.find_one(sort=[("uploaded_at", -1)])
            if last_m:
                last_updated_mistakes = last_m.get("uploaded_at")

            col_s = db["telemetry_sessions"]
            doc_count_sessions = col_s.count_documents({})
            last_s = col_s.find_one(sort=[("uploaded_at", -1)])
            if last_s:
                last_updated_sessions = last_s.get("uploaded_at")
        except Exception:
            pass

    data = {
        "generated_at": datetime.now(IST).isoformat(),
        "mongo_available": mongo_available,
        "database": MONGO_DB_NAME,
        "collections": {
            "mistake_annotations": {"document_count": doc_count_mistakes, "last_updated": last_updated_mistakes},
            "telemetry_sessions":  {"document_count": doc_count_sessions, "last_updated": last_updated_sessions},
        },
        "last_upload": upload_summary or {},
    }

    try:
        DATA_LAKE_DIR.mkdir(parents=True, exist_ok=True)
        with open(MONGO_STATUS_LOG, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Wrote MongoDB status log to {MONGO_STATUS_LOG}")
    except Exception as e:
        logger.error(f"Failed to write MongoDB status log: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse CLI arguments and execute the appropriate MongoDB pipeline actions."""
    parser = argparse.ArgumentParser(description="MongoDB Manager for ApexHunter")
    parser.add_argument("--upload", action="store_true", help="Upload data to MongoDB")
    parser.add_argument("--hdfs", action="store_true", help="Read session data from HDFS instead of local")
    parser.add_argument("--force", action="store_true", help="Overwrite existing documents")
    parser.add_argument("--stats", action="store_true", help="Generate MongoDB stats and JSON log")
    parser.add_argument("--query", choices=["mistakes"], help="Query data and print to terminal")

    args = parser.parse_args()

    is_avail = check_mongo_available()
    if not is_avail:
        logger.error("MongoDB is not available. Aborting.")
        if args.stats or args.upload:
            write_status_log(False, {})
        sys.exit(1)

    upload_summary: Dict[str, int] = {}
    try:
        if args.upload:
            logger.info("Uploading mistake annotations...")
            mistakes_res = upload_mistake_outputs(args.force)
            logger.info(f"Uploading session metadata ({'HDFS' if args.hdfs else 'local'} mode)...")
            sessions_res = upload_session_metadata(use_hdfs=args.hdfs)

            upload_summary = {
                "inserted": mistakes_res["inserted"] + sessions_res["inserted"],
                "updated": mistakes_res["updated"] + sessions_res["updated"],
                "skipped": mistakes_res["skipped"] + sessions_res["skipped"],
                "failed": mistakes_res["failed"] + sessions_res["failed"],
            }
            logger.info(f"Upload summary: {upload_summary}")

        if args.query == "mistakes":
            try:
                client = get_client()
                db = client[MONGO_DB_NAME]
                leaderboard = get_mistake_leaderboard(db)
                print(json.dumps(leaderboard, indent=2))
            except Exception as e:
                logger.error(f"Failed to query mistakes: {e}")

    finally:
        if args.upload or args.stats:
            logger.info("Gathering MongoDB stats...")
            write_status_log(is_avail, upload_summary)


if __name__ == "__main__":
    main()
