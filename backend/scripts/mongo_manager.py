import argparse
import json
import os
import re
import subprocess
import sys
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from utils import setup_logger, DATA_LAKE_DIR

logger = setup_logger(__name__)

HADOOP_CMD = "hadoop.cmd" if os.name == "nt" else "hadoop"
HDFS_SEASON_PATH = "/apexhunter/season_data"
HDFS_CLEAN_PATH = "/apexhunter/clean_data"

def get_client() -> MongoClient:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    return client

def check_mongo_available() -> bool:
    try:
        get_client()
        return True
    except Exception as e:
        logger.error(f"MongoDB not available: {e}")
        return False

def upload_mistake_outputs(force: bool = False) -> dict:
    stats = {"inserted": 0, "updated": 0, "skipped": 0, "failed": 0}
    try:
        client = get_client()
        db = client["apexhunter"]
        col = db["mistake_annotations"]
        
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
                    
                filter_key = {"session_file": meta.get("session_file"), "driver": meta.get("driver")}
                
                if not force:
                    existing = col.find_one(filter_key)
                    if existing:
                        stats["skipped"] += 1
                        continue
                        
                df = pd.read_parquet(p)
                
                def _safe_float(val):
                    try:
                        f = float(val)
                        return 0.0 if math.isnan(f) else f
                    except (ValueError, TypeError):
                        return 0.0

                summary = {
                    "mean_anomaly_score": _safe_float(df["anomaly_score"].mean()) if "anomaly_score" in df.columns and not df.empty else 0.0,
                    "std_anomaly_score":  _safe_float(df["anomaly_score"].std()) if "anomaly_score" in df.columns and not df.empty else 0.0,
                    "mistake_rows": int(df["is_mistake"].sum()) if "is_mistake" in df.columns else 0,
                    "total_rows": len(df),
                    "mistake_rate_pct": _safe_float(round(float(df["is_mistake"].sum()) / len(df) * 100, 2)) if "is_mistake" in df.columns and len(df) > 0 else 0.0,
                    "mean_speed_kmh": _safe_float(df["Speed"].mean()) if "Speed" in df.columns and not df.empty else 0.0,
                    "max_speed_kmh": _safe_float(df["Speed"].max()) if "Speed" in df.columns and not df.empty else 0.0,
                    "mean_speed_at_mistake": _safe_float(df[df["is_mistake"]]["Speed"].mean()) if "is_mistake" in df.columns and df["is_mistake"].any() and "Speed" in df.columns else 0.0,
                }
                
                doc = {
                    **meta,
                    **summary,
                    "source_parquet": str(p.name),
                    "uploaded_at": datetime.now(timezone.utc).isoformat()
                }
                
                res = col.update_one(filter_key, {"$set": doc}, upsert=True)
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

def _list_hdfs_files(hdfs_path: str) -> list:
    """List files in an HDFS directory. Returns list of {name, size_bytes, hdfs_path} dicts."""
    result = subprocess.run(
        [HADOOP_CMD, "fs", "-ls", hdfs_path],
        capture_output=True, text=True, shell=False
    )
    files = []
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
    """Get total size in bytes of an HDFS directory."""
    result = subprocess.run(
        [HADOOP_CMD, "fs", "-du", "-s", hdfs_path],
        capture_output=True, text=True, shell=False
    )
    if result.returncode == 0 and result.stdout.strip():
        try:
            return int(result.stdout.split()[0])
        except (ValueError, IndexError):
            pass
    return 0


def upload_session_metadata(use_hdfs: bool = False) -> dict:
    stats = {"inserted": 0, "updated": 0, "skipped": 0, "failed": 0}
    try:
        client = get_client()
        db = client["apexhunter"]
        col = db["telemetry_sessions"]

        # Get the total size of the Spark-cleaned output
        clean_data_size_bytes = 0

        if use_hdfs:
            # List individual session files from HDFS season_data (named files)
            hdfs_files = _list_hdfs_files(HDFS_SEASON_PATH)
            if not hdfs_files:
                logger.warning("No files found in HDFS season_data")
                return stats
            clean_data_size_bytes = _get_hdfs_dir_size(HDFS_CLEAN_PATH)
            logger.info(f"HDFS clean_data total size: {round(clean_data_size_bytes / (1024*1024), 2)} MB")
            logger.info(f"Found {len(hdfs_files)} session files in HDFS")

            for hf in hdfs_files:
                try:
                    filename = hf["name"]
                    year, round_num, session_type = 0, 0, ""
                    match = re.match(r"(\d{4})_(\d+)_([a-zA-Z0-9]+)\.parquet", filename)
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
                        "uploaded_at": datetime.now(timezone.utc).isoformat()
                    }

                    res = col.update_one(filter_key, {"$set": doc}, upsert=True)
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
            # Local mode — scan local clean_data directory
            clean_dir = DATA_LAKE_DIR / "clean_data"
            if not clean_dir.exists():
                return stats

            for p in clean_dir.glob("*.parquet"):
                try:
                    filename = p.name
                    year, round_num, session_type = 0, 0, ""
                    match = re.match(r"(\d{4})_(\d+)_([a-zA-Z0-9]+)\.parquet", filename)
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
                        "uploaded_at": datetime.now(timezone.utc).isoformat()
                    }

                    res = col.update_one(filter_key, {"$set": doc}, upsert=True)
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

def get_mistake_leaderboard(db) -> list:
    col = db["mistake_annotations"]
    cursor = col.find({}, {"driver": 1, "session_file": 1, "mistake_rate_pct": 1, "mean_anomaly_score": 1, "total_rows": 1, "_id": 0})
    return sorted(list(cursor), key=lambda x: (x.get("mistake_rate_pct") or 0), reverse=True)

def get_anomaly_score_distribution(db) -> dict:
    col = db["mistake_annotations"]
    cursor = col.find({}, {"mean_anomaly_score": 1, "_id": 0})
    scores = [doc.get("mean_anomaly_score") for doc in cursor if doc.get("mean_anomaly_score") is not None]
    return {"scores": scores}

def get_session_summary(db) -> dict:
    col_sessions = db["telemetry_sessions"]
    col_mistakes = db["mistake_annotations"]
    
    total_sessions = col_sessions.count_documents({})
    
    drivers = col_mistakes.distinct("driver")
    total_drivers = len(drivers)
    
    pipeline = [
        {"$group": {"_id": None, "avg_mistake_rate": {"$avg": "$mistake_rate_pct"}}}
    ]
    res = list(col_mistakes.aggregate(pipeline))
    overall_mistake_rate_pct = res[0].get("avg_mistake_rate") if res and res[0].get("avg_mistake_rate") is not None else 0.0
    
    q_count = col_sessions.count_documents({"session_type": "Q"})
    r_count = col_sessions.count_documents({"session_type": "R"})
    
    return {
        "total_sessions": total_sessions,
        "total_drivers_analyzed": total_drivers,
        "overall_mistake_rate_pct": round(overall_mistake_rate_pct, 2) if overall_mistake_rate_pct else 0.0,
        "sessions_by_type": {"Q": q_count, "R": r_count}
    }

def get_mistakes_by_driver(db, driver: str) -> list:
    col = db["mistake_annotations"]
    cursor = col.find({"driver": driver}, {"_id": 0}).sort("session_file", 1)
    return list(cursor)

def write_status_log(mongo_available: bool, upload_summary: dict = None) -> None:
    log_path = DATA_LAKE_DIR / "mongo_status.json"
    
    doc_count_mistakes = 0
    doc_count_sessions = 0
    last_updated_mistakes = None
    last_updated_sessions = None
    
    if mongo_available:
        try:
            client = get_client()
            db = client["apexhunter"]
            
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mongo_available": mongo_available,
        "database": "apexhunter",
        "collections": {
            "mistake_annotations": { "document_count": doc_count_mistakes, "last_updated": last_updated_mistakes },
            "telemetry_sessions":  { "document_count": doc_count_sessions, "last_updated": last_updated_sessions }
        },
        "last_upload": upload_summary or {}
    }
    
    try:
        DATA_LAKE_DIR.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Wrote MongoDB status log to {log_path}")
    except Exception as e:
        logger.error(f"Failed to write MongoDB status log: {e}")

def main():
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
        
    upload_summary = {}
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
                db = client["apexhunter"]
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
