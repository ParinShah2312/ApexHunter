"""Fetch ONLY 2023 Bahrain Q/R for cars 1 and 11 from OpenF1; never train or run inference.

Use this when prepare_demo.py cannot run: F1's live-timing host, which FastF1 reads, answered
401 Unauthorized as of 2026-11-02. Writes the same bundle layout, with OpenF1 as its source.
Run manually from the repository root using the dashboard requirements.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
from etl.clean_telemetry import clean_telemetry_file

API = "https://api.openf1.org/v1"
SESSIONS = {"Q": "Qualifying", "R": "Race"}


def fetch(endpoint: str, **params) -> pd.DataFrame:
    """GET one endpoint, waiting out OpenF1's free-tier rate limit."""
    for attempt in range(6):
        response = requests.get(f"{API}/{endpoint}", params=params, timeout=120)
        if response.status_code != 429:
            response.raise_for_status()
            return pd.DataFrame(response.json())
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"OpenF1 rate limit persisted for {endpoint}")


def utc_naive(column: pd.Series) -> pd.Series:
    return pd.to_datetime(column, format="ISO8601", utc=True).dt.tz_localize(None)


def lap_boundaries(laps: pd.DataFrame, t0: pd.Timestamp) -> list[dict]:
    """Lap windows in seconds since t0. Laps without a start or duration are left out."""
    timed = laps.dropna(subset=["date_start", "lap_duration"]).sort_values("lap_number")
    starts = (utc_naive(timed["date_start"]) - t0).dt.total_seconds()
    return [{"lap_number": int(number),
             "start_seconds": float(start),
             "end_seconds": float(start + duration)}
            for number, start, duration in zip(timed["lap_number"], starts, timed["lap_duration"])]


def driver_telemetry(session_key: int, driver: str, code: str, t0: pd.Timestamp,
                     boundaries: list[dict]) -> pd.DataFrame:
    """Car data merged with the nearest position sample, in the raw parquet's schema."""
    car = fetch("car_data", session_key=session_key, driver_number=driver)
    time.sleep(1)
    location = fetch("location", session_key=session_key, driver_number=driver)
    time.sleep(1)
    car["Date"] = utc_naive(car["date"])
    location["Date"] = utc_naive(location["date"])
    merged = pd.merge_asof(car.sort_values("Date"),
                           location.sort_values("Date")[["Date", "x", "y", "z"]],
                           on="Date", direction="nearest", tolerance=pd.Timedelta("1s"))
    seconds = (merged["Date"] - t0).dt.total_seconds()
    first = min(lap["start_seconds"] for lap in boundaries)
    last = max(lap["end_seconds"] for lap in boundaries)
    merged = merged[seconds.between(first, last)].copy()
    merged["SessionTime"] = merged["Date"] - t0
    return pd.DataFrame({
        "Date": merged["Date"],
        "SessionTime": merged["SessionTime"],
        "Time": merged["SessionTime"],
        "Speed": merged["speed"].astype(float),
        "RPM": merged["rpm"].astype(float),
        "nGear": merged["n_gear"].astype(int),
        "Throttle": merged["throttle"].astype(float),
        "Brake": (merged["brake"] > 0).astype(float),  # OpenF1 reports 0-100; the dashboard uses 0/1
        "X": merged["x"].astype(float),
        "Y": merged["y"].astype(float),
        "Z": merged["z"].astype(float),
        "Driver": driver,
        "Round": 1,
        "Session": code,
        "Year": 2023,
    })


def main():
    bundle = ROOT / "demo_data"
    (bundle / "clean_data").mkdir(parents=True, exist_ok=True)
    raw_dir = ROOT / "data_lake" / "season_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "source": "OpenF1 API (https://api.openf1.org)",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "event_names": {"2023": {"1": "Bahrain Grand Prix"}},
        "sessions": {},
        "files": [],
    }
    for code, name in SESSIONS.items():
        matches = fetch("sessions", year=2023, country_name="Bahrain", session_name=name)
        if len(matches) != 1:
            raise RuntimeError(f"Expected one OpenF1 session for {name}, found {len(matches)}")
        session_key = int(matches["session_key"].iloc[0])
        t0 = utc_naive(matches["date_start"]).iloc[0]
        frames, drivers = [], {}
        for driver in ("1", "11"):
            laps = fetch("laps", session_key=session_key, driver_number=driver)
            time.sleep(1)
            boundaries = lap_boundaries(laps, t0)
            if not boundaries:
                raise RuntimeError(f"Missing lap timing for {code}, car {driver}")
            telemetry = driver_telemetry(session_key, driver, code, t0, boundaries)
            if telemetry.empty:
                raise RuntimeError(f"Missing telemetry for {code}, car {driver}")
            frames.append(telemetry)
            drivers[driver] = {"lap_boundaries": boundaries,
                               "fastest_lap_seconds": float(laps["lap_duration"].min())}
        filename = f"2023_1_{code}.parquet"
        raw = raw_dir / filename
        pd.concat(frames, ignore_index=True).to_parquet(raw, index=False)
        target = bundle / "clean_data" / filename
        clean_telemetry_file(raw, target)
        cleaned = pd.read_parquet(target)
        if cleaned.empty or set(cleaned.Driver.astype(str).unique()) != {"1", "11"}:
            raise RuntimeError("Cleaned demo must contain both drivers")
        manifest["sessions"][target.stem] = {
            "year": 2023, "round": 1, "session": code,
            "event_name": "Bahrain Grand Prix",
            "openf1_session_key": session_key,
            "drivers": drivers,
            "precomputed_outputs": [],
        }
        manifest["files"].append({
            "path": f"clean_data/{filename}",
            "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "rows": len(cleaned),
        })
        print(f"Prepared {filename}: {len(cleaned):,} rows, {target.stat().st_size:,} bytes")
    pending = bundle / "manifest.json.tmp"
    pending.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(bundle / "manifest.json")


if __name__ == "__main__":
    main()
