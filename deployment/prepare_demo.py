"""Fetch ONLY 2023 Bahrain Q/R for cars 1 and 11; never train or run inference.

Run manually from the repository root using the dashboard requirements.
The existing ETL functions clean the genuine FastF1 telemetry unchanged.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import fastf1
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
from etl.download_season_data import _process_driver
from etl.clean_telemetry import clean_telemetry_file
from utils import enable_fastf1_cache


def main():
    enable_fastf1_cache()
    bundle = ROOT / "demo_data"
    (bundle / "clean_data").mkdir(parents=True, exist_ok=True)
    raw_dir = ROOT / "data_lake" / "season_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    schedule = fastf1.get_event_schedule(2023)
    manifest = {
        "schema_version": 1,
        "source": "Formula 1 timing data accessed through FastF1",
        "fastf1_version": fastf1.__version__,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "event_names": {"2023": {str(int(r.RoundNumber)): r.EventName
                                    for r in schedule.itertuples() if r.RoundNumber > 0}},
        "sessions": {},
        "files": [],
    }
    for code in ("Q", "R"):
        session = fastf1.get_session(2023, 1, code)
        session.load(laps=True, telemetry=True, weather=False)
        frames, drivers = [], {}
        for driver in ("1", "11"):
            telemetry = _process_driver(session, driver, 1, code, 2023)
            if telemetry is None or telemetry.empty:
                raise RuntimeError(f"Missing genuine telemetry for {code}, car {driver}")
            frames.append(telemetry)
            laps = session.laps.pick_drivers(driver)
            boundaries = [{"lap_number": int(lap.LapNumber),
                           "start_seconds": lap.LapStartTime.total_seconds(),
                           "end_seconds": lap.Time.total_seconds()}
                          for lap in laps.itertuples()
                          if pd.notna(lap.LapStartTime) and pd.notna(lap.Time)]
            fastest = laps.pick_fastest()
            drivers[driver] = {"lap_boundaries": boundaries,
                               "fastest_lap_seconds": fastest["LapTime"].total_seconds()}
        filename = f"2023_1_{code}.parquet"
        raw = raw_dir / filename
        pd.concat(frames, ignore_index=True).to_parquet(raw, index=False)
        target = bundle / "clean_data" / filename
        clean_telemetry_file(raw, target)
        cleaned = pd.read_parquet(target)
        if cleaned.empty or set(cleaned.Driver.astype(str).unique()) != {"1", "11"}:
            raise RuntimeError("Cleaned demo must contain both genuine drivers")
        manifest["sessions"][target.stem] = {
            "year": 2023, "round": 1, "session": code,
            "event_name": session.event["EventName"],
            "api_path": session.api_path,
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
