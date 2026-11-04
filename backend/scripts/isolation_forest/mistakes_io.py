"""Data loading, validation, and output writing for the ApexHunter
Isolation Forest pipeline. All file I/O operations are isolated here.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from isolation_forest.mistakes_features import FEATURE_COLUMNS, project_to_track
from isolation_forest.mistakes_model import N_ESTIMATORS
from lstm.tyre_data import representative_lap_mask
from utils import IST, enable_fastf1_cache

# ── Required columns for session parquet files ────────────────────────────────
REQUIRED_COLUMNS: List[str] = [
    "Driver", "Speed", "Throttle", "Brake",
    "X", "Y", "SessionTime", "nGear", "Year", "Round", "Session",
]

# ── Lap and reference selection ───────────────────────────────────────────────
LAP_TIME_CUTOFF: float = 1.07           # laps slower than 107 % of the driver's best are dropped
N_REFERENCE_DRIVERS: int = 5            # agreed: median of the 5 fastest other drivers
MIN_REFERENCE_LAPS: int = 3

# ── Traffic filter (races only) ───────────────────────────────────────────────
# Samples with DRS open or less than 1 s behind another car are left out (driver and reference):
# the tow and DRS add speed and dirty air costs it, so the field comparison would flag them as
# mistakes. 1 s is the DRS detection gap. Qualifying is not filtered: everyone uses DRS there.
DRS_OPEN_MIN: int = 10                  # FastF1 DRS channel: 0/1 off, 8 eligible, 10/12/14 open
TRAFFIC_GAP_S: float = 1.0


def load_and_validate(
    session_path: Path,
    driver: str,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load a session parquet, validate columns, and filter to the target driver.

    Args:
        session_path: Path to the cleaned session parquet file.
        driver: Driver code string (e.g. "1", "44").
        logger: Logger instance for messages.

    Returns:
        Tuple of (df_session_full, df_driver).

    Raises:
        ValueError: If required columns are missing or the driver is not found.
    """
    logger.info(f"Loading session file: {session_path}")
    try:
        df_session = pd.read_parquet(session_path)
    except Exception as e:
        raise ValueError(f"Failed to load session file: {e}") from e

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df_session.columns]
    if missing_cols:
        for col in missing_cols:
            logger.error(f"Missing required column: {col}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    df_driver = df_session[df_session["Driver"] == driver].copy()
    if df_driver.empty:
        logger.error(f"Driver '{driver}' not found in session file.")
        raise ValueError(f"Driver '{driver}' not found in session file.")

    row_count = len(df_driver)
    filename = Path(session_path).name
    logger.info(f"Loaded {row_count} rows for driver {driver} from {filename}")

    return df_session, df_driver


def load_fastf1_session(df_session: pd.DataFrame, logger: logging.Logger):
    """Load the session from the FastF1 cache, with telemetry (year/round/session come from the
    parquet columns). Used for the lap table, the traffic filter and the corner positions.

    Raises:
        ValueError: If FastF1 cannot load the session.
    """
    import fastf1
    year, rnd, name = (df_session[c].iloc[0] for c in ("Year", "Round", "Session"))
    try:
        enable_fastf1_cache()
        session = fastf1.get_session(int(year), int(rnd), str(name))
        session.load(weather=False, messages=False)
    except Exception as e:
        raise ValueError(f"Could not load FastF1 laps for {year} round {rnd} {name}: {e}") from e
    logger.info(f"FastF1 session loaded: {len(session.laps)} laps")
    return session


def traffic_mask(session, driver: str, df_driver: pd.DataFrame) -> np.ndarray:
    """True for the driver's samples with DRS open or less than TRAFFIC_GAP_S behind another car.

    DRS and the gap to the car ahead are not in the cleaned parquet; they come from FastF1's
    telemetry for the same samples (matched on SessionTime; unmatched samples count as clear).
    The gap is FastF1's distance to the car ahead divided by the driver's speed.
    """
    tel = session.laps.pick_drivers(driver).get_telemetry()
    has_car_ahead = tel["DriverAhead"].fillna("").astype(str).str.len() > 0
    gap_s = tel["DistanceToDriverAhead"] / np.maximum(tel["Speed"] / 3.6, 1.0)
    in_traffic = (tel["DRS"] >= DRS_OPEN_MIN) | (has_car_ahead & (gap_s < TRAFFIC_GAP_S))
    by_time = pd.Series(in_traffic.to_numpy(), index=tel["SessionTime"]).groupby(level=0).any()
    return by_time.reindex(df_driver["SessionTime"]).fillna(False).to_numpy(dtype=bool)


def corner_positions(session, line_points: np.ndarray, line_s: np.ndarray) -> pd.DataFrame:
    """The circuit's corners (FastF1 circuit info) as label + track position in metres along the
    track line; empty if FastF1 has no circuit info."""
    try:
        corners = session.get_circuit_info().corners
    except Exception:
        return pd.DataFrame(columns=["label", "track_pos"])
    labels = [f"Turn {int(n)}{letter}" for n, letter in zip(corners["Number"], corners["Letter"].fillna(""))]
    return pd.DataFrame({"label": labels, "track_pos": project_to_track(corners, line_points, line_s)}
                        ).sort_values("track_pos").reset_index(drop=True)


def select_laps(laps: pd.DataFrame, is_race: bool) -> pd.DataFrame:
    """Laps that show normal racing: representative (no lap 1, pit laps, SC/VSC/red flag,
    deleted or inaccurate laps) and within 107 % of the driver's best such lap (drops
    qualifying cool-down laps and laps spoiled by an incident). In a race, each driver's last
    lap is dropped too: drivers ease off on it, which was the top "mistake" event for both race
    winners checked (2023 R1 #1, 2024 R8 #16)."""
    if is_race:
        last = laps.groupby("DriverNumber")["LapNumber"].transform("max")
        laps = laps[laps["LapNumber"] < last]
    laps = laps[representative_lap_mask(laps)]
    best = laps.groupby("DriverNumber")["LapTime"].transform("min")
    return laps[laps["LapTime"] <= best * LAP_TIME_CUTOFF]


def select_reference_drivers(
    laps: pd.DataFrame, target_driver: str, logger: logging.Logger
) -> List[str]:
    """The N_REFERENCE_DRIVERS fastest drivers by median selected lap time, never the target.

    `laps` must already be filtered by select_laps. Drivers with fewer than MIN_REFERENCE_LAPS
    laps are skipped (one lucky lap is not a reference).

    Raises:
        ValueError: If no other driver has enough laps.
    """
    others = laps[laps["DriverNumber"].astype(str) != target_driver]
    counts = others.groupby("DriverNumber").size()
    pace = others.groupby("DriverNumber")["LapTime"].median()[counts >= MIN_REFERENCE_LAPS]
    if pace.empty:
        raise ValueError("No other driver has enough representative laps for a reference.")
    refs = [str(d) for d in pace.sort_values().index[:N_REFERENCE_DRIVERS]]
    logger.info(f"Reference drivers (fastest by median lap): {refs}")
    return refs


def slice_laps(df_driver: pd.DataFrame, driver_laps: pd.DataFrame) -> pd.DataFrame:
    """Keep only the samples inside the given laps and add their LapNumber.

    `df_driver` must be one driver's telemetry sorted by SessionTime. A sample belongs to a lap
    when its SessionTime lies between the lap's LapStartTime and end Time.
    """
    bounds = (driver_laps[["LapNumber", "LapStartTime", "Time"]]
              .rename(columns={"Time": "LapEndTime"})
              .dropna().sort_values("LapStartTime"))
    merged = pd.merge_asof(df_driver, bounds, left_on="SessionTime", right_on="LapStartTime",
                           direction="backward")
    inside = merged["LapNumber"].notna() & (merged["SessionTime"] <= merged["LapEndTime"])
    out = merged[inside].drop(columns=["LapStartTime", "LapEndTime"])
    out["LapNumber"] = out["LapNumber"].astype("int32")
    return out.reset_index(drop=True)


def build_meta(
    session_file: str,
    driver: str,
    ref_drivers: List[str],
    n_laps: int,
    evaluation: dict,
    df_annotated: pd.DataFrame,
) -> dict:
    """Construct the metadata dictionary for a completed run.

    Args:
        session_file: Path to the input session file as string.
        driver: Target driver code.
        ref_drivers: Reference driver codes (the field profile is built from their laps).
        n_laps: Number of the driver's laps analysed.
        evaluation: Output of mistakes_model.evaluate_settings (chosen settings, planted-mistake
            metrics and the search table).
        df_annotated: The annotated driver DataFrame.

    Returns:
        Metadata dictionary with all required keys.
    """
    total_rows = len(df_annotated)
    total_mistakes = int(df_annotated["is_mistake"].sum())
    mistake_rate_pct = round(total_mistakes / total_rows * 100, 2) if total_rows > 0 else 0.0

    return {
        "session_file": session_file,
        "driver": driver,
        "reference_drivers": ref_drivers,
        "n_laps": n_laps,
        "best_contamination": evaluation["best"]["contamination"],
        "best_max_samples": evaluation["best"]["max_samples"],
        "best_max_features": evaluation["best"]["max_features"],
        "evaluation": {k: evaluation["best"][k] for k in (
            "precision", "recall", "f1", "recall_by_type", "counts")}
            | {"n_folds": evaluation["n_folds"], "injection_seeds": evaluation["injection_seeds"]},
        "search": evaluation["search"],
        "total_rows": total_rows,
        "total_mistakes": total_mistakes,
        "mistake_rate_pct": mistake_rate_pct,
        "n_estimators": N_ESTIMATORS,
        "feature_columns": FEATURE_COLUMNS,
        "timestamp": datetime.now(IST).isoformat(),
    }


def save_outputs(
    df_annotated: pd.DataFrame,
    meta: dict,
    events: List[dict],
    output_dir: Path,
    output_stem: str,
    logger: logging.Logger,
    other_events: Optional[Dict[str, List[dict]]] = None,
) -> Tuple[Path, Path]:
    """Save the annotated parquet, metadata JSON and events JSON (most severe event first).

    Args:
        df_annotated: The annotated driver DataFrame.
        meta: Metadata dictionary.
        events: Output of mistakes_events.build_events.
        output_dir: Directory to write into.
        output_stem: Base name for the output files.
        logger: Logger instance.
        other_events: With --compare, the Z-score's and autoencoder's own events
            ({"z_score": [...], "lstm_autoencoder": [...]}), saved under "detectors".

    Returns:
        Tuple of (parquet_path, json_path).
    """
    parquet_path = output_dir / f"{output_stem}_mistakes.parquet"
    json_path = output_dir / f"{output_stem}_mistakes_meta.json"

    df_annotated.to_parquet(parquet_path, compression="snappy")
    logger.info(f"Saved annotated parquet: {parquet_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata JSON: {json_path}")

    events_path = output_dir / f"{output_stem}_mistakes_events.json"
    with open(events_path, "w", encoding="utf-8") as f:
        out = {"session_file": meta["session_file"], "driver": meta["driver"],
               "n_events": len(events), "events": events}
        if other_events:
            out["detectors"] = other_events
        json.dump(out, f, indent=2)
    logger.info(f"Saved events JSON: {events_path} ({len(events)} events)")

    return parquet_path, json_path
