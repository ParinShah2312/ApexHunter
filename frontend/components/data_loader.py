"""
ApexHunter Frontend - Data Loader
Handles all data loading with caching. Every function that reads from
disk uses @st.cache_data. Uses string paths as cache keys.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import fastf1
import numpy as np
import pandas as pd
import streamlit as st

# FastF1 lookups that fail (offline, rate limit) return an empty result; without an expiry that
# empty result was cached until the app restarted
FASTF1_LOOKUP_TTL_S = 600


@st.cache_data(show_spinner=False, ttl=FASTF1_LOOKUP_TTL_S)
def get_lap_boundaries(year: int, round_num: int, session_type: str, driver_number: str) -> List[dict]:
    """Fetch lap start/end times from FastF1 for a specific driver.

    Returns a list of dicts with keys: lap_number, start_seconds, end_seconds.
    Returns an empty list on failure.
    """
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load(laps=True, telemetry=False, weather=False)
        driver_laps = session.laps.pick_drivers(driver_number)
        boundaries = []
        for _, lap in driver_laps.iterrows():
            lap_num = int(lap["LapNumber"])
            start = lap["LapStartTime"]
            end = lap["Time"]  # This is the lap end time (cumulative)
            if pd.isna(start) or pd.isna(end):
                continue
            boundaries.append({
                "lap_number": lap_num,
                "start_seconds": start.total_seconds(),
                "end_seconds": end.total_seconds(),
            })
        boundaries.sort(key=lambda x: x["lap_number"])
        return boundaries
    except Exception:
        return []


def assign_lap_numbers(df: pd.DataFrame, boundaries: List[dict]) -> pd.DataFrame:
    """Assign a LapNumber column to a DataFrame based on SessionTime and lap boundaries.

    Rows that don't fall within any lap boundary get LapNumber = 0.
    """
    df = df.copy()
    time_col = "SessionTime"
    if time_col not in df.columns:
        df["LapNumber"] = 0
        return df

    if pd.api.types.is_timedelta64_dtype(df[time_col]):
        seconds = df[time_col].dt.total_seconds().values
    else:
        seconds = df[time_col].astype(float).values

    lap_numbers = np.zeros(len(seconds), dtype=int)
    for b in boundaries:
        mask = (seconds >= b["start_seconds"]) & (seconds < b["end_seconds"])
        lap_numbers[mask] = b["lap_number"]

    df["LapNumber"] = lap_numbers
    return df


@st.cache_data(show_spinner=False, ttl=FASTF1_LOOKUP_TTL_S)


def get_event_schedule(year: int) -> Dict[int, str]:
    """Fetches the F1 event schedule for a given year and returns a round→name map."""
    try:
        schedule = fastf1.get_event_schedule(year)
        return dict(zip(schedule["RoundNumber"], schedule["EventName"]))
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=FASTF1_LOOKUP_TTL_S)
def get_session_info(year: int, round_num: int, session_type: str) -> dict:
    """Event and driver facts for one session from FastF1 (cached laps + results).

    Returns {"event", "location", "drivers": {number: {name, abbr, team, color, position}},
    "fastest_s": {number: seconds}}, or {} when FastF1 is unavailable. position = finishing
    (race) or qualifying position, None if not classified. Teams come from the
    session's own results, so a driver who changed team between seasons shows the right one (#55).
    Fastest laps use FastF1's personal-best laps, so deleted laps (track limits) do not count.
    """
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        drivers = {}
        for _, r in session.results.iterrows():
            color = str(r.get("TeamColor") or "")
            drivers[str(r["DriverNumber"])] = {
                "name": r["FullName"], "abbr": r["Abbreviation"], "team": r["TeamName"],
                "color": f"#{color}" if len(color) == 6 else "#8a94a6",
                "position": int(r["Position"]) if pd.notna(r.get("Position")) else None,
            }
        laps = session.laps
        best = laps[laps["IsPersonalBest"].eq(True)].dropna(subset=["LapTime"])
        fastest = best.groupby("DriverNumber")["LapTime"].min().dt.total_seconds()
        return {
            "event": str(session.event["EventName"]),
            "location": str(session.event["Location"]),
            "drivers": drivers,
            "fastest_s": {str(k): float(v) for k, v in fastest.items()},
        }
    except Exception:
        return {}


SC_VSC_RED_CODES = set("4567")      # FastF1 track status: 4 SC, 5 red flag, 6/7 VSC


def lap_table(laps: pd.DataFrame) -> pd.DataFrame:
    """FastF1 laps → driver, lap, start_s, lap_s, compound, kind ("pit" | "sc" | "normal"),
    deleted. Laps without a start or a time are left out. kind "pit" = in- or out-lap; "sc" = run
    at least partly under a safety car, VSC or red flag."""
    lap_s = laps["LapTime"].dt.total_seconds()
    lap_s = lap_s.fillna((laps["Time"] - laps["LapStartTime"]).dt.total_seconds())
    pit = laps["PitInTime"].notna() | laps["PitOutTime"].notna()
    sc = laps["TrackStatus"].fillna("").astype(str).map(lambda s: bool(set(s) & SC_VSC_RED_CODES))
    out = pd.DataFrame({
        "driver": laps["DriverNumber"].astype(str),
        "lap": laps["LapNumber"],
        "start_s": laps["LapStartTime"].dt.total_seconds(),
        "lap_s": lap_s,
        "compound": laps["Compound"].fillna("").astype(str),
        "kind": np.where(pit, "pit", np.where(sc, "sc", "normal")),
        "deleted": laps["Deleted"].eq(True) if "Deleted" in laps else False,
    })
    out = out.dropna(subset=["lap", "start_s", "lap_s"])
    out["lap"] = out["lap"].astype(int)
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=FASTF1_LOOKUP_TTL_S)
def get_session_laps(year: int, round_num: int, session_type: str) -> pd.DataFrame:
    """lap_table for every driver of a session (empty frame if FastF1 is unavailable)."""
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        return lap_table(session.laps)
    except Exception:
        return pd.DataFrame(columns=["driver", "lap", "start_s", "lap_s", "compound", "kind", "deleted"])


@st.cache_data(show_spinner=False, ttl=FASTF1_LOOKUP_TTL_S)
def get_corners(year: int, round_num: int, session_type: str) -> pd.DataFrame:
    """The circuit's corners from FastF1: label ("T1", "T9a"), X, Y (same 1/10 m units as the
    telemetry). Empty if FastF1 has no circuit info."""
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        c = session.get_circuit_info().corners
        return pd.DataFrame({"label": [f"T{int(n)}{letter if isinstance(letter, str) else ''}"
                                       for n, letter in zip(c["Number"], c["Letter"])],
                             "X": c["X"].to_numpy(dtype=float), "Y": c["Y"].to_numpy(dtype=float)})
    except Exception:
        return pd.DataFrame(columns=["label", "X", "Y"])


@st.cache_data(show_spinner="Loading telemetry...")


def load_session_data(filepath: str) -> pd.DataFrame:
    """Loads a parquet file and ensures all expected columns exist.

    Args:
        filepath: String path to the parquet file (strings are hashable for caching).

    Returns:
        DataFrame with all expected columns guaranteed to exist.
    """
    df = pd.read_parquet(filepath)

    expected_cols = [
        "Driver", "Speed", "RPM", "Throttle", "Brake",
        "X", "Y", "Time", "SessionTime", "nGear",
    ]
    for col in expected_cols:
        if col not in df.columns:
            if col == "Driver":
                df["Driver"] = "UNKNOWN"
            elif col in ("Time", "SessionTime"):
                df[col] = pd.to_timedelta(np.arange(len(df)), unit="s")
            elif col == "nGear":
                df["nGear"] = 8

    return df


@st.cache_data(show_spinner=False)
def load_mistake_data(filepath: str, mtime: float = 0) -> Optional[pd.DataFrame]:
    """Reads the Isolation Forest annotated parquet.

    The mtime parameter is the file modification time — used to bust the cache
    when the pipeline re-runs and overwrites the file.
    Returns None if file does not exist. Returns the DataFrame if it does.
    """
    if not Path(filepath).exists():
        return None
    return pd.read_parquet(filepath)


@st.cache_data(show_spinner=False)
def load_mistake_meta(filepath: str, mtime: float = 0) -> Optional[dict]:
    """Reads the JSON metadata file.

    The mtime parameter is the file modification time — used to bust the cache
    when the pipeline re-runs and overwrites the file.
    Returns None if file does not exist. Returns the parsed dict if it does.
    """
    p = Path(filepath)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)


def load_cv_metrics(filepath: str, mtime: float = 0) -> Optional[pd.DataFrame]:
    """Reads the CV metrics CSV and cleans the columns.

    Converts has_curb from string to bool.
    Extracts integer from distance_px strings like '123px'.

    Returns None if file does not exist.
    """
    if not Path(filepath).exists():
        return None
    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        return None

    # Clean has_curb: convert "True"/"False" strings to bool
    if "has_curb" in df.columns:
        df["has_curb"] = df["has_curb"].astype(str).str.strip().str.lower() == "true"

    # Backwards compatibility for older CSVs that still use distance_px
    if "distance_px" in df.columns:
        df.rename(columns={"distance_px": "distance_cm"}, inplace=True)

    # Clean distance_cm: extract integer from "123cm" or "123px" strings
    if "distance_cm" in df.columns:
        df["distance_cm"] = (
            df["distance_cm"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
        )

    return df


def load_turn_windows(filepath: str) -> Optional[List[dict]]:
    """Read the turn windows JSON written by cv/run_inference.py (one dict per corner with
    turn, start_s, apex_s, end_s, direction, speed_category, min_speed_kmh).

    Returns None if the file does not exist (e.g. a video processed before turn windows were saved).
    """
    if not Path(filepath).exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)["turns"]


def downsample(df: pd.DataFrame, max_points: int = 5000) -> pd.DataFrame:
    """Intelligently downsamples a DataFrame for chart rendering performance.

    Keeps every Nth row to stay under max_points while preserving the first and last row
    to maintain the full time range.

    Args:
        df: The input DataFrame to downsample.
        max_points: Maximum number of points to return.

    Returns:
        A downsampled DataFrame (or the original if already small enough).
    """
    if len(df) <= max_points:
        return df

    step = -(-len(df) // max_points)        # ceil: floor gave step 1 (no downsampling) for 5,001-9,999 rows
    sampled = df.iloc[::step]

    # Always include the last row to preserve full time range
    if sampled.index[-1] != df.index[-1]:
        sampled = pd.concat([sampled, df.iloc[[-1]]])

    return sampled


def load_racing_line(filepath: str) -> Optional[dict]:
    """Load a racing line JSON. Returns None if file does not exist."""
    if not Path(filepath).exists():
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_tyre_prediction(filepath: str) -> Optional[dict]:
    """Load a tyre prediction JSON. Returns None if the file does not exist or cannot be read
    (tyre_prediction_unreadable tells the two apart)."""
    if not Path(filepath).exists():
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def tyre_prediction_unreadable(filepath: str) -> bool:
    """True if the strategy file exists but is not valid JSON: it used to be shown as "no pit
    window for this driver" (the same as a missing file)."""
    return Path(filepath).exists() and load_tyre_prediction(filepath) is None
