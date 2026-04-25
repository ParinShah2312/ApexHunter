"""
ApexHunter Frontend - Data Loader
Handles all data loading with caching. Every function that reads from
disk uses @st.cache_data. Uses string paths as cache keys.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import fastf1
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def get_event_schedule(year: int) -> Dict[int, str]:
    """Fetches the F1 event schedule for a given year and returns a round→name map."""
    try:
        schedule = fastf1.get_event_schedule(year)
        return dict(zip(schedule["RoundNumber"], schedule["EventName"]))
    except Exception:
        return {}


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
def load_mistake_data(filepath: str) -> Optional[pd.DataFrame]:
    """Reads the Isolation Forest annotated parquet.

    Returns None if file does not exist. Returns the DataFrame if it does.
    """
    if not Path(filepath).exists():
        return None
    return pd.read_parquet(filepath)


@st.cache_data(show_spinner=False)
def load_mistake_meta(filepath: str) -> Optional[dict]:
    """Reads the JSON metadata file.

    Returns None if file does not exist. Returns the parsed dict if it does.
    """
    p = Path(filepath)
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_cv_metrics(filepath: str) -> Optional[pd.DataFrame]:
    """Reads the CV metrics CSV and cleans the columns.

    Converts has_curb from string to bool.
    Extracts integer from distance_px strings like '123px'.

    Returns None if file does not exist.
    """
    if not Path(filepath).exists():
        return None
    df = pd.read_csv(filepath)

    # Clean has_curb: convert "True"/"False" strings to bool
    if "has_curb" in df.columns:
        df["has_curb"] = df["has_curb"].astype(str).str.strip().str.lower() == "true"

    # Clean distance_px: extract integer from "123px" strings
    if "distance_px" in df.columns:
        df["distance_px"] = (
            df["distance_px"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
        )

    return df


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

    step = len(df) // max_points
    sampled = df.iloc[::step]

    # Always include the last row to preserve full time range
    if sampled.index[-1] != df.index[-1]:
        sampled = pd.concat([sampled, df.iloc[[-1]]])

    return sampled
