"""
ApexHunter Frontend - Data Loader
Handles event schedule fetching, parquet loading, column validation, and downsampling.
"""

from typing import Dict

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


@st.cache_data(show_spinner="Loading telemetry data...")
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


def downsample(df: pd.DataFrame, max_points: int = 5000) -> pd.DataFrame:
    """Intelligently downsamples a DataFrame for chart rendering performance.

    Keeps every Nth row to stay under max_points while preserving the first and last row
    to maintain the full time range. This is only used for chart rendering;
    metric calculations always use the full dataset.

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
