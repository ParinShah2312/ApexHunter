"""
ApexHunter Frontend - Sidebar
Renders all sidebar filters and returns the user's selections.
Uses st.session_state to persist selections across reruns.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from config import AVAILABLE_YEARS, DATA_LAKE_DIR, DRIVER_MAPPING, SESSION_LABEL_MAP
from components.data_loader import get_event_schedule, load_session_data


@dataclass
class SidebarSelections:
    """Container for all sidebar filter selections."""

    year: int
    session_label: str
    driver_number: str
    driver_name: str
    df_driver: pd.DataFrame


def _build_session_options(
    available_files: List[Path], event_names: Dict[int, str]
) -> Tuple[List[str], Dict[str, Path]]:
    """Builds sorted, human-readable session labels from parquet filenames."""
    session_options: List[str] = []
    file_mapping: Dict[str, Path] = {}

    for f in available_files:
        parts = f.stem.split("_")
        if len(parts) == 3:
            _year, round_num, session_type = parts
            event_name = event_names.get(int(round_num), "Unknown Race")
            full_session = SESSION_LABEL_MAP.get(session_type, session_type)
            label = f"Round {round_num}: {event_name} - {full_session}"
        else:
            label = f.stem
        session_options.append(label)
        file_mapping[label] = f

    # Sort numerically by round number
    def _round_key(x: str) -> int:
        try:
            if "Round " in x:
                return int(x.split("Round ")[1].split(":")[0])
        except (ValueError, IndexError):
            pass
        return 999

    session_options.sort(key=_round_key)
    return session_options, file_mapping


def _build_driver_options(df: pd.DataFrame) -> List[str]:
    """Builds a sorted list of 'Name (#Number)' labels for the driver selectbox."""
    available_numbers = sorted(df["Driver"].dropna().unique())
    driver_list = [
        (DRIVER_MAPPING.get(d, "Unknown Driver"), f"{DRIVER_MAPPING.get(d, 'Unknown Driver')} (#{d})")
        for d in available_numbers
    ]
    driver_list.sort(key=lambda x: x[0])
    return [label for _, label in driver_list]


def _get_default_index(key: str, options: list) -> int:
    """Returns the saved index from session_state, or 0 if not found or out of range."""
    saved = st.session_state.get(key)
    if saved is not None and saved in options:
        return options.index(saved)
    return 0


def render_sidebar() -> SidebarSelections:
    """Renders the full sidebar and returns user selections + filtered DataFrame.

    Uses st.session_state to remember the user's last selections so they
    persist across Streamlit reruns during development.

    Returns:
        SidebarSelections dataclass with all filter values and the filtered DataFrame.
    """
    st.sidebar.header("Filter Data")

    # ── Year ──────────────────────────────────────────────────────────────
    year_idx = _get_default_index("sel_year", AVAILABLE_YEARS)
    selected_year = st.sidebar.selectbox("Year", AVAILABLE_YEARS, index=year_idx, key="sel_year")

    available_files = list(DATA_LAKE_DIR.glob(f"{selected_year}_*.parquet"))
    if not available_files:
        st.error(f"No data files found for {selected_year} in {DATA_LAKE_DIR}.")
        st.stop()

    event_names = get_event_schedule(selected_year)
    session_options, file_mapping = _build_session_options(available_files, event_names)

    # ── Session ───────────────────────────────────────────────────────────
    session_idx = _get_default_index("sel_session", session_options)
    selected_session = st.sidebar.selectbox(
        "Session", session_options, index=session_idx, key="sel_session",
    )

    # Pass string path to load_session_data (strings are hashable for @st.cache_data)
    try:
        df = load_session_data(str(file_mapping[selected_session]))
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # ── Driver ────────────────────────────────────────────────────────────
    driver_labels = _build_driver_options(df)
    driver_idx = _get_default_index("sel_driver", driver_labels)
    selected_driver_label = st.sidebar.selectbox(
        "Driver", driver_labels, index=driver_idx, key="sel_driver",
    )
    driver_number = selected_driver_label.split(" (#")[1].replace(")", "")
    driver_name = selected_driver_label.split(" (#")[0]

    # ── Footer ────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("ApexHunter v1.0 | Dev: Parin Shah | ID: 23001091")

    # ── Filter by driver ──────────────────────────────────────────────────
    df_driver = df[df["Driver"] == driver_number].copy()

    return SidebarSelections(
        year=selected_year,
        session_label=selected_session,
        driver_number=driver_number,
        driver_name=driver_name,
        df_driver=df_driver,
    )
