"""
ApexHunter Frontend - Sidebar
Renders all sidebar filters, AI model status indicators, export buttons,
and returns the user's selections.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from config import (
    AVAILABLE_YEARS,
    DATA_LAKE_DIR,
    DRIVER_MAPPING,
    MISTAKE_DATA_DIR,
    PROCESSED_VIDEO_DIR,
    SESSION_LABEL_MAP,
)
from components.data_loader import get_event_schedule, load_session_data, load_mistake_data


@dataclass
class SidebarSelections:
    """Container for all sidebar filter selections."""

    year: int
    session_label: str
    driver_number: str
    driver_name: str
    df_driver: pd.DataFrame
    session_filepath: str
    mistake_parquet_path: str
    mistake_meta_path: str
    compare_driver_number: Optional[str]
    df_compare: Optional[pd.DataFrame]


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
        (
            DRIVER_MAPPING.get(d, "Unknown Driver"),
            f"{DRIVER_MAPPING.get(d, 'Unknown Driver')} (#{d})",
        )
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
    """Renders the full sidebar and returns user selections + filtered DataFrame."""

    # ── Filters ───────────────────────────────────────────────────────────
    st.sidebar.markdown("### Filter Data")

    # Year
    year_idx = _get_default_index("sel_year", AVAILABLE_YEARS)
    selected_year = st.sidebar.selectbox(
        "Year", AVAILABLE_YEARS, index=year_idx, key="sel_year"
    )

    available_files = list(DATA_LAKE_DIR.glob(f"{selected_year}_*.parquet"))
    if not available_files:
        st.error(f"No data files found for {selected_year} in {DATA_LAKE_DIR}.")
        st.stop()

    event_names = get_event_schedule(selected_year)
    session_options, file_mapping = _build_session_options(available_files, event_names)

    # Session
    session_idx = _get_default_index("sel_session", session_options)
    selected_session = st.sidebar.selectbox(
        "Session", session_options, index=session_idx, key="sel_session"
    )

    session_filepath = str(file_mapping[selected_session])
    try:
        df = load_session_data(session_filepath)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Driver
    driver_labels = _build_driver_options(df)
    driver_idx = _get_default_index("sel_driver", driver_labels)
    selected_driver_label = st.sidebar.selectbox(
        "Driver", driver_labels, index=driver_idx, key="sel_driver"
    )
    driver_number = selected_driver_label.split(" (#")[1].replace(")", "")
    driver_name = selected_driver_label.split(" (#")[0]

    # Compare Driver
    compare_labels = ["— None —"] + driver_labels
    compare_idx = _get_default_index("sel_compare", compare_labels)
    selected_compare_label = st.sidebar.selectbox(
        "Compare Driver", compare_labels, index=compare_idx, key="sel_compare"
    )

    compare_driver_number = None
    df_compare = None
    if selected_compare_label != "— None —":
        comp_num = selected_compare_label.split(" (#")[1].replace(")", "")
        if comp_num != driver_number:
            compare_driver_number = comp_num
            df_compare = df[df["Driver"] == comp_num].copy()

    # Filter primary driver
    df_driver = df[df["Driver"] == driver_number].copy()

    # ── Derive mistake file paths ─────────────────────────────────────────
    stem = Path(session_filepath).stem
    output_stem = f"{stem}_{driver_number}"
    mistake_parquet_path = str(MISTAKE_DATA_DIR / f"{output_stem}_mistakes.parquet")
    mistake_meta_path = str(MISTAKE_DATA_DIR / f"{output_stem}_mistakes_meta.json")

    # ── AI Models Status ──────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("**AI Models**")

    green_dot = (
        '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        'background:#00ff88;box-shadow:0 0 5px #00ff88;margin-right:6px"></span>'
    )
    gray_dot = (
        '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        'background:#3a4558;margin-right:6px"></span>'
    )

    # YOLOv11-Seg: any .mp4 file exists in PROCESSED_VIDEO_DIR
    has_video = any(PROCESSED_VIDEO_DIR.glob("*.mp4")) if PROCESSED_VIDEO_DIR.exists() else False
    st.sidebar.markdown(
        f"{green_dot if has_video else gray_dot} YOLOv11-Seg",
        unsafe_allow_html=True,
    )

    # Isolation Forest
    has_iso = Path(mistake_parquet_path).exists()
    st.sidebar.markdown(
        f"{green_dot if has_iso else gray_dot} Isolation Forest",
        unsafe_allow_html=True,
    )

    # LSTM Tyre: always gray
    st.sidebar.markdown(f"{gray_dot} LSTM Tyre", unsafe_allow_html=True)

    # A* Racing Line: always gray
    st.sidebar.markdown(f"{gray_dot} A* Racing Line", unsafe_allow_html=True)

    # ── Export Section ────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    col1, col2, col3 = st.sidebar.columns(3)

    # Export CSV
    with col1:
        if has_iso:
            df_mistakes = load_mistake_data(mistake_parquet_path)
            if df_mistakes is not None:
                csv_data = df_mistakes.to_csv(index=False).encode("utf-8")
                st.download_button("CSV", data=csv_data, file_name=f"{output_stem}_mistakes.csv", mime="text/csv")
            else:
                st.button("CSV", disabled=True)
        else:
            st.button("CSV", disabled=True)

    # Export Report
    with col2:
        top_speed = float(df_driver["Speed"].max()) if not df_driver.empty else 0.0
        report_lines = [
            "ApexHunter v2.0 — Session Report",
            "=" * 40,
            f"Session: {selected_session}",
            f"Year: {selected_year}",
            f"Driver: {driver_name} (#{driver_number})",
            f"Total rows: {len(df_driver)}",
            f"Top Speed: {top_speed:.1f} km/h",
        ]
        if has_iso:
            report_lines.append(f"Mistake parquet: {output_stem}_mistakes.parquet")
        report_text = "\n".join(report_lines)
        st.download_button(
            "Report",
            data=report_text.encode("utf-8"),
            file_name=f"{output_stem}_report.txt",
            mime="text/plain",
        )

    # Export Video
    with col3:
        if has_video:
            st.button("Video", disabled=True, help="Use file explorer to access videos")
        else:
            st.button("Video", disabled=True)

    # ── Footer ────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.caption("ApexHunter v2.0 · Parin Shah · 23001091")

    return SidebarSelections(
        year=selected_year,
        session_label=selected_session,
        driver_number=driver_number,
        driver_name=driver_name,
        df_driver=df_driver,
        session_filepath=session_filepath,
        mistake_parquet_path=mistake_parquet_path,
        mistake_meta_path=mistake_meta_path,
        compare_driver_number=compare_driver_number,
        df_compare=df_compare,
    )
