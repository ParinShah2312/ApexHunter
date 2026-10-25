"""
ApexHunter Frontend - Header Bar
Renders the persistent KPI strip above the tabs.
"""

from pathlib import Path
from typing import Tuple

import fastf1
import streamlit as st

from components.data_loader import FASTF1_LOOKUP_TTL_S
from config import TEAM_MAPPING


@st.cache_data(show_spinner=False, ttl=FASTF1_LOOKUP_TTL_S)


def _get_fastest_lap(year: int, round_num: int, session_type: str, driver_number: str) -> str:
    """Fetches the fastest lap time for a driver in a session via FastF1."""
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load(laps=True, telemetry=False, weather=False)
        fastest = session.laps.pick_drivers(driver_number).pick_fastest()
        lap_time = fastest["LapTime"]
        total_seconds = lap_time.total_seconds()
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:06.3f}"
    except Exception:
        return "—"


def session_from_file(session_filepath: str) -> Tuple[int, str]:
    """Return (round, session code) from a clean_data file name like '2024_1_Q.parquet'.

    The file name is authoritative; parsing the label text fell back to Race when the FastF1
    schedule was unavailable and turned "Sprint Shootout" into "Sprint".
    """
    parts = Path(session_filepath).stem.split("_")
    return int(parts[1]), parts[2]


def _driver_identity_html(driver_number: str, driver_name: str, team_name: str) -> str:
    """Build the styled HTML div for the driver identity block."""
    return (
        f'<div style="text-align:right;padding:4px 0">'
        f'<div style="font-family:\'Courier New\',monospace;font-size:2rem;'
        f'font-weight:700;color:#00d4ff;line-height:1">{driver_number}</div>'
        f'<div style="font-size:1rem;font-weight:600;letter-spacing:1px;'
        f'color:#e8edf5">{driver_name.upper()}</div>'
        f'<div style="font-size:0.75rem;color:#6b7890;letter-spacing:1px">'
        f'{team_name}</div>'
        f'</div>'
    )


def render_header_bar(sel) -> None:
    """Renders the persistent KPI strip above the tabs.

    Args:
        sel: SidebarSelections dataclass.
    """
    cols = st.columns([3, 1.5, 1.5, 1.5])

    # Derive round_num and session_type
    round_num, session_type = session_from_file(sel.session_filepath)

    # Extract track name from session label ("Round N: TrackName - Session")
    try:
        track_name = sel.session_label.split(": ", 1)[1].rsplit(" - ", 1)[0]
    except (IndexError, ValueError):
        track_name = "—"

    # KPI 1 — Track
    with cols[0]:
        st.metric(label="Track", value=track_name)

    # KPI 2 — Lap Time
    with cols[1]:
        lap_time_str = _get_fastest_lap(sel.year, round_num, session_type, sel.driver_number)
        st.metric(label="Fastest Lap Time", value=lap_time_str)

    # KPI 3 — Top Speed
    with cols[2]:
        top_speed = float(sel.df_driver["Speed"].max()) if not sel.df_driver.empty else 0.0
        st.metric("Top Speed", f"{top_speed:.1f} km/h")

    # Column 4 — Driver Identity
    with cols[3]:
        team_name = TEAM_MAPPING.get(sel.driver_number, "Unknown Team")
        st.markdown(
            _driver_identity_html(sel.driver_number, sel.driver_name, team_name),
            unsafe_allow_html=True,
        )

    st.markdown("---")

