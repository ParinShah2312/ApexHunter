"""
ApexHunter Frontend - Header Bar
Renders the persistent KPI strip above the tabs.
"""

from pathlib import Path
from typing import Optional, Tuple

import fastf1
import streamlit as st

from config import DRIVER_MAPPING, TEAM_MAPPING


@st.cache_data(show_spinner=False)
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


def _parse_session_label(session_label: str) -> Tuple[int, str]:
    """Parse 'Round N: EventName - SessionType' → (round_num, session_type_code).
    Returns (0, 'Q') on parse failure."""
    try:
        round_num = int(session_label.split("Round ")[1].split(":")[0])
        if "Race" in session_label: session_type = "R"
        elif "Qualifying" in session_label: session_type = "Q"
        elif "Sprint" in session_label: session_type = "Sprint"
        else: session_type = "Q"
        return (round_num, session_type)
    except Exception:
        return (0, "Q")


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


def render_header_bar(sel, mistake_meta: Optional[dict]) -> None:
    """Renders the persistent KPI strip above the tabs.

    Args:
        sel: SidebarSelections dataclass.
        mistake_meta: Parsed mistake metadata dict, or None.
    """
    cols = st.columns([1, 1, 1, 1, 1.2, 1.2, 1.8])

    # Derive round_num and session_type
    round_num, session_type = _parse_session_label(sel.session_label)

    # KPI 1 — Lap Time
    with cols[0]:
        lap_time_str = _get_fastest_lap(sel.year, round_num, session_type, sel.driver_number)
        st.metric(label="Lap Time", value=lap_time_str)

    # KPI 2 — Top Speed
    with cols[1]:
        top_speed = float(sel.df_driver["Speed"].max()) if not sel.df_driver.empty else 0.0
        st.metric("Top Speed", f"{top_speed:.1f} km/h")

    # KPI 3 — AI Deviation / Mistake Rate
    with cols[2]:
        if mistake_meta is not None and "reference_driver" in mistake_meta:
            st.metric("Mistake Rate", f"{mistake_meta['mistake_rate_pct']:.1f}%")
        else:
            st.metric("AI Deviation", "—")

    # KPI 4 — Mistakes
    with cols[3]:
        if mistake_meta is not None:
            st.metric("Mistakes", str(mistake_meta["total_mistakes"]))
        else:
            st.metric("Mistakes", "—")

    # KPI 5 — Reference Driver
    with cols[4]:
        if mistake_meta is not None:
            ref = mistake_meta["reference_driver"]
            ref_name = DRIVER_MAPPING.get(ref, ref)
            st.metric("Reference", ref_name)
        else:
            st.metric("Reference", "—")

    # KPI 6 — Best Contamination
    with cols[5]:
        if mistake_meta is not None:
            st.metric("Contamination", str(mistake_meta["best_contamination"]))
        else:
            st.metric("Contamination", "—")

    # Column 7 — Driver Identity
    with cols[6]:
        team_name = TEAM_MAPPING.get(sel.driver_number, "Unknown Team")
        st.markdown(
            _driver_identity_html(sel.driver_number, sel.driver_name, team_name),
            unsafe_allow_html=True,
        )

    st.markdown("---")
