"""
ApexHunter Frontend - Header Bar
Renders the persistent KPI strip above the tabs.
"""

from pathlib import Path
from typing import Optional

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
        formatted = str(lap_time).split(" ")[-1][:10]
        return formatted
    except Exception:
        return "—"


def render_header_bar(sel, mistake_meta: Optional[dict]) -> None:
    """Renders the persistent KPI strip above the tabs.

    Args:
        sel: SidebarSelections dataclass.
        mistake_meta: Parsed mistake metadata dict, or None.
    """
    cols = st.columns([1, 1, 1, 1, 1, 1, 1.4])

    # Derive round_num and session_type from session_filepath
    filepath_stem = Path(sel.session_filepath).stem  # e.g. "2024_1_Q"
    parts = filepath_stem.split("_")
    if len(parts) >= 3:
        round_num = int(parts[1])
        session_type = parts[2]
    else:
        round_num = 1
        session_type = "Q"

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
            f'<div style="text-align:right;padding:4px 0">'
            f'<div style="font-family:\'Courier New\',monospace;font-size:2rem;'
            f'font-weight:700;color:#00d4ff;line-height:1">{sel.driver_number}</div>'
            f'<div style="font-size:1rem;font-weight:600;letter-spacing:1px;'
            f'color:#e8edf5">{sel.driver_name.upper()}</div>'
            f'<div style="font-size:0.75rem;color:#6b7890;letter-spacing:1px">'
            f'{team_name}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
