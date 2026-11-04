"""
ApexHunter Frontend - Header Bar
One slim card above the tabs: Grand Prix and session, the driver in team colour, and four numbers
(finishing or qualifying position, fastest lap, gap to the session's fastest lap and who set it,
top speed).
"""

from html import escape
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st

from config import SESSION_LABEL_MAP


def session_from_file(session_filepath: str) -> Tuple[int, str]:
    """Return (round, session code) from a clean_data file name like '2024_1_Q.parquet'.

    The file name is authoritative; parsing the label text fell back to Race when the FastF1
    schedule was unavailable and turned "Sprint Shootout" into "Sprint".
    """
    parts = Path(session_filepath).stem.split("_")
    return int(parts[1]), parts[2]


def lap_time_text(seconds: Optional[float]) -> str:
    """1:31.447 style, or an em dash."""
    if seconds is None:
        return "—"
    return f"{int(seconds // 60)}:{seconds % 60:06.3f}"


def gap_text(driver_s: Optional[float], fastest: dict) -> str:
    """Gap of the driver's fastest lap to the session's fastest, e.g. '+0.412 s' or 'Fastest'."""
    if driver_s is None or not fastest:
        return "—"
    gap = driver_s - min(fastest.values())
    return "Fastest" if gap <= 0 else f"+{gap:.3f} s"


def fastest_owner(fastest: dict, drivers: dict) -> str:
    """Short name of whoever set the session's fastest lap, e.g. 'ZHO'."""
    if not fastest:
        return ""
    num = min(fastest, key=fastest.get)
    return drivers.get(num, {}).get("abbr", f"#{num}")


def _stat(label: str, value: str, title: str = "") -> str:
    return (f'<div style="text-align:right;min-width:100px" title="{escape(title)}">'
            f'<div style="font-size:0.7rem;color:#8a94a6;text-transform:uppercase;letter-spacing:.6px">'
            f'{escape(label)}</div><div style="font-size:1.25rem;font-weight:600;color:#e8edf5;'
            f'font-variant-numeric:tabular-nums">{value}</div></div>')


def render_header_bar(sel) -> None:
    """Renders the slim header card. sel: SidebarSelections."""
    info = sel.session_info or {}
    driver = info.get("drivers", {}).get(sel.driver_number, {})
    color = driver.get("color", "#8a94a6")
    team = driver.get("team", "")
    fastest = info.get("fastest_s", {})
    driver_s = fastest.get(sel.driver_number)
    position = driver.get("position")
    owner = fastest_owner(fastest, info.get("drivers", {}))
    is_race = sel.session_type == "R"
    top_speed = float(sel.df_driver["Speed"].max()) if not sel.df_driver.empty else None
    place = " · ".join(p for p in (str(sel.year), f"Round {sel.round_num}",
                                   SESSION_LABEL_MAP.get(sel.session_type, sel.session_type),
                                   info.get("location", "")) if p)
    html = (
        '<div style="display:flex;align-items:center;gap:28px;flex-wrap:wrap;padding:12px 18px;'
        'background:#131924;border:1px solid #ffffff14;border-radius:10px;margin-bottom:8px">'
        f'<div style="flex:1;min-width:220px"><div style="font-size:1.35rem;font-weight:700;'
        f'color:#e8edf5">{escape(sel.event_name)}</div>'
        f'<div style="font-size:0.85rem;color:#8a94a6">{escape(place)}</div></div>'
        f'<div style="border-left:4px solid {color};padding-left:10px;min-width:180px">'
        f'<div style="font-size:1.05rem;font-weight:600;color:#e8edf5">'
        f'{escape(sel.driver_name)} <span style="color:#8a94a6;font-weight:400">#{sel.driver_number}'
        f'</span></div><div style="font-size:0.85rem;color:{color}">{escape(team)}</div></div>'
        + _stat("Finished" if is_race else "Qualified", f"P{position}" if position else "—")
        + _stat("Fastest lap", lap_time_text(driver_s), "This driver's best lap of the session")
        + _stat(f"Gap to fastest ({owner})" if owner else "Gap to fastest", gap_text(driver_s, fastest),
                "Gap to the fastest single lap of the session"
                + (". In a race this is often set late on fresh tyres, not by the winner" if is_race else ""))
        + _stat("Top speed", "—" if top_speed is None else f"{top_speed:.0f} km/h")
        + '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
