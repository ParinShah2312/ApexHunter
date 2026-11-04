"""
ApexHunter Frontend - Sidebar
Season switch → Grand Prix → session switch → driver, then badges saying whether model results
exist for the chosen driver. Returns the selections.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from config import (
    AVAILABLE_YEARS,
    DATA_LAKE_DIR,
    DRIVER_MAPPING,
    MISTAKE_DATA_DIR,
    SESSION_LABEL_MAP,
    RACING_LINES_DIR,
    TYRE_PREDICTIONS_DIR,
)
from components.data_loader import get_event_schedule, get_session_info, load_session_data

# Opens on a session that has both mistake and pit-window results
DEFAULT_YEAR, DEFAULT_ROUND, DEFAULT_SESSION, DEFAULT_DRIVER = 2023, 1, "R", "1"
SESSION_ORDER = ("Q", "R")


@dataclass
class SidebarSelections:
    """Container for all sidebar filter selections."""

    year: int
    round_num: int
    session_type: str
    event_name: str
    driver_number: str
    driver_name: str
    df_driver: pd.DataFrame
    session_filepath: str
    mistake_parquet_path: str
    mistake_meta_path: str
    racing_line_path: str
    tyre_prediction_path: str
    df_full: pd.DataFrame
    session_info: dict


def session_files(year: int) -> Dict[int, Dict[str, Path]]:
    """{round: {session code: clean_data file}} for one season, from the file names."""
    rounds: Dict[int, Dict[str, Path]] = {}
    for f in DATA_LAKE_DIR.glob(f"{year}_*.parquet"):
        parts = f.stem.split("_")
        if len(parts) == 3 and parts[1].isdigit():
            rounds.setdefault(int(parts[1]), {})[parts[2]] = f
    return dict(sorted(rounds.items()))


def _derive_output_paths(session_filepath: str, driver_number: str) -> Dict[str, str]:
    """Derives output paths for models based on the session and driver."""
    output_stem = f"{Path(session_filepath).stem}_{driver_number}"
    return {
        "mistake_parquet": str(MISTAKE_DATA_DIR / f"{output_stem}_mistakes.parquet"),
        "mistake_meta": str(MISTAKE_DATA_DIR / f"{output_stem}_mistakes_meta.json"),
        "racing_line": str(RACING_LINES_DIR / f"{output_stem}_racing_line.json"),
        "tyre_prediction": str(TYRE_PREDICTIONS_DIR / f"{output_stem}_strategy.json"),
    }


def _badge(label: str, ok: bool) -> str:
    color, mark = ("#2ecc71", "✓") if ok else ("#5b6577", "✗")
    return (f'<span style="display:inline-block;margin:0 6px 6px 0;padding:3px 10px;border-radius:12px;'
            f'border:1px solid {color};color:{color};font-size:0.8rem">{label} {mark}</span>')


def _init_default(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def render_sidebar() -> SidebarSelections:
    """Renders the full sidebar and returns user selections + filtered DataFrame."""
    sb = st.sidebar
    sb.markdown("### ApexHunter")

    _init_default("sel_year", DEFAULT_YEAR)
    year = sb.segmented_control("Season", AVAILABLE_YEARS, key="sel_year", required=True)

    rounds = session_files(year)
    if not rounds:
        st.error(f"No data files found for {year} in {DATA_LAKE_DIR}.")
        st.stop()
    names = get_event_schedule(year)
    round_options = list(rounds)
    _init_default("sel_round", DEFAULT_ROUND)
    if st.session_state["sel_round"] not in round_options:       # e.g. round 24 → 2023
        st.session_state["sel_round"] = round_options[0]
    round_num = sb.selectbox(
        "Grand Prix", round_options, key="sel_round", filter_mode=None,
        format_func=lambda r: f"R{r:02d} · {names.get(r, 'Round ' + str(r))}",
    )

    codes = [c for c in SESSION_ORDER if c in rounds[round_num]]
    _init_default("sel_session_type", DEFAULT_SESSION)
    if st.session_state["sel_session_type"] not in codes:
        st.session_state["sel_session_type"] = codes[-1]
    session_type = sb.segmented_control(
        "Session", codes, key="sel_session_type", required=True,
        format_func=lambda c: SESSION_LABEL_MAP.get(c, c),
    )

    session_filepath = str(rounds[round_num][session_type])
    try:
        df = load_session_data(session_filepath)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    info = get_session_info(year, round_num, session_type)
    drivers_info = info.get("drivers", {})

    def driver_name(num: str) -> str:
        return drivers_info.get(num, {}).get("name") or DRIVER_MAPPING.get(num, f"Car #{num}")

    driver_options = sorted(df["Driver"].dropna().astype(str).unique(), key=driver_name)
    if not driver_options:
        st.error("No drivers found in this session.")
        st.stop()
    _init_default("sel_driver_number", DEFAULT_DRIVER)
    if st.session_state["sel_driver_number"] not in driver_options:
        st.session_state["sel_driver_number"] = driver_options[0]
    driver_number = sb.selectbox(
        "Driver", driver_options, key="sel_driver_number", filter_mode=None,
        format_func=lambda n: f"{driver_name(n)} · {drivers_info.get(n, {}).get('team', f'#{n}')}",
    )

    df_driver = df[df["Driver"].astype(str) == driver_number].copy()
    paths = _derive_output_paths(session_filepath, driver_number)

    # ── Results badges ────────────────────────────────────────────────────
    sb.markdown("---")
    sb.markdown("**Results for this driver**")
    sb.markdown(
        _badge("Mistakes", Path(paths["mistake_parquet"]).exists())
        + _badge("Pit window", Path(paths["tyre_prediction"]).exists()),
        unsafe_allow_html=True,
    )

    sb.markdown("---")
    sb.caption("ApexHunter · Parin Shah · 23001091")

    return SidebarSelections(
        year=year,
        round_num=round_num,
        session_type=session_type,
        event_name=info.get("event") or names.get(round_num, f"Round {round_num}"),
        driver_number=driver_number,
        driver_name=driver_name(driver_number),
        df_driver=df_driver,
        session_filepath=session_filepath,
        mistake_parquet_path=paths["mistake_parquet"],
        mistake_meta_path=paths["mistake_meta"],
        racing_line_path=paths["racing_line"],
        tyre_prediction_path=paths["tyre_prediction"],
        df_full=df,
        session_info=info,
    )
