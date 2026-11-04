"""
================================================================================
  ApexHunter - F1 Telemetry Analytics Dashboard
  Entry Point: app.py
--------------------------------------------------------------------------------
  Purpose : Orchestrator that assembles the dashboard from components.
            Run with: streamlit run frontend/app.py
================================================================================
"""

import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ApexHunter",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
_DARK_THEME_CSS: str = """
    <style>
    /* 1. Dark background */
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: #0a0c0f;
    }

    /* 2. Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #0f1217;
        border-right: 1px solid #ffffff12;
    }

    /* 3. Metric containers */
    [data-testid="stMetric"] {
        background: #1a2030;
        border: 1px solid #ffffff12;
        border-radius: 8px;
        padding: 10px 12px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    [data-testid="stMetric"]:hover {
        background: #1e2538;
        border-color: #00d4ff33;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }

    /* 4. Metric values and labels */
    [data-testid="stMetricValue"] {
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.5rem !important;
        color: #00d4ff !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        color: #6b7890 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }

    /* 5. Tab styling (tab headers only: a plain "button" here also greyed out every switch and
       button inside a tab, e.g. Compare with, Colour track by, the More switch) */
    [data-testid="stTabs"] button[role="tab"] {
        background: transparent;
        border: none;
        color: #6b7890;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #00d4ff;
        border-bottom: 2px solid #00d4ff;
    }

    /* 6. Plotly chart containers */
    [data-testid="stPlotlyChart"] {
        background: transparent;
    }

    /* 7. Slider thumb */
    [data-testid="stSlider"] input[type="range"]::-webkit-slider-thumb {
        background: #00d4ff;
    }

    /* 8. Button styling */
    [data-testid="stButton"] button {
        background: #1a2030;
        border: 1px solid #ffffff22;
        color: #e8edf5;
        border-radius: 4px;
    }
    [data-testid="stButton"] button:hover {
        border-color: #00d4ff44;
        color: #00d4ff;
    }

    /* 9. Segmented controls (season, session, More tab): make the selected option obvious */
    [data-testid="stBaseButton-segmented_control"] {
        background: #1a2030;
        color: #8a94a6;
        border-color: #ffffff22;
    }
    [data-testid="stBaseButton-segmented_controlActive"] {
        background: #00d4ff26;
        color: #00d4ff;
        border-color: #00d4ff;
        font-weight: 600;
    }
    </style>
"""


# ── Global CSS Injection ──────────────────────────────────────────────────────
st.markdown(_DARK_THEME_CSS, unsafe_allow_html=True)


# ── Dropdown keyboard lock ────────────────────────────────────────────────────
# Every selectbox passes filter_mode=None, which stops typing, but Backspace/Delete still clear
# the shown selection in Streamlit's dropdown. This blocks those two keys inside dropdowns.
# st.html runs in the page itself (not an iframe), replacing the deprecated st.components.v1.html.
_DROPDOWN_KEY_LOCK_JS: str = """
<script>
if (!window.__apexHunterDropdownLock) {
  window.__apexHunterDropdownLock = true;
  document.addEventListener("keydown", (e) => {
    if ((e.key === "Backspace" || e.key === "Delete")
        && e.target.closest && e.target.closest('div[data-baseweb="select"]')) {
      e.preventDefault();
      e.stopImmediatePropagation();
    }
  }, true);
}
</script>
"""
st.html(_DROPDOWN_KEY_LOCK_JS, unsafe_allow_javascript=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from components.sidebar import render_sidebar
from components.header_bar import render_header_bar
from components.telemetry_charts import render_telemetry
from components.cv_feed import render_cv_feed
from components.strategy_tab import render_strategy_tab
from components.mistakes_tab import render_mistakes_tab
from components.overview_tab import render_overview
from components.data_loader import load_mistake_data, load_mistake_meta, load_racing_line, load_tyre_prediction, tyre_prediction_unreadable, get_lap_boundaries, assign_lap_numbers
from components.racing_line import render_racing_line
from components.bigdata_tab import render_bigdata_tab

# ── Step 1: Sidebar ───────────────────────────────────────────────────────────
sel = render_sidebar()

# ── Step 2: Load model outputs ────────────────────────────────────────────────
import os as _os
_parquet_mtime = _os.path.getmtime(sel.mistake_parquet_path) if _os.path.exists(sel.mistake_parquet_path) else 0
_meta_mtime = _os.path.getmtime(sel.mistake_meta_path) if _os.path.exists(sel.mistake_meta_path) else 0
df_mistakes = load_mistake_data(sel.mistake_parquet_path, mtime=_parquet_mtime)
meta = load_mistake_meta(sel.mistake_meta_path, mtime=_meta_mtime)
_events_path = sel.mistake_meta_path.replace("_mistakes_meta.json", "_mistakes_events.json")
_events_mtime = _os.path.getmtime(_events_path) if _os.path.exists(_events_path) else 0
mistake_events = load_mistake_meta(_events_path, mtime=_events_mtime)     # same JSON loader
tyre_data = load_tyre_prediction(sel.tyre_prediction_path)

# The backend writes LapNumber with its own lap rule, which events.json uses too; re-assigning it
# here (with a slightly different boundary rule) put boundary samples in other laps (#53). Only
# outputs written before the rework lack the column.
if df_mistakes is not None and "LapNumber" not in df_mistakes.columns:
    lap_boundaries = get_lap_boundaries(sel.year, sel.round_num, sel.session_type, sel.driver_number)
    if lap_boundaries:
        df_mistakes = assign_lap_numbers(df_mistakes, lap_boundaries)

# ── Step 3: Header bar ────────────────────────────────────────────────────────
render_header_bar(sel)

# ── Step 4: Tabs (lazy: only the open tab's code runs) ───────────────────────
TAB_NAMES = ["Overview", "Driver Mistakes", "Tyres & Pit Window", "Telemetry", "More"]
tab_overview, tab_mistakes, tab_tyres, tab_telemetry, tab_more = st.tabs(
    TAB_NAMES, key="main_tab", on_change="rerun")

if tab_overview.open:
    with tab_overview:
        render_overview(sel, df_mistakes, meta, mistake_events, tyre_data)

if tab_mistakes.open:
    with tab_mistakes:
        render_mistakes_tab(sel, df_mistakes, meta, mistake_events)

if tab_tyres.open:
    with tab_tyres:
        if tyre_prediction_unreadable(sel.tyre_prediction_path):
            st.error("The pit-window file for this driver cannot be read. Re-run "
                     "`python backend/scripts/predict_strategy.py --session <path> --driver <number> "
                     "--source all --force`.")
        else:
            render_strategy_tab(tyre_data, f"{sel.year}_{sel.round_num}_{sel.session_type}_{sel.driver_number}")

if tab_telemetry.open:
    with tab_telemetry:
        render_telemetry(sel)

if tab_more.open:
    with tab_more:
        part = st.segmented_control("Part", ["Apex detection", "Racing line", "Big Data"],
                                    default="Apex detection", required=True, key="more_part",
                                    label_visibility="collapsed")
        if part == "Apex detection":
            render_cv_feed()
        elif part == "Racing line":
            render_racing_line(racing_line_data=load_racing_line(sel.racing_line_path),
                               driver_number=sel.driver_number, df_full=sel.df_full)
        else:
            render_bigdata_tab(df_full=sel.df_full, racing_line_path=sel.racing_line_path)
