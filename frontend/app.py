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
import pandas as pd
from typing import Tuple

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

    /* 5. Tab styling */
    [data-testid="stTabs"] button {
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
    </style>
"""


def _get_session_time_range(df: pd.DataFrame) -> Tuple[float, float]:
    """Calculate the minimum and maximum session time in seconds."""
    time_col = "SessionTime" if "SessionTime" in df.columns else "Time"
    if pd.api.types.is_timedelta64_dtype(df[time_col]):
        min_t = float(df[time_col].dt.total_seconds().min())
        max_t = float(df[time_col].dt.total_seconds().max())
    else:
        min_t = float(df[time_col].min())
        max_t = float(df[time_col].max())
    return min_t, max_t

# ── Global CSS Injection ──────────────────────────────────────────────────────
st.markdown(_DARK_THEME_CSS, unsafe_allow_html=True)


import streamlit.components.v1 as components

# ── Disable typing in selectboxes (JS injection) ─────────────────────────────
components.html(
    """
    <script>
    const doc = window.parent.document;

    // Inject CSS into parent to hide the text cursor
    const style = doc.createElement('style');
    style.textContent = `
        div[data-baseweb="select"] input {
            caret-color: transparent !important;
            cursor: pointer !important;
        }
    `;
    doc.head.appendChild(style);

    function lockSelects() {
        doc.querySelectorAll('div[data-baseweb="select"] input').forEach(el => {
            if (el._locked) return;
            el._locked = true;

            // 1. Make readonly
            el.setAttribute('readonly', 'true');

            // 2. Block typing (letters, numbers, space) and deletions
            el.addEventListener('keydown', e => {
                if (e.key.length === 1 || e.key === 'Backspace' || e.key === 'Delete') {
                    e.preventDefault();
                    e.stopImmediatePropagation();
                }
            }, true);
        });
    }

    lockSelects();
    const observer = new MutationObserver(lockSelects);
    observer.observe(doc.body, { childList: true, subtree: true });
    </script>
    """,
    height=0
)

# ── Imports ───────────────────────────────────────────────────────────────────
from components.sidebar import render_sidebar
from components.header_bar import render_header_bar
from components.telemetry_charts import render_telemetry
from components.track_map import render_track_map
from components.cv_feed import render_cv_feed
from components.ai_analysis import render_ai_analysis
from components.data_loader import load_mistake_data, load_mistake_meta, load_racing_line, load_tyre_prediction
from components.racing_line import render_racing_line
from components.bigdata_tab import render_bigdata_tab

# ── Step 1: Sidebar ──────────────────────────────────────────────────────────
sel = render_sidebar()

# ── Step 2: Load AI data ─────────────────────────────────────────────────────
df_mistakes = load_mistake_data(sel.mistake_parquet_path)
meta = load_mistake_meta(sel.mistake_meta_path)
racing_line_data = load_racing_line(sel.racing_line_path)
tyre_data = load_tyre_prediction(sel.tyre_prediction_path)

# ── Step 3: Header bar ───────────────────────────────────────────────────────
render_header_bar(sel, meta)

# ── Step 4: Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏎 Race Intelligence",
    "📊 Telemetry",
    "🧠 AI Analysis",
    "🛣 Racing Line",
    "🗄️ Big Data"
])

# ── Tab 1: Race Intelligence ─────────────────────────────────────────────────
with tab1:
    # ── Master scrubber at the TOP of Race Intelligence ────────────────────
    st.markdown("**⏱ Session Time — Master Scrubber**")

    import datetime
    min_t, max_t = _get_session_time_range(sel.df_driver)

    if min_t > 0:
        gap_dt = datetime.datetime(2000, 1, 1) + datetime.timedelta(seconds=min_t)
        formatted_gap = gap_dt.strftime('%H:%M:%S.%f')[:-5]
        st.info(f"Telemetry data for this session begins at **{formatted_gap}**. *This initial gap typically occurs because the driver was waiting in the garage before their first out-lap.*", icon="ℹ️")

    base_date = datetime.datetime(2000, 1, 1)
    min_dt = base_date + datetime.timedelta(seconds=min_t)
    max_dt = base_date + datetime.timedelta(seconds=max_t)

    # Initialize scrub_dt if needed
    if "scrub_dt" not in st.session_state:
        st.session_state["scrub_dt"] = base_date + datetime.timedelta(seconds=st.session_state.get("scrub_seconds", max_t))

    # Clamp session state value to valid range
    st.session_state["scrub_dt"] = max(min_dt, min(st.session_state["scrub_dt"], max_dt))

    selected_dt = st.slider(
        "Master session time",
        min_value=min_dt,
        max_value=max_dt,
        format="HH:mm:ss.S",
        key="scrub_dt",
        label_visibility="collapsed",
        step=datetime.timedelta(milliseconds=100)
    )
    
    # Sync back to scrub_seconds so the rest of the code works
    st.session_state["scrub_seconds"] = (selected_dt - base_date).total_seconds()

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        scrub = st.session_state.get("scrub_seconds", 0.0)
        render_cv_feed(scrub_seconds=scrub, min_t=min_t)

    with col_right:
        st.markdown("**LIVE TRACK MAP**")

        # Map mode (existing)
        map_mode = st.radio(
            "Map mode",
            options=["Speed", "Mistakes"],
            horizontal=True,
            label_visibility="collapsed",
        )
        mode_key = "speed" if map_mode == "Speed" else "mistakes"

        render_track_map(
            df_filtered=sel.df_driver,
            mode=mode_key,
            df_mistakes=df_mistakes,
            scrub_seconds=st.session_state.get("scrub_seconds", 0.0),
            racing_line_data=racing_line_data,
            show_optimal_line=False,
            show_ghost=False
        )

# ── Tab 2: Telemetry ─────────────────────────────────────────────────────────
with tab2:
    result = render_telemetry(
        df_driver=sel.df_driver,
        driver_name=sel.driver_name,
        driver_number=sel.driver_number,
        df_compare=sel.df_compare,
        compare_number=sel.compare_driver_number,
    )

# ── Tab 3: AI Analysis ───────────────────────────────────────────────────────
with tab3:
    render_ai_analysis(
        df_mistakes=df_mistakes,
        meta=meta,
        df_session=sel.df_driver,
        scrub_seconds=st.session_state.get("scrub_seconds", 0.0),
        tyre_data=tyre_data,
    )

with tab4:
    render_racing_line(
        racing_line_data=racing_line_data,
        driver_number=sel.driver_number,
        df_full=sel.df_full
    )

with tab5:
    render_bigdata_tab(df_full=sel.df_full)
