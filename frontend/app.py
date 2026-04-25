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

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ApexHunter",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS Injection ──────────────────────────────────────────────────────
st.markdown(
    """
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
        border-radius: 6px;
        padding: 12px;
    }

    /* 4. Monospace metric values */
    [data-testid="stMetricValue"] {
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.4rem;
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
    """,
    unsafe_allow_html=True,
)

# ── Disable typing in selectboxes (JS injection) ─────────────────────────────
import streamlit.components.v1 as components
components.html(
    """
    <script>
    const doc = window.parent.document;

    // Inject CSS into parent to hide the text cursor
    const style = doc.createElement('style');
    style.textContent = `
        div[data-baseweb="select"] input {
            caret-color: transparent !important;
        }
    `;
    doc.head.appendChild(style);

    function lockSelects() {
        doc.querySelectorAll('div[data-baseweb="select"] input').forEach(el => {
            if (el._locked) return;
            el._locked = true;
            el.setAttribute('readonly', 'true');

            // Block Backspace/Delete — BaseWeb handles them before readonly kicks in
            el.addEventListener('keydown', e => {
                if (e.key === 'Backspace' || e.key === 'Delete') {
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
    height=0,
)

# ── Imports ───────────────────────────────────────────────────────────────────
from components.sidebar import render_sidebar
from components.header_bar import render_header_bar
from components.telemetry_charts import render_telemetry
from components.track_map import render_track_map
from components.cv_feed import render_cv_feed
from components.ai_analysis import render_ai_analysis
from components.data_loader import load_mistake_data, load_mistake_meta

# ── Step 1: Sidebar ──────────────────────────────────────────────────────────
sel = render_sidebar()

# ── Step 2: Load AI data ─────────────────────────────────────────────────────
df_mistakes = load_mistake_data(sel.mistake_parquet_path)
meta = load_mistake_meta(sel.mistake_meta_path)

# ── Step 3: Header bar ───────────────────────────────────────────────────────
render_header_bar(sel, meta)

# ── Step 4: Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🏎 Race Intelligence", "📊 Telemetry", "🧠 AI Analysis"])

# ── Tab 1: Race Intelligence ─────────────────────────────────────────────────
with tab1:
    # ── Master scrubber at the TOP of Race Intelligence ────────────────────
    st.markdown("**⏱ Session Time — Master Scrubber**")

    time_col = "SessionTime" if "SessionTime" in sel.df_driver.columns else "Time"
    if pd.api.types.is_timedelta64_dtype(sel.df_driver[time_col]):
        min_t = float(sel.df_driver[time_col].dt.total_seconds().min())
        max_t = float(sel.df_driver[time_col].dt.total_seconds().max())
    else:
        min_t = float(sel.df_driver[time_col].min())
        max_t = float(sel.df_driver[time_col].max())

    # Clamp initial value to valid range
    scrub_init = st.session_state.get("scrub_seconds", max_t)
    scrub_init = max(min_t, min(scrub_init, max_t))

    st.slider(
        "Master session time",
        min_value=min_t,
        max_value=max_t,
        value=scrub_init,
        format="%.1f s",
        key="scrub_seconds",
        label_visibility="collapsed",
    )

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        scrub = st.session_state.get("scrub_seconds", 0.0)
        render_cv_feed(scrub_seconds=scrub)

    with col_right:
        st.markdown("**LIVE TRACK MAP**")

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
    )
