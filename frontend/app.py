"""
================================================================================
  ApexHunter - F1 Data Explorer Dashboard
  Entry Point: app.py
--------------------------------------------------------------------------------
  Purpose : Slim orchestrator that assembles the dashboard from components.
            Run with: streamlit run frontend/app.py
================================================================================
"""

import streamlit as st
import streamlit.components.v1 as components

from components.sidebar import render_sidebar
from components.telemetry_charts import render_telemetry
from components.track_map import render_track_map

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ApexHunter Dashboard", layout="wide")

# ── Disable typing in selectboxes (force dropdown-only) ──────────────────────
components.html(
    """
<script>
const doc = window.parent.document;
doc.addEventListener('keydown', function(e) {
    if (e.target && e.target.matches('div[data-baseweb="select"] input')) {
        if (!['Tab', 'Enter', 'Escape', 'ArrowUp', 'ArrowDown'].includes(e.key)) {
            e.preventDefault();
            e.stopPropagation();
            return false;
        }
    }
}, true);
const applyReadOnly = () => {
    const inputs = doc.querySelectorAll('div[data-baseweb="select"] input');
    inputs.forEach(input => {
        if (!input.hasAttribute('readonly')) {
            input.setAttribute('readonly', 'readonly');
        }
    });
};
const observer = new MutationObserver(applyReadOnly);
if (doc.body) { observer.observe(doc.body, { childList: true, subtree: true }); }
applyReadOnly();
</script>
""",
    height=0,
    width=0,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏎️ ApexHunter F1 Data Explorer")

# ── Sidebar → selections + filtered DataFrame ────────────────────────────────
sel = render_sidebar()

# ── Tabbed Layout: only the active tab renders (performance boost) ────────────
tab_telemetry, tab_track = st.tabs(["📊 Telemetry", "🗺️ Track Map"])

with tab_telemetry:
    result = render_telemetry(
        df_driver=sel.df_driver,
        driver_name=sel.driver_name,
        driver_number=sel.driver_number,
    )

with tab_track:
    if result is not None:
        df_filtered, labels, hover_tpl = result
        render_track_map(
            df_filtered=df_filtered,
            speed_label=labels["speed"],
            hover_template=hover_tpl,
        )
    else:
        st.info("Select a driver with available data to view the track map.")
