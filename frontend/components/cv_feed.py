"""
ApexHunter Frontend - CV Feed
Renders the YOLO-processed video player, status badge, and CV stat cards.
"""

import math
from pathlib import Path

import pandas as pd
import streamlit as st

from config import PROCESSED_CSV_DIR, PROCESSED_VIDEO_DIR
from components.data_loader import load_cv_metrics


def render_cv_feed(scrub_seconds: float) -> None:
    """Renders the CV video feed panel with status badge and stat cards."""

    if not PROCESSED_VIDEO_DIR.exists():
        st.info("No processed videos found. Run backend/scripts/run_inference.py first.")
        return

    mp4_files = sorted(PROCESSED_VIDEO_DIR.glob("*.mp4"))
    if not mp4_files:
        st.info("No processed videos found. Run backend/scripts/run_inference.py first.")
        return

    selected_filename = st.selectbox(
        "Select pole lap video", options=[f.name for f in mp4_files]
    )

    csv_name = selected_filename.replace("_HUD.mp4", "_metrics.csv")
    csv_path = PROCESSED_CSV_DIR / csv_name
    df_cv = load_cv_metrics(str(csv_path))

    # Status badge
    if df_cv is not None and not df_cv.empty:
        closest_idx = (df_cv["timestamp_sec"] - scrub_seconds).abs().idxmin()
        current_status = df_cv.loc[closest_idx, "status"]
        badge_map = {
            "Hitting Apex": ("#00ff8822", "#00ff88", "#00ff8844"),
            "Near Apex": ("#ffb80022", "#ffb800", "#ffb80044"),
            "Missing Apex": ("#ff3a3a22", "#ff3a3a", "#ff3a3a44"),
            "Straight": ("#3a455822", "#6b7890", "#3a455844"),
        }
        bg, text, border = badge_map.get(current_status, ("#3a455822", "#6b7890", "#3a455844"))
        st.markdown(
            f'<div style="display:inline-block;padding:4px 12px;border-radius:4px;'
            f'border:1px solid {border};background:{bg};color:{text};font-weight:600;'
            f'font-size:13px;margin-bottom:8px">{current_status}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="display:inline-block;padding:4px 12px;border-radius:4px;'
            'border:1px solid #3a455844;background:#3a455822;color:#6b7890;font-weight:600;'
            'font-size:13px;margin-bottom:8px">NO METRICS DATA</div>',
            unsafe_allow_html=True,
        )

    video_path = PROCESSED_VIDEO_DIR / selected_filename
    
    # Pass the absolute file path directly to st.video so Streamlit uses its internal media server
    # This enables efficient streaming and seeking (range requests) instead of crashing the websocket with 279MB of bytes.
    st.video(str(video_path.absolute()), format="video/mp4")
    st.caption("Note: video plays from start. Use the scrubber to navigate telemetry and map data.")

    # Stat cards
    if df_cv is not None and not df_cv.empty:
        hitting = int((df_cv["status"] == "Hitting Apex").sum())
        near = int((df_cv["status"] == "Near Apex").sum())
        missing = int((df_cv["status"] == "Missing Apex").sum())
        total_curb_frames = int(df_cv["has_curb"].sum())
        closest_idx = (df_cv["timestamp_sec"] - scrub_seconds).abs().idxmin()
        current_dist = df_cv.loc[closest_idx, "distance_px"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            dist_val = "N/A"
            try:
                if not math.isnan(float(current_dist)):
                    dist_val = f"{float(current_dist):.0f} px"
            except (ValueError, TypeError):
                pass
            st.metric("Dist to Apex", dist_val)
        with c2:
            st.metric("Hitting Apex", str(hitting))
        with c3:
            st.metric("Near Apex", str(near))
        with c4:
            st.metric("Missing Apex", str(missing))

        if total_curb_frames > 0:
            st.markdown(
                f'<div style="display:flex;height:6px;border-radius:3px;overflow:hidden;margin-top:4px">'
                f'<div style="flex:{hitting};background:#00ff88"></div>'
                f'<div style="flex:{near};background:#ffb800"></div>'
                f'<div style="flex:{missing};background:#ff3a3a"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Dist to Apex", "—")
        with c2:
            st.metric("Hitting Apex", "—")
        with c3:
            st.metric("Near Apex", "—")
        with c4:
            st.metric("Missing Apex", "—")
