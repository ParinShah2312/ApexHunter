"""
ApexHunter Frontend - CV Feed
Renders the YOLO-processed video player, status badge, and CV stat cards.
"""

import math
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from config import PROCESSED_CSV_DIR, PROCESSED_VIDEO_DIR
from components.data_loader import load_cv_metrics


def _find_available_videos() -> List[Path]:
    """Return sorted list of .mp4 files in PROCESSED_VIDEO_DIR."""
    if not PROCESSED_VIDEO_DIR.exists():
        return []
    return sorted(PROCESSED_VIDEO_DIR.glob("*.mp4"))


def _get_current_status(df_cv: pd.DataFrame, scrub_seconds: float) -> str:
    """Return the apex status string for the frame closest to scrub_seconds."""
    closest_idx = (df_cv["timestamp_sec"] - scrub_seconds).abs().idxmin()
    return str(df_cv.loc[closest_idx, "status"])


def _status_badge_html(status: str) -> str:
    """Return the styled HTML badge for the given apex status string."""
    badge_map = {
        "Hitting Apex": ("#00ff8822", "#00ff88", "#00ff8844"),
        "Near Apex": ("#ffb80022", "#ffb800", "#ffb80044"),
        "Wide Line": ("#b142f522", "#b142f5", "#b142f544"),
        "Missing Apex": ("#ff3a3a22", "#ff3a3a", "#ff3a3a44"),
        "Tracking": ("#6b789022", "#8aa1c4", "#6b789044"),
        "Straight": ("#3a455822", "#6b7890", "#3a455844"),
    }
    bg, text, border = badge_map.get(status, ("#3a455822", "#6b7890", "#3a455844"))
    return (
        f'<div style="display:inline-block;padding:4px 12px;border-radius:4px;'
        f'border:1px solid {border};background:{bg};color:{text};font-weight:600;'
        f'font-size:13px;margin-bottom:8px">{status}</div>'
    )


def _render_proportion_bar(hitting: int, near: int, wide: int, missing: int, total_curb: int) -> None:
    """Render the colored proportion bar as st.markdown."""
    st.markdown(
        f'<div style="display:flex;height:6px;border-radius:3px;overflow:hidden;margin-top:4px">'
        f'<div style="flex:{hitting};background:#00ff88"></div>'
        f'<div style="flex:{near};background:#ffb800"></div>'
        f'<div style="flex:{wide};background:#b142f5"></div>'
        f'<div style="flex:{missing};background:#ff3a3a"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_cv_feed(scrub_seconds: float, min_t: float = 0.0) -> None:
    """Renders the CV video feed panel with status badge and stat cards."""
    mp4_files = _find_available_videos()
    if not mp4_files:
        st.info("No processed videos found. Run backend/scripts/run_inference.py first.")
        return

    selected_filename = st.selectbox(
        "Select pole lap video", options=[f.name for f in mp4_files]
    )

    csv_name = selected_filename.replace("_HUD.mp4", "_metrics.csv")
    csv_path = PROCESSED_CSV_DIR / csv_name
    df_cv = load_cv_metrics(str(csv_path))

    # Convert absolute session time to lap-relative video time (0-100s)
    video_scrub = max(0.0, scrub_seconds - min_t)

    # Status badge
    if df_cv is not None and not df_cv.empty:
        current_status = _get_current_status(df_cv, video_scrub)
        st.markdown(_status_badge_html(current_status), unsafe_allow_html=True)
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
        wide = int((df_cv["status"] == "Wide Line").sum())
        missing = int((df_cv["status"] == "Missing Apex").sum())
        total_curb_frames = int(df_cv["has_curb"].sum())
        closest_idx = (df_cv["timestamp_sec"] - video_scrub).abs().idxmin()
        current_dist = df_cv.loc[closest_idx, "distance_cm"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Hitting Apex", str(hitting))
        with c2:
            st.metric("Near Apex", str(near))
        with c3:
            st.metric("Wide Line", str(wide))
        with c4:
            st.metric("Missing Apex", str(missing))

        if total_curb_frames > 0:
            _render_proportion_bar(hitting, near, wide, missing, total_curb_frames)

        st.markdown("---")
        st.subheader("📋 Turn-by-Turn Report Card")

        BAHRAIN_TURN_WINDOWS = [
            (13.0, 16.0, "Right", "Slow", "Turn 1"),
            (17.0, 19.0, "Left", "Medium", "Turn 2"),
            (19.5, 20.5, "Right", "Medium", "Turn 3"),
            (27.0, 29.0, "Right", "Medium", "Turn 4"),
            (33.0, 34.0, "Left", "Fast", "Turn 5"),
            (34.0, 36.0, "Right", "Fast", "Turn 6"),
            (36.0, 38.0, "Left", "Fast", "Turn 7"),
            (41.0, 43.0, "Right", "Slow", "Turn 8"),
            (48.0, 50.0, "Left", "Fast", "Turn 9"),
            (51.0, 53.0, "Left", "Slow", "Turn 10"),
            (62.0, 67.0, "Left", "Fast", "Turn 11"),
            (68.0, 71.0, "Right", "Fast", "Turn 12"),
            (72.0, 77.0, "Right", "Fast", "Turn 13"),
            (85.0, 87.0, "Right", "Slow", "Turn 14"),
            (88.0, 90.0, "Right", "Medium", "Turn 15")
        ]

        report_data = []
        for start_t, end_t, direction, speed_cat, turn_name in BAHRAIN_TURN_WINDOWS:
            # Filter df_cv for this turn window
            df_turn = df_cv[(df_cv['timestamp_sec'] >= start_t) & (df_cv['timestamp_sec'] <= end_t)]
            df_turn_valid = df_turn.dropna(subset=['distance_cm'])

            if not df_turn_valid.empty:
                min_dist = df_turn_valid['distance_cm'].min()

                # Evaluate corner success by their closest point to the apex,
                # not the average of the whole 3-second corner entry/exit
                if min_dist <= 30.0:
                    dominant_status = "Hitting Apex"
                elif min_dist <= 60.0:
                    dominant_status = "Near Apex"
                elif min_dist <= 150.0:
                    dominant_status = "Wide Line"
                else:
                    dominant_status = "Missing Apex"
            else:
                min_dist = float('nan')
                dominant_status = "No Data"

            report_data.append({
                "Turn": turn_name,
                "Type": f"{direction} ({speed_cat})",
                "Closest Proximity (cm)": f"{min_dist:.1f}" if not pd.isna(min_dist) else "-",
                "Status": dominant_status
            })

        st.dataframe(
            pd.DataFrame(report_data),
            width='stretch',
            hide_index=True
        )

    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Hitting Apex", "—")
        with c2:
            st.metric("Near Apex", "—")
        with c3:
            st.metric("Wide Line", "—")
        with c4:
            st.metric("Missing Apex", "—")
