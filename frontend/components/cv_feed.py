"""
ApexHunter Frontend - CV Feed
Renders the YOLO-processed video player, status badge, and CV stat cards.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

from config import PROCESSED_CSV_DIR, PROCESSED_VIDEO_DIR
from components.data_loader import load_cv_metrics, load_turn_windows, load_video_lap


def _find_available_videos() -> List[Path]:
    """Return sorted list of .mp4 files in PROCESSED_VIDEO_DIR."""
    if not PROCESSED_VIDEO_DIR.exists():
        return []
    return sorted(PROCESSED_VIDEO_DIR.glob("*.mp4"))


def _get_current_status(df_cv: pd.DataFrame, scrub_seconds: float) -> str:
    """Return the apex status string for the frame closest to scrub_seconds."""
    closest_idx = (df_cv["timestamp_sec"] - scrub_seconds).abs().idxmin()
    return str(df_cv.loc[closest_idx, "status"])


def video_time_for(scrub_seconds: float, video_lap: Optional[dict], year: int, round_num: int,
                   session_type: str, driver_number: str) -> Optional[float]:
    """Map the session-time scrubber onto the processed video, or None if it can't be.

    The video shows one lap (video_lap: year, round, session, driver_number, lap_start_s,
    lap_end_s, video_offset_s). Only when that session and driver are selected and the scrubber is
    inside the lap is the video time = scrub - lap start + video offset. The old mapping
    (scrub - start of the selected session's telemetry) gave times far past the end of the video,
    so the badge always showed the last frame.
    """
    if video_lap is None:
        return None
    same_lap = (int(video_lap["year"]) == int(year) and int(video_lap["round"]) == int(round_num)
                and video_lap["session"] == session_type
                and str(video_lap["driver_number"]) == str(driver_number))
    if not same_lap or not video_lap["lap_start_s"] <= scrub_seconds <= video_lap["lap_end_s"]:
        return None
    return scrub_seconds - video_lap["lap_start_s"] + video_lap["video_offset_s"]


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


def render_cv_feed(scrub_seconds: float, year: int, round_num: int, session_type: str,
                   driver_number: str) -> None:
    """Renders the CV video feed panel with status badge and stat cards."""
    mp4_files = _find_available_videos()
    if not mp4_files:
        st.info("No processed videos found. Run backend/scripts/run_inference.py first.")
        return

    selected_filename = st.selectbox(
        "Select pole lap video", options=[f.name for f in mp4_files], filter_mode=None
    )

    csv_name = selected_filename.replace("_HUD.mp4", "_metrics.csv")
    csv_path = PROCESSED_CSV_DIR / csv_name
    # mtime in the cache key: a re-processed video's CSV used to be served stale until restart
    df_cv = load_cv_metrics(str(csv_path), mtime=csv_path.stat().st_mtime if csv_path.exists() else 0)
    turns_path = str(PROCESSED_CSV_DIR / selected_filename.replace("_HUD.mp4", "_turns.json"))
    video_lap = load_video_lap(turns_path)
    video_t = video_time_for(scrub_seconds, video_lap, year, round_num, session_type, driver_number)

    # Status badge: the frame at the scrubber's moment, when the scrubber is inside the video's lap
    if df_cv is not None and not df_cv.empty and video_t is not None:
        current_status = _get_current_status(df_cv, video_t)
        st.markdown(_status_badge_html(current_status), unsafe_allow_html=True)
    elif df_cv is not None and not df_cv.empty:
        if video_lap is None:
            hint = "re-run cv/run_inference.py to record which lap this video shows"
        else:
            hint = (f"this video is {video_lap['year']} round {video_lap['round']} qualifying, "
                    f"car #{video_lap['driver_number']}; select it and move the scrubber into that lap")
        st.markdown(_status_badge_html("Not in this lap"), unsafe_allow_html=True)
        st.caption(f"Live status follows the scrubber only inside the video's lap: {hint}.")
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

        turns = load_turn_windows(turns_path)
        if turns is None:
            st.info("No turn windows for this video. Re-run cv/run_inference.py with --video-offset "
                    "to generate them from FastF1 circuit data.")
            turns = []

        report_data = []
        for turn in turns:
            # Filter df_cv for this turn window
            df_turn = df_cv[(df_cv['timestamp_sec'] >= turn["start_s"]) & (df_cv['timestamp_sec'] <= turn["end_s"])]
            df_turn_valid = df_turn.dropna(subset=['distance_cm'])

            if not df_turn_valid.empty:
                # Rate the corner by its closest approach, using the status the backend gave that
                # frame (pixel thresholds scaled by corner speed). The card used to apply its own
                # 30/60/150 cm cut-offs, which disagreed with the per-frame classification.
                closest = df_turn_valid['distance_cm'].idxmin()
                min_dist = df_turn_valid.loc[closest, 'distance_cm']
                dominant_status = str(df_turn_valid.loc[closest, 'status'])
            else:
                min_dist = float('nan')
                dominant_status = "No Data"

            report_data.append({
                "Turn": turn["turn"],
                "Type": f"{turn['direction']} ({turn['speed_category']})",
                "Closest Proximity (cm)": f"{min_dist:.1f}" if not pd.isna(min_dist) else "-",
                "Status": dominant_status
            })

        if report_data:
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
