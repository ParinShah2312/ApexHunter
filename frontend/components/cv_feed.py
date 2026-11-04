"""
ApexHunter Frontend - CV Feed
Renders the YOLO-processed video player, the corner counts and the corner-by-corner table.
"""

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

from config import PROCESSED_CSV_DIR, PROCESSED_VIDEO_DIR
from components.data_loader import load_cv_metrics, load_turn_windows


def _find_available_videos() -> List[Path]:
    """Return sorted list of .mp4 files in PROCESSED_VIDEO_DIR."""
    if not PROCESSED_VIDEO_DIR.exists():
        return []
    return sorted(PROCESSED_VIDEO_DIR.glob("*.mp4"))


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


def render_cv_feed() -> None:
    """Renders the processed pole-lap video (it has its own time slider), how many corners hit the
    apex, and the corner-by-corner table."""
    mp4_files = _find_available_videos()
    if not mp4_files:
        st.info("No processed videos found. Run backend/scripts/cv/run_inference.py first.")
        return

    selected_filename = st.selectbox(
        "Select pole lap video", options=[f.name for f in mp4_files], filter_mode=None
    )

    csv_name = selected_filename.replace("_HUD.mp4", "_metrics.csv")
    csv_path = PROCESSED_CSV_DIR / csv_name
    # mtime in the cache key: a re-processed video's CSV used to be served stale until restart
    df_cv = load_cv_metrics(str(csv_path), mtime=csv_path.stat().st_mtime if csv_path.exists() else 0)
    turns_path = str(PROCESSED_CSV_DIR / selected_filename.replace("_HUD.mp4", "_turns.json"))

    video_path = PROCESSED_VIDEO_DIR / selected_filename

    # Pass the absolute file path directly to st.video so Streamlit uses its internal media server
    # This enables efficient streaming and seeking (range requests) instead of crashing the websocket with 279MB of bytes.
    st.video(str(video_path.absolute()), format="video/mp4")
    st.caption("The video shows each frame's apex status on screen; use its own slider to move through the lap.")

    turns = load_turn_windows(turns_path)
    lap = _video_lap(turns_path)
    if lap:
        st.caption(f"This video: {lap['year']} round {lap['round']} "
                   f"{'qualifying' if lap['session'] == 'Q' else 'race'}, {lap['driver']} "
                   f"(#{lap['driver_number']}), fastest lap.")
    if df_cv is None or df_cv.empty:
        st.info("No CV metrics for this video. Run backend/scripts/cv/run_inference.py first.")
        return
    if turns is None:
        st.info("No turn windows for this video. Re-run cv/run_inference.py with --video-offset "
                "to generate them from FastF1 circuit data.")
        return

    report = corner_report(df_cv, turns)
    counts = report["Status"].value_counts()
    rated = int(report["Status"].isin(STATUSES).sum())
    cols = st.columns(len(STATUSES))
    for col, status in zip(cols, STATUSES):
        col.metric(status, f"{int(counts.get(status, 0))} of {rated}",
                   help="Corners rated by the closest the wheel got to the apex curb in that corner")
    if rated:
        _render_proportion_bar(*(int(counts.get(s_, 0)) for s_ in STATUSES), rated)

    st.markdown("##### Corner by corner")
    st.dataframe(report, width="stretch", hide_index=True)


STATUSES = ("Hitting Apex", "Near Apex", "Wide Line", "Missing Apex")


def corner_report(df_cv: pd.DataFrame, turns: List[dict]) -> pd.DataFrame:
    """One row per corner: rated by its closest approach to the apex curb, with the status the
    backend gave that frame (pixel thresholds scaled by corner speed). The cards count these
    corners; they used to count video frames."""
    rows = []
    for turn in turns:
        df_turn = df_cv[(df_cv["timestamp_sec"] >= turn["start_s"]) & (df_cv["timestamp_sec"] <= turn["end_s"])]
        valid = df_turn.dropna(subset=["distance_cm"])
        if not valid.empty:
            closest = valid["distance_cm"].idxmin()
            min_dist, status = valid.loc[closest, "distance_cm"], str(valid.loc[closest, "status"])
        else:
            min_dist, status = float("nan"), "No Data"
        rows.append({"Turn": turn["turn"], "Type": f"{turn['direction']} ({turn['speed_category']})",
                     "Closest to the apex curb (cm)": "-" if pd.isna(min_dist) else f"{min_dist:.1f}",
                     "Status": status})
    return pd.DataFrame(rows, columns=["Turn", "Type", "Closest to the apex curb (cm)", "Status"])


def _video_lap(turns_path: str) -> Optional[dict]:
    """Which lap the video shows (recorded by cv/run_inference.py in the turns file), or None."""
    try:
        with open(turns_path, encoding="utf-8") as f:
            return json.load(f).get("lap")
    except (OSError, ValueError):
        return None
