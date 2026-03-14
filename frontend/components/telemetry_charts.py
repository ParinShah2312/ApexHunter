"""
ApexHunter Frontend - Telemetry Charts
Renders the metrics row and the multi-subplot telemetry chart.
Uses downsampling for smooth chart rendering with large datasets.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.data_loader import downsample


def _get_labels(beginner_mode: bool) -> Dict[str, str]:
    """Returns display labels based on beginner mode setting.

    Args:
        beginner_mode: If True, returns simplified labels for non-technical users.

    Returns:
        Dictionary mapping metric keys to display strings.
    """
    if beginner_mode:
        return {
            "speed": "Vehicle Speed",
            "throttle": "Gas Pedal",
            "brake": "Brakes Applied",
            "rpm": "Engine Revs",
            "gear": "Current Gear",
        }
    return {
        "speed": "Speed (km/h)",
        "throttle": "Throttle (%)",
        "brake": "Brake",
        "rpm": "RPM",
        "gear": "nGear",
    }


def _get_time_col(df: pd.DataFrame) -> str:
    """Determines the best time column to use for time-based operations."""
    if "SessionTime" in df.columns and not df["SessionTime"].isnull().all():
        return "SessionTime"
    return "Time"


def _render_time_scrubber(df_driver: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Renders the time scrubber slider and returns filtered data + total seconds.

    Args:
        df_driver: DataFrame filtered to a single driver.

    Returns:
        Tuple of (filtered DataFrame up to scrub point, scrub time in seconds).
    """
    time_col = _get_time_col(df_driver)

    st.markdown("### Telemetry Playback")

    if pd.api.types.is_timedelta64_dtype(df_driver[time_col]):
        min_t = df_driver[time_col].dt.total_seconds().min()
        max_t = df_driver[time_col].dt.total_seconds().max()
        scrub = st.slider(
            "Scrub Session Time",
            min_value=float(min_t), max_value=float(max_t),
            value=float(max_t), format="%.1f s",
        )
        df_filtered = df_driver[df_driver[time_col].dt.total_seconds() <= scrub].copy()
    else:
        min_t = float(df_driver[time_col].min())
        max_t = float(df_driver[time_col].max())
        scrub = st.slider(
            "Scrub Session",
            min_value=min_t, max_value=max_t,
            value=max_t, format="%.1f",
        )
        df_filtered = df_driver[df_driver[time_col] <= scrub].copy()

    return df_filtered, scrub


def _render_metrics(df_filtered: pd.DataFrame, total_seconds: float, beginner_mode: bool) -> None:
    """Renders the Session Time, Corner Score, and Top Speed metric cards.

    Note: Metrics always use the FULL filtered dataset (no downsampling)
    to ensure accuracy.

    Args:
        df_filtered: DataFrame filtered to the scrub time range.
        total_seconds: Current scrub position in seconds.
        beginner_mode: Whether to use simplified labels.
    """
    hrs, remainder = divmod(total_seconds, 3600)
    mins, secs = divmod(remainder, 60)
    ms = (secs - int(secs)) * 1000
    time_str = f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}.{int(ms):03d}"

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    col1.metric("Session Time", time_str)

    top_speed = float(df_filtered["Speed"].max())
    col3.metric("Top Speed (km/h)" if not beginner_mode else "Max Speed", f"{top_speed:.1f}")

    # Perfect Corner Score: % of time NOT overlapping brake + throttle
    overlap = (df_filtered["Brake"] > 0) & (df_filtered["Throttle"] > 0)
    raw_score = 100 - (overlap.sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 100
    corner_score = float(np.clip(raw_score, 0, 100))

    with col2:
        st.metric("Perfect Corner Score", f"{corner_score:.1f}")
        st.progress(int(corner_score), text="Apex Accuracy")

    st.markdown("---")


def render_telemetry(
    df_driver: pd.DataFrame,
    driver_name: str,
    driver_number: str,
    beginner_mode: bool,
) -> Optional[Tuple[pd.DataFrame, Dict[str, str], str]]:
    """Main entry point: renders the telemetry playback section.

    Args:
        df_driver: DataFrame filtered to a single driver.
        driver_name: Human-readable driver name.
        driver_number: Driver's car number.
        beginner_mode: Whether to use simplified labels.

    Returns:
        Tuple of (filtered DataFrame, labels dict, hover template) for track map,
        or None if no data is available.
    """
    if df_driver.empty:
        st.warning(f"No telemetry data found for driver {driver_name} (#{driver_number}).")
        st.stop()
        return None

    # Info expander
    with st.expander("ℹ️ Why is there a gap in the telemetry data early in the session?"):
        st.write(
            "**Telemetry timestamps measure from when the official session window starts.** "
            "In a Race, the actual 'lights out' often occurs ~1 hour into the official "
            "SessionTime (formation lap, buildup, etc.). In Qualifying, data begins slightly "
            "before cars are released. You'll see a gap before the first recorded points."
        )

    labels = _get_labels(beginner_mode)
    df_filtered, total_seconds = _render_time_scrubber(df_driver)

    if df_filtered.empty:
        st.warning("No data in selected time range.")
        st.stop()
        return None

    # Metrics use FULL data for accuracy
    _render_metrics(df_filtered, total_seconds, beginner_mode)

    # Downsample for chart rendering performance only
    df_chart = downsample(df_filtered)
    time_col = _get_time_col(df_chart)

    x_data = (
        df_chart[time_col].dt.total_seconds()
        if pd.api.types.is_timedelta64_dtype(df_chart[time_col])
        else df_chart[time_col]
    )

    # ── Build subplots ────────────────────────────────────────────────────
    st.subheader(f"Telemetry Data - {driver_name} (#{driver_number})")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(labels["speed"], labels["throttle"], labels["brake"]),
    )

    hover_tpl = (
        "At this moment, the car is traveling at %{customdata[0]:.1f} km/h "
        "in %{customdata[1]} gear.<extra></extra>"
    )
    custom_data = np.stack((df_chart["Speed"], df_chart["nGear"]), axis=-1)

    for row, (y_col, color) in enumerate(
        [("Speed", "cyan"), ("Throttle", "green"), ("Brake", "red")], start=1
    ):
        fig.add_trace(
            go.Scatter(
                x=x_data, y=df_chart[y_col],
                name=labels.get(y_col.lower(), y_col),
                line=dict(color=color),
                customdata=custom_data,
                hovertemplate=hover_tpl,
            ),
            row=row, col=1,
        )

    fig.update_layout(height=650, showlegend=False, margin=dict(t=40, b=40, l=40, r=40))
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Return filtered data (full, not downsampled) for track map
    return df_filtered, labels, hover_tpl
