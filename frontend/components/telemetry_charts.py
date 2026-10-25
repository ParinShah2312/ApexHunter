"""
ApexHunter Frontend - Telemetry Charts
Renders the metrics row, time scrubber, and the five-panel telemetry chart.
"""

from typing import Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.data_loader import downsample
from components.scrubber import current_scrub, render_telemetry_scrubber

# Brake overlap only counts samples at or above this speed. Below it the car is in the garage, in
# its pit box or on the grid, where brake and throttle are often held together on purpose (2024 R1
# Q #11: 1,447 of 1,541 overlap samples were at 0 km/h, turning 0.67 % into 6.05 %).
MIN_OVERLAP_SPEED_KMH = 50.0


def _get_time_col(df: pd.DataFrame) -> str:
    """Return the primary time column name."""
    if "SessionTime" in df.columns and not df["SessionTime"].isnull().all():
        return "SessionTime"
    return "Time"


def _compute_brake_overlap(df: pd.DataFrame) -> float:
    """Return the percentage of on-track samples (Speed >= MIN_OVERLAP_SPEED_KMH) where the brake
    is on AND Throttle > 5%.

    Brake is a binary on/off flag in FastF1; the 5% throttle threshold matches the
    Isolation Forest brake_throttle_overlap feature and filters out pedal sensor noise.
    """
    on_track = df[df["Speed"] >= MIN_OVERLAP_SPEED_KMH]
    if len(on_track) == 0:
        return 0.0
    overlap = (on_track["Brake"] >= 0.5) & (on_track["Throttle"] > 5)
    return float(overlap.sum() / len(on_track) * 100)


def _render_metric_cards(df_driver: pd.DataFrame, scrub_seconds: float) -> None:
    """Render the three KPI metric cards row."""
    # Whole milliseconds first: truncating the float showed e.g. 1622.599 s as "27:02.598"
    total_s, ms = divmod(round(scrub_seconds * 1000), 1000)
    hours, remainder = divmod(total_s, 3600)
    mins, secs = divmod(remainder, 60)
    if hours > 0:
        time_str = f"{hours:02d}:{mins:02d}:{secs:02d}.{ms:03d}"
    else:
        time_str = f"{mins:02d}:{secs:02d}.{ms:03d}"

    top_speed = float(df_driver["Speed"].max()) if not df_driver.empty else 0.0
    overlap_pct = _compute_brake_overlap(df_driver)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Session Time", time_str)
    with mc2:
        st.metric("Top Speed", f"{top_speed:.1f} km/h")
    with mc3:
        st.metric("Brake Overlap", f"{overlap_pct:.2f}%")
        st.caption(f"brake + throttle at once, samples ≥ {MIN_OVERLAP_SPEED_KMH:.0f} km/h · lower is better")


def _build_telemetry_figure(
    df_chart: pd.DataFrame,
    x_data: pd.Series,
    driver_number: str,
    scrub_val: float
) -> go.Figure:
    """Build the 5-panel Plotly subplot figure."""
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Speed (km/h)", "Throttle %", "Brake (on/off)", "Gear", "RPM"),
    )

    # Primary driver traces
    trace_configs = [
        (1, "Speed",    "#00d4ff", "rgba(0,212,255,0.09)", "tozeroy", "lines", None,
         "At <b>%{x:.1f}s</b>, the car was travelling at <b>%{y:.1f} km/h</b><extra></extra>"),
        (2, "Throttle", "#00ff88", "rgba(0,255,136,0.09)", "tozeroy", "lines", None,
         "At <b>%{x:.1f}s</b>, throttle was at <b>%{y:.0f}%</b><extra></extra>"),
        (3, "Brake",    "#ff3a3a", "rgba(255,58,58,0.09)", "tozeroy", "lines", "hv",
         "At <b>%{x:.1f}s</b>, brake was <b>%{y:.0f}</b> (1 = on, 0 = off)<extra></extra>"),
        (4, "nGear",    "#6b7890", None, None, "lines", "hv",
         "At <b>%{x:.1f}s</b>, the car was in <b>gear %{y:.0f}</b><extra></extra>"),
        (5, "RPM",      "#a855f7", None, None, "lines", None,
         "At <b>%{x:.1f}s</b>, the engine was at <b>%{y:,.0f} RPM</b><extra></extra>"),
    ]

    for row, col_name, color, fillcolor, fill, mode, line_shape, htpl in trace_configs:
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=df_chart[col_name],
                mode=mode,
                fill=fill,
                line=dict(color=color, shape=line_shape) if line_shape else dict(color=color),
                fillcolor=fillcolor,
                name=f"{col_name} #{driver_number}",
                hovertemplate=htpl,
            ),
            row=row,
            col=1,
        )


    # Scrubber vertical line
    shapes = []
    for row in range(1, 6):
        xref = "x" if row == 1 else f"x{row}"
        yref = "y domain" if row == 1 else f"y{row} domain"
        shapes.append(
            dict(
                type="line",
                x0=scrub_val,
                x1=scrub_val,
                y0=0,
                y1=1,
                xref=xref,
                yref=yref,
                line=dict(color="rgba(255,255,255,0.4)", width=1, dash="dot"),
            )
        )

    fig.update_layout(
        shapes=shapes,
        height=580,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f1217",
        font=dict(color="#6b7890", size=11),
        showlegend=False,
        margin=dict(t=40, b=20, l=50, r=20),
    )

    for i in range(1, 6):
        xaxis_key = f"xaxis{i}" if i > 1 else "xaxis"
        yaxis_key = f"yaxis{i}" if i > 1 else "yaxis"
        fig.update_layout(
            **{
                xaxis_key: dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.07)"),
                yaxis_key: dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.07)"),
            }
        )

    return fig


def render_telemetry(
    df_driver: pd.DataFrame,
    driver_name: str,
    driver_number: str,
) -> Optional[Tuple[pd.DataFrame, str]]:
    """Main entry point: renders the telemetry playback section.

    Returns:
        Tuple of (filtered DataFrame, hover template) for use by the track map,
        or None if empty.
    """
    if df_driver.empty:
        # Not st.stop(): that stopped every tab, not just this one
        st.warning(f"No telemetry data found for driver {driver_name} (#{driver_number}).")
        return None

    time_col = _get_time_col(df_driver)

    if pd.api.types.is_timedelta64_dtype(df_driver[time_col]):
        min_t = float(df_driver[time_col].dt.total_seconds().min())
        max_t = float(df_driver[time_col].dt.total_seconds().max())
    else:
        min_t = float(df_driver[time_col].min())
        max_t = float(df_driver[time_col].max())

    # Cards, slider, chart cut-off and scrub line all use the one shared position
    scrub = current_scrub(min_t, max_t)
    _render_metric_cards(df_driver, scrub)
    render_telemetry_scrubber(min_t, max_t)

    # Filter to scrub position
    if pd.api.types.is_timedelta64_dtype(df_driver[time_col]):
        df_filtered = df_driver[df_driver[time_col].dt.total_seconds() <= scrub].copy()
    else:
        df_filtered = df_driver[df_driver[time_col] <= scrub].copy()

    if df_filtered.empty:
        st.warning("No data in selected time range.")
        return None

    # Downsample and prepare chart data
    df_chart = downsample(df_filtered)
    if pd.api.types.is_timedelta64_dtype(df_chart[time_col]):
        x_data = df_chart[time_col].dt.total_seconds()
    else:
        x_data = df_chart[time_col]

    # Build and render figure
    fig = _build_telemetry_figure(
        df_chart=df_chart,
        x_data=x_data,
        driver_number=driver_number,
        scrub_val=scrub
    )
    st.plotly_chart(fig, width='stretch')

    hover_template = "Speed: %{y:.1f} km/h<extra></extra>"
    return (df_filtered, hover_template)
