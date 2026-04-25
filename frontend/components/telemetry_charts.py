"""
ApexHunter Frontend - Telemetry Charts
Renders the metrics row, time scrubber, and the five-panel telemetry chart.
Supports compare driver overlay.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.data_loader import downsample


def render_telemetry(
    df_driver: pd.DataFrame,
    driver_name: str,
    driver_number: str,
    df_compare: Optional[pd.DataFrame] = None,
    compare_number: Optional[str] = None,
) -> Optional[Tuple[pd.DataFrame, str]]:
    """Main entry point: renders the telemetry playback section.

    Returns:
        Tuple of (filtered DataFrame, hover template) for use by the track map,
        or None if empty.
    """
    if df_driver.empty:
        st.warning(f"No telemetry data found for driver {driver_name} (#{driver_number}).")
        st.stop()
        return None

    # ── Metric Cards Row ──────────────────────────────────────────────────
    scrub_val = st.session_state.get("scrub_seconds", 0.0)
    hrs, remainder = divmod(scrub_val, 3600)
    mins, secs = divmod(remainder, 60)
    ms = (secs - int(secs)) * 1000
    time_str = f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}.{int(ms):03d}"

    top_speed = float(df_driver["Speed"].max())

    overlap_pct = 0.0
    if len(df_driver) > 0:
        overlap_pct = (
            ((df_driver["Brake"] > 0) & (df_driver["Throttle"] > 0)).sum()
            / len(df_driver)
            * 100
        )

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Session Time", time_str)
    with mc2:
        st.metric("Top Speed", f"{top_speed:.1f} km/h")
    with mc3:
        st.metric("Brake Overlap", f"{overlap_pct:.1f}%")
        st.caption("lower is better")

    # ── Time Scrubber ─────────────────────────────────────────────────────
    time_col = "SessionTime" if ("SessionTime" in df_driver.columns and not df_driver["SessionTime"].isnull().all()) else "Time"

    if pd.api.types.is_timedelta64_dtype(df_driver[time_col]):
        min_t = float(df_driver[time_col].dt.total_seconds().min())
        max_t = float(df_driver[time_col].dt.total_seconds().max())
    else:
        min_t = float(df_driver[time_col].min())
        max_t = float(df_driver[time_col].max())

    def _sync_scrub():
        st.session_state["scrub_seconds"] = st.session_state["telemetry_scrub_seconds"]

    # Clamp scrub value to valid range
    scrub_init = st.session_state.get("scrub_seconds", max_t)
    scrub_init = max(min_t, min(scrub_init, max_t))

    scrub = st.slider(
        "Session Time",
        min_value=min_t,
        max_value=max_t,
        value=scrub_init,
        format="%.1f s",
        key="telemetry_scrub_seconds",
        on_change=_sync_scrub,
    )

    # Filter to scrub position
    if pd.api.types.is_timedelta64_dtype(df_driver[time_col]):
        df_filtered = df_driver[df_driver[time_col].dt.total_seconds() <= scrub].copy()
    else:
        df_filtered = df_driver[df_driver[time_col] <= scrub].copy()

    if df_filtered.empty:
        st.warning("No data in selected time range.")
        return None

    # ── Build Five-Panel Chart ────────────────────────────────────────────
    df_chart = downsample(df_filtered)

    if pd.api.types.is_timedelta64_dtype(df_chart[time_col]):
        x_data = df_chart[time_col].dt.total_seconds()
    else:
        x_data = df_chart[time_col]

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Speed (km/h)", "Throttle %", "Brake %", "Gear", "RPM"),
    )

    # Primary driver traces  (row, column, color, fillcolor, fill, mode, line_shape, hovertemplate)
    trace_configs = [
        (1, "Speed",    "#00d4ff", "rgba(0,212,255,0.09)", "tozeroy", "lines", None,
         "At <b>%{x:.1f}s</b>, the car was travelling at <b>%{y:.1f} km/h</b><extra></extra>"),
        (2, "Throttle", "#00ff88", "rgba(0,255,136,0.09)", "tozeroy", "lines", None,
         "At <b>%{x:.1f}s</b>, throttle was at <b>%{y:.0f}%</b><extra></extra>"),
        (3, "Brake",    "#ff3a3a", "rgba(255,58,58,0.09)", "tozeroy", "lines", None,
         "At <b>%{x:.1f}s</b>, brake pressure was <b>%{y:.0f}%</b><extra></extra>"),
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

    # Compare driver traces
    if df_compare is not None and compare_number is not None:
        # Filter compare to same time window
        if pd.api.types.is_timedelta64_dtype(df_compare[time_col]):
            df_comp_filt = df_compare[df_compare[time_col].dt.total_seconds() <= scrub].copy()
        else:
            df_comp_filt = df_compare[df_compare[time_col] <= scrub].copy()

        if not df_comp_filt.empty:
            df_comp_chart = downsample(df_comp_filt)

            if pd.api.types.is_timedelta64_dtype(df_comp_chart[time_col]):
                x_comp = df_comp_chart[time_col].dt.total_seconds()
            else:
                x_comp = df_comp_chart[time_col]

            comp_configs = [
                (1, "Speed",    "rgba(0,212,255,0.53)", "lines", None,
                 "[Compare] At <b>%{x:.1f}s</b> — <b>%{y:.1f} km/h</b><extra></extra>"),
                (2, "Throttle", "rgba(0,255,136,0.53)", "lines", None,
                 "[Compare] At <b>%{x:.1f}s</b> — throttle <b>%{y:.0f}%</b><extra></extra>"),
                (3, "Brake",    "rgba(255,58,58,0.53)", "lines", None,
                 "[Compare] At <b>%{x:.1f}s</b> — brake <b>%{y:.0f}%</b><extra></extra>"),
                (4, "nGear",    "rgba(107,120,144,0.53)", "lines", "hv",
                 "[Compare] At <b>%{x:.1f}s</b> — <b>gear %{y:.0f}</b><extra></extra>"),
                (5, "RPM",      "rgba(168,85,247,0.53)", "lines", None,
                 "[Compare] At <b>%{x:.1f}s</b> — <b>%{y:,.0f} RPM</b><extra></extra>"),
            ]

            for row, col_name, color, mode, line_shape, htpl in comp_configs:
                fig.add_trace(
                    go.Scatter(
                        x=x_comp,
                        y=df_comp_chart[col_name],
                        mode=mode,
                        opacity=0.6,
                        line=dict(color=color, dash="dash", shape=line_shape) if line_shape else dict(color=color, dash="dash"),
                        name=f"{col_name} #{compare_number}",
                        hovertemplate=htpl,
                    ),
                    row=row,
                    col=1,
                )

    # Scrubber vertical line on all five subplots
    scrub_val = st.session_state.get("scrub_seconds", float(max_t))
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
        showlegend=True if df_compare is not None else False,
        margin=dict(t=40, b=20, l=50, r=20),
    )

    # Style grid lines for all axes
    for i in range(1, 6):
        xaxis_key = f"xaxis{i}" if i > 1 else "xaxis"
        yaxis_key = f"yaxis{i}" if i > 1 else "yaxis"
        fig.update_layout(
            **{
                xaxis_key: dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.07)"),
                yaxis_key: dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.07)"),
            }
        )

    st.plotly_chart(fig, width='stretch')

    hover_template = "Speed: %{y:.1f} km/h<extra></extra>"
    return (df_filtered, hover_template)
