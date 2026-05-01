"""
ApexHunter Frontend - Track Map
Renders the scatter-plot track visualization with speed or mistakes coloring.
Includes driver position dot linked to the master scrubber.
"""

from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.data_loader import downsample


def _interpolate_path_position(
    path: List[List[float]],
    fraction: float
) -> Optional[Tuple[float, float]]:
    """Return the (x, y) coordinate at a given fraction along a path.
    fraction=0.0 is the start, fraction=1.0 is the end.
    Uses linear interpolation between path points by arc-length."""
    if not path or len(path) < 2:
        return None
    fraction = max(0.0, min(1.0, fraction))
    pts = np.array(path)
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]
    if total == 0:
        return (float(pts[0][0]), float(pts[0][1]))
    target = fraction * total
    idx = np.searchsorted(cumulative, target, side="right") - 1
    idx = min(idx, len(pts) - 2)
    seg_start = cumulative[idx]
    seg_end = cumulative[idx + 1]
    if seg_end == seg_start:
        return (float(pts[idx][0]), float(pts[idx][1]))
    t = (target - seg_start) / (seg_end - seg_start)
    x = float(pts[idx][0] + t * (pts[idx + 1][0] - pts[idx][0]))
    y = float(pts[idx][1] + t * (pts[idx + 1][1] - pts[idx][1]))
    return (x, y)


def _add_mistakes_traces(fig: go.Figure, df_mistakes: pd.DataFrame) -> None:
    """Add mistake analysis traces to the figure."""
    df_map = downsample(df_mistakes, max_points=8000)

    fig.add_trace(
        go.Scattergl(
            x=df_map["X"],
            y=df_map["Y"],
            mode="markers",
            marker=dict(
                color=df_map["anomaly_score"].values,
                colorscale=["#00ff88", "#ffb800", "#ff3a3a"],
                reversescale=False,
                cmin=-0.3,
                cmax=0.3,
                colorbar=dict(
                    title="Anomaly Score",
                    thickness=12,
                    tickfont=dict(color="#6b7890"),
                ),
                size=3,
                opacity=0.7,
            ),
            hovertemplate="Score: %{marker.color:.3f}<br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>",
            name="All points",
            showlegend=False,
        )
    )

    df_mistake_rows = df_mistakes[df_mistakes["is_mistake"] == True]
    if not df_mistake_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=df_mistake_rows["X"],
                y=df_mistake_rows["Y"],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=10,
                    color="#ff3a3a",
                    line=dict(color="#ff3a3a", width=2),
                ),
                name="Mistake",
                customdata=df_mistake_rows["anomaly_score"].values,
                hovertemplate="MISTAKE<br>Score: %{customdata:.3f}<extra></extra>",
            )
        )

def _add_speed_trace(fig: go.Figure, df_filtered: pd.DataFrame) -> None:
    """Add the speed-colored track trace to the figure."""
    df_map = downsample(df_filtered, max_points=8000)

    fig.add_trace(
        go.Scattergl(
            x=df_map["X"],
            y=df_map["Y"],
            mode="markers",
            marker=dict(
                color=df_map["Speed"].values,
                colorscale=["#ff3a3a", "#ffb800", "#00ff88"],
                colorbar=dict(
                    title="Speed (km/h)",
                    thickness=12,
                    tickfont=dict(color="#6b7890"),
                ),
                size=3,
                opacity=0.8,
            ),
            hovertemplate="Speed: %{marker.color:.1f} km/h<br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>",
            showlegend=False,
        )
    )

def _add_optimal_line_traces(
    fig: go.Figure,
    df_filtered: pd.DataFrame,
    racing_line_data: dict,
    scrub_seconds: float,
    show_optimal_line: bool,
    show_ghost: bool
) -> None:
    """Add optimal line and ghost dot traces to the figure."""
    astar = racing_line_data.get("algorithms", {}).get("astar", {})
    if not (astar.get("found", False) and astar.get("path")):
        return

    path = astar["path"]

    if show_optimal_line:
        opt_x = [p[0] for p in path]
        opt_y = [p[1] for p in path]
        fig.add_trace(go.Scatter(
            x=opt_x,
            y=opt_y,
            mode="lines",
            line=dict(color="#00d4ff", width=1.5, dash="dash"),
            name="A* Optimal Line",
            hovertemplate="Optimal line<extra></extra>",
            opacity=0.8
        ))

    if show_ghost and not df_filtered.empty:
        time_col = "SessionTime" if "SessionTime" in df_filtered.columns else "Time"
        if pd.api.types.is_timedelta64_dtype(df_filtered[time_col]):
            min_t = df_filtered[time_col].dt.total_seconds().min()
            max_t = df_filtered[time_col].dt.total_seconds().max()
        else:
            min_t = float(df_filtered[time_col].min())
            max_t = float(df_filtered[time_col].max())
            
        fraction = (scrub_seconds - min_t) / (max_t - min_t) if max_t > min_t else 0.0
        ghost_pos = _interpolate_path_position(path, fraction)
        
        if ghost_pos is not None:
            fig.add_trace(go.Scatter(
                x=[ghost_pos[0]],
                y=[ghost_pos[1]],
                mode="markers",
                marker=dict(
                    size=14,
                    color="rgba(255,255,255,0.35)",
                    symbol="circle",
                    line=dict(color="rgba(255,255,255,0.7)", width=1.5)
                ),
                name="Ghost (optimal)",
                hovertemplate="Ghost position (A* optimal)<extra></extra>"
            ))

def _add_driver_position(fig: go.Figure, df_filtered: pd.DataFrame, scrub_seconds: float) -> None:
    """Add the driver's current position to the figure based on scrub_seconds."""
    time_col = "SessionTime" if ("SessionTime" in df_filtered.columns and not df_filtered["SessionTime"].isnull().all()) else "Time"

    if pd.api.types.is_timedelta64_dtype(df_filtered[time_col]):
        time_seconds = df_filtered[time_col].dt.total_seconds()
    else:
        time_seconds = df_filtered[time_col].astype(float)

    if not time_seconds.empty:
        closest_idx = (time_seconds - scrub_seconds).abs().idxmin()
        driver_x = df_filtered.loc[closest_idx, "X"]
        driver_y = df_filtered.loc[closest_idx, "Y"]

        fig.add_trace(
            go.Scatter(
                x=[driver_x],
                y=[driver_y],
                mode="markers",
                marker=dict(
                    size=14,
                    color="#3b82f6",
                    symbol="circle",
                    line=dict(color="#93c5fd", width=2),
                ),
                name="Driver position",
                hovertemplate="Driver position<extra></extra>",
                showlegend=False,
            )
        )

def _build_track_figure(show_legend: bool) -> go.Figure:
    """Create the base figure layout for the track map."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1520",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title="",
            scaleanchor="x",
            scaleratio=1,
        ),
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=show_legend,
        height=420,
    )
    return fig


def render_track_map(
    df_filtered: pd.DataFrame,
    mode: str,
    df_mistakes: Optional[pd.DataFrame],
    scrub_seconds: float,
    racing_line_data: Optional[dict] = None,
    show_optimal_line: bool = False,
    show_ghost: bool = False
) -> None:
    """Renders the track map in speed or mistakes mode.

    Args:
        df_filtered: DataFrame filtered to current scrub range.
        mode: "speed" or "mistakes".
        df_mistakes: Isolation Forest annotated DataFrame, or None.
        scrub_seconds: Current scrub position in seconds.
        racing_line_data: JSON output from optimal_line.py or None.
        show_optimal_line: Whether to draw the A* dashed line.
        show_ghost: Whether to draw the ghost dot.
    """
    show_legend = False

    if mode == "mistakes" and df_mistakes is not None:
        show_legend = True
    elif mode == "mistakes" and df_mistakes is None:
        mode = "speed"
        st.warning("No Isolation Forest output found. Run detect_mistakes.py first.")

    fig = _build_track_figure(show_legend=show_legend)

    if mode == "mistakes" and df_mistakes is not None:
        _add_mistakes_traces(fig, df_mistakes)
    elif mode == "speed":
        _add_speed_trace(fig, df_filtered)

    if racing_line_data is not None and (show_optimal_line or show_ghost):
        _add_optimal_line_traces(fig, df_filtered, racing_line_data, scrub_seconds, show_optimal_line, show_ghost)

    _add_driver_position(fig, df_filtered, scrub_seconds)

    st.plotly_chart(fig, width='stretch')
