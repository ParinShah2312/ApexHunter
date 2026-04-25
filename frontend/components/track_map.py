"""
ApexHunter Frontend - Track Map
Renders the scatter-plot track visualization with speed or mistakes coloring.
Includes driver position dot linked to the master scrubber.
"""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.data_loader import downsample


def render_track_map(
    df_filtered: pd.DataFrame,
    mode: str,
    df_mistakes: Optional[pd.DataFrame],
    scrub_seconds: float,
) -> None:
    """Renders the track map in speed or mistakes mode.

    Args:
        df_filtered: DataFrame filtered to current scrub range.
        mode: "speed" or "mistakes".
        df_mistakes: Isolation Forest annotated DataFrame, or None.
        scrub_seconds: Current scrub position in seconds.
    """
    fig = go.Figure()
    show_legend = False

    if mode == "mistakes" and df_mistakes is not None:
        # ── Mistakes Mode ─────────────────────────────────────────────────
        show_legend = True
        df_map = downsample(df_mistakes, max_points=8000)

        # Trace 1 — all points colored by anomaly_score
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

        # Trace 2 — mistake markers only (no downsampling — sparse and important)
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

    elif mode == "mistakes" and df_mistakes is None:
        # Fall back to speed mode and show warning
        mode = "speed"
        st.warning("No Isolation Forest output found. Run detect_mistakes.py first.")

    if mode == "speed":
        # ── Speed Mode ────────────────────────────────────────────────────
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

    # ── Driver Position Dot ───────────────────────────────────────────────
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

    # ── Layout ────────────────────────────────────────────────────────────
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

    st.plotly_chart(fig, width='stretch')
