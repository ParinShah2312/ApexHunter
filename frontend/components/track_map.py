"""
ApexHunter Frontend - Track Map
Renders the scatter-plot track visualization colored by speed.
Uses WebGL (Scattergl) for GPU-accelerated rendering of large point clouds.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.data_loader import downsample


def render_track_map(
    df_filtered: pd.DataFrame,
    speed_label: str,
    hover_template: str,
) -> None:
    """Renders the traffic-light style track map using GPU-accelerated WebGL.

    The data is downsampled to max 8000 points for buttery smooth pan/zoom,
    and rendered with Scattergl (WebGL) instead of standard SVG scatter.

    Args:
        df_filtered: DataFrame with X, Y, Speed, nGear columns.
        speed_label: Display label for the speed axis/colorbar.
        hover_template: Plotly hover template string.
    """
    st.subheader("Traffic Light Track Map")

    # Downsample for rendering (higher limit than charts since map is single trace)
    df_map = downsample(df_filtered, max_points=8000)

    speed_vals = df_map["Speed"].values
    custom_data = list(zip(df_map["Speed"], df_map["nGear"]))

    fig = go.Figure(
        go.Scattergl(
            x=df_map["X"],
            y=df_map["Y"],
            mode="markers",
            marker=dict(
                color=speed_vals,
                colorscale=["red", "yellow", "green"],
                colorbar=dict(title=speed_label),
                size=4,
            ),
            customdata=custom_data,
            hovertemplate=hover_template,
        )
    )

    fig.update_layout(
        xaxis=dict(title="X Coordinate"),
        yaxis=dict(title="Y Coordinate", scaleanchor="x", scaleratio=1),
        margin=dict(t=40, b=40, l=40, r=40),
    )

    st.plotly_chart(fig, use_container_width=True)
