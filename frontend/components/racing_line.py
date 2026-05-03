"""
ApexHunter Frontend - Racing Line
Renders the racing line analysis tab with algorithm path overlays,
track boundary polygons, and per-corner deviation charts.
"""
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ALGO_COLORS = {
    "astar": "#a855f7",     # purple
    "dijkstra": "#00d4ff",  # cyan
    "bfs": "#ef4444",       # bright red (distinct from cyan)
}
ALGO_LABELS = {
    "astar": "A*",
    "dijkstra": "Dijkstra",
    "bfs": "BFS",
}


def _build_racing_line_figure(
    racing_line_data: dict,
    algo_key: str,
    selected_algo: str,
    driver_number: str
) -> go.Figure:
    """Build the Plotly figure showing driver path and algorithm paths.
    Includes deviation shading and turn labels."""
    pts = np.array(racing_line_data.get("driver_path", []))
    if len(pts) > 0:
        pts_new = np.zeros_like(pts)
        pts_new[:, 0] = -pts[:, 1]
        pts_new[:, 1] = pts[:, 0]
        pts = pts_new

    algo_data = racing_line_data.get("algorithms", {}).get(algo_key, {})

    # Compute normal vectors to build inner/outer physical track edges
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    length = np.hypot(dx, dy)
    length[length == 0] = 1 # Avoid division by zero
    nx = -dy / length
    ny = dx / length

    # 300 units makes the track very wide, providing massive visibility
    # for the driver and algorithm paths inside it.
    half_width = 300

    outer_x = pts[:, 0] + nx * half_width
    outer_y = pts[:, 1] + ny * half_width
    inner_x = pts[:, 0] - nx * half_width
    inner_y = pts[:, 1] - ny * half_width

    # Close the loops (this will draw a straight line across the start/finish)
    outer_x = np.append(outer_x, outer_x[0])
    outer_y = np.append(outer_y, outer_y[0])
    inner_x = np.append(inner_x, inner_x[0])
    inner_y = np.append(inner_y, inner_y[0])

    # Combine outer and reversed inner to create a single filled polygon
    track_x = np.concatenate([outer_x, inner_x[::-1]])
    track_y = np.concatenate([outer_y, inner_y[::-1]])

    fig = go.Figure()

    # 1. Track Surface Polygon
    fig.add_trace(go.Scatter(
        x=track_x, y=track_y,
        fill="toself",
        fillcolor="#333333",  # Dark asphalt
        mode="lines",
        line=dict(color="#ff8c00", width=2), # Thin, crisp orange track limits
        name="Track Limits",
        hoverinfo="skip"
    ))

    # 2. Driver Path (Fastest Lap)
    fig.add_trace(go.Scatter(
        x=pts[:, 0], y=pts[:, 1],
        mode="lines",
        line=dict(color="#ffffff", width=2.5),
        name=f"Driver #{driver_number} (Fast Lap)"
    ))

    # 3. Add selected algorithm path
    opt_path = algo_data.get("path", [])
    # Rotate algorithm path to match the track rotation
    opt_x = [-p[1] for p in opt_path]
    opt_y = [p[0] for p in opt_path]
    fig.add_trace(go.Scatter(
        x=opt_x, y=opt_y,
        mode="lines",
        line=dict(color=ALGO_COLORS[algo_key], width=2.5),
        name=f"{selected_algo} Path"
    ))

    # 4. Add Annotations for Start/Finish and Turns
    if len(pts) >= 65:
        # We manually map the 15 Bahrain turns and provide exact X/Y coordinate offsets
        # to prevent any overlap and place them perfectly like the F1 graphic.
        # (index, label, dx, dy)
        bahrain_corners = [
            (7,  "T1",  -1500, -1500),  # bottom-left
            (9,  "T2",  -1500, 0),      # moved physically up the track
            (11, "T3",  -1500, 1000),   # left, moved physically further up the straight
            (17, "T4",  -1500, 1500),   # top-left
            (18, "T5",  0,     -1500),  # bottom (inside)
            (20, "T6",  0,     1500),   # moved further down the track
            (21, "T7",  -1000, -1000),  # bottom-left (inside)
            (27, "T8",  1500,  0),      # moved to right side
            (32, "T9",  0,     1500),   # top (inside)
            (34, "T10", 0,     -1500),  # moved straight down to the bottom
            (42, "T11", 1500,  -1000),  # moved to the right side
            (46, "T12", -1500, 500),    # left, moved up
            (50, "T13", 1500,  1500),   # top-right (outside)
            (59, "T14", 1500,  -1500),  # bottom-right (outside)
            (61, "T15", 0,     -1500)   # bottom (outside)
        ]

        # Start/Finish Line
        fig.add_annotation(
            x=pts[0, 0],
            y=pts[0, 1],
            ax=pts[0, 0],
            ay=pts[0, 1] - 2000,
            axref="x",
            ayref="y",
            text="Start / Finish",
            showarrow=True,
            arrowcolor="#ffffff",
            arrowwidth=1.5,
            arrowhead=0,
            font=dict(color="#ffffff", size=11, weight="bold"),
            bgcolor="rgba(15, 23, 42, 0.9)",
            bordercolor="#334155",
            borderwidth=1,
            borderpad=4
        )

        # Plot T1-T15
        for idx, label, dx, dy in bahrain_corners:
            fig.add_annotation(
                x=pts[idx, 0],
                y=pts[idx, 1],
                ax=pts[idx, 0] + dx,
                ay=pts[idx, 1] + dy,
                axref="x",
                ayref="y",
                text=label,
                showarrow=True,
                arrowcolor="#6b7890",
                arrowwidth=1,
                arrowhead=0,
                font=dict(color="#9ca3af", size=11, weight="bold"),
                bgcolor="rgba(15, 23, 42, 0.9)",
                bordercolor="#334155",
                borderwidth=1,
                borderpad=4
            )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1520",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        margin=dict(t=10, b=10, l=10, r=80),
        height=600,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#6b7890", size=11),
            x=1.01, y=1
        )
    )
    return fig


def _build_deviation_chart(
    racing_line_data: dict,
    algo_key: str
) -> go.Figure:
    """Build the per-corner deviation bar chart."""
    pts = np.array(racing_line_data.get("driver_path", []))
    if len(pts) > 0:
        pts_new = np.zeros_like(pts)
        pts_new[:, 0] = -pts[:, 1]
        pts_new[:, 1] = pts[:, 0]
        pts = pts_new

    algo_data = racing_line_data.get("algorithms", {}).get(algo_key, {})
    opt_path = algo_data.get("path", [])
    opt_x = [-p[1] for p in opt_path]
    opt_y = [p[0] for p in opt_path]
    scale = racing_line_data.get("coordinate_scale", 1.0)

    corners = []
    deviations = []

    bahrain_corners = [
        (7,  "T1",  -1500, -1500),
        (9,  "T2",  -1500, 0),
        (11, "T3",  -1500, 1000),
        (17, "T4",  -1500, 1500),
        (18, "T5",  0,     -1500),
        (20, "T6",  0,     1500),
        (21, "T7",  -1000, -1000),
        (27, "T8",  1500,  0),
        (32, "T9",  0,     1500),
        (34, "T10", 0,     -1500),
        (42, "T11", 1500,  -1000),
        (46, "T12", -1500, 500),
        (50, "T13", 1500,  1500),
        (59, "T14", 1500,  -1500),
        (61, "T15", 0,     -1500)
    ]

    opt_arr = np.column_stack((opt_x, opt_y))

    from scipy.interpolate import interp1d
    opt_dist = np.zeros(len(opt_arr))
    opt_dist[1:] = np.linalg.norm(opt_arr[1:] - opt_arr[:-1], axis=1)
    opt_cum_dist = np.cumsum(opt_dist)

    if opt_cum_dist[-1] > 0:
        f_x = interp1d(opt_cum_dist, opt_arr[:, 0], kind='linear')
        f_y = interp1d(opt_cum_dist, opt_arr[:, 1], kind='linear')
        fine_dist = np.linspace(0, opt_cum_dist[-1], 2000)
        opt_arr_fine = np.column_stack((f_x(fine_dist), f_y(fine_dist)))
    else:
        opt_arr_fine = opt_arr

    for idx, label, _, _ in bahrain_corners:
        corners.append(label)
        window_pts = pts[max(0, idx-1):min(len(pts), idx+2)]
        dists = []
        for dp in window_pts:
            dists.append(np.linalg.norm(opt_arr_fine - dp, axis=1).min())

        max_dev_units = max(dists) if dists else 0
        dev_m = max_dev_units * scale
        deviations.append(dev_m)

    dev_fig = go.Figure(go.Bar(
        x=corners,
        y=deviations,
        marker_color=ALGO_COLORS[algo_key],
        text=[f"{d:.1f}m" for d in deviations],
        textposition="auto"
    ))

    dev_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1520",
        xaxis=dict(title="Turn / Sector", showgrid=False, tickfont=dict(color="#6b7890")),
        yaxis=dict(title="Deviation (m)", showgrid=True, gridcolor="#1e293b", tickfont=dict(color="#6b7890")),
        margin=dict(t=20, b=20, l=20, r=20),
        height=300
    )
    return dev_fig


def render_racing_line(
    racing_line_data: Optional[dict],
    driver_number: str,
    df_full: Optional[pd.DataFrame] = None
) -> None:
    """
    Step 1 of the new UI build:
    Perfect Track Boundaries. We calculate normal vectors to plot a physical track
    polygon that won't overlap when zoomed out.
    """
    st.header("Racing Line Analysis")

    # ── Algorithm Selection ───────────────────────────────────────────────────
    selected_algo = st.radio("Select Algorithm", ["A* Search", "Dijkstra", "BFS"], horizontal=True)
    if selected_algo == "A* Search":
        algo_key = "astar"
    elif selected_algo == "Dijkstra":
        algo_key = "dijkstra"
    else:
        algo_key = "bfs"

    if not racing_line_data:
        st.warning(
            "No optimal racing line output found for this session and driver.\n\n"
            "Run: `python backend/scripts/optimal_line.py --session <path> --driver <code>`"
        )
        return

    pts = np.array(racing_line_data.get("driver_path", []))
    if len(pts) == 0:
        st.warning(
            "No optimal racing line output found for this session and driver.\n\n"
            "Run: `python backend/scripts/optimal_line.py --session <path> --driver <code>`"
        )
        return

    algo_data = racing_line_data.get("algorithms", {}).get(algo_key, {})

    if not algo_data or not algo_data.get("found"):
        st.warning(
            "No optimal racing line output found for this session and driver.\n\n"
            "Run: `python backend/scripts/optimal_line.py --session <path> --driver <code>`"
        )
        return

    fig = _build_racing_line_figure(racing_line_data, algo_key, selected_algo, driver_number)
    st.plotly_chart(fig, width="stretch")

    # ── Corner Deviation Chart ────────────────────────────────────────────────
    if len(pts) >= 65:
        st.markdown(f"### Corner-by-Corner Deviation ({selected_algo})")
        st.caption(f"Physical distance between the driver's path and the {selected_algo} line at each specific turn apex.")

        dev_fig = _build_deviation_chart(racing_line_data, algo_key)
        st.plotly_chart(dev_fig, width="stretch")
