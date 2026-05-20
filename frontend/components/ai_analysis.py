"""
ApexHunter Frontend - AI Analysis
Renders the AI Analysis tab with Isolation Forest panel (left) and LSTM placeholder (right).
"""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DRIVER_MAPPING
from components.data_loader import downsample



def _build_tyre_cliff_figure(stint: dict) -> go.Figure:
    """Build the full tyre cliff Plotly figure for one stint."""
    n_laps = stint["n_laps"]
    lap_nums = list(range(1, n_laps + 1))
    actual_time = stint["actual_laps"]
    predicted_time = stint["predicted_laps"]
    upper_time = stint["confidence_upper"]
    lower_time = stint["confidence_lower"]
    cliff_lap = stint.get("cliff_lap")

    def lap_color(i: int, total: int) -> str:
        ratio = i / max(1, total - 1)
        r = int(ratio * 255)
        g = int((1 - ratio) * 255)
        return f"rgb({r},{g},100)"

    fig = go.Figure()

    # Actual pace
    for i in range(len(actual_time) - 1):
        fig.add_trace(go.Scatter(
            x=[lap_nums[i], lap_nums[i + 1]],
            y=[actual_time[i], actual_time[i + 1]],
            mode="lines",
            line=dict(color=lap_color(i, len(actual_time)), width=2.5),
            showlegend=(i == 0),
            name="Actual lap time" if i == 0 else None,
            hovertemplate=f"Lap {lap_nums[i]}: %{{y:.2f}} s<extra></extra>",
        ))

    # Predicted trace
    pred_x = [lap_nums[i] for i, v in enumerate(predicted_time) if v is not None]
    pred_y = [v for v in predicted_time if v is not None]
    if pred_x:
        fig.add_trace(go.Scatter(
            x=pred_x, y=pred_y,
            mode="lines",
            line=dict(color="#a855f7", width=1.5, dash="dash"),
            name="LSTM prediction",
            hovertemplate="Predicted: %{y:.2f} s<extra></extra>",
        ))

    # Confidence band
    upper_x = [lap_nums[i] for i, v in enumerate(upper_time) if v is not None]
    upper_y = [v for v in upper_time if v is not None]
    lower_x = [lap_nums[i] for i, v in enumerate(lower_time) if v is not None]
    lower_y = [v for v in lower_time if v is not None]
    if upper_x and lower_x:
        fig.add_trace(go.Scatter(
            x=upper_x + lower_x[::-1],
            y=upper_y + lower_y[::-1],
            fill="toself",
            fillcolor="rgba(168,85,247,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="confidence_band",
        ))

    # Cliff vertical line
    if cliff_lap is not None:
        cliff_x = lap_nums[cliff_lap] if cliff_lap < len(lap_nums) else lap_nums[-1]
        fig.add_vline(
            x=cliff_x,
            line=dict(color="#ff3a3a", width=1.5, dash="dot"),
            annotation_text=f"CLIFF LAP {cliff_lap + 1}",
            annotation_position="top right",
            annotation_font=dict(color="#ff3a3a", size=10),
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1520",
        height=300,
        margin=dict(t=20, b=30, l=50, r=20),
        xaxis=dict(
            title="Lap in stint",
            tickfont=dict(color="#6b7890"),
            gridcolor="rgba(255,255,255,0.04)",
        ),
        yaxis=dict(
            title="Lap Time (s)",
            tickfont=dict(color="#6b7890"),
            gridcolor="rgba(255,255,255,0.04)",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#6b7890", size=11),
        ),
        showlegend=True,
    )
    return fig


def _render_isolation_forest_column(df_mistakes: Optional[pd.DataFrame], meta: Optional[dict]) -> None:
    """Render the full Isolation Forest card including map, stats, and CV scores."""
    if df_mistakes is None or meta is None:
        st.warning(
            "No Isolation Forest output found for this session and driver.\n\n"
            "Run: `python backend/scripts/detect_mistakes.py --session <path> --driver <code>`"
        )
        return

    contamination = meta["best_contamination"]
    total_mistakes = int(df_mistakes["is_mistake"].sum())
    total_rows = len(df_mistakes)
    rate = (total_mistakes / total_rows * 100) if total_rows > 0 else 0.0
    ref_name = DRIVER_MAPPING.get(meta["reference_driver"], meta["reference_driver"])

    st.markdown("**ISOLATION FOREST · Mistake Detection**")
    st.caption(
        f"contamination={contamination:.3f} · {total_mistakes} anomalies "
        f"({contamination:.3f}) · ref: {ref_name}"
    )

    # Lap selector — filter mistakes to a single lap
    df_display = df_mistakes
    if "LapNumber" in df_mistakes.columns:
        available_laps = sorted(df_mistakes[df_mistakes["LapNumber"] > 0]["LapNumber"].unique())
        if available_laps:
            lap_options = [f"Lap {lap}" for lap in available_laps]
            selected_lap_label = st.selectbox(
                "Select Lap", lap_options, key="ai_mistake_lap_selector"
            )
            selected_lap = available_laps[lap_options.index(selected_lap_label)]
            df_display = df_mistakes[df_mistakes["LapNumber"] == selected_lap]

    # Track map — full track background in neutral blue
    fig_map = go.Figure()
    df_map = downsample(df_mistakes, max_points=8000)

    fig_map.add_trace(
        go.Scattergl(
            x=df_map["X"],
            y=df_map["Y"],
            mode="markers",
            marker=dict(
                color="#3abdc7",
                size=3,
                opacity=0.5,
            ),
            showlegend=False,
            hovertemplate="X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>",
        )
    )

    # Mistake markers — only from the selected lap
    df_m = df_display[df_display["is_mistake"] == True]
    if not df_m.empty:
        fig_map.add_trace(
            go.Scatter(
                x=df_m["X"],
                y=df_m["Y"],
                mode="markers",
                marker=dict(symbol="x", size=10, color="#ff3a3a"),
                name="Mistake",
                customdata=df_m["anomaly_score"].values,
                text=df_m["Speed"].astype(str).values,
                hovertemplate="MISTAKE · score: %{customdata:.3f}<br>Speed: %{text}<extra></extra>",
            )
        )

    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1520",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="", scaleanchor="x", scaleratio=1),
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=True,
        height=300,
    )
    st.plotly_chart(fig_map, width='stretch')

    s1, s2, s3 = st.columns(3)
    with s1:
        lap_mistakes = int(df_display["is_mistake"].sum())
        st.metric("Anomalies", str(lap_mistakes))
    with s2:
        brake_override = int(
            (df_display["is_mistake"] & (df_display["brake_intensity"] > 0.5)).sum()
        )
        st.metric("Brake Override", str(brake_override))
    with s3:
        st.metric("Mistake Rate", f"{contamination:.3f}")




def _render_lstm_column(tyre_data: Optional[dict]) -> None:
    """Render the LSTM Tyre Cliff chart card including stint selector and stats."""
    st.markdown("**LSTM TYRE CLIFF PREDICTOR**")

    if tyre_data is None:
        st.warning(
            "No tyre prediction found for this session and driver.\n\n"
            "**Step 1 — Train the model** (only needed once):\n"
            "```\npython backend/scripts/train_lstm.py\n```\n\n"
            "**Step 2 — Run prediction**:\n"
            "```\npython backend/scripts/predict_cliff.py "
            "--session data_lake/clean_data/<file>.parquet "
            "--driver <code>\n```"
        )

        fig_placeholder = go.Figure()
        fig_placeholder.add_annotation(
            text="LSTM output will appear here",
            showarrow=False,
            font=dict(color="#3a4558", size=14),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        fig_placeholder.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0f1217",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=300,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_placeholder, width='stretch')
        return

    # ── Stint selector ────────────────────────────────────────────
    stints = tyre_data.get("stints", [])
    if len(stints) == 0:
        st.warning("Prediction file exists but contains no stint data.")
        return

    stint_options = [
        f"Stint {s['stint_index'] + 1} ({s['n_laps']} laps)"
        for s in stints
    ]
    selected_stint_label = st.selectbox(
        "Select stint", options=stint_options, key="tyre_stint_selector"
    )
    selected_idx = stint_options.index(selected_stint_label)
    stint = stints[selected_idx]

    fig = _build_tyre_cliff_figure(stint)
    st.plotly_chart(fig, width='stretch')

    # Stat cards - 2x2 grid for better readability
    c1, c2 = st.columns(2)
    with c1:
        cliff_lap = stint.get("cliff_lap")
        cliff_label = str(cliff_lap + 1) if cliff_lap is not None else "—"
        st.metric(
            "Predicted Cliff",
            f"Lap {cliff_label}" if cliff_lap is not None else "—",
        )
    with c2:
        lr = stint.get("laps_remaining")
        st.metric("Laps Remaining", str(lr) if lr is not None else "—")

    st.write("") # small spacing
    c3, c4 = st.columns(2)
    with c3:
        actual_time = stint["actual_laps"]
        valid_actuals = [t for t in actual_time if t is not None]
        current_pace = f"{valid_actuals[-1]:.2f}s" if valid_actuals else "—"
        st.metric("Current Pace", current_pace)
    with c4:
        predicted_time = stint["predicted_laps"]
        valid_preds = [t for t in predicted_time if t is not None]
        if valid_actuals and valid_preds:
            proj_pace = valid_preds[-1]
            diff = proj_pace - valid_actuals[-1]
            color = "normal" if diff <= 0 else "inverse"
            st.metric("Proj. End Pace", f"{proj_pace:.2f}s", f"{diff:+.2f}s", delta_color=color)
        else:
            st.metric("Proj. End Pace", "—")




def render_ai_analysis(
    df_mistakes: Optional[pd.DataFrame],
    meta: Optional[dict],
    df_session: pd.DataFrame,
    scrub_seconds: float,
    tyre_data: Optional[dict] = None,
) -> None:
    """Renders the AI Analysis tab content."""
    col_left, col_right = st.columns(2)

    with col_left:
        _render_isolation_forest_column(df_mistakes, meta)

    with col_right:
        _render_lstm_column(tyre_data)
