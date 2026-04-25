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


def render_ai_analysis(
    df_mistakes: Optional[pd.DataFrame],
    meta: Optional[dict],
    df_session: pd.DataFrame,
    scrub_seconds: float,
) -> None:
    """Renders the AI Analysis tab content."""
    col_left, col_right = st.columns(2)

    # ── LEFT COLUMN: Isolation Forest ─────────────────────────────────────
    with col_left:
        if df_mistakes is None or meta is None:
            st.warning(
                "No Isolation Forest output found for this session and driver.\n\n"
                "Run: `python backend/scripts/detect_mistakes.py --session <path> --driver <code>`"
            )
        else:
            contamination = meta["best_contamination"]
            total_mistakes = meta["total_mistakes"]
            rate = meta["mistake_rate_pct"]
            ref_name = DRIVER_MAPPING.get(meta["reference_driver"], meta["reference_driver"])

            st.markdown("**ISOLATION FOREST · Mistake Detection**")
            st.caption(
                f"contamination={contamination:.2f} · {total_mistakes} anomalies "
                f"({rate:.1f}%) · ref: {ref_name}"
            )

            # Track map colored by anomaly_score
            fig_map = go.Figure()
            df_map = downsample(df_mistakes, max_points=8000)

            fig_map.add_trace(
                go.Scattergl(
                    x=df_map["X"],
                    y=df_map["Y"],
                    mode="markers",
                    marker=dict(
                        color=df_map["anomaly_score"].values,
                        colorscale=["#00ff88", "#ffb800", "#ff3a3a"],
                        cmin=-0.3,
                        cmax=0.3,
                        colorbar=dict(title="Anomaly Score", thickness=12, tickfont=dict(color="#6b7890")),
                        size=3,
                        opacity=0.7,
                    ),
                    showlegend=False,
                    hovertemplate="Score: %{marker.color:.3f}<br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>",
                )
            )

            df_m = df_mistakes[df_mistakes["is_mistake"] == True]
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

            # Stats grid — Row 1
            s1, s2, s3 = st.columns(3)
            with s1:
                st.metric("Total Anomalies", str(meta["total_mistakes"]))
            with s2:
                brake_override = int(
                    (df_mistakes["is_mistake"] & (df_mistakes["brake_intensity"] > 0.5)).sum()
                )
                st.metric("Brake Override", str(brake_override))
            with s3:
                throttle_slip = int(
                    (
                        df_mistakes["is_mistake"]
                        & (df_mistakes["throttle_intensity"] < 0.2)
                        & (df_mistakes["speed_delta"] < -5.0)
                    ).sum()
                )
                st.metric("Throttle Slip", str(throttle_slip))

            # Stats grid — Row 2
            s4, s5, s6 = st.columns(3)
            with s4:
                st.metric("Contamination", str(meta["best_contamination"]))
            with s5:
                st.metric("K-Fold Score", f"{meta['best_cv_score']:.4f}")
            with s6:
                st.metric("Mistake Rate", f"{meta['mistake_rate_pct']:.1f}%")

            # CV Scores expander
            with st.expander("Grid Search CV Scores"):
                cv_scores = meta["cv_scores"]
                x_vals = [float(k) for k in cv_scores.keys()]
                y_vals = list(cv_scores.values())
                best_c = meta["best_contamination"]

                colors = [
                    "#00d4ff" if abs(x - best_c) < 1e-6 else "#3a4558"
                    for x in x_vals
                ]

                fig_cv = go.Figure(
                    go.Bar(
                        x=[str(x) for x in x_vals],
                        y=y_vals,
                        marker_color=colors,
                    )
                )
                fig_cv.update_layout(
                    height=180,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#0f1217",
                    font=dict(color="#6b7890", size=11),
                    margin=dict(t=10, b=30, l=40, r=10),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                )
                st.plotly_chart(fig_cv, width='stretch')

    # ── RIGHT COLUMN: LSTM Placeholder ────────────────────────────────────
    with col_right:
        st.markdown("**LSTM TYRE CLIFF PREDICTOR**")
        st.info("Coming in Phase 2. Run predict_cliff.py once the LSTM model is trained.")

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
