"""
ApexHunter Frontend - AI Analysis
Renders the AI Analysis tab: Isolation Forest mistake events (left) and the tyre pace / pit window
output of predict_strategy.py (right). Both are minimal views to check the models; the full design
comes with the UI redesign (ROADMAP §E).
"""

from typing import List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DRIVER_MAPPING
from components.data_loader import downsample



def mistake_rate_pct(df: pd.DataFrame) -> float:
    """Measured share of rows flagged as mistakes, in percent (0.0 for an empty frame).

    This is the model's actual output rate. It is NOT the contamination hyperparameter, which is
    only the threshold setting chosen during training.
    """
    if len(df) == 0:
        return 0.0
    return float(df["is_mistake"].sum()) / len(df) * 100


def _time_lost_text(event: dict) -> str:
    """'+0.18 s' (time lost vs the driver's own nearby laps), or 'n/a'."""
    lost = event.get("time_lost_s")
    return "n/a" if lost is None else f"{lost:+.2f} s"


def _render_isolation_forest_column(df_mistakes: Optional[pd.DataFrame], meta: Optional[dict],
                                    events: Optional[List[dict]] = None) -> None:
    """Render the Isolation Forest card: track map with one marker per mistake event, counts,
    and all events with their reasons, most costly first. Minimal version to check the rework; the
    full design comes with the UI redesign (ROADMAP §E)."""
    if df_mistakes is None or meta is None:
        st.warning(
            "No Isolation Forest output found for this session and driver.\n\n"
            "Run: `python backend/scripts/detect_mistakes.py --session <path> --driver <code>`"
        )
        return
    has_event_list = events is not None
    events = events or []

    contamination = meta["best_contamination"]
    total_mistakes = int(df_mistakes["is_mistake"].sum())
    session_rate = mistake_rate_pct(df_mistakes)
    ref_names = ", ".join(DRIVER_MAPPING.get(d, d) for d in meta["reference_drivers"])

    st.markdown("**ISOLATION FOREST · Mistake Detection**")
    st.caption(
        f"Session: {len(events)} events from {total_mistakes} flagged samples "
        f"({session_rate:.2f}% of samples) · contamination setting={contamination:.3f} · "
        f"field reference: {ref_names}"
    )

    # Lap selector — filter events to a single lap
    df_display, lap_events = df_mistakes, events
    if "LapNumber" in df_mistakes.columns:
        available_laps = sorted(df_mistakes[df_mistakes["LapNumber"] > 0]["LapNumber"].unique())
        if available_laps:
            lap_options = [f"Lap {lap}" for lap in available_laps]
            selected_lap_label = st.selectbox(
                "Select Lap", lap_options, key="ai_mistake_lap_selector", filter_mode=None
            )
            selected_lap = available_laps[lap_options.index(selected_lap_label)]
            df_display = df_mistakes[df_mistakes["LapNumber"] == selected_lap]
            lap_events = [e for e in events if e["lap"] == selected_lap]

    # Track map — full track background in neutral blue, one marker per event of the selected lap
    fig_map = go.Figure()
    df_map = downsample(df_mistakes, max_points=8000)
    fig_map.add_trace(
        go.Scattergl(
            x=df_map["X"], y=df_map["Y"], mode="markers",
            marker=dict(color="#3abdc7", size=3, opacity=0.5),
            showlegend=False, hovertemplate="X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>",
        )
    )
    if lap_events:
        fig_map.add_trace(
            go.Scatter(
                x=[e["x"] for e in lap_events], y=[e["y"] for e in lap_events], mode="markers",
                marker=dict(symbol="x", size=10, color="#ff3a3a"), name="Mistake event",
                text=[f"#{e['rank']} · {e.get('location', '')} · {e['reason']} · {_time_lost_text(e)}"
                      for e in lap_events],
                hovertemplate="%{text}<extra></extra>",
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
        st.metric("Events (lap)", str(len(lap_events)))
    with s2:
        st.metric("Flagged Samples (lap)", str(int(df_display["is_mistake"].sum())))
    with s3:
        st.metric("Mistake Rate", f"{mistake_rate_pct(df_display):.2f}%")

    if events:
        st.markdown(f"**All {len(events)} events, most costly first "
                    f"(time lost vs the driver's own nearby laps)**")
        top = pd.DataFrame(events)
        if "location" not in top.columns:                   # outputs from before the corner labels
            top["location"] = top["track_pos_m"].map(lambda m: f"{m:.0f} m into the lap")
        top["time_lost"] = [_time_lost_text(e) for e in events]
        top["why"] = [e["reason"] + (f" ({e['saving_note']})" if e.get("saving_note") else "")
                      for e in events]
        st.dataframe(
            top[["rank", "lap", "location", "time_lost", "why"]].rename(columns={
                "rank": "#", "lap": "Lap", "location": "Location", "time_lost": "Time lost",
                "why": "Why it was flagged"}),
            hide_index=True, width="stretch",
        )
    elif has_event_list:
        st.info("No mistake events in this session: no run of flags lasted 0.5 s or more.")
    else:
        st.info("No event list for this output. Re-run detect_mistakes.py with --force.")


SOURCE_NAMES = {"no_change": '"No change" (fuel only)', "laps_driven": "Laps driven (real lap times)",
                "ridge": "Ridge", "rnn": "RNN", "lstm": "LSTM"}
WINDOW_COLOUR = "rgba(0,255,136,0.13)"
NEUTRAL_COLOUR = "rgba(255,200,0,0.16)"


def lap_runs(laps: List[int]) -> List[Tuple[int, int]]:
    """[13, 14, 15, 20] -> [(13, 15), (20, 20)] (for shading)."""
    runs: List[Tuple[int, int]] = []
    for n in sorted(laps):
        if runs and n == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], n)
        else:
            runs.append((n, n))
    return runs


def lap_ranges_text(laps: List[int]) -> str:
    return ", ".join(f"{a}-{b}" if b > a else f"{a}" for a, b in lap_runs(laps))


def strategy_chart_laps(data: dict, stint: dict) -> List[dict]:
    """Lap rows shown for a stint: only the stint's own laps (user, 2026-11-03)."""
    return [r for r in data["laps"] if stint["first_lap"] <= r["lap"] <= stint["last_lap"]]


def is_racing_lap(row: dict, stint: dict) -> bool:
    """A normal racing lap: not lap 1, not the out-lap (stint's first lap after a stop), not the
    in-lap, not under SC/VSC/red. Only these are joined by the lap-time line."""
    out_lap = stint["first_lap"] if stint["first_lap"] > 1 else None
    return (row["lap"] != 1 and row["lap"] != out_lap and not row["pit_in"]
            and row["track"] == "green" and row["lap_time"] is not None)


def _build_strategy_figure(data: dict, stint: dict) -> go.Figure:
    rows = strategy_chart_laps(data, stint)
    first, last = stint["first_lap"], stint["last_lap"]
    fig = go.Figure()
    for a, b in lap_runs([r["lap"] for r in rows if r["track"] != "green"]):
        fig.add_vrect(x0=a - 0.5, x1=b + 0.5, fillcolor=NEUTRAL_COLOUR, line_width=0, layer="below")
    for a, b in lap_runs([n for n in stint.get("window", []) if first <= n <= last]):
        fig.add_vrect(x0=a - 0.5, x1=b + 0.5, fillcolor=WINDOW_COLOUR, line_width=0, layer="below")

    racing = [r for r in rows if is_racing_lap(r, stint)]
    green = sorted(r["lap_time"] for r in racing)
    yaxis = dict(title="Lap time (s)", tickfont=dict(color="#6b7890"), gridcolor="rgba(255,255,255,0.04)")
    top = None
    if len(green) >= 3:
        top = green[len(green) // 2] + 3.0
        yaxis["range"] = [green[0] - 0.5, top + 0.3]
    # the line joins only normal racing laps; it breaks across the other laps
    fig.add_trace(go.Scatter(
        x=[r["lap"] for r in rows], y=[r["lap_time"] if r in racing else None for r in rows],
        mode="lines+markers", line=dict(color="#e2e8f0", width=2), marker=dict(size=4),
        name="Real lap time", hovertemplate="Lap %{x}: %{y:.2f} s<extra></extra>", connectgaps=False))
    # lap 1, out-lap, in-lap and SC/VSC laps: grey markers, held at the top edge if off the chart
    other = [r for r in rows if r not in racing and r["lap_time"] is not None]
    if other:
        fig.add_trace(go.Scatter(
            x=[r["lap"] for r in other],
            y=[min(r["lap_time"], top) if top else r["lap_time"] for r in other],
            mode="markers", marker=dict(color="#6b7890", size=6, symbol="circle-open"),
            name="Lap 1 / pit / SC lap", text=[f"{r['lap_time']:.2f} s" for r in other],
            hovertemplate="Lap %{x}: %{text}<extra></extra>"))
    pred = [r for r in rows if r["predicted_lap_time"] is not None]
    if pred:
        fig.add_trace(go.Scatter(
            x=[r["lap"] for r in pred], y=[r["predicted_lap_time"] for r in pred], mode="markers",
            marker=dict(color="#a855f7", size=6, symbol="diamond"), name=f"Predicted ({SOURCE_NAMES.get(data['model'], data['model'])})",
            hovertemplate="Lap %{x}: predicted %{y:.2f} s<extra></extra>"))
    best = stint.get("best_lap")
    if best and first <= best <= last:
        fig.add_vline(x=best, line=dict(color="#00ff88", width=1.5, dash="dot"),
                      annotation_text=f"Best {best}", annotation_position="top left",
                      annotation_font=dict(color="#00ff88", size=10))
    if stint.get("actual_stop"):
        fig.add_vline(x=stint["actual_stop"], line=dict(color="#ff3a3a", width=1.5),
                      annotation_text=f"Pitted {stint['actual_stop']}", annotation_position="top right",
                      annotation_font=dict(color="#ff3a3a", size=10))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1520", height=300,
        margin=dict(t=25, b=30, l=50, r=20), yaxis=yaxis,
        xaxis=dict(title="Race lap", range=[first - 0.5, last + 0.5], tickfont=dict(color="#6b7890"),
                   gridcolor="rgba(255,255,255,0.04)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6b7890", size=11), orientation="h", y=-0.25),
    )
    return fig


def _render_strategy_column(data: Optional[dict]) -> None:
    """Tyre pace + pit window card (ROADMAP §C step 8): stint chart with the window (green),
    safety-car/VSC laps (yellow), best and real stop, the readable summary and a few numbers."""
    st.markdown("**TYRE PACE · PIT WINDOW**")
    if data is None:
        st.warning(
            "No pit strategy output for this session and driver.\n\n"
            "Run (races only; `check_strategy.py` must have been run once, it picks the wear source):\n"
            "```\npython backend/scripts/predict_strategy.py "
            "--session data_lake/clean_data/<file>.parquet --driver <number>\n```")
        return
    cost = data["pit_cost"]
    trained = {True: " · race was in the model's training data", False: " · held-out test race"}
    st.caption(
        f"Wear from: {SOURCE_NAMES.get(data['model'], data['model'])}"
        + trained.get(data.get("in_training_data"), "")
        + f" · pit stop costs {cost['normal_s']:.1f} s ({cost['sc_vsc_s']:.1f} s under SC/VSC)")
    scores = data.get("strategy_scores")
    if scores:
        table = pd.DataFrame([
            {"Wear from": SOURCE_NAMES.get(k, k) + (" (shown)" if k == data["model"] else ""),
             "Real stop in window": f"{v['hit_pct']:.0f} %", "Random 5-lap window": f"{v['random_pct']:.0f} %",
             "Margin": round(v["hit_pct"] - v["random_pct"], 1)}
            for k, v in scores.items()]).sort_values("Margin", ascending=False)
        with st.expander("Pit window scores on the test races (best margin is shown)"):
            st.dataframe(table, hide_index=True, width="stretch")

    stints = data.get("stints", [])
    if not stints:
        st.warning("The output file has no stints.")
        return
    labels = [f"Stint {s['stint']} · {s['compound']} · laps {s['first_lap']}-{s['last_lap']}" for s in stints]
    stint = stints[labels.index(st.selectbox("Select stint", labels, key="tyre_stint_selector",
                                             filter_mode=None))]

    st.plotly_chart(_build_strategy_figure(data, stint), width="stretch")
    if "summary" not in stint:
        st.info("No pit advice for this stint (last stint, wet tyres, or no prediction yet).")
        return
    st.markdown("\n".join(f"- {line}" for line in stint["summary"]))
    after = [n for n in stint["window"] if n > stint["last_lap"]]
    if after:
        st.caption(f"Window laps after the real stop ({lap_ranges_text(after)}) assume he had stayed "
                   f"out; the chart shows only this stint's laps.")
    # wear the model believed when the driver really pitted (after the stop it stays frozen)
    at_stop = next((a for a in stint["advice"] if a["lap"] == stint["actual_stop"]), None)
    wear_text = (f"{at_stop['wear_s']:.3f} s/lap" if at_stop and not at_stop.get("no_wear_signal")
                 else "no signal")
    n_window = len(stint["window"])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Real stop", f"Lap {stint['actual_stop']}", "in window" if stint["hit"] else "outside window",
                  delta_color="normal" if stint["hit"] else "inverse")
    with c2:
        st.metric("Window", f"{n_window} lap{'s' if n_window != 1 else ''}",
                  f"random hit {100 * stint['random_hit_chance']:.0f} %", delta_color="off")
    with c3:
        st.metric("Wear at real stop", wear_text)


def render_ai_analysis(
    df_mistakes: Optional[pd.DataFrame],
    meta: Optional[dict],
    df_session: pd.DataFrame,
    scrub_seconds: float,
    tyre_data: Optional[dict] = None,
    mistake_events: Optional[dict] = None,
) -> None:
    """Renders the AI Analysis tab content (tyre_data = the *_strategy.json output)."""
    col_left, col_right = st.columns(2)

    with col_left:
        _render_isolation_forest_column(df_mistakes, meta, (mistake_events or {}).get("events"))

    with col_right:
        _render_strategy_column(tyre_data)
