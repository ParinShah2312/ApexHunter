"""
ApexHunter Frontend - Tyres & Pit Window tab
Model switch (Ridge / RNN / LSTM; opens on the one with the best pit-window margin) -> stint buttons -> cards -> the stint's lap
times with the pit window (green), safety-car laps (yellow), best and real stop, the predicted lap
times -> how the models compare (lap-time error on the test races, pit window vs a random window).
Reads predict_strategy.py --source all outputs (*_strategy.json) and the tyre models' configs.
"""

import json
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st

from config import MODELS_DIR, TYRE_PREDICTIONS_DIR

SOURCE_NAMES = {"no_change": '"No change" (fuel only)', "laps_driven": "Laps driven (real lap times)",
                "ridge": "Ridge", "rnn": "RNN", "lstm": "LSTM"}
SOURCE_SHORT = {"no_change": "No change", "laps_driven": "Laps driven", "ridge": "Ridge", "rnn": "RNN",
                "lstm": "LSTM"}
SOURCE_KIND = {"no_change": "baseline", "laps_driven": "rule", "ridge": "ML", "rnn": "DL", "lstm": "DL"}
SOURCE_COLORS = {"no_change": "#8a94a6", "laps_driven": "#2ec4b6", "ridge": "#4f8fe0", "rnn": "#e07bb5",
                 "lstm": "#b45cff"}
# The tab shows the three trained models only (user, 2026-11-05); "no change" and "laps driven"
# stay in the outputs, check_strategy.py and the docs.
MODEL_ORDER = ["ridge", "rnn", "lstm"]
MODEL_CONFIGS = {"ridge": "tyre_ridge_config.json", "rnn": "tyre_rnn_config.json", "lstm": "tyre_config.json"}
WINDOW_COLOUR = "rgba(0,255,136,0.13)"
NEUTRAL_COLOUR = "rgba(255,200,0,0.16)"
PLOT_BG, GRID = "#0f1217", "rgba(255,255,255,0.05)"


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
        name="Real lap time", hovertemplate="Lap %{x}: real lap time %{y:.2f} s<extra></extra>",
        connectgaps=False))
    # lap 1, out-lap, in-lap and SC/VSC laps: grey markers, held at the top edge if off the chart
    other = [r for r in rows if r not in racing and r["lap_time"] is not None]
    if other:
        fig.add_trace(go.Scatter(
            x=[r["lap"] for r in other],
            y=[min(r["lap_time"], top) if top else r["lap_time"] for r in other],
            mode="markers", marker=dict(color="#6b7890", size=6, symbol="circle-open"),
            name="Lap 1 / pit / SC lap", text=[f"{r['lap_time']:.2f} s" for r in other],
            hovertemplate="Lap %{x}: %{text} (lap 1, pit or safety-car lap)<extra></extra>"))
    pred = [r for r in rows if r["predicted_lap_time"] is not None]
    if pred:
        fig.add_trace(go.Scatter(
            x=[r["lap"] for r in pred], y=[r["predicted_lap_time"] for r in pred], mode="markers",
            marker=dict(color=SOURCE_COLORS.get(data["model"], "#a855f7"), size=7, symbol="diamond"),
            name=f"Predicted by {SOURCE_SHORT.get(data['model'], data['model'])}",
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
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PLOT_BG, height=380,
        margin=dict(t=25, b=30, l=50, r=20), yaxis=yaxis,
        xaxis=dict(title="Race lap", range=[first - 0.5, last + 0.5], tickfont=dict(color="#6b7890"),
                   gridcolor="rgba(255,255,255,0.04)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8a94a6", size=11), orientation="h", y=-0.22),
    )
    return fig


def default_model(data: dict, available: List[str]) -> str:
    """The trained model with the largest pit-window margin over a random window on the test
    races (strategy_check.json scores in the file); the first available one without scores."""
    scores = data.get("strategy_scores") or {}
    scored = [m for m in available if m in scores]
    if not scored:
        return available[0]
    return max(scored, key=lambda m: scores[m]["hit_pct"] - scores[m]["random_pct"])


def normal_window(stint: dict) -> List[int]:
    """The stint's normal pit window: the run of window laps around the best stop lap (the window
    list also holds safety-car "pit now" laps, which the summary lists separately)."""
    runs = lap_runs(stint.get("window", []))
    best = stint.get("best_lap")
    run = next((r for r in runs if best is not None and r[0] <= best <= r[1]), None)   # none: SC call only
    return [] if run is None else list(range(run[0], run[1] + 1))


def source_view(data: dict, source: str) -> Optional[dict]:
    """The output for one wear source: the file's top part for the best source, else its entry under
    "sources" (predict_strategy.py --source all). None if the file has not got it."""
    if source == data.get("model"):
        return data
    entry = data.get("sources", {}).get(source)
    return None if entry is None else {**data, **entry}


def lap_time_error(view: dict) -> Optional[float]:
    """Average |predicted - real| lap time (s) over the laps that have a prediction (model sources
    only; equal to the error of the predicted change, as both start from the previous lap)."""
    errs = [abs(r["predicted_lap_time"] - r["lap_time"]) for r in view["laps"]
            if r.get("predicted_lap_time") is not None and r.get("lap_time") is not None]
    return sum(errs) / len(errs) if errs else None


@st.cache_data(show_spinner=False)
def test_errors() -> Dict[str, float]:
    """Lap-time error (MAE, s) of each model on the held-out test races, from the model configs."""
    out = {}
    for name, fname in MODEL_CONFIGS.items():
        try:
            with open(MODELS_DIR / fname, encoding="utf-8") as f:
                metrics = json.load(f)["metrics"]
        except (OSError, KeyError, ValueError):
            continue
        out[name] = metrics["test"]["mae_s"]
        out.setdefault("no_change", metrics["test_no_change"]["mae_s"])
    return out


def races_with_results() -> List[str]:
    """'2023 R1' style list of the races that have pit-window outputs."""
    races = {tuple(int(x) for x in f.name.split("_")[:2])
             for f in TYRE_PREDICTIONS_DIR.glob("*_*_R_*_strategy.json")}
    return [f"{y} R{r}" for y, r in sorted(races)]


def _comparison_figures(data: dict) -> Tuple[go.Figure, go.Figure]:
    errors = test_errors()
    order = [n for n in MODEL_ORDER if n in errors]
    err = go.Figure(go.Bar(
        x=[SOURCE_SHORT[n] for n in order], y=[errors[n] for n in order],
        marker_color=[SOURCE_COLORS[n] for n in order], text=[f"{errors[n]:.3f} s" for n in order],
        textposition="outside", hovertemplate="%{x}: average error %{y:.3f} s per lap<extra></extra>"))
    hi = max((errors[n] for n in order), default=1.0)
    err.update_layout(height=300, margin=dict(t=20, b=30, l=50, r=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor=PLOT_BG, font=dict(color="#8a94a6", size=11), showlegend=False,
                      yaxis=dict(title="Seconds per lap", gridcolor=GRID, range=[0, hi * 1.15]))
    scores = data.get("strategy_scores") or {}
    names = [n for n in MODEL_ORDER if n in scores]
    labels = [SOURCE_SHORT[n] for n in names]
    hit = go.Figure()
    hit.add_trace(go.Bar(x=labels, y=[scores[n]["hit_pct"] for n in names], name="Real stop inside the window",
                         marker_color=[SOURCE_COLORS[n] for n in names],
                         text=[f"{scores[n]['hit_pct']:.0f} %" for n in names], textposition="outside",
                         hovertemplate="%{x}: real stop inside the window %{y:.0f} % of the time<extra></extra>"))
    hit.add_trace(go.Bar(x=labels, y=[scores[n]["random_pct"] for n in names], name="Random window, same size",
                         marker_color="#3a4252", text=[f"{scores[n]['random_pct']:.0f} %" for n in names],
                         textposition="outside",
                         hovertemplate="%{x}: random window %{y:.0f} % of the time<extra></extra>"))
    hit.update_layout(barmode="group", height=300, margin=dict(t=20, b=30, l=50, r=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PLOT_BG, font=dict(color="#8a94a6", size=11),
                      legend=dict(orientation="h", y=1.15, x=0),
                      yaxis=dict(title="% of stints", gridcolor=GRID, range=[0, 60]))
    return err, hit


def render_strategy_tab(data: Optional[dict], ctx: str = "") -> None:
    """Renders the Tyres & Pit Window tab (data = the *_strategy.json output; ctx makes the widget
    keys unique per session and driver)."""
    if data is None:
        st.info("No pit window for this driver in this session. Pit windows are made for races only; "
                "they exist for: " + ", ".join(races_with_results()) + " (every driver).")
        return
    available = [m for m in MODEL_ORDER if source_view(data, m) is not None]
    if not available:                       # an older file with one rule-based source only
        available = [data["model"]]
    default = default_model(data, available)
    model_key = f"tyre_model_{ctx}"
    if st.session_state.get(model_key) not in available:
        st.session_state[model_key] = default
    source = st.segmented_control(
        "Tyre model", available, key=model_key, required=True,
        format_func=lambda m: f"{SOURCE_SHORT.get(m, m)} ({SOURCE_KIND.get(m, '')})")
    view = source_view(data, source)

    stints = view.get("stints", [])
    if not stints:
        st.warning("The output file has no stints.")
        return
    numbers = [s["stint"] for s in stints]
    by_number = {s["stint"]: s for s in stints}
    stint_key = f"tyre_stint_{ctx}"
    if st.session_state.get(stint_key) not in numbers:
        st.session_state[stint_key] = numbers[0]
    chosen = st.segmented_control(
        "Stint", numbers, key=stint_key, required=True,
        format_func=lambda n: f"Stint {n} · {by_number[n]['compound'].capitalize()} · laps "
                              f"{by_number[n]['first_lap']}–{by_number[n]['last_lap']}")
    stint = by_number[chosen]

    cost = view["pit_cost"]
    trained = {True: "this race was in the model's training data", False: "held-out test race"}
    note = trained.get(view.get("in_training_data"))
    advised = "summary" in stint
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Real stop", f"Lap {stint['actual_stop']}" if advised else "—",
              help="The lap he actually pitted on (FastF1)")
    window = normal_window(stint) if advised else []
    c2.metric("Pit window", f"Laps {lap_ranges_text(window)}" if window else "—",
              help="Best stop lap ± 2 laps. Safety-car 'pit now' calls are listed under the chart")
    c3.metric("Real stop in the window", ("Yes" if stint["hit"] else "No") if advised else "—",
              help=(f"A random window of the same size would contain it "
                    f"{100 * stint['random_hit_chance']:.0f} % of the time") if advised else None)
    err = lap_time_error(view)
    c4.metric("Lap-time error, this race", "—" if err is None else f"{err:.3f} s",
              help="Average difference between predicted and real lap time for this driver"
                   + (f" ({note})" if note else "") + ". Rule-based sources make no lap-time predictions.")

    st.plotly_chart(_build_strategy_figure(view, stint), width="stretch", key=f"tyre_chart_{ctx}_{source}")
    st.caption(f"Green = pit window, yellow = safety car / VSC, red line = real stop, dotted = best stop lap. "
               f"A stop costs {cost['normal_s']:.1f} s here ({cost['sc_vsc_s']:.1f} s under a safety car).")
    if not advised:
        st.info("No pit advice for this stint (last stint, wet tyres, or no prediction yet).")
    else:
        st.markdown("\n".join(f"- {line}" for line in stint["summary"]))
        after = [n for n in stint["window"] if n > stint["last_lap"]]
        if after:
            st.caption(f"Window laps after the real stop ({lap_ranges_text(after)}) assume he had stayed "
                       f"out; the chart shows only this stint's laps.")

    # ── How the models compare ────────────────────────────────────────────
    st.markdown("#### How the tyre models compare")
    err_fig, hit_fig = _comparison_figures(data)
    a, b = st.columns(2)
    with a:
        st.markdown("**Lap-time error on the 5 test races** (lower is better)")
        st.plotly_chart(err_fig, width="stretch", key=f"tyre_err_{ctx}", config={"displayModeBar": False})
        st.caption("Average difference between the predicted and the real next lap time, on races the "
                   "models never saw.")
    with b:
        st.markdown("**Pit window against a random window** (test races)")
        st.plotly_chart(hit_fig, width="stretch", key=f"tyre_hit_{ctx}", config={"displayModeBar": False})
        st.caption("How often the real stop fell inside the window, against a random window of the same "
                   "size. More accurate lap times do not give better windows: even perfect wear does not "
                   "(ceiling test), because real stops follow strategy, traffic and safety cars.")
