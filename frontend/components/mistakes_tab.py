"""
ApexHunter Frontend - Driver Mistakes tab
Detector switch (Isolation Forest / LSTM autoencoder / Z-score) → cards → track map with one marker
per mistake + the list of mistakes → the chosen mistake: this lap vs the field and vs the driver's
usual laps at that place → how the three detectors compare (planted-mistake test and real-lap check).
Reads detect_mistakes.py outputs: the parquet (per-reading flags), meta and events JSON.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DRIVER_MAPPING, MISTAKE_DATA_DIR, SESSION_LABEL_MAP
from components.data_loader import get_corners, get_session_laps
from components.lap_telemetry import lap_samples
from components.telemetry_charts import GRID, PLOT_BG, _dense_lap, fastest_lap

# name: (label, course, parquet column prefix; None = the Isolation Forest's own columns)
DETECTORS = {
    "isolation_forest": ("Isolation Forest", "ML", None),
    "lstm_autoencoder": ("LSTM autoencoder", "DL", "ae"),
    "z_score": ("Z-score", "baseline", "z"),
}
DETECTOR_COLORS = {"z_score": "#8a94a6", "isolation_forest": "#4f8fe0", "lstm_autoencoder": "#b45cff",
                   "random_spots": "#3a4252"}
LAP_COLOR, FIELD_COLOR, USUAL_COLOR = "#00d4ff", "#f5a623", "#8a94a6"
WINDOW_M = 250.0        # detail chart: this far before and after the flagged stretch
BIN_M = 10.0            # "his usual laps" = median per 10 m of track


def mistake_rate_pct(flags: np.ndarray) -> float:
    """Measured share of readings flagged, in percent (0.0 when empty). This is the detector's
    actual output, not the contamination setting (#4)."""
    return float(np.mean(flags) * 100) if len(flags) else 0.0


def detector_data(df: pd.DataFrame, events_json: Optional[dict], name: str):
    """(flags, event ids, events) of one detector, or None if this output has not got them
    (the Z-score and autoencoder ones are saved by detect_mistakes.py --compare)."""
    prefix = DETECTORS[name][2]
    if prefix is None:
        return df["is_mistake"].to_numpy(), df["event_id"].to_numpy(), (events_json or {}).get("events", [])
    others = (events_json or {}).get("detectors", {})
    if f"{prefix}_flag" not in df.columns or name not in others:
        return None
    return df[f"{prefix}_flag"].to_numpy(), df[f"{prefix}_event_id"].to_numpy(), others[name]


def costly_shares(meta: dict) -> Dict[str, Optional[float]]:
    """Share of each detector's events (and of random spots) that cost clearly more time than the
    driver's own nearby laps there (the real-lap check)."""
    check = meta.get("time_lost_check", {})
    comp = meta.get("comparison", {})
    return {"isolation_forest": check.get("isolation_forest", {}).get("costly_share"),
            "lstm_autoencoder": comp.get("lstm_autoencoder_time_lost_check", {}).get("costly_share"),
            "z_score": comp.get("z_score_time_lost_check", {}).get("costly_share"),
            "random_spots": check.get("random_spots", {}).get("costly_share")}


def sessions_with_results() -> List[str]:
    """'2023 R1 Race: Max Verstappen, Lewis Hamilton' for every session with mistake results."""
    found: Dict[tuple, List[str]] = {}
    for f in MISTAKE_DATA_DIR.glob("*_mistakes.parquet"):
        parts = f.name.split("_")
        if len(parts) >= 5 and parts[0].isdigit() and parts[1].isdigit():
            found.setdefault((int(parts[0]), int(parts[1]), parts[2]), []).append(
                DRIVER_MAPPING.get(parts[3], f"#{parts[3]}"))
    return [f"{y} R{r} {SESSION_LABEL_MAP.get(s, s)}: {', '.join(sorted(d))}"
            for (y, r, s), d in sorted(found.items())]


def _time_lost_text(e: dict) -> str:
    t = e.get("time_lost_s")
    return "not measured" if t is None else f"{t:+.2f} s"


def _what(e: dict) -> str:
    return e["reason"] + (f" ({e['saving_note']})" if e.get("saving_note") else "")


# ── Map ───────────────────────────────────────────────────────────────────────

def _track_line(sel) -> Optional[pd.DataFrame]:
    """The driver's fastest lap, re-sampled every 5 m (background of the map)."""
    laps = get_session_laps(sel.year, sel.round_num, sel.session_type)
    best = fastest_lap(laps[laps["driver"] == sel.driver_number])
    if best is None:
        return None
    lap = lap_samples(sel.df_driver, best["start_s"], best["lap_s"])
    if lap.empty:
        return None
    line = _dense_lap(lap)
    return pd.concat([line, line.iloc[[0]]], ignore_index=True)      # close the loop at the finish line


def _map_figure(line: Optional[pd.DataFrame], df: pd.DataFrame, corners: pd.DataFrame,
                events: List[dict], selected: Optional[int]) -> go.Figure:
    fig = go.Figure()
    if line is not None:
        fig.add_trace(go.Scatter(x=line["X"], y=line["Y"], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color="#3a4558", width=6)))
    else:                                                    # fall back to the analysed readings
        fig.add_trace(go.Scattergl(x=df["X"], y=df["Y"], mode="markers", hoverinfo="skip", showlegend=False,
                                   marker=dict(color="#3a4558", size=3)))
    if not corners.empty:
        fig.add_trace(go.Scatter(x=corners["X"], y=corners["Y"], mode="text", text=corners["label"],
                                 textposition="top right", textfont=dict(color="#8a94a6", size=10),
                                 hoverinfo="skip", showlegend=False))
    groups = [("Cost time", lambda e: e.get("costly") and not e.get("likely_saving"), "#ff4d4d"),
              ("Did not cost time", lambda e: not e.get("costly") and not e.get("likely_saving"), "#ffb800"),
              ("Likely saving tyres/fuel", lambda e: bool(e.get("likely_saving")), "#8a94a6")]
    for name, keep, color in groups:
        sub = [e for e in events if keep(e)]
        if not sub:
            continue
        fig.add_trace(go.Scatter(
            x=[e["x"] for e in sub], y=[e["y"] for e in sub], mode="markers", name=name,
            customdata=[e["event_id"] for e in sub],
            marker=dict(color=color, size=[10 + 40 * min(max(e.get("time_lost_s") or 0, 0), 0.5) for e in sub],
                        line=dict(color=["#ffffff" if e["event_id"] == selected else color for e in sub],
                                  width=[3 if e["event_id"] == selected else 1 for e in sub]), opacity=0.9),
            text=[f"<b>Lap {e['lap']} · {e.get('location', '')}</b><br>{_what(e)}<br>"
                  f"Time lost vs his nearby laps: {_time_lost_text(e)}" for e in sub],
            hovertemplate="%{text}<extra></extra>",
        ))
    fig.update_layout(height=470, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor=PLOT_BG, clickmode="event+select",
                      legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11)),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1))
    return fig


# ── Detail: this lap vs the field vs his usual laps ───────────────────────────

def corner_track_pos(df: pd.DataFrame, corners: pd.DataFrame) -> pd.DataFrame:
    """Corners placed on the parquet's track_pos axis (track_pos of the nearest analysed reading)."""
    if corners.empty or df.empty:
        return pd.DataFrame(columns=["label", "track_pos"])
    xy = df[["X", "Y"]].to_numpy(dtype=float)
    idx = [int(np.argmin(((xy - p) ** 2).sum(axis=1))) for p in corners[["X", "Y"]].to_numpy(dtype=float)]
    return pd.DataFrame({"label": corners["label"].to_numpy(), "track_pos": df["track_pos"].to_numpy()[idx]})


def usual_profile(df: pd.DataFrame, lap: int, lo: float, hi: float) -> pd.DataFrame:
    """Median speed and throttle of the driver's OTHER analysed laps per BIN_M metres in [lo, hi]."""
    other = df[(df["LapNumber"] != lap) & df["track_pos"].between(lo, hi)]
    if other.empty:
        return pd.DataFrame(columns=["track_pos", "Speed", "Throttle"])
    bins = (other["track_pos"] // BIN_M) * BIN_M + BIN_M / 2
    return other.groupby(bins)[["Speed", "Throttle"]].median().reset_index(names="track_pos")


def _detail_figure(df: pd.DataFrame, event: dict, corners_tp: pd.DataFrame) -> go.Figure:
    lo, hi = event["start_pos_m"] - WINDOW_M, event["end_pos_m"] + WINDOW_M
    lap = df[(df["LapNumber"] == event["lap"]) & df["track_pos"].between(lo, hi)].sort_values("track_pos")
    usual = usual_profile(df, event["lap"], lo, hi)
    x = lap["track_pos"]
    fig = go.Figure()
    dom = {"y": [0.55, 1.0], "y2": [0.2, 0.5], "y3": [0.0, 0.14]}

    def add(axis, **kw):
        fig.add_trace(go.Scatter(xaxis="x", yaxis=axis, **kw))

    add("y", x=usual["track_pos"], y=usual["Speed"], name="His usual laps (median)",
        line=dict(color=USUAL_COLOR, width=1.2, dash="dot"), hovertemplate="His usual: %{y:.0f} km/h<extra></extra>")
    add("y", x=x, y=lap["Speed"] - lap["speed_vs_field_kmh"], name="Field (5 fastest others)",
        line=dict(color=FIELD_COLOR, width=1.5), hovertemplate="Field: %{y:.0f} km/h<extra></extra>")
    add("y", x=x, y=lap["Speed"], name=f"This lap (lap {event['lap']})", line=dict(color=LAP_COLOR, width=2.2),
        hovertemplate="This lap: %{y:.0f} km/h<extra></extra>")
    add("y2", x=usual["track_pos"], y=usual["Throttle"], line=dict(color=USUAL_COLOR, width=1, dash="dot"),
        showlegend=False, hovertemplate="His usual: throttle %{y:.0f} %<extra></extra>")
    add("y2", x=x, y=lap["Throttle"] - lap["throttle_vs_field_pct"], line=dict(color=FIELD_COLOR, width=1.2),
        showlegend=False, hovertemplate="Field: throttle %{y:.0f} %<extra></extra>")
    add("y2", x=x, y=lap["Throttle"], line=dict(color=LAP_COLOR, width=1.8), showlegend=False,
        hovertemplate="This lap: throttle %{y:.0f} %<extra></extra>")
    add("y3", x=x, y=lap["field_brake_share"] * 100, fill="tozeroy", fillcolor="rgba(245,166,35,0.25)",
        line=dict(color=FIELD_COLOR, width=0.8), showlegend=False,
        hovertemplate="%{y:.0f} % of the field brakes here<extra></extra>")
    braking = np.where(lap["Brake"].to_numpy() >= 0.5, 100.0, np.nan)
    add("y3", x=x, y=braking, mode="lines", connectgaps=False, line=dict(color=LAP_COLOR, width=7),
        showlegend=False, hovertemplate="This lap: braking<extra></extra>")

    shapes = [dict(type="rect", xref="x", yref="paper", x0=event["start_pos_m"], x1=max(event["end_pos_m"],
                   event["start_pos_m"] + 5), y0=0, y1=1, fillcolor="rgba(255,77,77,0.15)", line=dict(width=0))]
    annotations = [dict(x=(event["start_pos_m"] + event["end_pos_m"]) / 2, y=1.0, xref="x", yref="paper",
                        text="flagged", showarrow=False, yanchor="bottom", font=dict(color="#ff4d4d", size=10))]
    for label, tp in zip(corners_tp["label"], corners_tp["track_pos"]):
        if lo <= tp <= hi:
            shapes.append(dict(type="line", xref="x", yref="paper", x0=tp, x1=tp, y0=0, y1=1,
                               line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot")))
            annotations.append(dict(x=tp, y=1.0, xref="x", yref="paper", text=label, showarrow=False,
                                    yanchor="bottom", font=dict(color="#c8d0dc", size=10)))
    fig.update_layout(
        height=420, margin=dict(t=30, b=40, l=70, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PLOT_BG,
        font=dict(color="#8a94a6", size=11), hovermode="x unified", hoversubplots="axis",
        legend=dict(orientation="h", y=1.12, x=0), shapes=shapes, annotations=annotations,
        xaxis=dict(title="Distance along the track", ticksuffix=" m", hoverformat=",.0f", gridcolor=GRID,
                   anchor="y3", range=[lo, hi]),
        yaxis=dict(domain=dom["y"], title=dict(text="Speed (km/h)", font=dict(size=10)), gridcolor=GRID),
        yaxis2=dict(domain=dom["y2"], title=dict(text="Throttle %", font=dict(size=10)), gridcolor=GRID,
                    range=[-3, 105]),
        yaxis3=dict(domain=dom["y3"], title=dict(text="Brake", font=dict(size=10)), range=[-5, 110],
                    showticklabels=False, showgrid=False),
    )
    return fig


# ── Detector comparison ───────────────────────────────────────────────────────

def _comparison_figures(meta: dict):
    planted = meta.get("comparison", {}).get("planted", {})
    order = ["z_score", "isolation_forest", "lstm_autoencoder"]
    bars = go.Figure()
    for name in order:
        if name not in planted:
            continue
        m = planted[name]
        bars.add_trace(go.Bar(x=["Precision", "Recall", "F1"], y=[m["precision"], m["recall"], m["f1"]],
                              name=DETECTORS[name][0], marker_color=DETECTOR_COLORS[name],
                              text=[f"{v:.2f}" for v in (m["precision"], m["recall"], m["f1"])], textposition="outside",
                              hovertemplate=DETECTORS[name][0] + ": %{x} %{y:.2f}<extra></extra>"))
    bars.update_layout(barmode="group", height=300, margin=dict(t=30, b=30, l=40, r=10),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PLOT_BG, font=dict(color="#8a94a6", size=11),
                       yaxis=dict(range=[0, 1.05], gridcolor=GRID), legend=dict(orientation="h", y=1.15, x=0))
    shares = costly_shares(meta)
    names = ["z_score", "isolation_forest", "lstm_autoencoder", "random_spots"]
    labels = [DETECTORS[n][0] if n in DETECTORS else "Random spots" for n in names]
    values = [shares[n] for n in names]
    real = go.Figure(go.Bar(
        x=labels, y=[v if v is not None else 0 for v in values],
        marker_color=[DETECTOR_COLORS[n] for n in names],
        text=["n/a" if v is None else f"{v * 100:.0f} %" for v in values], textposition="outside",
        hovertemplate="%{x}: %{text} of its moments cost time<extra></extra>"))
    real.update_layout(height=300, margin=dict(t=30, b=30, l=40, r=10), paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor=PLOT_BG, font=dict(color="#8a94a6", size=11), showlegend=False,
                       yaxis=dict(tickformat=".0%", range=[0, max([v or 0 for v in values] + [0.1]) * 1.3],
                                  gridcolor=GRID))
    return bars, real


# ── Tab ───────────────────────────────────────────────────────────────────────

def render_mistakes_tab(sel, df: Optional[pd.DataFrame], meta: Optional[dict], events_json: Optional[dict]) -> None:
    """Renders the Driver Mistakes tab. sel: SidebarSelections."""
    if df is None or meta is None:
        st.info("No mistake results for this driver in this session. Results exist for:\n\n"
                + "\n".join(f"- {s}" for s in sessions_with_results()))
        return
    ctx = f"{sel.year}_{sel.round_num}_{sel.session_type}_{sel.driver_number}"
    name = st.segmented_control(
        "Detector", list(DETECTORS), default="isolation_forest", required=True, key="mis_detector",
        format_func=lambda n: f"{DETECTORS[n][0]} ({DETECTORS[n][1]})")
    data = detector_data(df, events_json, name)
    if data is None:
        st.info(f"This output has no {DETECTORS[name][0]} results. Re-run detect_mistakes.py with "
                f"--compare --force.")
        return
    flags, _, events = data
    events = sorted(events, key=lambda e: e["rank"])
    shares = costly_shares(meta)
    lost = sum(max(e.get("time_lost_s") or 0, 0) for e in events if not e.get("likely_saving"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mistakes found", str(len(events)), help="Flagged stretches of at least 0.5 s")
    c2.metric("Time lost in total", f"{lost:.2f} s",
              help="Sum over the mistakes of the time lost against his own nearby laps at the same place")
    share = shares[name]
    random = shares["random_spots"]
    c3.metric("Mistakes that cost time", "n/a" if share is None else f"{share * 100:.0f} %",
              help="Real-lap check: share of flagged moments that lost clearly more time than his own "
                   "nearby laps" + ("" if random is None else
                                    f". Random places on the same laps: {random * 100:.0f} %"))
    c4.metric("Readings flagged", f"{int(flags.sum()):,}",
              help=f"{mistake_rate_pct(flags):.1f} % of the readings of {meta.get('n_laps', '?')} analysed laps: "
                   f"each detector flags its {meta.get('best_contamination', 0.02) * 100:.0f} % most unusual readings")

    if not events:
        st.info("No mistakes in this session: no run of flags lasted 0.5 s or more.")
        return
    lap_filter = st.selectbox("Show", ["All laps"] + [f"Lap {n}" for n in sorted({e["lap"] for e in events})],
                              key=f"mis_lap_{ctx}", filter_mode=None)
    shown = events if lap_filter == "All laps" else [e for e in events if e["lap"] == int(lap_filter[4:])]
    ids = [e["event_id"] for e in shown]
    sel_key = f"mis_sel_{ctx}_{name}"
    # Map and list keys carry the lap filter, so an old click does not carry over to another filter
    map_key, table_key = f"mis_map_{ctx}_{name}_{lap_filter}", f"mis_table_{ctx}_{name}_{lap_filter}"
    if st.session_state.get(sel_key + "_filter") != lap_filter:     # filter changed: start from the top
        st.session_state[sel_key + "_filter"] = lap_filter
        st.session_state[sel_key] = ids[0]

    # A click on the map or a row of the list since the last run picks that mistake
    clicked_map = (st.session_state.get(map_key) or {}).get("selection", {}).get("points", [])
    if clicked_map:
        cid = clicked_map[0].get("customdata")
        if cid is not None and cid != st.session_state.get(sel_key + "_lastmap"):
            st.session_state[sel_key + "_lastmap"] = cid
            st.session_state[sel_key] = int(cid)
    rows = (st.session_state.get(table_key) or {}).get("selection", {}).get("rows", [])
    if rows and rows != st.session_state.get(sel_key + "_lastrows") and rows[0] < len(shown):
        st.session_state[sel_key + "_lastrows"] = rows
        st.session_state[sel_key] = shown[rows[0]]["event_id"]
    if st.session_state.get(sel_key) not in ids:
        st.session_state[sel_key] = ids[0]
    selected = st.session_state[sel_key]

    corners = get_corners(sel.year, sel.round_num, sel.session_type)
    left, right = st.columns([1, 1.25])
    with left:
        st.plotly_chart(_map_figure(_track_line(sel), df, corners, shown, selected), width="stretch",
                        key=map_key, on_select="rerun", selection_mode="points", config={"displayModeBar": False})
        st.caption("Each marker is one mistake; bigger = more time lost. Click a marker or a row to see it below.")
    with right:
        table = pd.DataFrame({"#": [e["rank"] for e in shown], "Lap": [e["lap"] for e in shown],
                              "Where": [e.get("location", "") for e in shown],
                              "Time lost": [_time_lost_text(e) for e in shown],
                              "What happened": [_what(e) for e in shown]})
        st.dataframe(table, hide_index=True, width="stretch", height=470, key=table_key,
                     on_select="rerun", selection_mode="single-row",
                     column_config={"#": st.column_config.NumberColumn(width=40),
                                    "Lap": st.column_config.NumberColumn(width=45),
                                    "Where": st.column_config.TextColumn(width=140),
                                    "Time lost": st.column_config.TextColumn(width=80),
                                    "What happened": st.column_config.TextColumn(width="large")})

    event = next(e for e in shown if e["event_id"] == selected)
    st.markdown(f"#### Lap {event['lap']} · {event.get('location', '')}: {_what(event)}")
    st.caption(f"Time lost against his own nearby laps at this place: {_time_lost_text(event)} · "
               f"flagged for {event['duration_s']:.1f} s at {event['speed_kmh']:.0f} km/h")
    st.plotly_chart(_detail_figure(df, event, corner_track_pos(df, corners)), width="stretch",
                    key=f"mis_detail_{ctx}_{name}")
    st.caption("Field = average of the 5 fastest other drivers at the same point. His usual = median of his "
               "other analysed laps. Red band = the flagged stretch.")

    # ── How the three detectors compare ───────────────────────────────────
    st.markdown("#### How the three detectors compare (this session)")
    if "comparison" not in meta:
        st.info("Run detect_mistakes.py with --compare to compare the detectors.")
        return
    bars, real = _comparison_figures(meta)
    a, b = st.columns(2)
    with a:
        st.markdown("**Controlled test on synthetic mistakes**")
        st.plotly_chart(bars, width="stretch", key=f"mis_cmp_{ctx}", config={"displayModeBar": False})
        st.caption("Lock-ups, early lifts, late throttle, pedal overlap and missed downshifts were planted "
                   "in laps the detector had not seen. Shows how well it finds these patterns, not that "
                   "every real flag is a real mistake (there are no labelled mistakes in F1 data).")
    with b:
        st.markdown("**Real laps: do flagged moments lose time?**")
        st.plotly_chart(real, width="stretch", key=f"mis_real_{ctx}", config={"displayModeBar": False})
        st.caption("Share of each detector's mistakes that lost clearly more time than the driver's own "
                   "nearby laps there, against random places on the same laps.")
    with st.expander("Found per mistake type (controlled test)"):
        planted = meta["comparison"]["planted"]
        types = list(next(iter(planted.values()))["recall_by_type"])
        st.dataframe(pd.DataFrame({
            "Mistake type": [t.replace("_", " ").capitalize() for t in types],
            **{DETECTORS[n][0]: [f"{planted[n]['recall_by_type'][t] * 100:.0f} %" for t in types]
               for n in ("z_score", "isolation_forest", "lstm_autoencoder") if n in planted}}),
            hide_index=True, width="stretch")
