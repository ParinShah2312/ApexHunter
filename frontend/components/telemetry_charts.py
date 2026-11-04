"""
ApexHunter Frontend - Telemetry tab
One lap at a time, plotted against distance from the start line, optionally against another lap
(same driver) or another driver's fastest lap:
  lap picker (one dot per lap, click to open) → compare switch → cards → position slider →
  stacked charts (speed, time delta, throttle, brake bands, gear; RPM in a closed panel) + track map.
The maths lives in lap_telemetry.py.
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.data_loader import get_corners, get_session_laps
from components.header_bar import lap_time_text
from components.lap_telemetry import (corner_distances, full_throttle_pct, lap_samples, location_text,
                                      time_delta, to_lap_distance, value_at)

LAP_COLOR = "#00d4ff"
REF_COLOR = "#f5a623"
KIND_COLORS = {"normal": "#4f8fe0", "pit": "#8a94a6", "sc": "#e5c100"}
FASTEST_COLOR = "#b45cff"
PLOT_BG = "#0f1217"
GRID = "rgba(255,255,255,0.05)"
STRIP_CAP = 1.08          # lap picker: laps slower than 108 % of the best are drawn at the top edge
COMPARE_MODES = ["Another driver", "Another lap", "None"]
POSITION_STEP_M = 10


def _clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    return laps[(laps["kind"] == "normal") & ~laps["deleted"]]


def fastest_lap(laps: pd.DataFrame) -> Optional[pd.Series]:
    """A driver's fastest clean lap (no pit, SC/VSC or deleted lap); any lap if none is clean."""
    pool = _clean_laps(laps)
    if pool.empty:
        pool = laps
    return None if pool.empty else pool.loc[pool["lap_s"].idxmin()]


def _lap_label(row: pd.Series, best_lap: Optional[int]) -> str:
    tags = {"pit": " · pit lap", "sc": " · safety car"}.get(row["kind"], "")
    if row["deleted"]:
        tags += " · deleted"
    if best_lap is not None and int(row["lap"]) == best_lap:
        tags += " · fastest"
    return f"Lap {int(row['lap'])} · {lap_time_text(row['lap_s'])}{tags}"


# ── Lap picker strip ──────────────────────────────────────────────────────────

def _strip_figure(my_laps: pd.DataFrame, best_lap: Optional[int], selected: int) -> go.Figure:
    best_s = float(my_laps["lap_s"].min()) if best_lap is None else float(
        my_laps.loc[my_laps["lap"] == best_lap, "lap_s"].iloc[0])
    cap = best_s * STRIP_CAP
    fig = go.Figure()
    groups = [("Normal", my_laps["kind"].eq("normal") & my_laps["lap"].ne(best_lap), KIND_COLORS["normal"]),
              ("Pit in/out", my_laps["kind"].eq("pit"), KIND_COLORS["pit"]),
              ("Safety car / VSC", my_laps["kind"].eq("sc"), KIND_COLORS["sc"]),
              ("Fastest", my_laps["lap"].eq(best_lap), FASTEST_COLOR)]
    for name, mask, color in groups:
        sub = my_laps[mask]
        if sub.empty:
            continue
        over = sub["lap_s"] > cap
        fig.add_trace(go.Scatter(
            x=sub["lap"], y=np.minimum(sub["lap_s"], cap), mode="markers", name=name,
            marker=dict(color=color, size=np.where(sub["lap"].eq(selected), 15, 9),
                        symbol=np.where(over, "triangle-up", "circle"),
                        line=dict(color=np.where(sub["lap"].eq(selected), "#ffffff", color), width=2)),
            text=[lap_time_text(s) for s in sub["lap_s"]],
            hovertemplate="Lap %{x}: %{text}<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        height=170, margin=dict(t=8, b=30, l=50, r=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG, font=dict(color="#8a94a6", size=11), clickmode="event+select",
        legend=dict(orientation="h", y=1.18, x=0, font=dict(size=10)),
        xaxis=dict(title="Lap", gridcolor=GRID, dtick=5),
        yaxis=dict(title="Lap time (s)", gridcolor=GRID, range=[best_s - 0.6, cap + 0.4]),
    )
    return fig


def _clicked_lap(chart_key: str) -> Optional[int]:
    """The lap of the dot last clicked in the picker (from the chart's saved selection)."""
    state = st.session_state.get(chart_key)
    try:
        points = state["selection"]["points"]
    except (KeyError, TypeError):
        return None
    return int(points[0]["x"]) if points else None


# ── Stacked charts ────────────────────────────────────────────────────────────

def delta_text(delta_s: float, lap_name: str, ref_name: str) -> str:
    """'HAM lap 14 is 0.12 s behind VER lap 14' (or ahead / level)."""
    if abs(delta_s) < 0.005:
        return f"{lap_name} is level with {ref_name}"
    return f"{lap_name} is {abs(delta_s):.2f} s {'behind' if delta_s > 0 else 'ahead of'} {ref_name}"


def _row_domains(names: list) -> dict:
    """Vertical space per chart (top to bottom), with small gaps."""
    weights = {"speed": 4.0, "delta": 1.6, "throttle": 1.6, "brake": 0.8, "gear": 1.4}
    gap = 0.035
    total = sum(weights[n] for n in names)
    usable = 1 - gap * (len(names) - 1)
    domains, top = {}, 1.0
    for n in names:
        h = usable * weights[n] / total
        domains[n] = [max(top - h, 0.0), top]
        top -= h + gap
    return domains


def _stacked_figure(lap: pd.DataFrame, ref: Optional[pd.DataFrame], lap_name: str, ref_name: str,
                    corners: pd.DataFrame, position_m: float) -> go.Figure:
    names = ["speed"] + (["delta"] if ref is not None else []) + ["throttle", "brake", "gear"]
    dom = _row_domains(names)
    axis_of = {n: ("y" if i == 0 else f"y{i + 1}") for i, n in enumerate(names)}
    x = lap["dist"]
    ref_x = to_lap_distance(ref, lap) if ref is not None else None
    fig = go.Figure()

    def add(row, **kw):
        fig.add_trace(go.Scatter(xaxis="x", yaxis=axis_of[row], **kw))

    # Hover text is plain language; the place on the lap heads the list (on the chosen lap's speed)
    where = [location_text(d, corners) for d in x]
    hl, hr = lap_name, ref_name

    # Speed
    if ref is not None:
        add("speed", x=ref_x, y=ref["Speed"], name=ref_name, line=dict(color=REF_COLOR, width=1.3),
            hovertemplate=f"{hr}: %{{y:.0f}} km/h<extra></extra>")
    add("speed", x=x, y=lap["Speed"], name=lap_name, line=dict(color=LAP_COLOR, width=2), customdata=where,
        hovertemplate=f"<b>%{{customdata}}</b><br>{hl}: %{{y:.0f}} km/h<extra></extra>")
    # Time delta (+ = losing time to the comparison lap)
    if ref is not None:
        d = time_delta(lap, ref)
        add("delta", x=x, y=np.maximum(d, 0), fill="tozeroy", fillcolor="rgba(255,80,80,0.25)",
            line=dict(width=0), hoverinfo="skip", showlegend=False)
        add("delta", x=x, y=np.minimum(d, 0), fill="tozeroy", fillcolor="rgba(46,204,113,0.25)",
            line=dict(width=0), hoverinfo="skip", showlegend=False)
        add("delta", x=x, y=d, name="Time delta", line=dict(color="#e8edf5", width=1.5), showlegend=False,
            customdata=[delta_text(v, lap_name, ref_name) for v in d],
            hovertemplate="%{customdata}<extra></extra>")
    # Throttle
    if ref is not None:
        add("throttle", x=ref_x, y=ref["Throttle"], line=dict(color=REF_COLOR, width=1), showlegend=False,
            name=ref_name, hovertemplate=f"{hr}: throttle %{{y:.0f}} %<extra></extra>")
    add("throttle", x=x, y=lap["Throttle"], fill="tozeroy", fillcolor="rgba(0,212,255,0.18)",
        line=dict(color=LAP_COLOR, width=1.2), showlegend=False, name=lap_name,
        hovertemplate=f"{hl}: throttle %{{y:.0f}} %<extra></extra>")
    # Brake: on/off, so bands (one lane per lap), not a line
    lanes = [(lap, x, 1.0, LAP_COLOR, lap_name)]
    if ref is not None:
        lanes.append((ref, ref_x, 0.0, REF_COLOR, ref_name))
    for frame, xs, lane, color, name in lanes:
        on = frame["Brake"].to_numpy() >= 0.5
        add("brake", x=xs, y=np.where(on, lane, np.nan), mode="lines", connectgaps=False,
            line=dict(color=color, width=9), showlegend=False, name=name,
            hovertemplate=f"{name}: braking<extra></extra>")
    # Gear
    if ref is not None:
        add("gear", x=ref_x, y=ref["nGear"], line=dict(color=REF_COLOR, width=1, shape="hv"),
            showlegend=False, name=ref_name, hovertemplate=f"{hr}: gear %{{y}}<extra></extra>")
    add("gear", x=x, y=lap["nGear"], line=dict(color=LAP_COLOR, width=1.6, shape="hv"),
        showlegend=False, name=lap_name, hovertemplate=f"{hl}: gear %{{y}}<extra></extra>")

    titles = {"speed": "Speed (km/h)", "delta": "Delta (s)", "throttle": "Throttle %",
              "brake": "Brake", "gear": "Gear"}
    layout = dict(
        height=640, margin=dict(t=46, b=40, l=70, r=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG, font=dict(color="#8a94a6", size=11),
        hovermode="x unified", hoversubplots="axis",
        legend=dict(orientation="h", y=1.07, x=0),
        xaxis=dict(title="Distance from the start line", gridcolor=GRID, anchor=axis_of["gear"],
                   hoverformat=",.0f", ticksuffix=" m",
                   showspikes=True, spikemode="across", spikethickness=1, spikecolor="#8a94a6",
                   spikedash="dot", range=[0, float(x.iloc[-1])]),
    )
    for n in names:
        key = "yaxis" if axis_of[n] == "y" else f"yaxis{axis_of[n][1:]}"
        ax = dict(domain=dom[n], title=dict(text=titles[n], font=dict(size=10)), gridcolor=GRID,
                  anchor="x", zeroline=n == "delta", zerolinecolor="#5b6577")
        if n == "brake":
            ax.update(range=[-0.6, 1.6], showticklabels=False, showgrid=False)
        if n == "throttle":
            ax.update(range=[-3, 105])
        layout[key] = ax
    shapes = [dict(type="line", xref="x", yref="paper", x0=d, x1=d, y0=0, y1=1,
                   line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dot"))
              for d in corners["dist"]]
    shapes.append(dict(type="line", xref="x", yref="paper", x0=position_m, x1=position_m, y0=0, y1=1,
                       line=dict(color="rgba(255,255,255,0.7)", width=1.2)))
    layout["shapes"] = shapes
    layout["annotations"] = [dict(x=d, y=1.0, xref="x", yref="paper", text=label, showarrow=False,
                                  yanchor="bottom", font=dict(size=9, color="#8a94a6"))
                             for label, d in zip(corners["label"], corners["dist"])]
    fig.update_layout(**layout)
    return fig


def _rpm_figure(lap: pd.DataFrame, ref: Optional[pd.DataFrame], lap_name: str, ref_name: str) -> go.Figure:
    fig = go.Figure()
    if ref is not None:
        fig.add_trace(go.Scatter(x=to_lap_distance(ref, lap), y=ref["RPM"], name=ref_name,
                                 line=dict(color=REF_COLOR, width=1)))
    fig.add_trace(go.Scatter(x=lap["dist"], y=lap["RPM"], name=lap_name, line=dict(color=LAP_COLOR, width=1.4)))
    fig.update_layout(height=200, margin=dict(t=10, b=40, l=70, r=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor=PLOT_BG, font=dict(color="#8a94a6", size=11), hovermode="x unified",
                      showlegend=False, xaxis=dict(title="Distance (m)", gridcolor=GRID),
                      yaxis=dict(title="RPM", gridcolor=GRID))
    return fig


MAP_STEP_M = 5.0          # track map: the lap is redrawn every 5 m (readings are ~70 m apart at top speed)


def _dense_lap(lap: pd.DataFrame) -> pd.DataFrame:
    """The lap re-sampled every MAP_STEP_M metres (position and speed interpolated, gear held)."""
    d = np.arange(0.0, float(lap["dist"].iloc[-1]), MAP_STEP_M)
    src = lap["dist"].to_numpy()
    gear_idx = np.clip(np.searchsorted(src, d, side="right") - 1, 0, len(lap) - 1)
    return pd.DataFrame({"dist": d, "X": np.interp(d, src, lap["X"]), "Y": np.interp(d, src, lap["Y"]),
                         "Speed": np.interp(d, src, lap["Speed"]), "nGear": lap["nGear"].to_numpy()[gear_idx]})


def _track_figure(lap: pd.DataFrame, corners: pd.DataFrame, color_by: str, position_m: float) -> go.Figure:
    col = "Speed" if color_by == "Speed" else "nGear"
    lap = _dense_lap(lap)
    fig = go.Figure(go.Scatter(
        x=lap["X"], y=lap["Y"], mode="markers",
        marker=dict(color=lap[col], colorscale="Turbo" if col == "Speed" else "Viridis", size=5,
                    colorbar=dict(title=dict(text="km/h" if col == "Speed" else "Gear", side="right"),
                                  thickness=10, len=0.7)),
        customdata=np.stack([[location_text(d, corners) for d in lap["dist"]], lap[col].round(0)], axis=1),
        hovertemplate="%{customdata[0]}<br>" + ("%{customdata[1]} km/h" if col == "Speed"
                                               else "gear %{customdata[1]}") + "<extra></extra>",
        showlegend=False,
    ))
    if not corners.empty:
        idx = [int(np.argmin(np.abs(lap["dist"].to_numpy() - d))) for d in corners["dist"]]
        fig.add_trace(go.Scatter(x=lap["X"].to_numpy()[idx], y=lap["Y"].to_numpy()[idx], mode="text",
                                 text=corners["label"], textposition="top right", hoverinfo="skip",
                                 textfont=dict(color="#c8d0dc", size=10), showlegend=False))
    i = int(np.argmin(np.abs(lap["dist"].to_numpy() - position_m)))
    fig.add_trace(go.Scatter(x=[lap["X"].iloc[i]], y=[lap["Y"].iloc[i]], mode="markers", hoverinfo="skip",
                             marker=dict(color="#ffffff", size=14, line=dict(color="#000000", width=2)),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[lap["X"].iloc[0]], y=[lap["Y"].iloc[0]], mode="markers+text", text=["Start"],
                             textposition="bottom center", hoverinfo="skip", showlegend=False,
                             marker=dict(symbol="square", color="#e8edf5", size=8),
                             textfont=dict(color="#e8edf5", size=10)))
    fig.update_layout(height=430, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor=PLOT_BG,
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1))
    return fig


# ── Tab ───────────────────────────────────────────────────────────────────────

def _driver_label(info: dict, num: str) -> str:
    d = info.get("drivers", {}).get(num, {})
    return d.get("abbr") or f"#{num}"


def render_telemetry(sel) -> None:
    """Renders the Telemetry tab. sel: SidebarSelections."""
    laps = get_session_laps(sel.year, sel.round_num, sel.session_type)
    my_laps = laps[laps["driver"] == sel.driver_number].sort_values("lap")
    if my_laps.empty or sel.df_driver.empty:
        st.warning("No lap data for this driver in this session (FastF1 laps or telemetry missing).")
        return
    info = sel.session_info or {}
    ctx = f"{sel.year}_{sel.round_num}_{sel.session_type}_{sel.driver_number}"
    best = fastest_lap(my_laps)
    best_lap = None if best is None else int(best["lap"])
    lap_options = [int(n) for n in my_laps["lap"]]
    lap_key, strip_key = f"tel_lap_{ctx}", f"tel_strip_{ctx}"

    # A click in the picker since the last run selects that lap
    clicked = _clicked_lap(strip_key)
    if clicked is not None and clicked != st.session_state.get(f"tel_click_{ctx}"):
        st.session_state[f"tel_click_{ctx}"] = clicked
        if clicked in lap_options:
            st.session_state[lap_key] = clicked
    if st.session_state.get(lap_key) not in lap_options:
        st.session_state[lap_key] = best_lap if best_lap is not None else lap_options[0]

    # ── Lap picker ────────────────────────────────────────────────────────
    st.plotly_chart(_strip_figure(my_laps, best_lap, st.session_state[lap_key]), width="stretch",
                    key=strip_key, on_select="rerun", selection_mode="points",
                    config={"displayModeBar": False})
    st.caption("Each dot is one lap (click to open it): purple = fastest, grey = pit in/out lap, "
               "yellow = safety car or VSC. Triangles at the top are slower than shown.")

    by_lap = my_laps.set_index("lap")

    def label_of(n: int) -> str:
        return _lap_label(pd.Series({**by_lap.loc[n].to_dict(), "lap": n}), best_lap)

    c1, c2, c3 = st.columns([1.2, 1.1, 1.3])
    with c1:
        lap_no = st.selectbox("Lap", lap_options, key=lap_key, filter_mode=None, format_func=label_of)
    with c2:
        mode = st.segmented_control("Compare with", COMPARE_MODES, default=COMPARE_MODES[0],
                                    required=True, key="tel_compare_mode")
    ref_row, ref_driver, ref_name = None, None, ""
    with c3:
        if mode == "Another driver":
            others = []
            for num, grp in laps[laps["driver"] != sel.driver_number].groupby("driver"):
                f = fastest_lap(grp)
                if f is not None:
                    others.append((num, f))
            others.sort(key=lambda p: p[1]["lap_s"])
            if others:
                options = [num for num, _ in others]
                session_best = float(laps["lap_s"][laps["kind"].eq("normal") & ~laps["deleted"]].min())
                fast = dict(others)
                ref_driver = st.selectbox(
                    "Driver (their fastest lap)", options, key=f"tel_ref_driver_{ctx}", filter_mode=None,
                    format_func=lambda n: f"{_driver_label(info, n)} · {lap_time_text(fast[n]['lap_s'])}"
                                          + (" · session fastest" if fast[n]["lap_s"] <= session_best else ""))
                ref_row = fast[ref_driver]
                ref_name = f"{_driver_label(info, ref_driver)} lap {int(ref_row['lap'])}"
        elif mode == "Another lap":
            choices = [n for n in lap_options if n != lap_no]
            if choices:
                clean = _clean_laps(my_laps[my_laps["lap"] != lap_no])
                default = int(clean.loc[clean["lap_s"].idxmin(), "lap"]) if not clean.empty else choices[0]
                ref_key = f"tel_ref_lap_{ctx}"
                if st.session_state.get(ref_key) not in choices:
                    st.session_state[ref_key] = default
                ref_lap = st.selectbox("Other lap", choices, key=ref_key, filter_mode=None,
                                       format_func=label_of)
                ref_row, ref_driver = by_lap.loc[ref_lap].copy(), sel.driver_number
                ref_row["lap"] = ref_lap
                ref_name = f"{_driver_label(info, sel.driver_number)} lap {ref_lap}"

    row = by_lap.loc[lap_no]
    lap = lap_samples(sel.df_driver, row["start_s"], row["lap_s"])
    if lap.empty:
        st.warning(f"No telemetry readings inside lap {lap_no}.")
        return
    ref = None
    if ref_row is not None:
        ref_df = sel.df_full[sel.df_full["Driver"].astype(str) == ref_driver]
        ref = lap_samples(ref_df, ref_row["start_s"], ref_row["lap_s"])
        if ref.empty:
            st.info(f"No telemetry readings for {ref_name}; showing this lap alone.")
            ref = None
    lap_name = f"{_driver_label(info, sel.driver_number)} lap {lap_no}"

    # ── Cards ─────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lap time", lap_time_text(row["lap_s"]))
    if ref is not None:
        gap = row["lap_s"] - ref_row["lap_s"]
        m2.metric(f"Gap to {ref_name}", f"{gap:+.3f} s", help="+ = this lap is slower")
    else:
        m2.metric("Gap", "—", help="Pick a lap or driver to compare with")
    m3.metric("Top speed", f"{lap['Speed'].max():.0f} km/h")
    m4.metric("Full throttle", f"{full_throttle_pct(lap):.0f} %", help="Share of the lap's time at ≥ 98 % throttle")

    # ── Position slider ───────────────────────────────────────────────────
    length = float(lap["dist"].iloc[-1])
    corners = corner_distances(lap, get_corners(sel.year, sel.round_num, sel.session_type))
    stops = list(range(0, int(length) + 1, POSITION_STEP_M))
    pos_key = f"tel_pos_{ctx}"
    if st.session_state.get(pos_key) not in stops:
        st.session_state[pos_key] = 0
    # Labels must be unique: Streamlit maps a label back to the FIRST option with that label, so
    # "Straight: Turn 4 → Turn 5" alone made the dot snap to one point of the straight
    pos = st.select_slider("Position in the lap", stops, key=pos_key,
                           format_func=lambda m: f"{location_text(m, corners)} · {m:,} m")
    parts = [f"**{location_text(pos, corners)}** ({pos:,} m)", f"{value_at(lap, pos, 'Speed'):.0f} km/h",
             f"throttle {value_at(lap, pos, 'Throttle'):.0f} %",
             "braking" if value_at(lap, pos, "Brake") >= 0.5 else "no brake",
             f"gear {value_at(lap, pos, 'nGear'):.0f}"]
    if ref is not None:
        d = time_delta(lap, ref)
        parts.append(f"delta {d[int(np.argmin(np.abs(lap['dist'].to_numpy() - pos)))]:+.3f} s")
    st.markdown(" · ".join(parts))

    # ── Charts + track map ────────────────────────────────────────────────
    left, right = st.columns([2.3, 1])
    with left:
        st.plotly_chart(_stacked_figure(lap, ref, lap_name, ref_name, corners, pos), width="stretch",
                        key=f"tel_stack_{ctx}")
        st.caption("x = distance from the start line; dotted lines = corners. Delta above zero = this "
                   "lap is losing time to the comparison lap at that point. Brake is on/off only. "
                   "About 4 readings a second, so very short events look slightly rounded.")
        with st.expander("More channels: engine RPM"):
            st.plotly_chart(_rpm_figure(lap, ref, lap_name, ref_name), width="stretch", key=f"tel_rpm_{ctx}")
    with right:
        color_by = st.segmented_control("Colour track by", ["Speed", "Gear"], default="Speed",
                                        required=True, key="tel_map_color")
        st.plotly_chart(_track_figure(lap, corners, color_by, pos), width="stretch", key=f"tel_map_{ctx}",
                        config={"displayModeBar": False})
        st.caption("White dot = position from the slider.")
