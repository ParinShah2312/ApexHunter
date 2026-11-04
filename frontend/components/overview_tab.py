"""
ApexHunter Frontend - Overview tab
One screen for the chosen session and driver: the driver's mistakes (count, time lost, top 3) and
pit windows (per stint: window, real stop, inside or not), then the model scoreboard over all
scored sessions (mistakes: planted-mistake F1 and real-lap check; tyres: lap-time error and pit
window vs a random window). Uses the same helpers as the Driver Mistakes and Tyres tabs, so the
numbers always match theirs. Shows the trained models only (user, 2026-11-05).
"""

import json
from typing import Dict, Optional

import numpy as np
import streamlit as st

from config import MISTAKE_DATA_DIR, MODELS_DIR, TYRE_PREDICTIONS_DIR
from components.mistakes_tab import (DETECTORS, _time_lost_text, _what, costly_shares, detector_data,
                                     sessions_with_results)
from components.strategy_tab import (MODEL_CONFIGS, SOURCE_KIND, SOURCE_SHORT, default_model, lap_ranges_text,
                                     normal_window, races_with_results, source_view, test_errors)

MISTAKE_MODELS = ("isolation_forest", "lstm_autoencoder")      # trained detectors (Z-score = baseline)
TYRE_MODELS = ("ridge", "rnn", "lstm")
CHECK_FILE = TYRE_PREDICTIONS_DIR / "strategy_check.json"


@st.cache_data(show_spinner=False)
def mistake_scoreboard() -> Dict[str, dict]:
    """Mean planted-mistake precision/recall/F1 and real-lap 'cost time' share per detector over
    every saved session (meta files with --compare), plus the number of sessions."""
    rows: Dict[str, list] = {}
    for f in sorted(MISTAKE_DATA_DIR.glob("*_mistakes_meta.json")):
        with open(f, encoding="utf-8") as fh:
            meta = json.load(fh)
        planted = meta.get("comparison", {}).get("planted", {})
        shares = costly_shares(meta)
        for name in DETECTORS:
            if name in planted:
                rows.setdefault(name, []).append({**planted[name], "costly": shares.get(name),
                                                  "random": shares.get("random_spots")})
    out = {}
    for name, rs in rows.items():
        def mean(key):
            vals = [r[key] for r in rs if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        out[name] = {"precision": mean("precision"), "recall": mean("recall"), "f1": mean("f1"),
                     "costly": mean("costly"), "random": mean("random"), "sessions": len(rs)}
    return out


@st.cache_data(show_spinner=False)
def tyre_scoreboard() -> Dict[str, dict]:
    """Per trained model: test-race lap-time error and R² (model configs) and pit-window hit rate vs
    a random window (strategy_check.json)."""
    out: Dict[str, dict] = {}
    errors = test_errors()
    for name, fname in MODEL_CONFIGS.items():
        try:
            with open(MODELS_DIR / fname, encoding="utf-8") as f:
                test = json.load(f)["metrics"]["test"]
        except (OSError, KeyError, ValueError):
            continue
        out[name] = {"mae": errors.get(name), "r2": test.get("r2")}
    try:
        with open(CHECK_FILE, encoding="utf-8") as f:
            summary = json.load(f)["summary"]
    except (OSError, KeyError, ValueError):
        summary = {}
    for name in out:
        if name in summary:
            out[name].update(hit=summary[name]["hit_pct"], random=summary[name]["random_pct"])
    return out


def _mistakes_card(df, meta: Optional[dict], events_json: Optional[dict]) -> None:
    st.markdown("##### Driver mistakes")
    if df is None or meta is None:
        st.info("No mistake results for this driver here. They exist for: "
                + "; ".join(sessions_with_results()) + ".")
        return
    data = detector_data(df, events_json, "isolation_forest")
    events = sorted(data[2], key=lambda e: e["rank"]) if data else []
    lost = sum(max(e.get("time_lost_s") or 0, 0) for e in events if not e.get("likely_saving"))
    share = costly_shares(meta)["isolation_forest"]
    a, b, c = st.columns(3)
    a.metric("Mistakes", str(len(events)), help="Isolation Forest (ML), flagged stretches of 0.5 s or more")
    b.metric("Time lost", f"{lost:.2f} s", help="Against his own nearby laps at the same places")
    c.metric("Cost time", "n/a" if share is None else f"{share * 100:.0f} %",
             help="Share of the mistakes that lost clearly more time than his nearby laps")
    if events:
        st.markdown("**Most costly**")
        st.markdown("\n".join(f"{i}. Lap {e['lap']} · {e.get('location', '')}: {_what(e)} "
                              f"({_time_lost_text(e)})" for i, e in enumerate(events[:3], start=1)))
    st.caption("All mistakes, the other detectors and the charts: Driver Mistakes tab.")


def _pit_card(data: Optional[dict], ctx: str) -> None:
    st.markdown("##### Pit windows")
    if data is None:
        st.info("No pit windows for this driver here (races only). They exist for: "
                + ", ".join(races_with_results()) + " (every driver).")
        return
    available = [m for m in TYRE_MODELS if source_view(data, m) is not None] or [data["model"]]
    model = st.session_state.get(f"tyre_model_{ctx}")         # follow the Tyres tab's choice
    if model not in available:
        model = default_model(data, available)
    view = source_view(data, model)
    advised = [s for s in view.get("stints", []) if "summary" in s]
    hits = sum(bool(s["hit"]) for s in advised)
    a, b = st.columns(2)
    a.metric("Stops in window", f"{hits} of {len(advised)}" if advised else "—",
             help=f"Windows from {SOURCE_SHORT.get(model, model)}; the last stint has no stop")
    b.metric("Model", f"{SOURCE_SHORT.get(model, model)} ({SOURCE_KIND.get(model, '')})")
    lines = []
    for s in view.get("stints", []):
        head = f"Stint {s['stint']} · {s['compound'].capitalize()} · laps {s['first_lap']}–{s['last_lap']}"
        if "summary" not in s:
            lines.append(f"- {head}: no stop advice (last stint)")
            continue
        window = normal_window(s)
        win = f"window {lap_ranges_text(window)}" if window else "safety-car call only"
        lines.append(f"- {head}: {win}, pitted lap {s['actual_stop']} "
                     f"{'✓ inside' if s['hit'] else '✗ outside'}")
    st.markdown("\n".join(lines))
    st.caption("Charts and the other models: Tyres & Pit Window tab.")


def _scoreboard() -> None:
    st.markdown("#### Model scoreboard (all scored sessions)")
    mistakes = mistake_scoreboard()
    tyres = tyre_scoreboard()
    left, right = st.columns(2)
    with left:
        n = max((m["sessions"] for m in mistakes.values()), default=0)
        st.markdown(f"**Driver mistakes** · controlled test on synthetic mistakes, {n} sessions")
        cols = st.columns(len(MISTAKE_MODELS))
        for col, name in zip(cols, MISTAKE_MODELS):
            m = mistakes.get(name)
            label, kind, _ = DETECTORS[name]
            col.metric(f"{label} ({kind})", "—" if not m else f"F1 {m['f1']:.2f}",
                       help=None if not m else
                       f"Precision {m['precision']:.2f}, recall {m['recall']:.2f}. Real laps: "
                       f"{(m['costly'] or 0) * 100:.0f} % of its mistakes cost time "
                       f"(random places {(m['random'] or 0) * 100:.0f} %)")
        st.caption("F1 = balance of finding the planted mistakes (recall) and being right when it flags "
                   "(precision). It shows these patterns are found, not that every real flag is a mistake.")
    with right:
        st.markdown("**Tyre pace** · 5 held-out test races")
        cols = st.columns(len(TYRE_MODELS))
        for col, name in zip(cols, TYRE_MODELS):
            m = tyres.get(name)
            col.metric(f"{SOURCE_SHORT[name]} ({SOURCE_KIND[name]})",
                       "—" if not m or m.get("mae") is None else f"{m['mae']:.3f} s",
                       help=None if not m else
                       f"Average lap-time error per lap; R² {m.get('r2', 0):.2f}. Pit window: real stop "
                       f"inside {m.get('hit', 0):.0f} % of the time (random window {m.get('random', 0):.0f} %)")
        st.caption("Average error of the predicted next lap time (lower is better). Pit-window results "
                   "are in the ⓘ notes.")


def render_overview(sel, df_mistakes, meta: Optional[dict], events_json: Optional[dict],
                    tyre_data: Optional[dict]) -> None:
    """Renders the Overview tab. sel: SidebarSelections."""
    ctx = f"{sel.year}_{sel.round_num}_{sel.session_type}_{sel.driver_number}"
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            _mistakes_card(df_mistakes, meta, events_json)
    with right:
        with st.container(border=True):
            _pit_card(tyre_data, ctx)
    _scoreboard()
