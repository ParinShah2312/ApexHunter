"""Strategy check of a tyre pace model on held-out races (ROADMAP §C step 6).

A model's predicted lap-time changes go into the pit window (pit_window.driver_strategy) and the
result is scored against what the drivers really did: is the real stop inside the window (vs a
random window of the same size), and did real safety-car / VSC stops get a "pit now" call. Every
source is scored on the same stints, so the numbers are directly comparable.
"""

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from lstm.tyre_data import SEQUENCE_LENGTH, _consecutive_runs, make_window, window_positions


def predicted_changes(stints: List[dict], predict: Callable[[np.ndarray], np.ndarray]) -> Dict[int, float]:
    """Model predictions for one driver, keyed by the race lap they predict.

    The prediction for lap L comes from laps L-5..L-1, so it is known before lap L is driven and
    may inform the decision to pit on lap L (the key, as pit_window expects). Lap L is predicted
    whenever those 5 laps are consecutive clean laps, whether or not lap L itself turns out clean
    (an in-lap, a safety-car lap): that is only known after lap L, so it must not decide whether a
    prediction exists (#52; before, the lap after every run of clean laps had none).
    """
    known = {}
    for stint in stints:
        lf = pd.DataFrame(stint["lap_features"])
        laps = lf["race_lap_number"]
        targets = {int(laps.iloc[j]): j for j in window_positions(laps)}
        for run in _consecutive_runs(laps):            # the lap after each run of >= 5 clean laps
            if len(run) >= SEQUENCE_LENGTH:
                targets.setdefault(int(laps.iloc[run[-1]]) + 1, int(run[-1]) + 1)
        if not targets:
            continue
        preds = predict(np.array([make_window(lf, j) for j in targets.values()], dtype=np.float32))
        for lap, p in zip(targets, preds):
            known[lap] = float(p)
    return known


def stint_rows(source: str, race: str, driver: str, plans: List[dict]) -> List[dict]:
    """One flat row per planned stint (for scoring), incl. the SC/VSC laps it had advice on."""
    rows = []
    for st in plans:
        rows.append({
            "source": source, "race": race, "driver": driver, "stint": st["stint"],
            "actual_stop": st["actual_stop"], "hit": st["hit"],
            "random_hit_chance": st["random_hit_chance"], "window_size": len(st["window"]),
            "advised_laps": len(st["advice"]), "actual_under_sc_vsc": st["actual_under_sc_vsc"],
            "sc_called": st["actual_stop"] in st["sc_vsc_calls"],
            "sc_laps": [a["lap"] for a in st["advice"] if a["sc_vsc"]], "sc_calls": st["sc_vsc_calls"],
        })
    return rows


def summarise(rows: pd.DataFrame) -> Dict[str, dict]:
    """Scores per source on the stints that every source could advise on.

    hit_pct: real stop inside the window; random_pct: expected hits of a random window of the same
    size; sc_stops_called_pct: real SC/VSC stops with a "pit now" call; sc_laps_called_pct: share of
    the SC/VSC laps that every source advised on with a "pit now" call (if near 100 %, the call says
    little on its own). A model needs 5 clean laps before it can advise, so it has no advice on
    early-race safety cars; counting only shared laps keeps the comparison fair.
    """
    key = ["race", "driver", "stint"]
    n_sources = rows["source"].nunique()
    rows = rows[rows.groupby(key)["source"].transform("nunique") == n_sources]
    shared_sc = rows.groupby(key)["sc_laps"].agg(lambda lists: set.intersection(*map(set, lists)))
    out = {}
    for source, g in rows.groupby("source"):
        sc = g[g["actual_under_sc_vsc"]]
        shared = [shared_sc[tuple(r[k] for k in key)] for _, r in g.iterrows()]
        n_shared = sum(len(laps) for laps in shared)
        n_called = sum(len(laps & set(calls)) for laps, calls in zip(shared, g["sc_calls"]))
        out[source] = {
            "stints": int(len(g)),
            "hit_pct": float(100 * g["hit"].mean()),
            "random_pct": float(100 * g["random_hit_chance"].mean()),
            "median_window_laps": float(g["window_size"].median()),
            "median_advised_laps": float(g["advised_laps"].median()),
            "sc_vsc_stops": int(len(sc)),
            "sc_stops_called_pct": float(100 * sc["sc_called"].mean()) if len(sc) else float("nan"),
            "shared_sc_laps": int(n_shared),
            "sc_laps_called_pct": float(100 * n_called / n_shared) if n_shared else float("nan"),
        }
    return out


def best_source(summary: Dict[str, dict]) -> str:
    """The source whose windows beat a random window of the same size by the most points
    (hit_pct - random_pct). Raw hit rate is not used: a wider window hits more often by luck."""
    return max(summary, key=lambda s: summary[s]["hit_pct"] - summary[s]["random_pct"])
