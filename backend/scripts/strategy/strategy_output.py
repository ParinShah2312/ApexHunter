"""Per-driver pit-strategy output for one race (ROADMAP §C step 7; replaces the "tyre cliff").

The JSON written by predict_strategy.py holds every lap (real and predicted lap time, compound,
tyre age, track status) and every stint (pit window, best lap, safety-car calls, the real stop and
whether it was inside the window), plus a short readable summary per stint, e.g.

    Normal window: laps 33-36, best: lap 35
    VSC on lap 28 -> pit now (best option)
    Actual: pitted lap 28 (under VSC) - inside the window
"""

from typing import Dict, List, Optional

import pandas as pd

from strategy.pit_cost import lap_kind
from strategy.pit_window import WINDOW_HALF_LAPS, driver_strategy


def lap_ranges(laps: List[int]) -> str:
    """[13, 14, 15, 20] -> "13-15, 20"."""
    parts, start, prev = [], None, None
    for n in sorted(laps):
        if start is not None and n == prev + 1:
            prev = n
            continue
        if start is not None:
            parts.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = n
    if start is not None:
        parts.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ", ".join(parts)


def _neutral_name(status: str) -> str:
    return "Safety car" if "4" in status else "VSC"


def lap_table(driver_laps: pd.DataFrame, known_changes: Optional[Dict[int, float]]) -> List[dict]:
    """One row per race lap. predicted_lap_time = previous lap's real time + the predicted change
    (None where the model had no 5-lap history or the previous lap has no time, and for every lap
    when known_changes is None, i.e. a rule-based wear source with no lap predictions)."""
    known_changes = known_changes or {}
    laps = driver_laps.sort_values("LapNumber")
    times = dict(zip(laps["LapNumber"].astype(int), laps["LapTime"].dt.total_seconds()))
    rows = []
    for _, lap in laps.iterrows():
        n = int(lap["LapNumber"])
        status = str(lap["TrackStatus"]) if pd.notna(lap["TrackStatus"]) else ""
        prev = times.get(n - 1)
        predicted = (prev + known_changes[n]) if n in known_changes and prev is not None and pd.notna(prev) else None
        actual = times.get(n)
        rows.append({
            "lap": n, "stint": int(lap["Stint"]) if pd.notna(lap["Stint"]) else None,
            "compound": str(lap["Compound"]) if pd.notna(lap["Compound"]) else None,
            "tyre_age": float(lap["TyreLife"]) if pd.notna(lap["TyreLife"]) else None,
            "lap_time": float(actual) if actual is not None and pd.notna(actual) else None,
            "predicted_lap_time": float(predicted) if predicted is not None else None,
            "track": lap_kind(status), "neutralised": _neutral_name(status) if lap_kind(status) == "sc_vsc" else None,
            "pit_in": bool(pd.notna(lap["PitInTime"])),
        })
    return rows


def stint_summary(plan: dict, laps_by_number: Dict[int, dict]) -> List[str]:
    """The readable lines for one planned stint."""
    # normal window = the window laps around the normal best lap (+-2), SC/VSC laps included;
    # safety-car calls elsewhere get their own line
    if plan["best_lap"]:
        green = [n for n in plan["window"] if abs(n - plan["best_lap"]) <= WINDOW_HALF_LAPS]
    else:
        green = [n for n in plan["window"] if n not in plan["sc_vsc_calls"]]
    lines = [f"Normal window: lap{'s' if len(green) > 1 else ''} {lap_ranges(green)}" + (f", best: lap {plan['best_lap']}" if plan["best_lap"] else "")
             if green else "Normal window: none"]
    best_now = {a["lap"] for a in plan["advice"] if a["best_lap"] == a["lap"]}
    for run in lap_ranges(plan["sc_vsc_calls"]).split(", ") if plan["sc_vsc_calls"] else []:
        a, _, b = run.partition("-")
        first, last = int(a), int(b or a)
        name = laps_by_number[first]["neutralised"] or "SC/VSC"
        best = any(n in best_now for n in range(first, last + 1))
        lines.append(f"{name} on lap{'s' if last > first else ''} {run} -> pit now"
                     + (" (best option)" if best else ""))
    stop = plan["actual_stop"]
    under = laps_by_number[stop]["neutralised"] if plan["actual_under_sc_vsc"] else None
    under_text = {"Safety car": "the safety car", "VSC": "VSC"}.get(under)
    lines.append(f"Actual: pitted lap {stop}" + (f" (under {under_text})" if under else "")
                 + (" - inside the window" if plan["hit"] else " - outside the window"))
    return lines


def build_strategy_output(driver_laps: pd.DataFrame, known_changes: Dict[int, float], cost: dict,
                          model: str, in_training_data: Optional[bool],
                          lap_predictions: bool = True) -> dict:
    """Everything the dashboard shows for one driver in one race (see module doc).

    lap_predictions=False for the rule-based sources ("no change", "laps driven"): their changes
    set the wear but are not lap-time predictions, so no predicted lap times are stored."""
    laps = lap_table(driver_laps, known_changes if lap_predictions else None)
    by_number = {row["lap"]: row for row in laps}
    stints = []
    for _, g in driver_laps.sort_values("LapNumber").groupby("Stint", sort=True):
        stints.append({"stint": int(g["Stint"].iloc[0]), "compound": str(g["Compound"].iloc[0]),
                       "first_lap": int(g["LapNumber"].min()), "last_lap": int(g["LapNumber"].max())})
    plans = {p["stint"]: p for p in driver_strategy(driver_laps, known_changes, cost)}
    for s in stints:
        plan = plans.get(s["stint"])
        if plan is None:            # last stint, wet tyres, or no advice possible
            continue
        s.update({k: plan[k] for k in ("actual_stop", "horizon_end", "window", "best_lap",
                                       "sc_vsc_calls", "hit", "actual_under_sc_vsc", "random_hit_chance")})
        s["advice"] = plan["advice"]
        s["summary"] = stint_summary(plan, by_number)
    return {"model": model, "in_training_data": in_training_data, "pit_cost": cost,
            "laps": laps, "stints": stints}
