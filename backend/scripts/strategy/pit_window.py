"""Tyres-only pit window for past races (ROADMAP §C step 4).

At every lap k of a stint, every possible stop lap x of the stint is costed by the time lost to tyre
wear plus the pit stop, up to the end of the following stint (the driver's next real stop or the
finish, so the driver's real number of stops is kept). Fuel, track and car are the same on a given
lap whichever option is chosen, so they cancel: only tyre age and the lap of the stop differ.

    cost(x) = wear * (sum of old-tyre ages on laps start..x + new-tyre ages on laps x+1..end) + pit(x)

pit(k) uses lap k's track status (pit cost x 0.7 under SC/VSC, no advice on a red flag); every other
lap uses the normal cost, because future safety cars are not known. The best stop lap is chosen over
all stop laps of the stint (earlier ones too, so an overdue stop drops out), and lap k is in the
window when it is within WINDOW_HALF_LAPS of that best lap (user, 2026-11-04: a 5-lap window; the
earlier "within 1 s of the best total" rule spanned 9+ laps because wear is small).

Wear per lap of tyre age = mean of all lap-time changes known at lap k within the current stint
(the whole race so far until the stint has MIN_STINT_CHANGES) + the fuel gain (the changes mix tyre
wear with the car getting lighter). Averaging the whole stint, not the last 5 laps, cancels most
lap-to-lap noise (user, 2026-11-04). The changes can come from a model (predictions) or from the
laps already driven (observed_changes); keys are the first lap whose decision may use them, so
nothing from the future leaks in.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from lstm.tyre_data import SLICK_COMPOUNDS, representative_lap_mask, slick_lap_mask
from strategy.pit_cost import lap_kind, pit_cost_for_lap

# Lap-time gain per race lap from fuel burn + track rubbering in, fitted on the 40 training races
# (2023 R1 - 2024 R18, 37,883 dry laps, with driver-race base and compound wear terms): 0.051-0.056
# s/lap across lap filters (§C step 3).
FUEL_GAIN_S_PER_LAP: float = 0.053
WINDOW_HALF_LAPS: int = 2      # user decision 2026-11-04: window = best stop lap +-2 (5 laps)
MIN_STINT_CHANGES: int = 3     # fewer changes in the stint so far: average the whole race so far


def wear_rate(known_changes: Dict[int, float], k: int, stint_start: int = 1) -> Optional[float]:
    """Tyre wear (s per lap of age) usable at lap k, or None before any change is known: the mean
    of the changes known by lap k since stint_start (or of all known ones while the stint has fewer
    than MIN_STINT_CHANGES) + the fuel gain. Clipped at 0: tyres do not get faster with age."""
    known = [known_changes[j] for j in sorted(known_changes) if j <= k]
    in_stint = [known_changes[j] for j in sorted(known_changes) if stint_start <= j <= k]
    vals = in_stint if len(in_stint) >= MIN_STINT_CHANGES else known
    if not vals:
        return None
    return max(0.0, float(np.mean(vals)) + FUEL_GAIN_S_PER_LAP)


def option_costs(first_lap: int, age_first: float, horizon_end: int, wear: float, k: int,
                 pit_now_s: float, pit_other_s: float) -> Dict[int, float]:
    """Time lost (s) for pitting on each lap x = first_lap .. horizon_end-1 of the stint, using the
    wear known at lap k. Lap k costs pit_now_s (its own track status), every other lap pit_other_s.

    Earlier laps are included so that a stop that is already overdue is seen as worse than the best
    total, not as "the best option still left"."""
    costs = {}
    for x in range(first_lap, horizon_end):
        old = sum(age_first + (lap - first_lap) for lap in range(first_lap, x + 1))
        new = sum(lap - x for lap in range(x + 1, horizon_end + 1))
        costs[x] = wear * (old + new) + (pit_now_s if x == k else pit_other_s)
    return costs


def stint_advice(stint_laps: pd.DataFrame, horizon_end: int,
                 known_changes: Dict[int, float], cost: dict) -> List[dict]:
    """Advice for every lap of one stint (LapNumber, TyreLife, TrackStatus rows, in race order).

    Laps with no known wear yet or under a red flag get no row."""
    rows = []
    first_lap = int(stint_laps["LapNumber"].iloc[0])
    age_first = float(stint_laps["TyreLife"].iloc[0])
    for _, lap in stint_laps.iterrows():
        k = int(lap["LapNumber"])
        if k >= horizon_end:
            break
        status = str(lap["TrackStatus"]) if pd.notna(lap["TrackStatus"]) else ""
        pit_now = pit_cost_for_lap(cost, status)
        wear = wear_rate(known_changes, k, first_lap)
        if pit_now is None or wear is None:
            continue
        costs = option_costs(first_lap, age_first, horizon_end, wear, k, pit_now, cost["normal_s"])
        best = min(costs, key=costs.get)
        gap = costs[k] - costs[best]
        # No wear signal (user, 2026-11-03): with wear 0 every stop lap costs the same, so there is
        # no best lap and a green lap is NOT in the window (it used to be, giving "pit now" right
        # after a stop). A cheaper SC/VSC stop still beats every other lap, so it is the best lap.
        no_wear = wear == 0
        if no_wear and pit_now >= cost["normal_s"]:
            best = None
        rows.append({"lap": k, "wear_s": wear, "best_lap": best, "gap_s": gap,
                     "in_window": best is not None and abs(k - best) <= WINDOW_HALF_LAPS,
                     "no_wear_signal": no_wear, "sc_vsc": lap_kind(status) == "sc_vsc"})
    return rows


def observed_changes(driver_laps: pd.DataFrame) -> Dict[int, float]:
    """Lap-time changes already driven, keyed by the first lap whose decision may use them.

    The change into lap j (both j-1 and j representative dry laps) is complete when lap j ends, so
    it can inform the decision to pit on lap j+1. Used as a no-model baseline."""
    laps = driver_laps.sort_values("LapNumber")
    keep = representative_lap_mask(laps) & slick_lap_mask(laps)
    t = dict(zip(laps.loc[keep, "LapNumber"].astype(int), laps.loc[keep, "LapTime"].dt.total_seconds()))
    return {j + 1: t[j] - t[j - 1] for j in t if j - 1 in t}


def driver_strategy(driver_laps: pd.DataFrame, known_changes: Dict[int, float], cost: dict) -> List[dict]:
    """Pit window for every stint of one driver that ends in a real stop between two dry stints.

    Returns per stint: stint, actual_stop (in-lap), horizon_end, advice (stint_advice rows),
    window (laps in the window), best_lap (normal best stop lap, announced on the first green
    window lap; SC/VSC calls are reported separately),
    sc_vsc_calls (SC/VSC laps where pitting now was in the window), hit (actual stop in window),
    actual_under_sc_vsc and random_hit_chance (window size / laps with advice). Advice covers every
    lap from the stint's start to the end of the following stint (see the stayed-out laps below).
    """
    laps = driver_laps.sort_values("LapNumber")
    stints = [g for _, g in laps.groupby("Stint", sort=True)]
    out = []
    for cur, nxt in zip(stints, stints[1:]):
        compounds = {str(c).upper() for c in pd.concat([cur["Compound"], nxt["Compound"]])}
        if not compounds <= set(SLICK_COMPOUNDS):
            continue
        actual = int(cur["LapNumber"].max())
        horizon_end = int(nxt["LapNumber"].max())
        # Advice also on the laps after the real stop, as if the driver had stayed out: the old
        # tyres keep ageing, the track status is that lap's, and the wear stays at its last value
        # before the stop (nothing more is learned about the old tyres). Without these laps the
        # window would end at the real stop and always look like it found it.
        stayed_out = nxt[nxt["LapNumber"] < horizon_end][["LapNumber", "TrackStatus"]].assign(
            TyreLife=lambda d: float(cur["TyreLife"].iloc[-1]) + d["LapNumber"] - actual)
        lap_rows = pd.concat([cur[["LapNumber", "TyreLife", "TrackStatus"]], stayed_out])
        known = {j: v for j, v in known_changes.items() if j <= actual}
        advice = stint_advice(lap_rows, horizon_end, known, cost)
        if not advice:
            continue
        window = [a["lap"] for a in advice if a["in_window"]]
        stop_row = cur[cur["LapNumber"] == actual].iloc[0]
        out.append({
            "stint": int(cur["Stint"].iloc[0]), "actual_stop": actual, "horizon_end": horizon_end,
            "advice": advice, "window": window,
            "best_lap": next((a["best_lap"] for a in advice
                              if a["in_window"] and not a["sc_vsc"] and a["best_lap"] is not None), None),
            "sc_vsc_calls": [a["lap"] for a in advice if a["in_window"] and a["sc_vsc"]],
            "hit": actual in window,
            "actual_under_sc_vsc": lap_kind(str(stop_row["TrackStatus"])) == "sc_vsc",
            "random_hit_chance": len(window) / len(advice),
        })
    return out
