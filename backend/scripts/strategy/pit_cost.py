"""Time lost by a pit stop, measured from a race's own FastF1 laps (ROADMAP §C step 2).

Cost of one stop = the driver's in-lap + out-lap time minus the median of the same two laps for the
cars that did not pit on either lap. Each race uses the median of its own normal (green-flag)
stops; a stop under safety car / VSC costs SC_VSC_FACTOR of that; a red flag lets teams change tyres
for free, so red-flag laps get no advice.

The pit-lane loss is a fixed property of the track that teams know before the race, so using the
race's own stops to measure it does not leak anything a strategist would not have known.
"""

from typing import Optional

import numpy as np
import pandas as pd

# SC/VSC stops cost 0.69 x a normal stop (median ratio over the 21 races in 2023-24 that had both;
# SC/VSC medians per race are too noisy to use directly). User agreed to 0.7 (2026-11-01).
SC_VSC_FACTOR: float = 0.7
# Fewer measured stops than this and the race median is not trusted; DEFAULT_PIT_COST_S is used.
MIN_STOPS: int = 3
# Median normal stop over all 2023-24 races (1,170 stops). Only used when a race has < MIN_STOPS.
DEFAULT_PIT_COST_S: float = 22.8
# Cars that did not pit needed on both laps to form the comparison.
MIN_FIELD_CARS: int = 3

SC_VSC_CODES = ("4", "6", "7")   # FastF1 TrackStatus: safety car, VSC, VSC ending
RED_FLAG_CODE = "5"


def lap_kind(track_status: str) -> str:
    """'red', 'sc_vsc' or 'green' from the TrackStatus strings of the laps involved."""
    if RED_FLAG_CODE in track_status:
        return "red"
    if any(code in track_status for code in SC_VSC_CODES):
        return "sc_vsc"
    return "green"


def stop_costs(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per tyre stop in a race: Driver, LapNumber (the in-lap), kind, cost_s.

    A stop is an in-lap (PitInTime set) followed by the driver's next lap in a new Stint, so
    drive-through penalties and retirements in the pit lane are not counted. Lap-1 stops are left
    out (damage stops compared with standing-start laps; 27 in 2023-24, they moved the SC/VSC
    ratio from 0.69 to 0.88). Stops whose in- or
    out-lap has no lap time, or with fewer than MIN_FIELD_CARS cars to compare with, are skipped.
    """
    laps = laps.assign(t=laps["LapTime"].dt.total_seconds(),
                       status=laps["TrackStatus"].fillna("").astype(str))
    times = laps.pivot_table(index="LapNumber", columns="Driver", values="t")
    in_pits = laps["PitInTime"].notna() | laps["PitOutTime"].notna()
    by_driver_lap = laps.set_index(["Driver", "LapNumber"])

    rows = []
    for _, lap in laps[laps["PitInTime"].notna()].iterrows():
        driver, n = lap["Driver"], lap["LapNumber"]
        if n <= 1 or (driver, n + 1) not in by_driver_lap.index:
            continue
        out_lap = by_driver_lap.loc[(driver, n + 1)]
        if out_lap["Stint"] != lap["Stint"] + 1:
            continue
        mine = lap["t"] + out_lap["t"]
        if np.isnan(mine):
            continue
        pitted = set(laps.loc[laps["LapNumber"].isin([n, n + 1]) & in_pits, "Driver"])
        field = times.loc[[n, n + 1], [d for d in times.columns if d not in pitted]].dropna(axis=1)
        if field.shape[1] < MIN_FIELD_CARS:
            continue
        rows.append({"Driver": driver, "LapNumber": int(n),
                     "kind": lap_kind(lap["status"] + out_lap["status"]),
                     "cost_s": float(mine - field.sum().median())})
    return pd.DataFrame(rows, columns=["Driver", "LapNumber", "kind", "cost_s"])


def race_pit_cost(laps: pd.DataFrame) -> dict:
    """Pit cost for one race: normal stop, SC/VSC stop and how the normal value was found."""
    costs = stop_costs(laps)
    normal = costs.loc[costs["kind"] == "green", "cost_s"]
    if len(normal) >= MIN_STOPS:
        normal_s, source = float(normal.median()), "race median"
    else:
        normal_s, source = DEFAULT_PIT_COST_S, "default (too few normal stops)"
    return {"normal_s": normal_s, "sc_vsc_s": normal_s * SC_VSC_FACTOR,
            "n_normal_stops": int(len(normal)), "source": source}


def pit_cost_for_lap(cost: dict, track_status: str) -> Optional[float]:
    """Cost of pitting on a lap with this TrackStatus; None on a red-flag lap (no advice)."""
    kind = lap_kind(track_status)
    if kind == "red":
        return None
    return cost["sc_vsc_s"] if kind == "sc_vsc" else cost["normal_s"]
