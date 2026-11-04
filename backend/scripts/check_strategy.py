"""
================================================================================
  ApexHunter - Pit strategy check (ROADMAP §C step 6)
  Script: check_strategy.py
--------------------------------------------------------------------------------
  Purpose : Run every saved tyre model (Ridge, and the RNN and LSTM when trained) through the pit
            window on the held-out test races and score them against "no change" (next lap = last
            lap), "laps driven" (wear from the real lap times so far) and a random window of the
            same size: share of real stops inside the window and real SC/VSC stops called. Records
            the best source. Writes data_lake/tyre_predictions/strategy_check.json.

  Usage   : python backend/scripts/check_strategy.py
================================================================================
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from lstm.tyre_data import extract_stints
from lstm.tyre_io import load_model_artifacts
from lstm.tyre_model import CELLS, sequence_predictor
from lstm.tyre_ridge import flatten, load_ridge
from strategy.pit_cost import race_pit_cost
from strategy.pit_window import driver_strategy, observed_changes
from strategy.strategy_check import best_source, predicted_changes, stint_rows, summarise
from utils import DATA_LAKE_DIR, PROJECT_ROOT, enable_fastf1_cache, setup_logger

logger = setup_logger(__name__)


Predictor = Callable[[np.ndarray], np.ndarray]


def load_predictors(models_dir: Path, log: logging.Logger) -> Tuple[Dict[str, Predictor], List[str]]:
    """Every trained tyre model in models_dir as {name: windows -> lap-time change (s)}, and the
    test races (Ridge's; the RNN/LSTM use the same split and must report the same races)."""
    ridge, config = load_ridge(models_dir)
    predictors: Dict[str, Predictor] = {"ridge": lambda w: ridge.predict(flatten(w))}
    for cell in CELLS:
        try:
            model, scaler, cfg = load_model_artifacts(models_dir, log, cell=cell)
        except (FileNotFoundError, ValueError) as e:
            log.info(f"{cell}: not scored ({str(e).splitlines()[0][:90]})")
            continue
        if cfg.get("test_races") != config["test_races"]:
            raise ValueError(f"{cell} was tested on {cfg.get('test_races')}, Ridge on {config['test_races']}")
        predictors[cell] = sequence_predictor(model, scaler, cfg["y_mean"], cfg["y_std"])
    return predictors, config["test_races"]


def check_races(sessions_dir: Path, races: List[str], predictors: Dict[str, Predictor],
                log: logging.Logger) -> pd.DataFrame:
    """Stint rows (strategy_check.stint_rows) for every driver of the given races and every source:
    each model in predictors, plus the rule-based "no change" and "laps driven"."""
    import fastf1

    enable_fastf1_cache()
    rows = []
    for race in races:
        year, rnd, kind = race.split("_")
        session = fastf1.get_session(int(year), int(rnd), kind)
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps
        cost = race_pit_cost(laps)
        telemetry = pd.read_parquet(sessions_dir / f"{race}.parquet")
        for driver in telemetry["Driver"].unique():
            driver_laps = laps.pick_drivers(driver)
            if len(driver_laps) == 0:
                continue
            tel = telemetry[telemetry["Driver"] == driver].sort_values("SessionTime").reset_index(drop=True)
            stints = extract_stints(driver_laps, tel, log)
            sources = {name: predicted_changes(stints, predict) for name, predict in predictors.items()}
            sources.update({
                "no_change": {lap: 0.0 for lap in range(1, int(laps["LapNumber"].max()) + 2)},
                # no-model baseline: the same stint average, of the lap-time changes really driven
                "laps_driven": observed_changes(driver_laps),
            })
            for source, known in sources.items():
                plans = driver_strategy(driver_laps, known, cost)
                rows += stint_rows(source, race, str(driver_laps["Driver"].iloc[0]), plans)
        log.info(f"{race}: pit cost {cost['normal_s']:.1f} s ({cost['source']})")
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Score the tyre models' pit windows on the test races.")
    p.add_argument("--sessions-dir", type=str, default=None, help="Path to clean_data directory.")
    p.add_argument("--models-dir", type=str, default=None, help="Folder with the saved tyre models.")
    args = p.parse_args()
    sessions_dir = Path(args.sessions_dir) if args.sessions_dir else DATA_LAKE_DIR / "clean_data"
    models_dir = Path(args.models_dir) if args.models_dir else PROJECT_ROOT / "models"

    predictors, races = load_predictors(models_dir, logger)
    rows = check_races(sessions_dir, races, predictors, logger)
    summary = summarise(rows)

    logger.info(f"Test races: {', '.join(races)}")
    logger.info(f"{'source':10s}{'stints':>8s}{'hit %':>8s}{'random %':>10s}{'window':>8s}"
                f"{'SC stops':>10s}{'called %':>10s}{'SC laps called %':>18s}")
    for source, s in summary.items():
        logger.info(f"{source:10s}{s['stints']:8d}{s['hit_pct']:8.1f}{s['random_pct']:10.1f}"
                    f"{s['median_window_laps']:8.0f}{s['sc_vsc_stops']:10d}{s['sc_stops_called_pct']:10.1f}"
                    f"{s['sc_laps_called_pct']:18.1f}  (of {s['shared_sc_laps']} shared SC/VSC laps)")
    out = DATA_LAKE_DIR / "tyre_predictions" / "strategy_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"test_races": races, "summary": summary, "best_source": best_source(summary),
                   "stints": rows.to_dict(orient="records")}, f, indent=2)
    logger.info(f"Best source (margin over a random window): {best_source(summary)}; saved {out}")


if __name__ == "__main__":
    main()
