"""
================================================================================
  ApexHunter - Pit strategy output (ROADMAP §C step 7)
  Script: predict_strategy.py
--------------------------------------------------------------------------------
  Purpose : For one driver in one race: predict each lap with the tyre pace model, work out the
            pit window and safety-car calls, compare with the real stops, and write
            data_lake/tyre_predictions/{session}_{driver}_strategy.json for the dashboard.

  Usage   : python backend/scripts/predict_strategy.py --session data_lake/clean_data/2024_22_R.parquet --driver 1 [--source all]
================================================================================
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from lstm.tyre_data import extract_stints
from lstm.tyre_io import load_model_artifacts
from lstm.tyre_model import CELLS, sequence_predictor
from lstm.tyre_ridge import flatten, load_ridge
from strategy.pit_cost import race_pit_cost
from strategy.pit_window import observed_changes
from strategy.strategy_check import predicted_changes
from strategy.strategy_output import build_strategy_output
from utils import DATA_LAKE_DIR, IST, PROJECT_ROOT, enable_fastf1_cache, setup_logger

logger = setup_logger(__name__)
# Wear sources. "best" = the one with the largest margin over a random window in the last
# check_strategy.py run (user, 2026-11-04: show whichever is best). "all" = the best one on top
# plus every source under "sources", for the dashboard's model switch.
SOURCES = ("best", "all", "ridge", "rnn", "lstm", "no_change", "laps_driven")
EACH_SOURCE = ("ridge", "rnn", "lstm", "no_change", "laps_driven")
RULE_BASED = ("no_change", "laps_driven")
CHECK_FILE = DATA_LAKE_DIR / "tyre_predictions" / "strategy_check.json"


def _predictor(source: str, models_dir: Path):
    """(predict function, model config) for a model source; (None, {}) for the rule-based ones."""
    if source == "ridge":
        ridge, config = load_ridge(models_dir)
        return (lambda w: ridge.predict(flatten(w))), config
    if source in CELLS:
        model, scaler, config = load_model_artifacts(models_dir, logger, cell=source)
        return sequence_predictor(model, scaler, config["y_mean"], config["y_std"]), config
    return None, {}


def run_prediction(session_path: Path, driver: str, models_dir: Path, source: str = "best",
                   check_file: Path = CHECK_FILE) -> dict:
    """Build the strategy output for one driver (driver number as in the clean parquet)."""
    import fastf1

    if source not in SOURCES:
        raise ValueError(f"Unknown source '{source}' (available: {', '.join(SOURCES)})")
    scores, best = None, None
    if check_file.exists():
        with open(check_file, "r", encoding="utf-8") as f:
            check = json.load(f)
        scores, best = check["summary"], check["best_source"]
    if source in ("best", "all"):
        if best is None:
            raise FileNotFoundError(f"{check_file} not found: run check_strategy.py first, or pass --source")
    wanted = EACH_SOURCE if source == "all" else (best if source == "best" else source,)

    stem = session_path.stem
    year, rnd, kind = stem.split("_")
    enable_fastf1_cache()
    session = fastf1.get_session(int(year), int(rnd), kind)
    session.load(telemetry=False, weather=False, messages=False)
    laps = session.laps

    telemetry = pd.read_parquet(session_path)
    tel = telemetry[telemetry["Driver"] == driver].sort_values("SessionTime").reset_index(drop=True)
    driver_laps = laps.pick_drivers(driver)
    if tel.empty or len(driver_laps) == 0:
        raise ValueError(f"No data for driver '{driver}' in {session_path.name}")
    cost = race_pit_cost(laps)
    stints = None

    def build(src: str) -> dict:
        nonlocal stints
        predict, config = _predictor(src, models_dir)
        if src == "no_change":
            known = {lap: 0.0 for lap in range(1, int(laps["LapNumber"].max()) + 2)}
        elif src == "laps_driven":
            known = observed_changes(driver_laps)
        else:
            if stints is None:
                stints = extract_stints(driver_laps, tel, logger)
            known = predicted_changes(stints, predict)
        return build_strategy_output(
            driver_laps, known, cost, src,
            in_training_data=None if src in RULE_BASED else stem not in config.get("test_races", []),
            lap_predictions=src not in RULE_BASED)

    outputs = {src: build(src) for src in wanted}
    out = dict(outputs[best if source == "all" else wanted[0]])
    if source == "all":
        out["sources"] = {src: {k: o[k] for k in ("model", "in_training_data", "laps", "stints")}
                          for src, o in outputs.items()}
    out.update({"strategy_scores": scores, "best_source": best})
    out.update({"session_file": str(session_path), "driver": driver,
                "driver_code": str(driver_laps["Driver"].iloc[0]),
                "timestamp": datetime.now(IST).isoformat()})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Pit strategy output for one driver in one race.")
    p.add_argument("--session", type=str, required=True, help="Path to a clean race parquet file.")
    p.add_argument("--driver", type=str, required=True, help="Driver number, e.g. 1.")
    p.add_argument("--source", type=str, default="best", choices=SOURCES,
                   help="Wear source; 'best' = best in the last check_strategy.py run; 'all' = best "
                        "on top plus every source (for the dashboard's model switch).")
    p.add_argument("--models-dir", type=str, default=None, help="Folder with the saved model.")
    p.add_argument("--output-dir", type=str, default=None, help="Output folder.")
    p.add_argument("--force", action="store_true", default=False, help="Overwrite existing output.")
    args = p.parse_args()

    models_dir = Path(args.models_dir) if args.models_dir else PROJECT_ROOT / "models"
    output_dir = Path(args.output_dir) if args.output_dir else DATA_LAKE_DIR / "tyre_predictions"
    session_path = Path(args.session)
    output_path = output_dir / f"{session_path.stem}_{args.driver}_strategy.json"
    if output_path.exists() and not args.force:
        logger.info("Already processed. Use --force to re-run.")
        return
    try:
        out = run_prediction(session_path, args.driver, models_dir, args.source)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    tmp.replace(output_path)

    trained = {True: ", race was in the training data", False: ", held-out test race", None: ""}
    logger.info(f"{out['driver_code']} {session_path.stem} (wear from {out['model']}"
                f"{trained[out['in_training_data']]})")
    for s in out["stints"]:
        logger.info(f"  Stint {s['stint']} ({s['compound']}, laps {s['first_lap']}-{s['last_lap']})")
        for line in s.get("summary", ["  no pit advice (last stint or no data)"]):
            logger.info(f"    {line}")
    logger.info(f"Saved {output_path}")


if __name__ == "__main__":
    main()
