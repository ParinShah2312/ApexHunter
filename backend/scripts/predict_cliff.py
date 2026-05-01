"""Prediction orchestrator for the ApexHunter LSTM tyre cliff predictor.
Loads a trained model and predicts cliff laps for a session."""

import argparse
import gc
import sys
from pathlib import Path

from typing import Any

import numpy as np
import pandas as pd

from utils import setup_logger, DATA_LAKE_DIR, PROJECT_ROOT
from tyre_data import extract_stints, LAP_FEATURES
from tyre_model import INPUT_SIZE, monte_carlo_predict, predict_single_stint
from tyre_io import load_model_artifacts, build_prediction_output, save_prediction, log_prediction_complete

logger = setup_logger(__name__)


def main() -> None:
    """Parse CLI arguments and run the prediction pipeline."""
    p = argparse.ArgumentParser(description="Predict tyre cliff laps for a session.")
    p.add_argument("--session", type=str, required=True, help="Path to a clean parquet file.")
    p.add_argument("--driver", type=str, required=True, help="Driver code string.")
    p.add_argument("--models-dir", type=str, default=None, help="Path to model artifacts.")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory for predictions.")
    p.add_argument("--force", action="store_true", default=False, help="Overwrite existing output.")
    args = p.parse_args()

    models_dir = Path(args.models_dir) if args.models_dir else PROJECT_ROOT / "models"
    output_dir = Path(args.output_dir) if args.output_dir else DATA_LAKE_DIR / "tyre_predictions"
    stem = Path(args.session).stem
    output_path = output_dir / f"{stem}_{args.driver}_tyre.json"

    if output_path.exists() and not args.force:
        logger.info("Already processed. Use --force to re-run.")
        return

    try:
        model, scaler, config = load_model_artifacts(models_dir, logger)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    y_mean, y_std = config["y_mean"], config["y_std"]
    seq_len = config["sequence_length"]

    try:
        parts = stem.split("_")
        year, round_num, session_type = int(parts[0]), int(parts[1]), parts[2]
        import fastf1
        cache_dir = PROJECT_ROOT / "data_lake" / "cache"
        cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        session_f1 = fastf1.get_session(year, round_num, session_type)
        session_f1.load(telemetry=False, weather=False)
        f1_laps = session_f1.laps
    except Exception as e:
        logger.error(f"Failed to load fastf1 laps: {e}")
        sys.exit(1)

    try:
        df = pd.read_parquet(args.session)
        df_driver = df[df["Driver"] == args.driver].copy()
        if df_driver.empty:
            raise ValueError(f"No data for driver '{args.driver}' in {args.session}")
        df_driver.sort_values("SessionTime", inplace=True)
        df_driver.reset_index(drop=True, inplace=True)
        driver_laps = f1_laps.pick_drivers(args.driver)
        if len(driver_laps) == 0:
            raise ValueError(f"No fastf1 laps for driver '{args.driver}'")
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)

    stints_data = extract_stints(driver_laps, df_driver, logger)
    if len(stints_data) == 0:
        logger.error("No stints detected.")
        sys.exit(1)

    stint_results = []
    for stint_data in stints_data:
        res = predict_single_stint(stint_data, model, scaler, y_mean, y_std, seq_len)
        stint_results.append(res)

    data = build_prediction_output(str(args.session), args.driver, stint_results)
    save_prediction(data, output_path, logger)
    gc.collect()

    log_prediction_complete(args.driver, stem, stint_results, output_path, logger)


if __name__ == "__main__":
    main()
