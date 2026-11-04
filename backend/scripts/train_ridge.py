"""
================================================================================
  ApexHunter - Ridge tyre pace baseline (ROADMAP §C step 5)
  Script: train_ridge.py
--------------------------------------------------------------------------------
  Purpose : Train the Ridge regression baseline on the same windows, target and race split as the
            LSTM, score it on the held-out races against "no change" (next lap = last lap), and
            save it to models/tyre_ridge.pkl + tyre_ridge_config.json.

  Usage   : python backend/scripts/train_ridge.py [--seasons 2023 2024] [--test-split 0.1]
================================================================================
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from lstm.tyre_data import LAP_FEATURES, SEQUENCE_LENGTH, TARGET, build_training_dataset, race_files
from lstm.tyre_model import split_by_groups
from lstm.tyre_ridge import fit_ridge, flatten, regression_metrics, save_ridge, top_coefficients
from utils import DATA_LAKE_DIR, IST, PROJECT_ROOT, setup_logger

logger = setup_logger(__name__)


def run_ridge_training(sessions_dir: Path, models_dir: Path, seasons: List[str],
                       test_split: float, log: logging.Logger) -> dict:
    """Build the dataset, fit Ridge on the training races, evaluate, save. Returns the config."""
    t0 = time.time()
    X, y, groups, last_lap_time = build_training_dataset(sessions_dir, seasons, log)
    train_idx, test_idx = split_by_groups(len(X), groups, test_split)
    model, alpha, val_mse = fit_ridge(X[train_idx], y[train_idx], groups[train_idx])

    metrics = {}
    for part, idx in (("train", train_idx), ("test", test_idx)):
        metrics[part] = regression_metrics(y[idx], model.predict(flatten(X[idx])), last_lap_time[idx])
        metrics[f"{part}_no_change"] = regression_metrics(y[idx], np.zeros(len(idx)), last_lap_time[idx])

    config = {
        "model": "ridge", "alpha": alpha, "val_mse_by_alpha": {str(a): m for a, m in val_mse.items()},
        "features": LAP_FEATURES, "target": TARGET, "sequence_length": SEQUENCE_LENGTH,
        "metrics": metrics, "top_coefficients": top_coefficients(model),
        "seasons": seasons,
        "n_train_sessions": int(len(np.unique(groups[train_idx]))),
        "n_test_sessions": int(len(np.unique(groups[test_idx]))),
        "n_train_sequences": int(len(train_idx)), "n_test_sequences": int(len(test_idx)),
        # held-out races, so the strategy check (check_strategy.py) uses exactly these
        "test_races": [race_files(sessions_dir, seasons)[g].stem for g in np.unique(groups[test_idx])],
        "trained_at": datetime.now(IST).isoformat(),
    }
    save_ridge(model, config, models_dir)

    log.info(f"Ridge: alpha={alpha} (validation MSE by alpha: "
             + ", ".join(f"{a:g}={m:.4f}" for a, m in val_mse.items()) + ")")
    log.info(f"{'':12s}{'MAE s':>8s}{'RMSE s':>8s}{'R2':>8s}{'MAPE %':>8s}")
    for key, m in metrics.items():
        log.info(f"{key:12s}{m['mae_s']:8.3f}{m['rmse_s']:8.3f}{m['r2']:8.3f}{m['mape_pct']:8.3f}")
    log.info("Largest coefficients: " + "; ".join(f"{n} {c:+.3f}" for n, c in config["top_coefficients"]))
    log.info(f"Saved {models_dir / 'tyre_ridge.pkl'} in {time.time() - t0:.0f} s")
    return config


def main() -> None:
    p = argparse.ArgumentParser(description="Train the ApexHunter Ridge tyre pace baseline.")
    p.add_argument("--sessions-dir", type=str, default=None, help="Path to clean_data directory.")
    p.add_argument("--models-dir", type=str, default=None, help="Path to save model artifacts.")
    p.add_argument("--seasons", nargs="+", default=["2023", "2024"], help="Years to include.")
    p.add_argument("--test-split", type=float, default=0.1,
                   help="Fraction of (most recent) race sessions held out for test.")
    args = p.parse_args()
    sessions_dir = Path(args.sessions_dir) if args.sessions_dir else DATA_LAKE_DIR / "clean_data"
    models_dir = Path(args.models_dir) if args.models_dir else PROJECT_ROOT / "models"
    try:
        run_ridge_training(sessions_dir, models_dir, args.seasons, args.test_split, logger)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
