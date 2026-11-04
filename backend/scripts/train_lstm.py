"""
================================================================================
  ApexHunter - LSTM / vanilla RNN tyre pace trainer (ROADMAP §D)
  Script: train_lstm.py
--------------------------------------------------------------------------------
  Purpose : Train the LSTM or the vanilla RNN on the same windows, target and race split as the
            Ridge baseline; 3 seeds; metrics vs "no change"; saves models/tyre_{lstm,rnn}*.

  Usage   : python backend/scripts/train_lstm.py [--cell lstm|rnn] [--seasons 2024] [--force]
================================================================================
"""

import argparse
import gc
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from lstm.tyre_data import LAP_FEATURES, SEQUENCE_LENGTH, TARGET, build_training_dataset, race_files
from lstm.tyre_io import (
    artifact_names, log_dataset_stats, log_training_complete, log_training_start, save_model_artifacts,
)
from lstm.tyre_model import (
    CELLS, DROPOUT_RATE, INPUT_SIZE, N_EPOCHS_FINAL, N_EPOCHS_SEARCH, NUM_LAYERS,
    run_hyperparameter_search, sequence_predictor, split_by_groups, train_final_model,
)
from lstm.tyre_ridge import regression_metrics
from utils import DATA_LAKE_DIR, IST, PROJECT_ROOT, setup_logger

logger = setup_logger(__name__)

MIN_SEQUENCES: int = 50
SEEDS: Tuple[int, ...] = (42, 43, 44)
"""Final model trained once per seed; results reported as mean +- spread (user: the proper run)."""


def fit_scalers(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[StandardScaler, float, float]:
    """Fit the feature scaler and target normalisation on TRAINING data only.

    Fitting on the full dataset would leak test-set statistics into training.

    Returns:
        Tuple of (feature_scaler, y_mean, y_std).
    """
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, INPUT_SIZE))
    return scaler, float(y_train.mean()), float(y_train.std())


def apply_scalers(
    X: np.ndarray, y: np.ndarray, scaler: StandardScaler, y_mean: float, y_std: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale features with a fitted scaler and z-normalise targets."""
    X_scaled = scaler.transform(X.reshape(-1, INPUT_SIZE)).reshape(X.shape)
    y_scaled = (y - y_mean) / (y_std + 1e-8)
    return X_scaled, y_scaled


def run_training(
    sessions_dir: Path,
    models_dir: Path,
    seasons: List[str],
    test_split: float,
    log: logging.Logger,
    grid: Optional[List[dict]] = None,
    n_epochs_search: int = N_EPOCHS_SEARCH,
    n_epochs_final: int = N_EPOCHS_FINAL,
    cell: str = "lstm",
    seeds: Sequence[int] = SEEDS,
) -> dict:
    """Build the dataset, tune, train, evaluate and save the LSTM or vanilla RNN. Returns the config.

    The most recent `test_split` fraction of race sessions is held out for testing; scalers are
    fitted on the remaining (training) sessions only. The settings are searched once; the final
    model is then trained once per seed and the test metrics are reported per seed and as mean and
    spread, so a lucky start cannot pass for a result. The seed with the best validation MSE is saved.

    Raises:
        ValueError: If no data is found or there are too few sequences to train on.
    """
    pipeline_t0 = time.time()
    models_dir.mkdir(parents=True, exist_ok=True)

    log.info("")
    log.info("=" * 56)
    log.info(f"  PHASE 1 / 4 -- DATASET CONSTRUCTION ({cell.upper()})")
    log.info("=" * 56)
    data_t0 = time.time()
    X, y, groups, last_lap_time = build_training_dataset(sessions_dir, seasons, log)
    data_time = time.time() - data_t0
    if len(X) < MIN_SEQUENCES:
        raise ValueError(f"Not enough sequences for training (got {len(X)}, need >= {MIN_SEQUENCES}).")
    log_dataset_stats(X, y, data_time, log, SEQUENCE_LENGTH, INPUT_SIZE)

    # Chronological split by whole sessions: the latest sessions form the test set
    train_idx, test_idx = split_by_groups(len(X), groups, test_split)
    groups_train = groups[train_idx]
    log.info(f"  Train sessions   : {len(np.unique(groups_train))}  ({len(train_idx):,} sequences)")
    log.info(f"  Test sessions    : {len(np.unique(groups[test_idx]))}  ({len(test_idx):,} sequences)")

    # Fit scalers on training data only (no test-set leakage)
    scaler, y_mean, y_std = fit_scalers(X[train_idx], y[train_idx])
    X_scaled, y_scaled = apply_scalers(X, y, scaler, y_mean, y_std)
    X_train, y_train = X_scaled[train_idx], y_scaled[train_idx]
    log.info(f"  Target norm      : mean={y_mean:.4f}, std={y_std:.4f} (train only)")

    best_hidden, best_lr, search_val_mse = run_hyperparameter_search(
        X_train, y_train, log, groups=groups_train, grid=grid, n_epochs=n_epochs_search, cell=cell,
    )

    # Final model once per seed; metrics on the real scale (s), vs "no change" (next lap = last lap)
    runs = []
    for seed in seeds:
        model, val_mse = train_final_model(X_train, y_train, best_hidden, best_lr, log,
                                           groups=groups_train, n_epochs=n_epochs_final,
                                           cell=cell, seed=seed)
        predict = sequence_predictor(model, scaler, y_mean, y_std)
        metrics = {part: regression_metrics(y[idx], predict(X[idx]), last_lap_time[idx])
                   for part, idx in (("train", train_idx), ("test", test_idx))}
        runs.append({"seed": int(seed), "val_mse": float(val_mse), "metrics": metrics, "model": model})
        log.info(f"[Seed {seed}] val_MSE {val_mse:.4f} | test MAE {metrics['test']['mae_s']:.3f} s, "
                 f"RMSE {metrics['test']['rmse_s']:.3f} s, R2 {metrics['test']['r2']:.3f}")
        gc.collect()
    best_run = min(runs, key=lambda r: r["val_mse"])
    metrics = dict(best_run["metrics"])
    for part, idx in (("train", train_idx), ("test", test_idx)):
        metrics[f"{part}_no_change"] = regression_metrics(y[idx], np.zeros(len(idx)), last_lap_time[idx])
    test_mae = [r["metrics"]["test"]["mae_s"] for r in runs]
    test_r2 = [r["metrics"]["test"]["r2"] for r in runs]

    files = race_files(sessions_dir, seasons)
    config = {
        "cell": cell,
        "hidden_size": best_hidden,
        "learning_rate": best_lr,
        "input_size": INPUT_SIZE,
        "features": LAP_FEATURES,
        "target": TARGET,
        "sequence_length": SEQUENCE_LENGTH,
        "num_layers": NUM_LAYERS,
        "dropout_rate": DROPOUT_RATE,
        "y_mean": y_mean,
        "y_std": y_std,
        "search_val_mse": search_val_mse,
        "val_mse": best_run["val_mse"],          # #46: the saved (final) model's own score
        "seed": best_run["seed"],
        "metrics": metrics,                      # saved model + "no change", like the Ridge config
        "seed_runs": [{k: r[k] for k in ("seed", "val_mse", "metrics")} for r in runs],
        "test_mae_s_mean": float(np.mean(test_mae)), "test_mae_s_std": float(np.std(test_mae)),
        "test_r2_mean": float(np.mean(test_r2)), "test_r2_std": float(np.std(test_r2)),
        "seasons": seasons,
        "test_races": [files[g].stem for g in np.unique(groups[test_idx])],
        "n_train_sessions": int(len(np.unique(groups_train))),
        "n_test_sessions": int(len(np.unique(groups[test_idx]))),
        "n_train_sequences": int(len(train_idx)),
        "n_test_sequences": int(len(test_idx)),
        "trained_at": datetime.now(IST).isoformat(),
    }
    save_model_artifacts(best_run["model"], scaler, config, models_dir, log, cell=cell)
    gc.collect()

    total_time = time.time() - pipeline_t0
    log_training_complete(len(X), best_hidden, best_lr, best_run["val_mse"], metrics["test"]["mae_s"],
                          total_time, models_dir, log, cell=cell)
    log.info(f"  {cell.upper()} test over {len(runs)} seeds: MAE {np.mean(test_mae):.3f} +- {np.std(test_mae):.3f} s, "
             f"R2 {np.mean(test_r2):.3f} +- {np.std(test_r2):.3f} | no change MAE "
             f"{metrics['test_no_change']['mae_s']:.3f} s")
    return config


def main() -> None:
    """Parse CLI arguments and run the full training pipeline."""
    p = argparse.ArgumentParser(description="Train the ApexHunter LSTM or vanilla RNN tyre pace model.")
    p.add_argument("--cell", type=str, default="lstm", choices=CELLS, help="lstm or rnn (vanilla RNN).")
    p.add_argument("--sessions-dir", type=str, default=None, help="Path to clean_data directory.")
    p.add_argument("--models-dir", type=str, default=None, help="Path to save model artifacts.")
    p.add_argument("--seasons", nargs="+", default=["2023", "2024"], help="Years to include.")
    p.add_argument("--test-split", type=float, default=0.1,
                   help="Fraction of (most recent) race sessions held out for test.")
    p.add_argument("--force", action="store_true", default=False, help="Re-train even if model exists.")
    args = p.parse_args()

    sessions_dir = Path(args.sessions_dir) if args.sessions_dir else DATA_LAKE_DIR / "clean_data"
    models_dir = Path(args.models_dir) if args.models_dir else PROJECT_ROOT / "models"

    log_training_start(sessions_dir, models_dir, args.seasons, args.test_split, args.force, logger)

    if (models_dir / artifact_names(args.cell)[0]).exists() and not args.force:
        logger.info("Model already exists. Use --force to re-train.")
        return

    try:
        run_training(sessions_dir, models_dir, args.seasons, args.test_split, logger, cell=args.cell)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
