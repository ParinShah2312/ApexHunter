"""
================================================================================
  ApexHunter - LSTM Tyre Cliff Trainer
  Script: train_lstm.py
--------------------------------------------------------------------------------
  Purpose : Training orchestrator for the ApexHunter LSTM tyre cliff predictor.

  Usage   : python backend/scripts/train_lstm.py [--seasons 2024] [--force]
================================================================================
"""

import argparse
import gc
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from lstm.tyre_data import LAP_FEATURES, SEQUENCE_LENGTH, build_training_dataset
from lstm.tyre_io import log_dataset_stats, log_training_complete, log_training_start, save_model_artifacts
from lstm.tyre_model import (
    DROPOUT_RATE, INPUT_SIZE, N_EPOCHS_FINAL, N_EPOCHS_SEARCH, NUM_LAYERS,
    evaluate_on_test, run_hyperparameter_search, split_by_groups, train_final_model,
)
from utils import DATA_LAKE_DIR, IST, PROJECT_ROOT, setup_logger

logger = setup_logger(__name__)

MIN_SEQUENCES: int = 50


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
) -> dict:
    """Build the dataset, tune, train, evaluate and save the LSTM. Returns the saved config.

    The most recent `test_split` fraction of race sessions is held out for testing; scalers
    are fitted on the remaining (training) sessions only.

    Raises:
        ValueError: If no data is found or there are too few sequences to train on.
    """
    pipeline_t0 = time.time()
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── PHASE 1: Dataset Construction ─────────────────────────────────────
    log.info("")
    log.info("=" * 56)
    log.info("  PHASE 1 / 4 -- DATASET CONSTRUCTION")
    log.info("=" * 56)

    data_t0 = time.time()
    X, y, groups = build_training_dataset(sessions_dir, seasons, log)
    data_time = time.time() - data_t0

    if len(X) < MIN_SEQUENCES:
        raise ValueError(f"Not enough sequences for training (got {len(X)}, need >= {MIN_SEQUENCES}).")

    log_dataset_stats(X, y, data_time, log, SEQUENCE_LENGTH, INPUT_SIZE)

    # Chronological split by whole sessions: the latest sessions form the test set
    train_idx, test_idx = split_by_groups(len(X), groups, test_split)
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train_raw, y_test_raw = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    log.info(f"  Train sessions   : {len(np.unique(groups_train))}  ({len(X_train_raw):,} sequences)")
    log.info(f"  Test sessions    : {len(np.unique(groups[test_idx]))}  ({len(X_test_raw):,} sequences)")

    # Fit scalers on training data only (no test-set leakage)
    scaler, y_mean, y_std = fit_scalers(X_train_raw, y_train_raw)
    X_train, y_train = apply_scalers(X_train_raw, y_train_raw, scaler, y_mean, y_std)
    X_test, y_test = apply_scalers(X_test_raw, y_test_raw, scaler, y_mean, y_std)
    log.info(f"  Target norm      : mean={y_mean:.4f}, std={y_std:.4f} (train only)")
    log.info("=" * 56)

    # ── PHASE 2: Hyperparameter Search (logged inside tyre_model) ────────
    best_hidden, best_lr, best_val_mse = run_hyperparameter_search(
        X_train, y_train, log, groups=groups_train, grid=grid, n_epochs=n_epochs_search,
    )

    # ── PHASE 3: Final Model Training (logged inside tyre_model) ─────────
    final_model = train_final_model(
        X_train, y_train, best_hidden, best_lr, log, groups=groups_train, n_epochs=n_epochs_final,
    )
    gc.collect()

    # ── PHASE 4: Evaluation (logged inside tyre_model) ───────────────────
    mae = evaluate_on_test(final_model, X_test, y_test, y_mean, y_std)

    # ── Save Artifacts ────────────────────────────────────────────────────
    config = {
        "hidden_size": best_hidden,
        "learning_rate": best_lr,
        "input_size": INPUT_SIZE,
        "features": LAP_FEATURES,
        "sequence_length": SEQUENCE_LENGTH,
        "num_layers": NUM_LAYERS,
        "dropout_rate": DROPOUT_RATE,
        "y_mean": y_mean,
        "y_std": y_std,
        "val_mse": best_val_mse,
        "test_mae_kmh": mae,
        "seasons": seasons,
        "n_train_sessions": int(len(np.unique(groups_train))),
        "n_test_sessions": int(len(np.unique(groups[test_idx]))),
        "n_train_sequences": int(len(X_train)),
        "n_test_sequences": int(len(X_test)),
        "trained_at": datetime.now(IST).isoformat(),
    }
    save_model_artifacts(final_model, scaler, config, models_dir, log)
    gc.collect()

    total_time = time.time() - pipeline_t0
    log_training_complete(len(X), best_hidden, best_lr, best_val_mse, mae, total_time, models_dir, log)
    return config


def main() -> None:
    """Parse CLI arguments and run the full training pipeline."""
    p = argparse.ArgumentParser(description="Train the ApexHunter LSTM tyre cliff predictor.")
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

    if (models_dir / "tyre_lstm.pt").exists() and not args.force:
        logger.info("Model already exists. Use --force to re-train.")
        return

    try:
        run_training(sessions_dir, models_dir, args.seasons, args.test_split, logger)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
