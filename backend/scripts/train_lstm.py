"""Training orchestrator for the ApexHunter LSTM tyre cliff predictor.
Builds dataset, runs hyperparameter search, trains final model."""

import argparse
import gc
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import setup_logger, DATA_LAKE_DIR, PROJECT_ROOT
from tyre_data import build_training_dataset, SEQUENCE_LENGTH
from tyre_model import (
    INPUT_SIZE, NUM_LAYERS, DROPOUT_RATE,
    run_hyperparameter_search, train_final_model, evaluate_on_test,
)
from tyre_io import save_model_artifacts

logger = setup_logger(__name__)


def main() -> None:
    """Parse CLI arguments and run the full training pipeline."""
    pipeline_t0 = time.time()

    p = argparse.ArgumentParser(description="Train the ApexHunter LSTM tyre cliff predictor.")
    p.add_argument("--sessions-dir", type=str, default=None, help="Path to clean_data directory.")
    p.add_argument("--models-dir", type=str, default=None, help="Path to save model artifacts.")
    p.add_argument("--seasons", nargs="+", default=["2023", "2024"], help="Years to include.")
    p.add_argument("--test-split", type=float, default=0.1, help="Fraction held out for test.")
    p.add_argument("--force", action="store_true", default=False, help="Re-train even if model exists.")
    args = p.parse_args()

    sessions_dir = Path(args.sessions_dir) if args.sessions_dir else DATA_LAKE_DIR / "clean_data"
    models_dir = Path(args.models_dir) if args.models_dir else PROJECT_ROOT / "models"
    test_split = args.test_split

    logger.info("")
    logger.info("#" * 56)
    logger.info("#" + " " * 54 + "#")
    logger.info("#    ApexHunter -- LSTM Tyre Cliff Predictor" + " " * 10 + "#")
    logger.info("#" + " " * 54 + "#")
    logger.info("#" * 56)
    logger.info(f"  Sessions dir   : {sessions_dir}")
    logger.info(f"  Models dir     : {models_dir}")
    logger.info(f"  Seasons        : {args.seasons}")
    logger.info(f"  Test split     : {test_split}")
    logger.info(f"  Force retrain  : {args.force}")
    logger.info(f"  Started at     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("#" * 56)

    model_path = models_dir / "tyre_lstm.pt"
    if model_path.exists() and not args.force:
        logger.info("Model already exists. Use --force to re-train.")
        return

    # ── PHASE 1: Dataset Construction ─────────────────────────────────────
    logger.info("")
    logger.info("=" * 56)
    logger.info("  PHASE 1 / 4 -- DATASET CONSTRUCTION")
    logger.info("=" * 56)

    data_t0 = time.time()
    try:
        X, y = build_training_dataset(sessions_dir, args.seasons, logger)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    data_time = time.time() - data_t0

    if len(X) < 50:
        logger.error("Not enough sequences for training (need >= 50).")
        sys.exit(1)

    logger.info(f"  Dataset built in {data_time:.1f}s")
    logger.info(f"  X shape          : {X.shape}")
    logger.info(f"  y shape          : {y.shape}")
    logger.info(f"  Sequence length  : {SEQUENCE_LENGTH}")
    logger.info(f"  Features/step    : {INPUT_SIZE}")
    logger.info(f"  X range          : [{X.min():.2f}, {X.max():.2f}]")
    logger.info(f"  y range          : [{y.min():.2f}, {y.max():.2f}]")
    logger.info(f"  y mean           : {y.mean():.4f}")
    logger.info(f"  y std            : {y.std():.4f}")

    # Fit scaler on 2D features
    logger.info("  Fitting StandardScaler on features...")
    X_2d = X.reshape(-1, INPUT_SIZE)
    scaler = StandardScaler()
    X_scaled_2d = scaler.fit_transform(X_2d)
    X_scaled = X_scaled_2d.reshape(X.shape)
    logger.info(f"  Scaler means     : {np.array2string(scaler.mean_, precision=3)}")
    logger.info(f"  Scaler scales    : {np.array2string(scaler.scale_, precision=3)}")

    # Normalize targets
    y_mean = float(y.mean())
    y_std = float(y.std())
    y_scaled = (y - y_mean) / (y_std + 1e-8)
    logger.info(f"  Target norm      : mean={y_mean:.4f}, std={y_std:.4f}")

    # Chronological train/test split
    test_n = max(1, int(len(X_scaled) * test_split))
    X_train, X_test = X_scaled[:-test_n], X_scaled[-test_n:]
    y_train, y_test = y_scaled[:-test_n], y_scaled[-test_n:]
    logger.info(f"  Train sequences  : {len(X_train):,}")
    logger.info(f"  Test sequences   : {len(X_test):,}")
    logger.info("=" * 56)

    # ── PHASE 2: Hyperparameter Search (logged inside tyre_model) ────────
    best_hidden, best_lr, best_val_mse = run_hyperparameter_search(X_train, y_train, logger)

    # ── PHASE 3: Final Model Training (logged inside tyre_model) ─────────
    final_model = train_final_model(X_train, y_train, best_hidden, best_lr, logger)
    gc.collect()

    # ── PHASE 4: Evaluation (logged inside tyre_model) ───────────────────
    mae = evaluate_on_test(final_model, X_test, y_test, y_mean, y_std)

    # ── Save Artifacts ────────────────────────────────────────────────────
    config = {
        "hidden_size": best_hidden,
        "learning_rate": best_lr,
        "input_size": INPUT_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "num_layers": NUM_LAYERS,
        "dropout_rate": DROPOUT_RATE,
        "y_mean": y_mean,
        "y_std": y_std,
        "val_mse": best_val_mse,
        "test_mae_kmh": mae,
        "seasons": args.seasons,
        "trained_at": datetime.utcnow().isoformat(),
    }
    save_model_artifacts(final_model, scaler, config, models_dir, logger)
    gc.collect()

    total_time = time.time() - pipeline_t0
    mins, secs = divmod(int(total_time), 60)

    logger.info("")
    logger.info("#" * 56)
    logger.info("#" + " " * 54 + "#")
    logger.info("#    TRAINING COMPLETE" + " " * 33 + "#")
    logger.info("#" + " " * 54 + "#")
    logger.info("#" * 56)
    logger.info(f"  Sequences      : {len(X):,}")
    logger.info(f"  Best hidden    : {best_hidden}")
    logger.info(f"  Best LR        : {best_lr}")
    logger.info(f"  Val MSE        : {best_val_mse:.6f}")
    logger.info(f"  Test MAE       : {mae:.3f} km/h")
    logger.info(f"  Total time     : {mins}m {secs}s")
    logger.info("-" * 56)
    logger.info(f"  Model saved    : {models_dir / 'tyre_lstm.pt'}")
    logger.info(f"  Scaler saved   : {models_dir / 'tyre_scaler.pkl'}")
    logger.info(f"  Config saved   : {models_dir / 'tyre_config.json'}")
    logger.info("#" * 56)


if __name__ == "__main__":
    main()
