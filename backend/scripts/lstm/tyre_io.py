"""I/O operations for the ApexHunter LSTM tyre predictor.
Handles model saving/loading, scaler persistence, and prediction output writing."""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

from sklearn.preprocessing import StandardScaler

from utils import IST, setup_logger

logger = setup_logger(__name__)

# ── File names ────────────────────────────────────────────────────────────────
MODEL_FILENAME: str = "tyre_lstm.pt"
SCALER_FILENAME: str = "tyre_scaler.pkl"
CONFIG_FILENAME: str = "tyre_config.json"


def save_model_artifacts(
    model: Any,
    scaler: StandardScaler,
    config: dict,
    models_dir: Path,
    logger: logging.Logger,
) -> None:
    """Save the trained model, scaler, and config to the models/ directory.

    Args:
        model: Trained TyreCliffLSTM (nn.Module).
        scaler: Fitted StandardScaler for feature normalization.
        config: Training config dict with hyperparams and metrics.
        models_dir: Directory to save artifacts into.
        logger: Logger instance.
    """
    import torch

    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / MODEL_FILENAME
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model: {model_path}")

    scaler_path = models_dir / SCALER_FILENAME
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler: {scaler_path}")

    config_path = models_dir / CONFIG_FILENAME
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config: {config_path}")


def load_model_artifacts(
    models_dir: Path,
    logger: logging.Logger,
) -> Tuple[Any, StandardScaler, dict]:
    """Load the trained model, scaler, and config.

    Args:
        models_dir: Directory containing saved artifacts.
        logger: Logger instance.

    Returns:
        Tuple of (model, scaler, config).

    Raises:
        FileNotFoundError: If any artifact is missing.
    """
    import torch

    model_path = models_dir / MODEL_FILENAME
    scaler_path = models_dir / SCALER_FILENAME
    config_path = models_dir / CONFIG_FILENAME

    missing = []
    if not model_path.exists():
        missing.append(str(model_path))
    if not scaler_path.exists():
        missing.append(str(scaler_path))
    if not config_path.exists():
        missing.append(str(config_path))

    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts: {', '.join(missing)}"
        )

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)
    logger.info(f"Loaded config: {config_path}")

    # Reconstruct model
    from lstm.tyre_model import TyreCliffLSTM, INPUT_SIZE, NUM_LAYERS, DROPOUT_RATE

    model = TyreCliffLSTM(
        INPUT_SIZE, config["hidden_size"], NUM_LAYERS, DROPOUT_RATE
    )
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    logger.info(f"Loaded model: {model_path}")

    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logger.info(f"Loaded scaler: {scaler_path}")

    return (model, scaler, config)


def build_prediction_output(
    session_path: str,
    driver: str,
    stint_results: List[dict],
) -> dict:
    """Build the prediction output dict.

    Args:
        session_path: Path to the session parquet file.
        driver: Driver code string.
        stint_results: List of per-stint prediction dicts.

    Returns:
        Dict with session_file, driver, timestamp, and stints.
    """
    return {
        "session_file": session_path,
        "driver": driver,
        "timestamp": datetime.now(IST).isoformat(),
        "stints": stint_results,
    }


def save_prediction(
    data: dict,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Write prediction JSON to disk.

    Args:
        data: Prediction output dict from build_prediction_output.
        output_path: Destination path for the JSON file.
        logger: Logger instance.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved tyre prediction: {output_path}")


def log_prediction_complete(
    driver: str, stem: str, stint_results: List[dict], output_path: Path, logger: logging.Logger
) -> None:
    """Log the completion of prediction."""
    stint_lines = "\n".join(
        f"   Stint {s['stint_index']+1}: {s['n_laps']} laps, "
        f"cliff at lap {s['cliff_lap']+1 if s['cliff_lap'] is not None else '—'}, "
        f"{s['laps_remaining'] if s['laps_remaining'] is not None else '—'} laps remaining"
        for s in stint_results
    )
    logger.info(f"""
======================================================
   ApexHunter — Tyre Cliff Prediction Complete
======================================================
   Driver        : {driver}
   Session       : {stem}
   Stints found  : {len(stint_results)}
{stint_lines}
======================================================
   Output: {output_path}
======================================================""")


def log_training_start(sessions_dir, models_dir, seasons, test_split, force, logger):
    logger.info("")
    logger.info("#" * 56)
    logger.info("#" + " " * 54 + "#")
    logger.info("#    ApexHunter -- LSTM Tyre Cliff Predictor" + " " * 10 + "#")
    logger.info("#" + " " * 54 + "#")
    logger.info("#" * 56)
    logger.info(f"  Sessions dir   : {sessions_dir}")
    logger.info(f"  Models dir     : {models_dir}")
    logger.info(f"  Seasons        : {seasons}")
    logger.info(f"  Test split     : {test_split}")
    logger.info(f"  Force retrain  : {force}")
    logger.info(f"  Started at     : {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("#" * 56)


def log_dataset_stats(X, y, data_time, logger, seq_len, input_size):
    logger.info(f"  Dataset built in {data_time:.1f}s")
    logger.info(f"  X shape          : {X.shape}")
    logger.info(f"  y shape          : {y.shape}")
    logger.info(f"  Sequence length  : {seq_len}")
    logger.info(f"  Features/step    : {input_size}")
    logger.info(f"  X range          : [{X.min():.2f}, {X.max():.2f}]")
    logger.info(f"  y range          : [{y.min():.2f}, {y.max():.2f}]")
    logger.info(f"  y mean           : {y.mean():.4f}")
    logger.info(f"  y std            : {y.std():.4f}")


def log_training_complete(X_len, best_hidden, best_lr, best_val_mse, mae, total_time, models_dir, logger):
    mins, secs = divmod(int(total_time), 60)
    logger.info("")
    logger.info("#" * 56)
    logger.info("#" + " " * 54 + "#")
    logger.info("#    TRAINING COMPLETE" + " " * 33 + "#")
    logger.info("#" + " " * 54 + "#")
    logger.info("#" * 56)
    logger.info(f"  Sequences      : {X_len:,}")
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
