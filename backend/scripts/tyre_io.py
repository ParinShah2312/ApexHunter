"""I/O operations for the ApexHunter LSTM tyre predictor.
Handles model saving/loading, scaler persistence, and prediction output writing."""

import json
import gc
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import setup_logger

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
    from tyre_model import TyreCliffLSTM, INPUT_SIZE, NUM_LAYERS, DROPOUT_RATE

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
        "timestamp": datetime.utcnow().isoformat(),
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
