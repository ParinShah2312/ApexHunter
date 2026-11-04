"""I/O for the ApexHunter tyre pace models (LSTM and vanilla RNN): saving and loading the model,
scaler and config, and the training log banners."""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

from sklearn.preprocessing import StandardScaler

from utils import IST, setup_logger

logger = setup_logger(__name__)

# ── File names ────────────────────────────────────────────────────────────────
MODEL_FILENAME: str = "tyre_lstm.pt"
SCALER_FILENAME: str = "tyre_scaler.pkl"
CONFIG_FILENAME: str = "tyre_config.json"


def artifact_names(cell: str = "lstm") -> Tuple[str, str, str]:
    """(model, scaler, config) file names. The LSTM keeps the original names; the vanilla RNN
    (ROADMAP §D) gets its own, so both models can be saved side by side."""
    if cell == "lstm":
        return MODEL_FILENAME, SCALER_FILENAME, CONFIG_FILENAME
    return f"tyre_{cell}.pt", f"tyre_{cell}_scaler.pkl", f"tyre_{cell}_config.json"


def save_model_artifacts(
    model: Any,
    scaler: StandardScaler,
    config: dict,
    models_dir: Path,
    logger: logging.Logger,
    cell: str = "lstm",
) -> None:
    """Save the trained model, scaler, and config to the models/ directory (names per cell).

    Args:
        model: Trained TyrePaceRNN (nn.Module).
        scaler: Fitted StandardScaler for feature normalization.
        config: Training config dict with hyperparams and metrics.
        models_dir: Directory to save artifacts into.
        logger: Logger instance.
    """
    import torch

    models_dir.mkdir(parents=True, exist_ok=True)
    model_name, scaler_name, config_name = artifact_names(cell)

    model_path = models_dir / model_name
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model: {model_path}")

    scaler_path = models_dir / scaler_name
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler: {scaler_path}")

    config_path = models_dir / config_name
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config: {config_path}")


def load_model_artifacts(
    models_dir: Path,
    logger: logging.Logger,
    cell: str = "lstm",
) -> Tuple[Any, StandardScaler, dict]:
    """Load the trained model ("lstm" or "rnn"), scaler, and config.

    Args:
        models_dir: Directory containing saved artifacts.
        logger: Logger instance.

    Returns:
        Tuple of (model, scaler, config).

    Raises:
        FileNotFoundError: If any artifact is missing.
        ValueError: If the saved model was trained on a different feature set than the
            current code uses (the model must then be retrained).
    """
    import torch

    model_name, scaler_name, config_name = artifact_names(cell)
    model_path = models_dir / model_name
    scaler_path = models_dir / scaler_name
    config_path = models_dir / config_name

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
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info(f"Loaded config: {config_path}")

    # Reconstruct model
    from lstm.tyre_data import LAP_FEATURES, TARGET
    from lstm.tyre_model import TyrePaceRNN, INPUT_SIZE, NUM_LAYERS, DROPOUT_RATE

    saved_input_size = config.get("input_size")
    saved_features = config.get("features")
    if (saved_input_size != INPUT_SIZE or (saved_features is not None and saved_features != LAP_FEATURES)
            or config.get("target") != TARGET or config.get("cell", "lstm") != cell):
        raise ValueError(
            f"Saved {cell.upper()} model is incompatible with the current code: it was trained on "
            f"{saved_input_size} features {saved_features or '(names not recorded)'} and target "
            f"{config.get('target') or '(not recorded)'} ({config.get('cell', 'lstm')} cell), but the "
            f"code now uses {INPUT_SIZE} features {LAP_FEATURES} and target {TARGET} ({cell} cell). "
            f"Retrain with: python backend/scripts/train_lstm.py --cell {cell} --force"
        )

    model = TyrePaceRNN(
        INPUT_SIZE, config["hidden_size"], NUM_LAYERS, DROPOUT_RATE, cell=cell
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


def log_training_start(sessions_dir, models_dir, seasons, test_split, force, logger):
    logger.info("")
    logger.info("#" * 56)
    logger.info("#" + " " * 54 + "#")
    logger.info("#    ApexHunter -- tyre pace model training" + " " * 11 + "#")
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


def log_training_complete(X_len, best_hidden, best_lr, best_val_mse, mae, total_time, models_dir, logger,
                          cell: str = "lstm"):
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
    logger.info(f"  Test MAE       : {mae:.3f} s")
    logger.info(f"  Total time     : {mins}m {secs}s")
    logger.info("-" * 56)
    model_name, scaler_name, config_name = artifact_names(cell)        # #51: the RNN's own files
    logger.info(f"  Model saved    : {models_dir / model_name}")
    logger.info(f"  Scaler saved   : {models_dir / scaler_name}")
    logger.info(f"  Config saved   : {models_dir / config_name}")
    logger.info("#" * 56)
