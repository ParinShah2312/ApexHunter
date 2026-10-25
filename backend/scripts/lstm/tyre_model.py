"""LSTM architecture and training loop for the ApexHunter tyre cliff predictor.
Handles model definition, hyperparameter search, training, and evaluation."""

import copy
import gc
import logging
import math
import time
from typing import List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import setup_logger

logger = setup_logger(__name__)

# ── Module-level constants ────────────────────────────────────────────────────
INPUT_SIZE: int = 9
"""Number of features per lap (len(LAP_FEATURES))."""

NUM_LAYERS: int = 2
"""Stacked LSTM layers."""

DROPOUT_RATE: float = 0.3
"""Dropout between LSTM layers."""

RANDOM_STATE: int = 42
"""Random seed for reproducibility."""

N_EPOCHS_SEARCH: int = 20
"""Epochs per hyperparameter combination during search."""

N_EPOCHS_FINAL: int = 50
"""Epochs for final model training."""

BATCH_SIZE: int = 32
"""Mini-batch size for DataLoader."""

VAL_SPLIT: float = 0.2
"""Fraction of training data held out for validation."""

HYPERPARAM_GRID: List[dict] = [
    {"hidden_size": 32,  "learning_rate": 0.001},
    {"hidden_size": 64,  "learning_rate": 0.001},
    {"hidden_size": 128, "learning_rate": 0.001},
    {"hidden_size": 32,  "learning_rate": 0.0005},
    {"hidden_size": 64,  "learning_rate": 0.0005},
    {"hidden_size": 128, "learning_rate": 0.0001},
]
"""Six combinations — exhaustive grid over hidden_size and learning_rate."""


class TyreCliffLSTM(nn.Module):
    """LSTM model for predicting tyre pace degradation."""

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, dropout: float) -> None:
        """Initialize the LSTM model.

        Args:
            input_size: Number of features per time step.
            hidden_size: Number of LSTM hidden units.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability between LSTM layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x shape: (batch, seq_len, input_size)."""
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)  # shape: (batch,)


def split_by_groups(
    n_samples: int, groups: Optional[np.ndarray], holdout_frac: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Split sample indices into (keep, holdout), holding out the LAST groups (sessions).

    Sliding-window sequences from the same session overlap by several laps, so splitting by
    sample index leaks near-identical windows across the split. Holding out whole, most recent
    sessions avoids that and keeps the split chronological. Falls back to a chronological index
    split when groups is None or contains fewer than two sessions.

    Args:
        n_samples: Number of samples.
        groups: Session index per sample (chronologically ordered), or None.
        holdout_frac: Fraction of sessions (or samples, in the fallback) to hold out.

    Returns:
        Tuple of (keep_indices, holdout_indices).
    """
    indices = np.arange(n_samples)
    unique_groups = np.unique(groups) if groups is not None else np.array([])
    if len(unique_groups) >= 2:
        n_holdout = min(len(unique_groups) - 1, max(1, math.ceil(len(unique_groups) * holdout_frac)))
        holdout_groups = unique_groups[-n_holdout:]
        holdout_mask = np.isin(groups, holdout_groups)
        return indices[~holdout_mask], indices[holdout_mask]
    n_holdout = max(1, int(n_samples * holdout_frac))
    return indices[:-n_holdout], indices[-n_holdout:]


def _build_data_loader(
    X_train: np.ndarray, y_train: np.ndarray, batch_size: int = BATCH_SIZE
) -> DataLoader:
    """Create a DataLoader for training data."""
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _compute_val_mse(
    model: nn.Module, X_val_t: torch.Tensor, y_val_t: torch.Tensor, criterion: nn.Module
) -> float:
    """Evaluate model and return MSE."""
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_mse = float(criterion(val_preds, y_val_t).item())
    return val_mse


def _run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    X_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    logger: logging.Logger,
    verbose: bool,
    config_label: str
) -> Tuple[float, int]:
    """Run training loop and return (best validation MSE, best epoch).

    The model's weights are restored to the best-validation epoch before returning, so the
    returned model matches the reported best MSE (not whatever the last epoch produced).
    """
    best_val_mse = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    train_start = time.time()

    for epoch in range(n_epochs):
        ep_t0 = time.time()
        model.train()
        running_loss = 0.0
        batch_count = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1

        avg_loss = running_loss / max(batch_count, 1)

        val_mse = _compute_val_mse(model, X_val_t, y_val_t, criterion)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        elapsed = time.time() - ep_t0
        should_log = (
            verbose or epoch == 0 or epoch == n_epochs - 1
            or (epoch + 1) % 5 == 0
        )
        if should_log:
            bar_len = 20
            filled = int(bar_len * (epoch + 1) / n_epochs)
            bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
            pct = 100.0 * (epoch + 1) / n_epochs
            logger.info(
                f"{config_label}Epoch {epoch+1:3d}/{n_epochs} "
                f"|{bar}| {pct:5.1f}%  "
                f"loss={avg_loss:.6f}  val={val_mse:.6f}  "
                f"best={best_val_mse:.6f}(ep{best_epoch})  "
                f"[{elapsed:.2f}s]"
            )

    model.load_state_dict(best_state)
    model.eval()

    total_time = time.time() - train_start
    logger.info(
        f"{config_label}Done {n_epochs} epochs in {total_time:.1f}s  "
        f"| last val={val_mse:.6f}  best={best_val_mse:.6f}@ep{best_epoch} (weights restored)"
    )
    return best_val_mse, best_epoch


def train_one_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_size: int,
    learning_rate: float,
    n_epochs: int,
    logger: logging.Logger,
    *,
    verbose: bool = False,
    config_label: str = "",
) -> Tuple[TyreCliffLSTM, float]:
    """Train LSTM for one hyperparameter configuration.

    Args:
        X_train: Training features, shape (N, seq_len, input_size).
        y_train: Training targets, shape (N,).
        X_val: Validation features.
        y_val: Validation targets.
        hidden_size: Number of LSTM hidden units.
        learning_rate: Adam learning rate.
        n_epochs: Number of training epochs.
        logger: Logger instance.
        verbose: If True, log every epoch; otherwise every 5th.
        config_label: Label prefix for log messages.

    Returns:
        Tuple of (trained model, validation MSE).
    """
    torch.manual_seed(RANDOM_STATE)

    loader = _build_data_loader(X_train, y_train)

    model = TyreCliffLSTM(INPUT_SIZE, hidden_size, NUM_LAYERS, DROPOUT_RATE)
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        logger.info(f"{config_label}Model params: {total_params:,} trainable")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    best_val_mse, _ = _run_training_loop(
        model, loader, X_val_t, y_val_t, criterion, optimizer,
        n_epochs, logger, verbose, config_label
    )

    return (model, best_val_mse)


def run_hyperparameter_search(
    X: np.ndarray,
    y: np.ndarray,
    logger: logging.Logger,
    groups: Optional[np.ndarray] = None,
    grid: Optional[List[dict]] = None,
    n_epochs: int = N_EPOCHS_SEARCH,
) -> Tuple[int, float, float]:
    """Grid search over HYPERPARAM_GRID.

    Args:
        X: Training feature array, shape (N, seq_len, input_size).
        y: Training target array, shape (N,).
        logger: Logger instance.
        groups: Optional session index per sample; when given, the validation set is the
            last VAL_SPLIT fraction of sessions rather than of samples.
        grid: Optional list of {"hidden_size", "learning_rate"} dicts (defaults to HYPERPARAM_GRID).
        n_epochs: Epochs per configuration.

    Returns:
        Tuple of (best_hidden_size, best_learning_rate, best_val_mse).
    """
    grid = grid if grid is not None else HYPERPARAM_GRID
    train_idx, val_idx = split_by_groups(len(X), groups, VAL_SPLIT)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    best_hidden: int = grid[0]["hidden_size"]
    best_lr: float = grid[0]["learning_rate"]
    best_mse: float = float("inf")

    total_configs = len(grid)
    search_t0 = time.time()

    logger.info("")
    logger.info("=" * 56)
    logger.info("  PHASE 2 / 4 -- HYPERPARAMETER SEARCH")
    logger.info("=" * 56)
    logger.info(f"  Grid size        : {total_configs} configurations")
    logger.info(f"  Epochs per config: {n_epochs}")
    logger.info(f"  Train samples    : {len(X_train):,}")
    logger.info(f"  Val samples      : {len(X_val):,}")
    logger.info(f"  Batch size       : {BATCH_SIZE}")
    logger.info("=" * 56)

    results_table = []

    for idx, config in enumerate(grid, 1):
        h = config["hidden_size"]
        lr = config["learning_rate"]
        label = f"[{idx}/{total_configs}] "

        logger.info("")
        logger.info(f"-- {label}hidden={h}, lr={lr} --")

        cfg_t0 = time.time()
        _, val_mse = train_one_config(
            X_train, y_train, X_val, y_val,
            hidden_size=h, learning_rate=lr,
            n_epochs=n_epochs, logger=logger,
            config_label=label,
        )
        cfg_time = time.time() - cfg_t0

        is_best = val_mse < best_mse
        if is_best:
            best_mse = val_mse
            best_hidden = h
            best_lr = lr

        marker = " * BEST" if is_best else ""
        results_table.append((h, lr, val_mse, cfg_time, marker))
        gc.collect()

    search_time = time.time() - search_t0

    logger.info("")
    logger.info("-" * 56)
    logger.info("  SEARCH RESULTS (sorted by val MSE)")
    logger.info("-" * 56)
    logger.info(f"  {'Hidden':>8s}  {'LR':>10s}  {'Val MSE':>10s}  {'Time':>7s}")
    for h, lr, mse, t, marker in sorted(results_table, key=lambda r: r[2]):
        logger.info(f"  {h:>8d}  {lr:>10.5f}  {mse:>10.6f}  {t:>6.1f}s{marker}")
    logger.info("-" * 56)
    logger.info(
        f"  Winner: hidden={best_hidden}, lr={best_lr}, "
        f"val_MSE={best_mse:.6f}  ({search_time:.1f}s total)"
    )
    logger.info("-" * 56)
    return (best_hidden, best_lr, best_mse)


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    hidden_size: int,
    learning_rate: float,
    logger: logging.Logger,
    groups: Optional[np.ndarray] = None,
    n_epochs: int = N_EPOCHS_FINAL,
) -> TyreCliffLSTM:
    """Train the final model for n_epochs, keeping the best-validation-epoch weights.

    Args:
        X: Training feature array, shape (N, seq_len, input_size).
        y: Training target array, shape (N,).
        hidden_size: Best hidden size from search.
        learning_rate: Best learning rate from search.
        logger: Logger instance.
        groups: Optional session index per sample, used to hold out whole sessions for validation.
        n_epochs: Number of training epochs.

    Returns:
        Trained TyreCliffLSTM model.
    """
    train_idx, val_idx = split_by_groups(len(X), groups, VAL_SPLIT)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    logger.info("")
    logger.info("=" * 56)
    logger.info("  PHASE 3 / 4 -- FINAL MODEL TRAINING")
    logger.info("=" * 56)
    logger.info(f"  Architecture     : LSTM({INPUT_SIZE} -> {hidden_size} x {NUM_LAYERS})")
    logger.info(f"  Learning rate    : {learning_rate}")
    logger.info(f"  Dropout          : {DROPOUT_RATE}")
    logger.info(f"  Epochs           : {n_epochs}")
    logger.info(f"  Batch size       : {BATCH_SIZE}")
    logger.info(f"  Train samples    : {len(X_train):,}")
    logger.info(f"  Val samples      : {len(X_val):,}")
    logger.info("=" * 56)

    final_t0 = time.time()
    model, val_mse = train_one_config(
        X_train, y_train, X_val, y_val,
        hidden_size=hidden_size, learning_rate=learning_rate,
        n_epochs=n_epochs, logger=logger,
        verbose=True,
        config_label="[Final] ",
    )
    final_time = time.time() - final_t0

    logger.info(f"Final model val_MSE: {val_mse:.6f}  (trained in {final_time:.1f}s)")
    return model


def evaluate_on_test(
    model: TyreCliffLSTM,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_target_mean: float,
    scaler_target_std: float,
) -> float:
    """Compute MAE in km/h on the held-out test set.

    Args:
        model: Trained TyreCliffLSTM.
        X_test: Test features, shape (N, seq_len, input_size).
        y_test: Test targets (normalized), shape (N,).
        scaler_target_mean: Mean used for target normalization.
        scaler_target_std: Std used for target normalization.

    Returns:
        Mean Absolute Error in km/h.
    """
    logger.info("")
    logger.info("=" * 56)
    logger.info("  PHASE 4 / 4 -- TEST SET EVALUATION")
    logger.info("=" * 56)
    logger.info(f"  Test samples     : {len(X_test):,}")
    logger.info(f"  Target mean      : {scaler_target_mean:.4f}")
    logger.info(f"  Target std       : {scaler_target_std:.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).numpy()

    preds_denorm = preds * scaler_target_std + scaler_target_mean
    y_denorm = y_test * scaler_target_std + scaler_target_mean

    abs_errors = np.abs(preds_denorm - y_denorm)
    mae = float(np.mean(abs_errors))
    median_ae = float(np.median(abs_errors))
    max_ae = float(np.max(abs_errors))
    rmse = float(np.sqrt(np.mean((preds_denorm - y_denorm) ** 2)))

    logger.info(f"  MAE              : {mae:.3f} km/h")
    logger.info(f"  Median AE        : {median_ae:.3f} km/h")
    logger.info(f"  Max AE           : {max_ae:.3f} km/h")
    logger.info(f"  RMSE             : {rmse:.3f} km/h")
    logger.info("=" * 56)
    return mae


def monte_carlo_predict(
    model: TyreCliffLSTM,
    X_input: np.ndarray,
    n_samples: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run N forward passes with dropout enabled to estimate prediction uncertainty.

    Args:
        model: Trained TyreCliffLSTM (dropout will be enabled for MC sampling).
        X_input: Input features, shape (N, seq_len, input_size).
        n_samples: Number of Monte Carlo forward passes.

    Returns:
        Tuple of (mean_preds, lower_bound, upper_bound), each shape (N,).
    """
    model.train()  # Enable dropout for MC sampling
    X_t = torch.FloatTensor(X_input)

    preds_list = []
    for _ in range(n_samples):
        with torch.no_grad():
            preds_list.append(model(X_t).numpy())

    preds_arr = np.array(preds_list)  # shape (n_samples, N)

    mean_preds = preds_arr.mean(axis=0)
    std_preds = preds_arr.std(axis=0)
    lower = mean_preds - std_preds
    upper = mean_preds + std_preds

    model.eval()
    return (mean_preds, lower, upper)


def predict_single_stint(
    stint_data: dict, model: nn.Module, scaler: Any,
    y_mean: float, y_std: float, seq_len: int
) -> dict:
    """Predict cliff for a single stint and return the result dictionary."""
    from lstm.tyre_data import LAP_FEATURES
    stint_index = stint_data["stint_index"]
    lap_features = pd.DataFrame(stint_data["lap_features"])
    actual_speeds = lap_features["mean_speed"].tolist()
    actual_lap_times = lap_features["lap_time_seconds"].tolist()

    lap_numbers = [int(n) for n in lap_features["race_lap_number"]]

    # Out-laps are already excluded by extract_stints, so the first lap is a genuine racing lap
    # and can seed the padding for the first prediction.
    features_np = lap_features[LAP_FEATURES].values.copy()

    padded_features = np.vstack([np.tile(features_np[0], (seq_len, 1)), features_np])

    new_seqs = []
    for i in range(len(features_np)):
        new_seqs.append(padded_features[i : i + seq_len])

    sequences = np.array(new_seqs, dtype=np.float32)
    seq_2d = sequences.reshape(-1, INPUT_SIZE)
    seq_scaled = scaler.transform(seq_2d).reshape(sequences.shape)

    mean_preds, lower, upper = monte_carlo_predict(model, seq_scaled)

    mean_preds_speed = (mean_preds * y_std + y_mean).tolist()
    lower_speed = (lower * y_std + y_mean).tolist()
    upper_speed = (upper * y_std + y_mean).tolist()

    valid_pairs = [(s, t) for s, t in zip(actual_speeds, actual_lap_times) if not pd.isna(t) and t > 0]
    if valid_pairs:
        track_dist = float(np.mean([s * t for s, t in valid_pairs]))
    else:
        track_dist = actual_speeds[0] * 90.0

    def speed_to_time(s): return track_dist / max(s, 1.0)

    predicted_laps = [speed_to_time(s) for s in mean_preds_speed]
    confidence_lower = [speed_to_time(s) for s in upper_speed]
    confidence_upper = [speed_to_time(s) for s in lower_speed]
    actual_laps_clean = [t if not pd.isna(t) else None for t in actual_lap_times]

    best_lap_time = min(predicted_laps)
    best_lap_idx = predicted_laps.index(best_lap_time)
    cliff_lap = None
    for i in range(best_lap_idx + 1, len(predicted_laps)):
        if predicted_laps[i] > best_lap_time + 1.5:
            cliff_lap = i
            break

    laps_remaining = (len(actual_lap_times) - cliff_lap) if cliff_lap is not None else None

    return {
        "stint_index": stint_index, "n_laps": len(actual_lap_times),
        "lap_numbers": lap_numbers,
        "actual_laps": actual_laps_clean, "predicted_laps": predicted_laps,
        "confidence_upper": confidence_upper, "confidence_lower": confidence_lower,
        "cliff_lap": cliff_lap, "laps_remaining": laps_remaining,
    }
