"""LSTM architecture and training loop for the ApexHunter tyre cliff predictor.
Handles model definition, hyperparameter search, training, and evaluation."""

import gc
import logging
import time
from typing import List, Tuple

import numpy as np
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

    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TyreCliffLSTM(INPUT_SIZE, hidden_size, NUM_LAYERS, DROPOUT_RATE)
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        logger.info(f"{config_label}Model params: {total_params:,} trainable")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    best_val_mse = float("inf")
    best_epoch = 0
    train_start = time.time()

    for epoch in range(n_epochs):
        ep_t0 = time.time()
        model.train()
        running_loss = 0.0
        batch_count = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1

        avg_loss = running_loss / max(batch_count, 1)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_mse = float(criterion(val_preds, y_val_t).item())

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch + 1

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

    total_time = time.time() - train_start
    logger.info(
        f"{config_label}Done {n_epochs} epochs in {total_time:.1f}s  "
        f"| val={val_mse:.6f}  best={best_val_mse:.6f}@ep{best_epoch}"
    )
    return (model, best_val_mse)


def run_hyperparameter_search(
    X: np.ndarray,
    y: np.ndarray,
    logger: logging.Logger,
) -> Tuple[int, float, float]:
    """Grid search over HYPERPARAM_GRID.

    Args:
        X: Full feature array, shape (N, seq_len, input_size).
        y: Full target array, shape (N,).
        logger: Logger instance.

    Returns:
        Tuple of (best_hidden_size, best_learning_rate, best_val_mse).
    """
    split = int(len(X) * (1 - VAL_SPLIT))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    best_hidden: int = HYPERPARAM_GRID[0]["hidden_size"]
    best_lr: float = HYPERPARAM_GRID[0]["learning_rate"]
    best_mse: float = float("inf")

    total_configs = len(HYPERPARAM_GRID)
    search_t0 = time.time()

    logger.info("")
    logger.info("=" * 56)
    logger.info("  PHASE 2 / 4 -- HYPERPARAMETER SEARCH")
    logger.info("=" * 56)
    logger.info(f"  Grid size        : {total_configs} configurations")
    logger.info(f"  Epochs per config: {N_EPOCHS_SEARCH}")
    logger.info(f"  Train samples    : {len(X_train):,}")
    logger.info(f"  Val samples      : {len(X_val):,}")
    logger.info(f"  Batch size       : {BATCH_SIZE}")
    logger.info("=" * 56)

    results_table = []

    for idx, config in enumerate(HYPERPARAM_GRID, 1):
        h = config["hidden_size"]
        lr = config["learning_rate"]
        label = f"[{idx}/{total_configs}] "

        logger.info("")
        logger.info(f"-- {label}hidden={h}, lr={lr} --")

        cfg_t0 = time.time()
        _, val_mse = train_one_config(
            X_train, y_train, X_val, y_val,
            hidden_size=h, learning_rate=lr,
            n_epochs=N_EPOCHS_SEARCH, logger=logger,
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
) -> TyreCliffLSTM:
    """Train the final model on the full dataset for N_EPOCHS_FINAL epochs.

    Args:
        X: Full feature array, shape (N, seq_len, input_size).
        y: Full target array, shape (N,).
        hidden_size: Best hidden size from search.
        learning_rate: Best learning rate from search.
        logger: Logger instance.

    Returns:
        Trained TyreCliffLSTM model.
    """
    split = int(len(X) * (1 - VAL_SPLIT))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    logger.info("")
    logger.info("=" * 56)
    logger.info("  PHASE 3 / 4 -- FINAL MODEL TRAINING")
    logger.info("=" * 56)
    logger.info(f"  Architecture     : LSTM({INPUT_SIZE} -> {hidden_size} x {NUM_LAYERS})")
    logger.info(f"  Learning rate    : {learning_rate}")
    logger.info(f"  Dropout          : {DROPOUT_RATE}")
    logger.info(f"  Epochs           : {N_EPOCHS_FINAL}")
    logger.info(f"  Batch size       : {BATCH_SIZE}")
    logger.info(f"  Train samples    : {len(X_train):,}")
    logger.info(f"  Val samples      : {len(X_val):,}")
    logger.info("=" * 56)

    final_t0 = time.time()
    model, val_mse = train_one_config(
        X_train, y_train, X_val, y_val,
        hidden_size=hidden_size, learning_rate=learning_rate,
        n_epochs=N_EPOCHS_FINAL, logger=logger,
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
