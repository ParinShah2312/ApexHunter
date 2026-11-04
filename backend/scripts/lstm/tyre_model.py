"""Recurrent tyre pace models (LSTM and vanilla RNN, ROADMAP §D) and their training loop.
Handles model definition, hyperparameter search, training, evaluation and prediction."""

import copy
import gc
import logging
import math
import time
from typing import Any, Callable, List, Optional, Tuple

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
"""Stacked recurrent (LSTM or RNN) layers."""

DROPOUT_RATE: float = 0.3
"""Dropout between the recurrent layers."""

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
"""Six of the 3 x 3 hidden_size x learning_rate combinations (not exhaustive)."""


CELLS: Tuple[str, ...] = ("lstm", "rnn")
"""Recurrent layer types: LSTM, or a vanilla (Elman, tanh) RNN with the same size and setup."""


class TyrePaceRNN(nn.Module):
    """Recurrent model predicting the next lap's change in lap time from a 5-lap window.

    cell="lstm" (default) or "rnn": the vanilla RNN baseline of ROADMAP §D uses exactly the same
    layers, dropout, head, data and training, so the two differ only in the recurrent cell.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, dropout: float, cell: str = "lstm") -> None:
        """Initialize the model.

        Args:
            input_size: Number of features per time step.
            hidden_size: Number of hidden units.
            num_layers: Number of stacked recurrent layers.
            dropout: Dropout probability between recurrent layers.
            cell: "lstm" or "rnn".
        """
        super().__init__()
        if cell not in CELLS:
            raise ValueError(f"Unknown cell '{cell}' (use one of {CELLS})")
        self.cell = cell
        layer = nn.LSTM if cell == "lstm" else nn.RNN
        self.lstm = layer(
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
    cell: str = "lstm",
    seed: int = RANDOM_STATE,
) -> Tuple[TyrePaceRNN, float]:
    """Train the LSTM or vanilla RNN for one hyperparameter configuration.

    Args:
        X_train: Training features, shape (N, seq_len, input_size).
        y_train: Training targets, shape (N,).
        X_val: Validation features.
        y_val: Validation targets.
        hidden_size: Number of hidden units.
        learning_rate: Adam learning rate.
        n_epochs: Number of training epochs.
        logger: Logger instance.
        verbose: If True, log every epoch; otherwise every 5th.
        config_label: Label prefix for log messages.
        cell: "lstm" or "rnn".
        seed: Random seed (weights, dropout and batch order).

    Returns:
        Tuple of (trained model, validation MSE).
    """
    torch.manual_seed(seed)

    loader = _build_data_loader(X_train, y_train)

    model = TyrePaceRNN(INPUT_SIZE, hidden_size, NUM_LAYERS, DROPOUT_RATE, cell=cell)
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
    cell: str = "lstm",
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
            config_label=label, cell=cell,
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
    cell: str = "lstm",
    seed: int = RANDOM_STATE,
) -> Tuple[TyrePaceRNN, float]:
    """Train the final model for n_epochs, keeping the best-validation-epoch weights.

    Returns (model, the final model's own best validation MSE) — #46: the config used to record the
    search run's score instead.

    Args:
        X: Training feature array, shape (N, seq_len, input_size).
        y: Training target array, shape (N,).
        hidden_size: Best hidden size from search.
        learning_rate: Best learning rate from search.
        logger: Logger instance.
        groups: Optional session index per sample, used to hold out whole sessions for validation.
        n_epochs: Number of training epochs.
        cell: "lstm" or "rnn".
        seed: Random seed.
    """
    train_idx, val_idx = split_by_groups(len(X), groups, VAL_SPLIT)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    logger.info("")
    logger.info("=" * 56)
    logger.info("  PHASE 3 / 4 -- FINAL MODEL TRAINING")
    logger.info("=" * 56)
    logger.info(f"  Architecture     : {cell.upper()}({INPUT_SIZE} -> {hidden_size} x {NUM_LAYERS}), seed {seed}")
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
        config_label="[Final] ", cell=cell, seed=seed,
    )
    final_time = time.time() - final_t0

    logger.info(f"Final model val_MSE: {val_mse:.6f}  (trained in {final_time:.1f}s)")
    return model, val_mse


def sequence_predictor(model: nn.Module, scaler: Any, y_mean: float,
                       y_std: float) -> Callable[[np.ndarray], np.ndarray]:
    """windows (N, seq_len, n_features), unscaled -> predicted lap-time change (s) per window.
    Deterministic (dropout off); used by the strategy check and output like Ridge's predict."""
    def predict(windows: np.ndarray) -> np.ndarray:
        scaled = scaler.transform(windows.reshape(-1, windows.shape[-1])).reshape(windows.shape)
        model.eval()
        with torch.no_grad():
            out = model(torch.FloatTensor(scaled.astype(np.float32))).numpy()
        return out * y_std + y_mean
    return predict
