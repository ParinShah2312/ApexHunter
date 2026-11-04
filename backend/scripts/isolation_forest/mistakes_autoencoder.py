"""LSTM autoencoder mistake detector (ROADMAP §D, Deep Learning course).

Same five features as the Isolation Forest (deviations from the field at the same track position),
but read as ~2 s sequences: an encoder LSTM compresses a window of WINDOW consecutive samples into
a small state, a decoder LSTM rebuilds the window from it. Trained only on the driver's clean laps,
it learns what normal driving looks like; a window it cannot rebuild well is unusual. Each sample's
score is its reconstruction error averaged over every window that contains it. Windows never cross
a lap boundary (the detector is given each sample's lap number).

Plugs into mistakes_eval.compare_detectors like the Z-score and Isolation Forest detectors
(fit(X, laps) / score(X, laps), higher score = more anomalous).
"""

import copy
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

WINDOW: int = 16          # samples per window: ~2 s at the median 0.13 s spacing
HIDDEN: int = 16          # encoder state size (the bottleneck) and decoder width
EPOCHS: int = 20
BATCH_SIZE: int = 256
LEARNING_RATE: float = 1e-3
TRAIN_STRIDE: int = 2     # every 2nd window for training (neighbouring windows are near copies)
VAL_FRAC: float = 0.1     # latest windows held out to pick the best epoch
SEED: int = 42


class LSTMAutoencoder(nn.Module):
    """Encoder LSTM -> last hidden state, repeated per time step -> decoder LSTM -> linear layer."""

    def __init__(self, n_features: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)                               # h: (1, batch, hidden)
        z = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)           # same code at every step
        out, _ = self.decoder(z)
        return self.output(out)


def window_starts(laps: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """Start rows of all windows lying entirely inside one lap (rows of a lap are contiguous)."""
    breaks = np.flatnonzero(np.diff(laps) != 0) + 1
    starts: List[int] = []
    for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(laps)]):
        starts.extend(range(a, b - window + 1, stride))
    return np.asarray(starts, dtype=np.int64)


def _windows(X: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    return X[starts[:, None] + np.arange(window)[None, :]]


class LSTMAutoencoderDetector:
    """Detector interface for mistakes_eval.compare_detectors and detect_mistakes.py."""

    def __init__(self, window: int = WINDOW, hidden: int = HIDDEN, epochs: int = EPOCHS, seed: int = SEED):
        self.window, self.hidden, self.epochs, self.seed = window, hidden, epochs, seed

    def fit(self, X: np.ndarray, laps: Optional[np.ndarray] = None) -> "LSTMAutoencoderDetector":
        laps = np.zeros(len(X)) if laps is None else np.asarray(laps)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = ((X - self.mean_) / self.std_).astype(np.float32)
        starts = window_starts(laps, self.window, TRAIN_STRIDE)
        if len(starts) < 10:
            raise ValueError(f"Too few {self.window}-sample windows to train the autoencoder ({len(starts)}).")
        n_val = max(1, int(len(starts) * VAL_FRAC))
        train_w = torch.from_numpy(_windows(Xs, starts[:-n_val], self.window))
        val_w = torch.from_numpy(_windows(Xs, starts[-n_val:], self.window))

        torch.manual_seed(self.seed)
        self.model_ = LSTMAutoencoder(X.shape[1], self.hidden)
        opt = torch.optim.Adam(self.model_.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, self.history_ = float("inf"), None, []
        for _ in range(self.epochs):
            self.model_.train()
            order = torch.randperm(len(train_w), generator=gen)
            train_loss = 0.0
            for i in range(0, len(order), BATCH_SIZE):
                batch = train_w[order[i:i + BATCH_SIZE]]
                opt.zero_grad()
                loss = loss_fn(self.model_(batch), batch)
                loss.backward()
                opt.step()
                train_loss += loss.item() * len(batch)
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(val_w), val_w))
            self.history_.append((train_loss / len(train_w), val_loss))
            if val_loss < best:
                best, best_state = val_loss, copy.deepcopy(self.model_.state_dict())
        self.model_.load_state_dict(best_state)          # best-validation epoch
        self.model_.eval()
        return self

    def score(self, X: np.ndarray, laps: Optional[np.ndarray] = None) -> np.ndarray:
        """Per-sample reconstruction error, averaged over the windows containing the sample.
        Samples in a lap too short for one window score 0 (never flagged)."""
        laps = np.zeros(len(X)) if laps is None else np.asarray(laps)
        Xs = ((X - self.mean_) / self.std_).astype(np.float32)
        starts = window_starts(laps, self.window, 1)
        total, count = np.zeros(len(X)), np.zeros(len(X))
        if len(starts) == 0:
            return total
        with torch.no_grad():
            for i in range(0, len(starts), 4096):
                s = starts[i:i + 4096]
                w = torch.from_numpy(_windows(Xs, s, self.window))
                err = ((self.model_(w) - w) ** 2).mean(dim=2).numpy()        # (windows, window)
                rows = (s[:, None] + np.arange(self.window)[None, :]).ravel()
                np.add.at(total, rows, err.ravel())
                np.add.at(count, rows, 1)
        return np.divide(total, count, out=np.zeros(len(X)), where=count > 0)
