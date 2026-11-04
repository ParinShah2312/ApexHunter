"""Ridge regression baseline for the tyre pace model (ROADMAP §C step 5).

Same data, windows, target and race split as the LSTM: each 5-lap window (LAP_FEATURES, lap time and
telemetry means relative to the window's last lap) is flattened to 45 inputs and the model predicts
the next lap's change in lap time (s). Features are standardised inside the pipeline, so the scaler
only ever sees training races. Alpha is chosen on the most recent training races (validation).
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lstm.tyre_data import LAP_FEATURES, SEQUENCE_LENGTH, TARGET
from lstm.tyre_model import split_by_groups

# Validation MSE on 2023-24 falls up to ~3e4 and rises after it (the inputs are noisy, so strong
# shrinkage wins); the first grid stopped at 1e4 and picked its edge.
ALPHAS: List[float] = [1.0, 10.0, 100.0, 1e3, 3e3, 1e4, 3e4, 1e5, 1e6]
VAL_SPLIT: float = 0.2          # most recent 20 % of the training races, as for the LSTM
MODEL_FILENAME: str = "tyre_ridge.pkl"
CONFIG_FILENAME: str = "tyre_ridge_config.json"


def flatten(X: np.ndarray) -> np.ndarray:
    """(N, seq_len, n_features) windows -> (N, seq_len * n_features) rows for Ridge."""
    return X.reshape(len(X), -1)


def make_ridge(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def regression_metrics(change_true: np.ndarray, change_pred: np.ndarray,
                       last_lap_time: np.ndarray) -> Dict[str, float]:
    """MAE, RMSE and R² of the predicted lap-time change (s), and MAPE (%) of the predicted lap time.

    MAE and RMSE are the same for the change and the lap time (lap time = last lap + change). R² is
    on the change: on absolute lap times it is ~0.99 for any model, because it mostly measures the
    differences between tracks. MAPE is on the lap time (a % of a change near 0 is meaningless).
    """
    err = change_pred - change_true
    ss_tot = float(np.sum((change_true - change_true.mean()) ** 2))
    lap_time = last_lap_time + change_true
    return {
        "mae_s": float(np.mean(np.abs(err))),
        "rmse_s": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "mape_pct": float(100.0 * np.mean(np.abs(err) / lap_time)),
    }


def fit_ridge(X_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray,
              alphas: Optional[List[float]] = None) -> Tuple[Pipeline, float, Dict[float, float]]:
    """Choose alpha by validation MSE on the most recent training races, then refit on all of them.

    Returns (model fitted on all training races, best alpha, {alpha: validation MSE}).
    """
    alphas = alphas or ALPHAS
    fit_idx, val_idx = split_by_groups(len(X_train), groups_train, VAL_SPLIT)
    Xf = flatten(X_train)
    val_mse = {}
    for a in alphas:
        m = make_ridge(a).fit(Xf[fit_idx], y_train[fit_idx])
        val_mse[a] = float(np.mean((m.predict(Xf[val_idx]) - y_train[val_idx]) ** 2))
    best = min(val_mse, key=val_mse.get)
    return make_ridge(best).fit(Xf, y_train), best, val_mse


def top_coefficients(model: Pipeline, n: int = 8) -> List[Tuple[str, float]]:
    """Largest standardised coefficients as ("feature @ lap -k", coef); lap -1 = window's last lap."""
    names = [f"{f} @ lap -{SEQUENCE_LENGTH - i}" for i in range(SEQUENCE_LENGTH) for f in LAP_FEATURES]
    coef = model.named_steps["ridge"].coef_
    order = np.argsort(-np.abs(coef))[:n]
    return [(names[i], float(coef[i])) for i in order]


def save_ridge(model: Pipeline, config: dict, models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / MODEL_FILENAME, "wb") as f:
        pickle.dump(model, f)
    with open(models_dir / CONFIG_FILENAME, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_ridge(models_dir: Path) -> Tuple[Pipeline, dict]:
    """Load the saved Ridge model; raises ValueError if it was trained on other features/target."""
    with open(models_dir / CONFIG_FILENAME, "r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("features") != LAP_FEATURES or config.get("target") != TARGET:
        raise ValueError(
            f"Saved Ridge model was trained on features {config.get('features')} and target "
            f"{config.get('target')}, but the code now uses {LAP_FEATURES} and {TARGET}. "
            f"Retrain with: python backend/scripts/train_ridge.py")
    with open(models_dir / MODEL_FILENAME, "rb") as f:
        model = pickle.load(f)
    return model, config
