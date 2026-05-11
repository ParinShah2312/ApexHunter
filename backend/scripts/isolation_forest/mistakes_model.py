"""Isolation Forest training and grid search for ApexHunter mistake detection.
Handles cross-validated contamination selection and final model fitting.
"""

import gc
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold

# ── Model constants ───────────────────────────────────────────────────────────
CONTAMINATION_VALUES: list = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
N_FOLDS: int = 5
N_ESTIMATORS: int = 200
RANDOM_STATE: int = 42


def run_grid_search(
    X_train: np.ndarray,
    logger: logging.Logger,
) -> Tuple[float, Dict[str, float], float]:
    """Manual grid search over contamination values using K-Fold CV.

    For each contamination value, trains IsolationForest models on K folds
    and scores each fold by the mean absolute decision-function value on
    the held-out set.  Higher absolute scores → better class separation.

    Args:
        X_train: 2-D numpy array of shape (n_samples, n_features).
        logger: Logger instance for progress messages.

    Returns:
        Tuple of (best_contamination, cv_scores_dict, best_cv_score).
        cv_scores_dict maps str(contamination) → mean CV score for all 6 values.
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores: Dict[str, float] = {}

    for c in CONTAMINATION_VALUES:
        fold_scores: List[float] = []
        for train_idx, val_idx in kf.split(X_train):
            X_fold_train = X_train[train_idx]
            X_fold_val = X_train[val_idx]

            model = IsolationForest(
                contamination=c,
                random_state=RANDOM_STATE,
                n_estimators=N_ESTIMATORS,
            )
            model.fit(X_fold_train)

            decision_vals = model.decision_function(X_fold_val)
            fold_score = float(np.mean(np.abs(decision_vals)))
            fold_scores.append(fold_score)

        mean_score = float(np.mean(fold_scores))
        cv_scores[str(c)] = mean_score
        logger.info(f"contamination={c:.2f} → mean CV score={mean_score:.5f}")

    best_contamination = CONTAMINATION_VALUES[
        int(np.argmax([cv_scores[str(c)] for c in CONTAMINATION_VALUES]))
    ]
    best_cv_score = cv_scores[str(best_contamination)]
    logger.info(f"Best contamination: {best_contamination:.2f}")

    gc.collect()

    return best_contamination, cv_scores, best_cv_score


def train_final_model(
    X_train: np.ndarray,
    best_contamination: float,
    logger: logging.Logger,
) -> IsolationForest:
    """Fit and return the final IsolationForest model.

    Args:
        X_train: 2-D numpy array of training data.
        best_contamination: The contamination hyperparameter to use.
        logger: Logger instance for progress messages.

    Returns:
        A fitted IsolationForest model.
    """
    final_model = IsolationForest(
        contamination=best_contamination,
        n_estimators=N_ESTIMATORS,
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_model.fit(X_train)
    row_count = len(X_train)
    logger.info(f"Final model fitted on {row_count} rows")

    return final_model


def run_inference(
    model: IsolationForest,
    features: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run decision_function and predict on the features.

    Args:
        model: A fitted IsolationForest.
        features: DataFrame of engineered feature columns.

    Returns:
        Tuple of (raw_scores, predictions). raw_scores are floats,
        predictions are +1 (normal) or -1 (anomaly).
    """
    raw_scores = model.decision_function(features.values)
    predictions = model.predict(features.values)
    return raw_scores, predictions
