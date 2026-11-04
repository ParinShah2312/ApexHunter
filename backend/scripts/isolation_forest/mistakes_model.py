"""Isolation Forest training and model selection for ApexHunter mistake detection.

Settings are chosen with the planted-mistake test (mistakes_eval): lap-grouped folds, fake
mistakes planted in the held-out laps, and each setting scored by how well it finds them.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from isolation_forest.mistakes_eval import build_folds, fold_metrics, summarise

# ── Model constants ───────────────────────────────────────────────────────────
# Share of samples flagged. Fixed at 2 % for every session (user decision, Oct 2026) so mistake
# counts compare across drivers and races; best F1 on 2 of 3 races checked. The other values are
# still evaluated and reported in the meta "search" table (the precision/recall trade-off).
CONTAMINATION: float = 0.02
CONTAMINATION_VALUES: List[float] = [0.005, 0.01, 0.02, 0.03, 0.05]
# Rows each tree is built from: an int = that many rows (capped at the data size), a float = that
# fraction of the rows. 1.0 = all rows, fixed for every session (user decision, Oct 2026): in the
# 256-65536 sweep (ISOLATION_FOREST_REWORK.md §6) all rows was best or within 0.001 PR-AUC on every
# run, with no sign of overfitting. Still tunable: put several values here and the search picks one.
MAX_SAMPLES_VALUES: List[float] = [1.0]
MAX_FEATURES_VALUES: List[float] = [1.0, 0.6]
N_FOLDS: int = 5
INJECTION_SEEDS: Tuple[int, ...] = (0, 1, 2)
N_ESTIMATORS: int = 200
RANDOM_STATE: int = 42


def fit_forest(X: np.ndarray, max_samples: float, max_features: float, contamination: float = 0.01) -> IsolationForest:
    """Fit an IsolationForest (an int max_samples is capped at the number of rows)."""
    return IsolationForest(
        n_estimators=N_ESTIMATORS, max_samples=max_samples if isinstance(max_samples, float) else min(max_samples, len(X)), max_features=max_features,
        contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X)


def anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous (the negated IsolationForest score_samples)."""
    return -model.score_samples(X)


class ForestDetector:
    """Isolation Forest with fixed settings, in the fit/score form used by mistakes_eval."""

    def __init__(self, max_samples: float, max_features: float):
        self.max_samples, self.max_features = max_samples, max_features

    def fit(self, X: np.ndarray, laps=None) -> "ForestDetector":
        self.model_ = fit_forest(X, self.max_samples, self.max_features)
        return self

    def score(self, X: np.ndarray, laps=None) -> np.ndarray:
        return anomaly_scores(self.model_, X)


def evaluate_settings(df: pd.DataFrame, profile: pd.DataFrame, logger: logging.Logger) -> Dict[str, object]:
    """Planted-mistake model selection.

    For every (max_samples, max_features) pair and every lap-grouped fold: fit on the clean
    training laps, plant mistakes in the held-out laps (several seeds) and score them per mistake
    (mistakes_eval.mistake_metrics). The pair with the best mean F1 at CONTAMINATION wins; every
    value in CONTAMINATION_VALUES is still scored for the report. The threshold is the
    (1 - contamination) quantile of the training-lap scores, which is what
    IsolationForest(contamination=c).predict uses.

    Args:
        df: The driver's kept-lap samples with LapNumber, SessionTime, track_pos, accel and raw channels.
        profile: Field reference profile (mistakes_features.build_reference_profile).

    Returns:
        Dict with the chosen settings, the search table, the metrics of the chosen setting
        (mean over folds and seeds, plus recall per mistake type and the summed counts)
        and the planted folds themselves (reused by the detector comparison; not saved).
    """
    # Planted test sets are the same for every setting: build them once per fold and seed
    folds = build_folds(df, profile, N_FOLDS, INJECTION_SEEDS)

    table = []
    for ms in MAX_SAMPLES_VALUES:
        for mf in MAX_FEATURES_VALUES:
            per_c: Dict[float, List[dict]] = {c: [] for c in CONTAMINATION_VALUES}
            for X_train, tests, _ in folds:
                det = ForestDetector(ms, mf).fit(X_train)
                train_scores = det.score(X_train)
                for test in tests:
                    test_scores = det.score(test[0])
                    for c in CONTAMINATION_VALUES:
                        per_c[c].append(fold_metrics(train_scores, test_scores, test, c))
            summaries = {c: summarise(ms_list) for c, ms_list in per_c.items()}
            for c, summary in summaries.items():
                table.append({"max_samples": ms, "max_features": mf, "contamination": c, **summary})
            logger.info(f"max_samples={ms}, max_features={mf}: per-mistake F1={summaries[CONTAMINATION]['f1']:.3f}")

    candidates = [r for r in table if r["contamination"] == CONTAMINATION]
    best = max(candidates, key=lambda r: r["f1"])
    logger.info(f"Selected max_samples={best['max_samples']}, max_features={best['max_features']}, "
                f"contamination={best['contamination']:.3f}: per-mistake precision={best['precision']:.3f}, "
                f"recall={best['recall']:.3f}, F1={best['f1']:.3f} "
                f"({best['counts']['found']} of {best['counts']['planted']} planted mistakes found)")
    return {"best": best, "search": [{k: v for k, v in r.items() if k not in ("recall_by_type", "counts")}
                                     for r in table],
            "n_folds": len(folds), "injection_seeds": list(INJECTION_SEEDS), "folds": folds}


def train_final_model(X: np.ndarray, best: dict, logger: logging.Logger) -> IsolationForest:
    """Fit the final IsolationForest on all of the driver's kept laps with the chosen settings."""
    model = fit_forest(X, best["max_samples"], best["max_features"], best["contamination"])
    logger.info(f"Final model fitted on {len(X)} rows")
    return model


def run_inference(model: IsolationForest, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Run decision_function and predict on the features.

    Returns:
        (raw_scores, predictions): decision_function values (negative = anomalous) and
        +1 (normal) / -1 (anomaly).
    """
    return model.decision_function(features.values), model.predict(features.values)
