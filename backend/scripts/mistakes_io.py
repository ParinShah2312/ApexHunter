"""Data loading, validation, and output writing for the ApexHunter
Isolation Forest pipeline. All file I/O operations are isolated here.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from mistakes_features import FEATURE_COLUMNS
from mistakes_model import N_ESTIMATORS

# ── Required columns for session parquet files ────────────────────────────────
REQUIRED_COLUMNS: List[str] = [
    "Driver", "Speed", "RPM", "Throttle", "Brake",
    "X", "Y", "SessionTime", "nGear",
]


def load_and_validate(
    session_path: Path,
    driver: str,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load a session parquet, validate columns, and filter to the target driver.

    Args:
        session_path: Path to the cleaned session parquet file.
        driver: Driver code string (e.g. "1", "44").
        logger: Logger instance for messages.

    Returns:
        Tuple of (df_session_full, df_driver).

    Raises:
        ValueError: If required columns are missing or the driver is not found.
    """
    logger.info(f"Loading session file: {session_path}")
    try:
        df_session = pd.read_parquet(session_path)
    except Exception as e:
        raise ValueError(f"Failed to load session file: {e}") from e

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df_session.columns]
    if missing_cols:
        for col in missing_cols:
            logger.error(f"Missing required column: {col}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    df_driver = df_session[df_session["Driver"] == driver].copy()
    if df_driver.empty:
        logger.error(f"Driver '{driver}' not found in session file.")
        raise ValueError(f"Driver '{driver}' not found in session file.")

    row_count = len(df_driver)
    filename = Path(session_path).name
    logger.info(f"Loaded {row_count} rows for driver {driver} from {filename}")

    return df_session, df_driver


def select_reference_driver(
    df_session: pd.DataFrame,
    target_driver: str,
    logger: logging.Logger,
) -> str:
    """Select the fastest driver by mean Speed as the reference baseline.

    Args:
        df_session: Full session DataFrame with all drivers.
        target_driver: The target driver code to compare against.
        logger: Logger instance.

    Returns:
        The driver code string of the reference driver.
    """
    mean_speeds = df_session.groupby("Driver")["Speed"].mean()
    ref_driver_code = str(mean_speeds.idxmax())
    ref_mean_speed = float(mean_speeds.max())

    if ref_driver_code == target_driver:
        logger.warning(
            "Reference driver is the same as target driver — "
            "anomalies will be relative to the driver's own baseline."
        )

    logger.info(f"Reference driver: {ref_driver_code} with mean speed {ref_mean_speed:.1f} km/h")
    return ref_driver_code


def build_meta(
    session_file: str,
    driver: str,
    ref_driver: str,
    ref_file: str,
    best_contamination: float,
    cv_scores: dict,
    best_cv_score: float,
    df_annotated: pd.DataFrame,
) -> dict:
    """Construct the metadata dictionary for a completed run.

    Args:
        session_file: Path to the input session file as string.
        driver: Target driver code.
        ref_driver: Reference driver code.
        ref_file: Path to the reference file as string.
        best_contamination: Selected contamination hyperparameter.
        cv_scores: Dict mapping str(contamination) → mean CV score.
        best_cv_score: Score at the best contamination.
        df_annotated: The annotated driver DataFrame.

    Returns:
        Metadata dictionary with all required keys.
    """
    total_rows = len(df_annotated)
    total_mistakes = int(df_annotated["is_mistake"].sum())
    mistake_rate_pct = round(total_mistakes / total_rows * 100, 2) if total_rows > 0 else 0.0

    return {
        "session_file": session_file,
        "driver": driver,
        "reference_driver": ref_driver,
        "reference_file": ref_file,
        "best_contamination": best_contamination,
        "cv_scores": cv_scores,
        "best_cv_score": best_cv_score,
        "total_rows": total_rows,
        "total_mistakes": total_mistakes,
        "mistake_rate_pct": mistake_rate_pct,
        "n_estimators": N_ESTIMATORS,
        "feature_columns": FEATURE_COLUMNS,
        "timestamp": datetime.utcnow().isoformat(),
    }


def save_outputs(
    df_annotated: pd.DataFrame,
    meta: dict,
    output_dir: Path,
    output_stem: str,
    logger: logging.Logger,
) -> Tuple[Path, Path]:
    """Save the annotated parquet and metadata JSON.

    Args:
        df_annotated: The annotated driver DataFrame.
        meta: Metadata dictionary.
        output_dir: Directory to write into.
        output_stem: Base name for the output files.
        logger: Logger instance.

    Returns:
        Tuple of (parquet_path, json_path).
    """
    parquet_path = output_dir / f"{output_stem}_mistakes.parquet"
    json_path = output_dir / f"{output_stem}_mistakes_meta.json"

    df_annotated.to_parquet(parquet_path, compression="snappy")
    logger.info(f"Saved annotated parquet: {parquet_path}")

    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata JSON: {json_path}")

    return parquet_path, json_path
