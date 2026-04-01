"""Orchestrator for Isolation Forest anomaly detection on F1 telemetry data."""

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

from utils import setup_logger, DATA_LAKE_DIR
from mistakes_features import FEATURE_COLUMNS, engineer_features
from mistakes_model import run_grid_search, train_final_model, run_inference, CONTAMINATION_VALUES
from mistakes_io import load_and_validate, select_reference_driver, build_meta, save_outputs

logger = setup_logger(__name__)


def run_pipeline(session_path: str, driver: str, reference_path: str | None,
                 output_dir: Path, force: bool) -> None:
    """End-to-end mistake-detection pipeline."""
    input_stem = Path(session_path).stem
    output_stem = f"{input_stem}_{driver}"
    parquet_out = output_dir / f"{output_stem}_mistakes.parquet"
    if parquet_out.exists() and not force:
        logger.info("Already processed. Use --force to re-run.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        df_session, df_driver = load_and_validate(Path(session_path), driver, logger)
    except ValueError:
        sys.exit(1)
    if reference_path is not None:
        try:
            df_ref_session = pd.read_parquet(reference_path)
        except Exception as e:
            logger.error(f"Failed to load reference file: {e}")
            sys.exit(1)
    else:
        df_ref_session = df_session
    ref_code = select_reference_driver(df_ref_session, driver, logger)
    df_ref = df_ref_session[df_ref_session["Driver"] == ref_code].copy()
    del df_session; gc.collect()
    for d in (df_driver, df_ref):
        d.sort_values("SessionTime", inplace=True)
        d.reset_index(drop=True, inplace=True)
    feat_drv, feat_ref = engineer_features(df_driver), engineer_features(df_ref)
    gc.collect()
    X_train = pd.concat([feat_ref, feat_drv], ignore_index=True).values
    best_c, cv_scores, best_cv = run_grid_search(X_train, logger)
    del X_train; gc.collect()
    X_final = pd.concat([feat_ref, feat_drv], ignore_index=True).values
    model = train_final_model(X_final, best_c, logger)
    del X_final; gc.collect()
    feat_tgt = engineer_features(df_driver)
    raw_scores, preds = run_inference(model, feat_tgt)
    df_driver["anomaly_score"] = raw_scores.astype("float32")
    df_driver["is_mistake"] = (preds == -1).astype(bool)
    for col in FEATURE_COLUMNS:
        df_driver[col] = feat_tgt[col].astype("float32")
    del feat_tgt, feat_drv, feat_ref; gc.collect()
    ref_str = str(reference_path) if reference_path is not None else str(session_path)
    meta = build_meta(str(session_path), driver, ref_code, ref_str, best_c, cv_scores, best_cv, df_driver)
    pq_path, js_path = save_outputs(df_driver, meta, output_dir, output_stem, logger)
    gc.collect()
    logger.info(f"Done: {meta['total_mistakes']}/{meta['total_rows']} mistakes ({meta['mistake_rate_pct']:.1f}%)")


def main() -> None:
    """Parse CLI arguments and launch the pipeline."""
    p = argparse.ArgumentParser(description="Detect driving mistakes via Isolation Forest.")
    p.add_argument("--session", type=str, required=True, help="Cleaned parquet session file.")
    p.add_argument("--driver", type=str, required=True, help="Driver code (e.g. '1' or '44').")
    p.add_argument("--reference", type=str, default=None, help="Optional reference parquet file.")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    p.add_argument("--force", action="store_true", default=False, help="Overwrite existing output.")
    args = p.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else DATA_LAKE_DIR / "mistake_data"
    try:
        run_pipeline(args.session, args.driver, args.reference, output_dir, args.force)
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
