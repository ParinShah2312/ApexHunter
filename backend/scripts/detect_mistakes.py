"""
================================================================================
  ApexHunter - Isolation Forest Anomaly Detection
  Script: detect_mistakes.py
--------------------------------------------------------------------------------
  Purpose : Orchestrator for Isolation Forest anomaly detection on F1
            telemetry data. The driver's representative laps are compared with
            the 5 fastest other drivers at the same point on track; in races,
            samples with DRS open or within 1 s of the car ahead are left out.
            Each event gets the time it cost vs the driver's own nearby laps.

  Usage   : python backend/scripts/detect_mistakes.py --session <path> --driver <code>
================================================================================
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from isolation_forest.mistakes_features import (
    FEATURE_COLUMNS, INFO_COLUMNS, build_reference_profile, build_track_line, compute_acceleration,
    engineer_features, project_to_track,
)
from isolation_forest.mistakes_io import (
    build_meta, corner_positions, load_and_validate, load_fastf1_session, save_outputs, select_laps,
    select_reference_drivers, slice_laps, traffic_mask,
)
from isolation_forest.mistakes_eval import ZScoreDetector, compare_detectors
from isolation_forest.mistakes_events import assign_events, build_events, event_overlap, location_label
from isolation_forest.mistakes_timeloss import (
    BASELINE_LAPS, add_time_lost, build_lap_timing, costly_share, event_windows, random_windows,
)
from isolation_forest.mistakes_model import (
    CONTAMINATION, ForestDetector, evaluate_settings, run_inference, train_final_model,
)
from utils import DATA_LAKE_DIR, setup_logger

logger = setup_logger(__name__)


def driver_lap_samples(df_session: pd.DataFrame, laps: pd.DataFrame, driver: str) -> pd.DataFrame:
    """One driver's samples on the selected laps, with LapNumber and acceleration."""
    df = df_session[df_session["Driver"] == driver].sort_values("SessionTime").reset_index(drop=True)
    df["accel"] = compute_acceleration(df)                  # before slicing: needs the preceding samples
    return slice_laps(df, laps[laps["DriverNumber"].astype(str) == driver])


def drop_traffic(df: pd.DataFrame, session, driver: str) -> pd.DataFrame:
    """Leave out the samples with DRS open or within 1 s of the car ahead (races only)."""
    return df[~traffic_mask(session, driver, df)].reset_index(drop=True)


def compare_with_baseline(evaluation: dict, X: np.ndarray, df_driver: pd.DataFrame, time_check) -> dict:
    """Z-score baseline vs Isolation Forest on the same planted folds and flag rate, how much the
    two detectors' flags agree on the real laps, and the Z-score events' time-lost check
    (`time_check(event_windows)` -> costly_share result)."""
    best = evaluation["best"]
    detectors = {"z_score": ZScoreDetector,
                 "isolation_forest": lambda: ForestDetector(best["max_samples"], best["max_features"])}
    planted = compare_detectors(evaluation["folds"], detectors, CONTAMINATION)

    z_scores = ZScoreDetector().fit(X).score(X)
    z_flags = z_scores > np.quantile(z_scores, 1 - CONTAMINATION)
    if_flags = df_driver["is_mistake"].to_numpy()
    z_events = assign_events(df_driver.assign(is_mistake=z_flags))
    return {
        "contamination": CONTAMINATION,
        "planted": planted,
        "z_score_time_lost_check": time_check(event_windows(df_driver, z_events)),
        "real_laps": {
            "z_score_flagged_samples": int(z_flags.sum()),
            "z_score_events": int(len(np.unique(z_events[z_events >= 0]))),
            "shared_flagged_samples": int((z_flags & if_flags).sum()),
            "if_events_with_a_z_flag": round(event_overlap(df_driver["event_id"].to_numpy(), z_flags), 3),
            "z_events_with_an_if_flag": round(event_overlap(z_events, if_flags), 3),
        },
    }


def log_comparison(comparison: dict) -> None:
    for name, m in comparison["planted"].items():
        logger.info(f"{name:>16}: per-mistake precision={m['precision']:.3f}, "
                    f"recall={m['recall']:.3f}, F1={m['f1']:.3f}")
    logger.info(f"Z-score events: costly share {comparison['z_score_time_lost_check']['costly_share']}")
    r = comparison["real_laps"]
    logger.info(f"Real laps: Z-score {r['z_score_events']} events; {r['if_events_with_a_z_flag']:.0%} of IF events "
                f"contain a Z-score flag, {r['z_events_with_an_if_flag']:.0%} of Z-score events an IF flag")


def run_pipeline(session_path: str, driver: str, output_dir: Path, force: bool, compare: bool = False) -> None:
    """End-to-end mistake-detection pipeline (compare=True adds the Z-score baseline comparison)."""
    input_stem = Path(session_path).stem
    output_stem = f"{input_stem}_{driver}"
    parquet_out = output_dir / f"{output_stem}_mistakes.parquet"
    if parquet_out.exists() and not force:
        logger.info("Already processed. Use --force to re-run.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        df_session, _ = load_and_validate(Path(session_path), driver, logger)
        session = load_fastf1_session(df_session, logger)
        is_race = str(df_session["Session"].iloc[0]) == "R"
        laps = select_laps(session.laps, is_race=is_race)
        ref_drivers = select_reference_drivers(laps, driver, logger)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    df_driver = driver_lap_samples(df_session, laps, driver)
    if df_driver.empty:
        logger.error(f"Driver {driver} has no representative laps in this session.")
        sys.exit(1)
    df_ref = pd.concat([drop_traffic(driver_lap_samples(df_session, laps, d), session, d) if is_race
                        else driver_lap_samples(df_session, laps, d) for d in ref_drivers], ignore_index=True)
    df_timing = df_driver.copy()                            # all kept-lap samples, for the lap times
    n_traffic = 0
    if is_race:
        n_before = len(df_driver)
        df_driver = drop_traffic(df_driver, session, driver)
        n_traffic = n_before - len(df_driver)
        logger.info(f"Traffic filter: {n_traffic} of {n_before} samples left out (DRS open or < 1 s behind a car)")

    # Track line = the session's fastest selected lap
    fastest = laps.loc[laps["LapTime"].idxmin()]
    fastest_driver = str(fastest["DriverNumber"])
    df_fastest = driver_lap_samples(df_session, laps, fastest_driver)
    line_points, line_s = build_track_line(df_fastest[df_fastest["LapNumber"] == fastest["LapNumber"]])
    corners = corner_positions(session, line_points, line_s)
    del df_session; gc.collect()
    logger.info(f"Track line: driver {fastest_driver} lap {int(fastest['LapNumber'])}, {line_s[-1]:.0f} m")

    for d in (df_driver, df_ref, df_timing):
        d["track_pos"] = project_to_track(d, line_points, line_s).astype("float32")
    lap_length = float(line_s[-1])
    timing = build_lap_timing(df_timing, lap_length)
    driver_laps = laps[laps["DriverNumber"].astype(str) == driver]
    # Qualifying: every run is its own stint, so all kept (push) laps are compared with each other
    stint_of = {int(lap): int(stint) if is_race else 0
                for lap, stint in zip(driver_laps["LapNumber"], driver_laps["Stint"].fillna(0))}
    nearby = BASELINE_LAPS if is_race else None
    profile = build_reference_profile(df_ref, float(line_s[-1]))
    features = engineer_features(df_driver, profile)
    logger.info(f"{df_driver['LapNumber'].nunique()} laps, {len(df_driver)} samples; "
                f"reference: {df_ref['LapNumber'].count()} samples from {len(ref_drivers)} drivers")

    try:
        evaluation = evaluate_settings(df_driver, profile, logger)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    model = train_final_model(features[FEATURE_COLUMNS].to_numpy(), evaluation["best"], logger)
    raw_scores, preds = run_inference(model, features[FEATURE_COLUMNS])
    df_driver = df_driver.drop(columns=["accel"])
    df_driver["anomaly_score"] = raw_scores.astype("float32")
    df_driver["is_mistake"] = (preds == -1).astype(bool)
    for col in FEATURE_COLUMNS + INFO_COLUMNS:
        df_driver[col] = features[col]
    df_driver["event_id"] = assign_events(df_driver)
    events = build_events(df_driver)
    for e in events:
        e["location"] = location_label(e["track_pos_m"], corners, lap_length)
    events = add_time_lost(events, df_driver, timing, stint_of, lap_length, nearby)
    meta = build_meta(str(session_path), driver, ref_drivers, int(df_driver["LapNumber"].nunique()),
                      evaluation, df_driver)
    meta["effective_max_samples"] = int(model.max_samples_)      # rows per tree after capping at the data size
    meta["n_train_rows"] = len(features)
    meta["n_events"] = len(events)
    meta["traffic_samples_removed"] = n_traffic
    meta["flags_outside_events"] = int((df_driver["is_mistake"] & (df_driver["event_id"] < 0)).sum())
    meta["time_lost_check"] = {
        "isolation_forest": costly_share(event_windows(df_driver, df_driver["event_id"].to_numpy()), timing, stint_of, lap_length, nearby),
        "random_spots": costly_share(random_windows(timing, lap_length), timing, stint_of, lap_length, nearby),
    }
    logger.info(f"Time-lost check: costly share {meta['time_lost_check']['isolation_forest']['costly_share']} "
                f"of events vs {meta['time_lost_check']['random_spots']['costly_share']} of random spots")
    if compare:
        meta["comparison"] = compare_with_baseline(evaluation, features[FEATURE_COLUMNS].to_numpy(), df_driver,
                                                   lambda windows: costly_share(windows, timing, stint_of, lap_length, nearby))
        log_comparison(meta["comparison"])
    save_outputs(df_driver, meta, events, output_dir, output_stem, logger)
    logger.info(f"Done: {meta['total_mistakes']}/{meta['total_rows']} samples flagged ({meta['mistake_rate_pct']:.1f}%) in {len(events)} events")


def main() -> None:
    """Parse CLI arguments and launch the pipeline."""
    p = argparse.ArgumentParser(description="Detect driving mistakes via Isolation Forest.")
    p.add_argument("--session", type=str, required=True, help="Cleaned parquet session file.")
    p.add_argument("--driver", type=str, required=True, help="Driver code (e.g. '1' or '44').")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    p.add_argument("--force", action="store_true", default=False, help="Overwrite existing output.")
    p.add_argument("--compare", action="store_true", default=False,
                   help="Also evaluate the Z-score baseline on the same planted mistakes (saved in the meta file).")
    args = p.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else DATA_LAKE_DIR / "mistake_data"
    try:
        run_pipeline(args.session, args.driver, output_dir, args.force, args.compare)
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
