"""Prediction orchestrator for the ApexHunter LSTM tyre cliff predictor.
Loads a trained model and predicts cliff laps for a session."""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import setup_logger, DATA_LAKE_DIR, PROJECT_ROOT
from tyre_data import extract_stints, CLIFF_SPEED_DROP_KMH, LAP_FEATURES
from tyre_model import INPUT_SIZE, monte_carlo_predict
from tyre_io import load_model_artifacts, build_prediction_output, save_prediction

logger = setup_logger(__name__)


def main() -> None:
    """Parse CLI arguments and run the prediction pipeline."""
    p = argparse.ArgumentParser(description="Predict tyre cliff laps for a session.")
    p.add_argument("--session", type=str, required=True, help="Path to a clean parquet file.")
    p.add_argument("--driver", type=str, required=True, help="Driver code string.")
    p.add_argument("--models-dir", type=str, default=None, help="Path to model artifacts.")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory for predictions.")
    p.add_argument("--force", action="store_true", default=False, help="Overwrite existing output.")
    args = p.parse_args()

    models_dir = Path(args.models_dir) if args.models_dir else PROJECT_ROOT / "models"
    output_dir = Path(args.output_dir) if args.output_dir else DATA_LAKE_DIR / "tyre_predictions"
    stem = Path(args.session).stem
    output_path = output_dir / f"{stem}_{args.driver}_tyre.json"

    if output_path.exists() and not args.force:
        logger.info("Already processed. Use --force to re-run.")
        return

    try:
        model, scaler, config = load_model_artifacts(models_dir, logger)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    y_mean, y_std = config["y_mean"], config["y_std"]
    seq_len = config["sequence_length"]

    try:
        parts = stem.split("_")
        year, round_num, session_type = int(parts[0]), int(parts[1]), parts[2]
        import fastf1
        cache_dir = PROJECT_ROOT / "data_lake" / "cache"
        cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        session_f1 = fastf1.get_session(year, round_num, session_type)
        session_f1.load(telemetry=False, weather=False)
        f1_laps = session_f1.laps
    except Exception as e:
        logger.error(f"Failed to load fastf1 laps: {e}")
        sys.exit(1)

    try:
        df = pd.read_parquet(args.session)
        df_driver = df[df["Driver"] == args.driver].copy()
        if df_driver.empty:
            raise ValueError(f"No data for driver '{args.driver}' in {args.session}")
        df_driver.sort_values("SessionTime", inplace=True)
        df_driver.reset_index(drop=True, inplace=True)
        driver_laps = f1_laps.pick_drivers(args.driver)
        if len(driver_laps) == 0:
            raise ValueError(f"No fastf1 laps for driver '{args.driver}'")
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)

    stints_data = extract_stints(driver_laps, df_driver, logger)
    if len(stints_data) == 0:
        logger.error("No stints detected.")
        sys.exit(1)

    stint_results = []
    for stint_data in stints_data:
        stint_index = stint_data["stint_index"]
        lap_features = pd.DataFrame(stint_data["lap_features"])
        actual_speeds = lap_features["mean_speed"].tolist()
        actual_lap_times = lap_features["lap_time_seconds"].tolist()
        
        features_np = lap_features[LAP_FEATURES].values.copy()
        
        # The first lap is an out-lap (slow). If we feed this slow lap into the LSTM history,
        # it falsely predicts a cliff. We overwrite lap 1 with lap 2's features for the input sequence.
        if len(features_np) > 1:
            features_np[0] = features_np[1]
        
        # Pad left with seq_len copies of the first row (now lap 2) to predict all laps smoothly
        padded_features = np.vstack([np.tile(features_np[0], (seq_len, 1)), features_np])
        
        # Build padded sequences
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

        # Compute track distance for this stint to convert speed back to lap time accurately
        valid_pairs = [(s, t) for s, t in zip(actual_speeds, actual_lap_times) if not pd.isna(t) and t > 0]
        if valid_pairs:
            track_dist = float(np.mean([s * t for s, t in valid_pairs]))
        else:
            track_dist = actual_speeds[0] * 90.0

        def speed_to_time(s): return track_dist / max(s, 1.0)

        predicted_laps = [speed_to_time(s) for s in mean_preds_speed]
        
        # Note: upper speed = lower time. We map to confidence_lower and confidence_upper for UI
        confidence_lower = [speed_to_time(s) for s in upper_speed]
        confidence_upper = [speed_to_time(s) for s in lower_speed]
        
        # Clean actual lap times replacing NaN with None
        actual_laps_clean = [t if not pd.isna(t) else None for t in actual_lap_times]
        
        # Detect cliff based on lap time
        best_lap_time = min(predicted_laps)
        best_lap_idx = predicted_laps.index(best_lap_time)
        cliff_lap = None
        # Start looking for the cliff AFTER the best lap is set, since the first laps are slow out-laps
        for i in range(best_lap_idx + 1, len(predicted_laps)):
            if predicted_laps[i] > best_lap_time + 1.5:  # 1.5s drop-off
                cliff_lap = i
                break

        laps_remaining = (len(actual_lap_times) - cliff_lap) if cliff_lap is not None else None

        stint_results.append({
            "stint_index": stint_index, "n_laps": len(actual_lap_times),
            "actual_laps": actual_laps_clean, "predicted_laps": predicted_laps,
            "confidence_upper": confidence_upper, "confidence_lower": confidence_lower,
            "cliff_lap": cliff_lap, "laps_remaining": laps_remaining,
        })

    data = build_prediction_output(str(args.session), args.driver, stint_results)
    save_prediction(data, output_path, logger)
    gc.collect()

    stint_lines = "\n".join(
        f"   Stint {s['stint_index']+1}: {s['n_laps']} laps, "
        f"cliff at lap {s['cliff_lap']+1 if s['cliff_lap'] is not None else '—'}, "
        f"{s['laps_remaining'] if s['laps_remaining'] is not None else '—'} laps remaining"
        for s in stint_results
    )
    logger.info(f"""
======================================================
   ApexHunter — Tyre Cliff Prediction Complete
======================================================
   Driver        : {args.driver}
   Session       : {stem}
   Stints found  : {len(stint_results)}
{stint_lines}
======================================================
   Output: {output_path}
======================================================""")


if __name__ == "__main__":
    main()
