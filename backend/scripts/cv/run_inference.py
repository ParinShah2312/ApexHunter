"""
================================================================================
  ApexHunter - YOLOv11-Seg Inference Pipeline
  Script: run_inference.py
--------------------------------------------------------------------------------
  Purpose : Runs YOLOv11-Seg on F1 video, outputs HUD video + metrics CSV.

  Usage   : python backend/scripts/cv/run_inference.py --input <video_path> --video-offset <seconds>

            --video-offset is how many seconds into the video the timed lap starts.
            Corner windows come from FastF1 (see cv/extract_turns.py); year, round and
            driver are read from the video's path unless --year/--round/--driver are given.
================================================================================
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

# Make backend/scripts importable so this file runs both by path and as `python -m <pkg>.<module>`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv.extract_turns import build_turn_windows_with_lap, save_turn_windows, turn_context_at
from cv.inference_geometry import classify_apex_status, compute_wheel_positions
from cv.inference_hud import HUDStateTracker, draw_hud
from cv.inference_io import create_csv_writer, create_video_writer, finalize_video, open_video, write_csv_row
from cv.inference_masking import process_masks
from utils import CONFIG, DATA_LAKE_DIR, PROJECT_ROOT, setup_logger

logger = setup_logger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics is not installed. Run: pip install ultralytics")
    sys.exit(1)

MODEL_PATH: Path = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_INPUT_VIDEO: Path = DATA_LAKE_DIR / "edited_videos" / "2024" / "01_bahrain_ver_pole - Trim.mp4"
OUTPUT_DIR: Path = DATA_LAKE_DIR / "processed_video"
OUTPUT_CSV_DIR: Path = DATA_LAKE_DIR / "processed_csv"
ALPHA: float = CONFIG.get("inference", {}).get("alpha", 0.5)


def process_video(
    input_video_path: Path,
    video_offset_s: float,
    force: bool = False,
    year: Optional[int] = None,
    round_num: Optional[int] = None,
    driver: Optional[str] = None,
) -> None:
    """Run YOLO inference on a video and produce HUD video, metrics CSV and turn-window JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PATH}"); sys.exit(1)
    if not input_video_path.exists():
        logger.error(f"Video not found: {input_video_path}"); sys.exit(1)
    stem = input_video_path.stem
    vid_out = OUTPUT_DIR / f"{stem}_HUD.mp4"
    csv_out = OUTPUT_CSV_DIR / f"{stem}_metrics.csv"
    turns_out = OUTPUT_CSV_DIR / f"{stem}_turns.json"
    outputs = (vid_out, csv_out, turns_out)
    if all(o.exists() for o in outputs) and not force:
        logger.info("Already processed. Use --force to re-run."); return
    turn_windows, lap_info = build_turn_windows_with_lap(video_offset_s, input_video_path, year, round_num, driver)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH).to(device)
    cap, w, h, fps, total = open_video(input_video_path)
    out = create_video_writer(vid_out, fps, w, h)
    csv_fh, csv_w = create_csv_writer(csv_out)
    try:
        save_turn_windows(turn_windows, turns_out, video_offset_s, lap_info)
        _process_frames(cap, out, csv_w, model, turn_windows, w, h, fps, total)
    except BaseException:
        # A crashed or interrupted run must not leave files that later count as "already processed"
        cap.release(); out.release(); csv_fh.close()
        for o in outputs:
            o.unlink(missing_ok=True)
        logger.error("Inference stopped; partial outputs removed.")
        raise
    cap.release(); out.release(); csv_fh.close()
    finalize_video(vid_out)
    logger.info(f"Done — Video: {vid_out.resolve()}, CSV: {csv_out.resolve()}")


def _process_frames(cap, out, csv_w, model, turn_windows, w: int, h: int, fps: float, total: int) -> None:
    """Run YOLO + geometry on every frame, writing the HUD video and the metrics CSV."""
    lw, rw = compute_wheel_positions(w, h)
    cx = w // 2
    tracker = HUDStateTracker(alpha=0.15)

    for idx in tqdm(range(total), desc="Processing", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = float(idx) / float(fps) if fps > 0 else 0.0
        expected_turn, speed_cat = turn_context_at(timestamp_sec, turn_windows)

        # retina_masks: full-resolution masks with the letterbox padding removed. Without it the
        # 384x640 network mask was stretched over the frame, shifting edges by up to ~29 px.
        res = model.predict(frame, conf=0.25, verbose=False, retina_masks=True)[0]
        hud, dist, cp, dw, curb, turn = process_masks(res, frame, lw, rw, cx, expected_turn, speed_cat)

        # Phase 1: Apply Temporal Smoothing to visual and data elements
        cp, dist = tracker.update(cp, dist, curb)

        status, color = classify_apex_status(dist, curb, expected_turn, speed_cat)

        PIXELS_TO_CM = 0.155
        ds = "N/A" if not curb or dist == float('inf') else f"{int(dist * PIXELS_TO_CM)}cm"
        write_csv_row(csv_w, idx, fps, ds, status, curb)
        out.write(draw_hud(frame, hud, lw, rw, cp, dw, curb, status, color, ds, turn, ALPHA, idx))


def main() -> None:
    """Parse CLI arguments and run YOLO inference."""
    p = argparse.ArgumentParser(description="Run YOLO inference on F1 pole lap video.")
    p.add_argument("--input", type=str, default=str(DEFAULT_INPUT_VIDEO), help="Input video path.")
    p.add_argument("--video-offset", type=float, required=True,
                   help="Seconds into the video at which the timed lap starts (e.g. 4.9 for 2024 Bahrain).")
    p.add_argument("--year", type=int, default=None, help="Override the year inferred from the video path.")
    p.add_argument("--round", type=int, default=None, help="Override the round inferred from the video path.")
    p.add_argument("--driver", type=str, default=None, help="Override the driver abbreviation (e.g. VER).")
    p.add_argument("--force", action="store_true", default=False, help="Re-run if output exists.")
    a = p.parse_args()
    process_video(Path(a.input), a.video_offset, force=a.force, year=a.year, round_num=a.round, driver=a.driver)


if __name__ == "__main__":
    main()
