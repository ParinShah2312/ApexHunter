"""Orchestrator: Runs YOLOv11-Seg on F1 video, outputs HUD video + metrics CSV."""

import sys
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from utils import setup_logger, DATA_LAKE_DIR, PROJECT_ROOT, CONFIG
from inference_geometry import classify_apex_status, compute_wheel_positions
from inference_masking import process_masks
from inference_hud import draw_hud
from inference_io import open_video, create_video_writer, create_csv_writer, write_csv_row

logger = setup_logger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics is not installed. Run: pip install ultralytics")
    sys.exit(1)

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_INPUT_VIDEO = DATA_LAKE_DIR / "edited_videos" / "2024" / "01_bahrain_ver_pole - Trim.mp4"
OUTPUT_DIR = DATA_LAKE_DIR / "processed_video"
OUTPUT_CSV_DIR = DATA_LAKE_DIR / "processed_csv"
ALPHA = CONFIG.get("inference", {}).get("alpha", 0.5)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)


def process_video(input_video_path: Path, force: bool = False) -> None:
    """Run YOLO inference on a video and produce HUD video + metrics CSV."""
    if not MODEL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PATH}"); sys.exit(1)
    if not input_video_path.exists():
        logger.error(f"Video not found: {input_video_path}"); sys.exit(1)
    stem = input_video_path.stem
    vid_out = OUTPUT_DIR / f"{stem}_HUD.mp4"
    csv_out = OUTPUT_CSV_DIR / f"{stem}_metrics.csv"
    if vid_out.exists() and csv_out.exists() and not force:
        logger.info("Already processed. Use --force to re-run."); return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH).to(device)
    cap, w, h, fps, total = open_video(input_video_path)
    out = create_video_writer(vid_out, fps, w, h)
    csv_fh, csv_w = create_csv_writer(csv_out)
    lw, rw = compute_wheel_positions(w, h)
    cx = w // 2
    for idx in tqdm(range(total), desc="Processing", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        res = model.predict(frame, conf=0.25, verbose=False)[0]
        hud, dist, cp, dw, curb, turn = process_masks(res, frame, lw, rw, cx)
        status, color = classify_apex_status(dist, curb)
        ds = "N/A" if not curb or dist == float('inf') else f"{int(dist)}px"
        write_csv_row(csv_w, idx, fps, ds, status, curb)
        out.write(draw_hud(frame, hud, lw, rw, cp, dw, curb, status, color, ds, turn, ALPHA))
    cap.release(); out.release(); csv_fh.close()
    logger.info(f"Done — Video: {vid_out.resolve()}, CSV: {csv_out.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run YOLO inference on F1 pole lap video.")
    p.add_argument('--input', type=str, default=str(DEFAULT_INPUT_VIDEO), help="Input video path.")
    p.add_argument('--force', action='store_true', default=False, help="Re-run if output exists.")
    a = p.parse_args()
    process_video(Path(a.input), force=a.force)
