"""Video I/O and CSV writing for the ApexHunter CV inference pipeline.
Handles VideoCapture setup, VideoWriter setup, and CSV row writing.
"""

import csv
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Tuple

import cv2

logger = logging.getLogger(__name__)


def open_video(input_path: Path) -> Tuple[cv2.VideoCapture, int, int, int, int]:
    """Open a video file and return capture object with metadata.

    Args:
        input_path: Path to the input video file.

    Returns:
        Tuple of (cap, width, height, fps, total_frames).

    Raises:
        ValueError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, width, height, fps, total_frames


def create_video_writer(
    output_path: Path, fps: int, width: int, height: int
) -> cv2.VideoWriter:
    """Create and return a VideoWriter using mp4v codec.

    OpenCV writes an intermediate file with mp4v; call finalize_video()
    after releasing the writer to re-encode to H.264 with proper compression.

    Args:
        output_path: Path for the output video file.
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        An opened cv2.VideoWriter.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def finalize_video(output_path: Path) -> None:
    """Re-encode the OpenCV output to a web-friendly H.264 MP4 using ffmpeg.

    OpenCV's built-in encoders on Windows produce enormous files (mp4v is
    uncompressed-ish, avc1 uses ~170 Mbps).  This step re-encodes to a
    properly compressed H.264 file at ~8 Mbps — small enough for Streamlit
    to serve and for browsers to stream.

    Args:
        output_path: Path to the mp4v file written by OpenCV.
    """
    if not shutil.which("ffmpeg"):
        logger.warning(
            "ffmpeg not found on PATH — skipping re-encode.  "
            "The output video will be very large and may not play in the dashboard."
        )
        return

    tmp_path = output_path.with_suffix(".tmp.mp4")
    output_path.rename(tmp_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(tmp_path),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info("Re-encoding video to H.264 (CRF 23) …")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        tmp_path.unlink()
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Re-encode complete — {output_path.name}: {size_mb:.1f} MB")
    except subprocess.CalledProcessError as exc:
        logger.error(f"ffmpeg re-encode failed: {exc.stderr}")
        # Restore the original so the pipeline doesn't lose data
        if tmp_path.exists():
            tmp_path.rename(output_path)


def create_csv_writer(csv_path: Path) -> Tuple[Any, Any]:
    """Open a CSV file and write the header row.

    Args:
        csv_path: Path for the output CSV file.

    Returns:
        Tuple of (file_handle, csv_writer). The caller is responsible for
        closing the file handle.
    """
    file_handle = open(csv_path, mode='w', newline='')
    writer = csv.writer(file_handle)
    writer.writerow(['frame_number', 'timestamp_sec', 'distance_cm', 'status', 'has_curb'])
    return file_handle, writer


def write_csv_row(
    csv_writer: Any,
    frame_idx: int,
    fps: int,
    distance_str: str,
    status: str,
    has_curb: bool,
) -> None:
    """Write a single row to the metrics CSV.

    Args:
        csv_writer: A csv.writer instance.
        frame_idx: Zero-based frame index.
        fps: Frames per second of the video.
        distance_str: Formatted distance string (e.g. "123px" or "N/A").
        status: Apex status string.
        has_curb: Whether a curb was detected.
    """
    timestamp = round(frame_idx / fps, 2)
    csv_writer.writerow([frame_idx, timestamp, distance_str, status, has_curb])
