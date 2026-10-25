"""
================================================================================
  ApexHunter - Turn Windows from FastF1 Circuit Data
  Script: extract_turns.py
--------------------------------------------------------------------------------
  Purpose : Builds the per-corner time windows used by the CV inference pipeline.

            Corners come from FastF1's official circuit information (number and
            distance along the lap). Each corner's time on the pole lap is read
            from that lap's telemetry, turn direction from the GPS heading change
            and speed class from the minimum speed through the corner. The only
            manual input is the video offset: how many seconds into the onboard
            video the timed lap starts.

            (This replaces a curvature-based corner detector that merged nearby
            corners, e.g. it found 11 of Bahrain's 15, and hand-typed windows.)

  Usage   : python backend/scripts/cv/extract_turns.py --video-offset 4.9 \
                --video "data_lake/edited_videos/2024/01_bahrain_ver_pole - Trim.mp4"
            python backend/scripts/cv/extract_turns.py --year 2024 --round 1 --video-offset 4.9
================================================================================
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Make backend/scripts importable so this file runs both by path and as `python -m <pkg>.<module>`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import enable_fastf1_cache, setup_logger

logger = setup_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
WINDOW_BEFORE_M: float = 50.0     # window starts this far (track metres) before the apex
WINDOW_AFTER_M: float = 50.0      # ... and ends this far after it
DIRECTION_SPAN_M: float = 30.0    # heading change is measured over apex ± this distance
SLOW_MAX_KMH: float = 100.0       # minimum corner speed below this → "Slow"
FAST_MIN_KMH: float = 200.0       # minimum corner speed above this → "Fast"
NO_TURN: Tuple[str, str] = ("-", "Medium")   # context returned on straights

# e.g. ".../edited_videos/2024/01_bahrain_ver_pole - Trim.mp4"
VIDEO_NAME_PATTERN = re.compile(r"^(?P<round>\d+)_[a-z]+_(?P<driver>[a-z]{3})_pole", re.IGNORECASE)


@dataclass
class TurnWindow:
    """One corner's time window, in seconds from the start of the onboard video."""
    turn: str
    start_s: float
    apex_s: float
    end_s: float
    direction: str            # "Left" or "Right"
    speed_category: str       # "Slow", "Medium" or "Fast"
    min_speed_kmh: float


def parse_video_path(video_path: Path) -> Tuple[int, int, str]:
    """Return (year, round, driver abbreviation) from an edited pole-lap video path.

    Expects `.../edited_videos/{year}/{round}_{venue}_{drv}_pole - Trim.mp4`.

    Raises:
        ValueError: If the year folder or file name does not follow that convention.
    """
    match = VIDEO_NAME_PATTERN.match(video_path.name)
    year_folder = video_path.parent.name
    if not match or not year_folder.isdigit():
        raise ValueError(
            f"Cannot infer year/round/driver from '{video_path}'. Expected "
            f"'<year>/<round>_<venue>_<drv>_pole - Trim.mp4', or pass --year/--round/--driver."
        )
    return int(year_folder), int(match.group("round")), match.group("driver").upper()


def _speed_category(min_speed_kmh: float) -> str:
    if min_speed_kmh < SLOW_MAX_KMH:
        return "Slow"
    if min_speed_kmh > FAST_MIN_KMH:
        return "Fast"
    return "Medium"


def _heading_change_deg(x: np.ndarray, y: np.ndarray, i0: int, i1: int) -> float:
    """Signed heading change between samples i0 and i1 (positive = anticlockwise = left turn)."""
    h0 = np.arctan2(y[i0 + 1] - y[i0], x[i0 + 1] - x[i0])
    h1 = np.arctan2(y[i1 + 1] - y[i1], x[i1 + 1] - x[i1])
    return float(np.degrees(np.angle(np.exp(1j * (h1 - h0)))))


def compute_turn_windows(
    corners: pd.DataFrame, telemetry: pd.DataFrame, video_offset_s: float
) -> List[TurnWindow]:
    """Build turn windows from circuit corners and the pole lap's telemetry.

    Args:
        corners: FastF1 circuit_info.corners (columns Number, Letter, Distance).
        telemetry: Pole-lap telemetry with Time (lap-relative timedelta), Distance, X, Y, Speed.
        video_offset_s: Seconds into the video at which the timed lap starts.

    Returns:
        Windows sorted by apex time. Windows of neighbouring corners never overlap: where
        they would, the boundary is placed halfway between the two apexes.
    """
    t = telemetry["Time"].dt.total_seconds().to_numpy()
    dist = telemetry["Distance"].to_numpy(dtype=float)
    x = telemetry["X"].to_numpy(dtype=float)
    y = telemetry["Y"].to_numpy(dtype=float)
    speed = telemetry["Speed"].to_numpy(dtype=float)
    last = len(dist) - 2

    windows: List[TurnWindow] = []
    for _, corner in corners.sort_values("Distance").iterrows():
        apex_d = float(corner["Distance"])
        letter = corner.get("Letter", "") or ""
        label = f"T{int(corner['Number'])}{letter if isinstance(letter, str) else ''}"

        i0 = min(int(np.searchsorted(dist, apex_d - DIRECTION_SPAN_M)), last)
        i1 = min(int(np.searchsorted(dist, apex_d + DIRECTION_SPAN_M)), last)
        direction = "Left" if _heading_change_deg(x, y, i0, i1) > 0 else "Right"

        w0 = int(np.searchsorted(dist, apex_d - WINDOW_BEFORE_M))
        w1 = max(int(np.searchsorted(dist, apex_d + WINDOW_AFTER_M)), w0 + 1)
        min_speed = float(speed[w0:w1].min())

        windows.append(TurnWindow(
            turn=label,
            start_s=float(np.interp(apex_d - WINDOW_BEFORE_M, dist, t)) + video_offset_s,
            apex_s=float(np.interp(apex_d, dist, t)) + video_offset_s,
            end_s=float(np.interp(apex_d + WINDOW_AFTER_M, dist, t)) + video_offset_s,
            direction=direction,
            speed_category=_speed_category(min_speed),
            min_speed_kmh=round(min_speed, 1),
        ))

    for prev, nxt in zip(windows, windows[1:]):
        if prev.end_s > nxt.start_s:
            boundary = (prev.apex_s + nxt.apex_s) / 2.0
            prev.end_s, nxt.start_s = boundary, boundary
    return windows


def load_pole_lap(year: int, round_num: int, driver: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load circuit corners and the pole lap's telemetry for a qualifying session via FastF1.

    Args:
        year, round_num: The event.
        driver: Driver abbreviation (e.g. "VER"). Defaults to the pole sitter.

    Returns:
        Tuple of (corners, telemetry with Distance, driver abbreviation).
    """
    import fastf1

    enable_fastf1_cache()
    session = fastf1.get_session(year, round_num, "Q")
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    if driver is None:
        driver = str(session.results.sort_values("Position").iloc[0]["Abbreviation"])
    lap = session.laps.pick_drivers(driver).pick_fastest()
    telemetry = lap.get_telemetry().add_distance()
    logger.info(f"Pole lap: {year} round {round_num} {driver} {lap['LapTime']}")
    return session.get_circuit_info().corners, telemetry, driver


def build_turn_windows(
    video_offset_s: float,
    video_path: Optional[Path] = None,
    year: Optional[int] = None,
    round_num: Optional[int] = None,
    driver: Optional[str] = None,
) -> List[TurnWindow]:
    """Turn windows for a pole-lap video; year/round/driver default to those in the video name."""
    if video_path is not None and (year is None or round_num is None or driver is None):
        v_year, v_round, v_driver = parse_video_path(video_path)
        year, round_num, driver = year or v_year, round_num or v_round, driver or v_driver
    if year is None or round_num is None:
        raise ValueError("Provide a video path or both year and round.")
    corners, telemetry, _ = load_pole_lap(year, round_num, driver)
    return compute_turn_windows(corners, telemetry, video_offset_s)


def turn_context_at(timestamp_s: float, windows: List[TurnWindow]) -> Tuple[str, str]:
    """Return (direction, speed_category) for a video timestamp, or ("-", "Medium") on straights."""
    for w in windows:
        if w.start_s <= timestamp_s <= w.end_s:
            return w.direction, w.speed_category
    return NO_TURN


def save_turn_windows(windows: List[TurnWindow], path: Path, video_offset_s: float) -> None:
    """Write windows to JSON (consumed by the dashboard's turn-by-turn report card)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"video_offset_s": video_offset_s, "turns": [asdict(w) for w in windows]}, f, indent=2)
    logger.info(f"Saved {len(windows)} turn windows to {path}")


def load_turn_windows(path: Path) -> List[TurnWindow]:
    """Read windows written by save_turn_windows."""
    with open(path, "r") as f:
        return [TurnWindow(**w) for w in json.load(f)["turns"]]


def main() -> None:
    p = argparse.ArgumentParser(description="Build CV turn windows from FastF1 circuit data.")
    p.add_argument("--video-offset", type=float, required=True,
                   help="Seconds into the video at which the timed lap starts.")
    p.add_argument("--video", type=str, default=None, help="Edited pole-lap video (infers year/round/driver).")
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--round", type=int, default=None)
    p.add_argument("--driver", type=str, default=None, help="Driver abbreviation; defaults to the pole sitter.")
    p.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    args = p.parse_args()

    windows = build_turn_windows(
        args.video_offset, Path(args.video) if args.video else None, args.year, args.round, args.driver,
    )
    print(f"\n{'Turn':<6}{'Start':>8}{'Apex':>8}{'End':>8}  {'Dir':<6}{'Class':<8}{'Min km/h':>9}")
    for w in windows:
        print(f"{w.turn:<6}{w.start_s:8.2f}{w.apex_s:8.2f}{w.end_s:8.2f}  {w.direction:<6}"
              f"{w.speed_category:<8}{w.min_speed_kmh:9.1f}")
    if args.output:
        save_turn_windows(windows, Path(args.output), args.video_offset)


if __name__ == "__main__":
    main()
