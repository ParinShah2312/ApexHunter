"""Tests for frontend/components/cv_feed.py: mapping the session-time scrubber onto the video (#35)."""

import sys
import unittest
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parents[4] / "frontend"
sys.path.insert(0, str(FRONTEND_DIR))

from components.cv_feed import video_time_for

# The processed Bahrain video: VER's pole lap, 4354.125-4443.304 s into 2024 R1 Q; the lap starts
# 4.9 s into the video
LAP = {"year": 2024, "round": 1, "session": "Q", "driver": "VER", "driver_number": "1",
       "lap_start_s": 4354.125, "lap_end_s": 4443.304, "video_offset_s": 4.9}


class TestVideoTime(unittest.TestCase):

    def test_scrubber_maps_into_the_video_only_inside_its_lap(self) -> None:
        """Regression for #35: the old mapping (scrub - start of telemetry, e.g. 4364 - 1622 s)
        always landed past the end of the 98 s video, so the badge showed the last frame."""
        self.assertAlmostEqual(video_time_for(4364.125, LAP, 2024, 1, "Q", "1"), 14.9)   # 10 s into the lap
        self.assertIsNone(video_time_for(4300.0, LAP, 2024, 1, "Q", "1"))              # before the lap
        self.assertIsNone(video_time_for(4364.125, LAP, 2024, 1, "Q", "16"))           # another driver
        self.assertIsNone(video_time_for(4364.125, LAP, 2024, 1, "R", "1"))            # the race
        self.assertIsNone(video_time_for(4364.125, None, 2024, 1, "Q", "1"))           # lap not recorded


if __name__ == "__main__":
    unittest.main()
