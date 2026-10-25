"""Unit tests for cv/select_training_frames.py: even frame spacing and the copied output set."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from cv import select_training_frames
from cv.select_training_frames import evenly_spaced

FRAMES_PER_VIDEO = 11
VIDEOS = {  # year/video folder → number of extracted frames
    "2023/01_bahrain_ver_pole": 50,
    "2023/02_saudi_nor_pole": 30,
    "2024/01_bahrain_per_pole": 40,
}


class TestEvenlySpaced(unittest.TestCase):

    def test_includes_first_and_last_and_spreads_evenly(self) -> None:
        picked = evenly_spaced(list(range(460)), FRAMES_PER_VIDEO)
        self.assertEqual(len(picked), FRAMES_PER_VIDEO)
        self.assertEqual((picked[0], picked[-1]), (0, 459))
        gaps = [b - a for a, b in zip(picked, picked[1:])]
        self.assertLessEqual(max(gaps) - min(gaps), 1)

    def test_short_list_returns_everything(self) -> None:
        self.assertEqual(evenly_spaced(list(range(5)), FRAMES_PER_VIDEO), list(range(5)))

    def test_no_duplicates_when_barely_longer_than_k(self) -> None:
        picked = evenly_spaced(list(range(12)), FRAMES_PER_VIDEO)
        self.assertEqual(len(set(picked)), FRAMES_PER_VIDEO)
        self.assertEqual(picked[-1], 11)


class TestSelectFrames(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.frames_root = self.tmp / "cv_frames"
        self.output_dir = self.tmp / "cv_dataset" / "upload_to_roboflow"
        for folder, n in VIDEOS.items():
            d = self.frames_root / folder
            d.mkdir(parents=True)
            for i in range(1, n + 1):
                (d / f"frame_{i:05d}.jpg").touch()
        (self.frames_root / "2023" / "03_empty_track").mkdir()
        self._patches = [
            patch.object(select_training_frames, "FRAMES_ROOT", self.frames_root),
            patch.object(select_training_frames, "OUTPUT_DIR", self.output_dir),
            patch.object(select_training_frames, "SEASONS", ["2023", "2024", "2025"]),  # 2025 missing
            patch.object(select_training_frames, "FRAMES_PER_VIDEO", FRAMES_PER_VIDEO),
        ]
        for p in self._patches:
            p.start()
        select_training_frames.select_frames()
        self.selected = sorted(f.name for f in self.output_dir.iterdir())

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _for(self, folder: str):
        year, video = folder.split("/")
        return [n for n in self.selected if n.startswith(f"{year}_{video}_")]

    def test_exactly_k_frames_per_video(self) -> None:
        for folder in VIDEOS:
            self.assertEqual(len(self._for(folder)), FRAMES_PER_VIDEO, folder)
        self.assertEqual(len(self.selected), FRAMES_PER_VIDEO * len(VIDEOS))

    def test_first_and_last_frame_of_each_video_selected(self) -> None:
        """Regression for #16: the old step-slicing never reached the end of a video."""
        for folder, n in VIDEOS.items():
            names = self._for(folder)
            self.assertTrue(any(name.endswith("frame_00001.jpg") for name in names), folder)
            self.assertTrue(any(name.endswith(f"frame_{n:05d}.jpg") for name in names), folder)

    def test_names_are_year_and_video_prefixed(self) -> None:
        self.assertEqual(len(self.selected), sum(len(self._for(f)) for f in VIDEOS))

    def test_empty_video_folder_and_missing_year_skipped(self) -> None:
        self.assertFalse(any("03_empty_track" in n for n in self.selected))
        self.assertFalse(any(n.startswith("2025_") for n in self.selected))


if __name__ == "__main__":
    unittest.main()
