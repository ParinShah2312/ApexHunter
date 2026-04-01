"""Unit tests for frame selection: empty folder, missing dir, reproducibility."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import select_training_frames


class TestFrameSelectionEdge(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.frames_root = Path(self.tmp) / "cv_frames"
        self.output_dir = Path(self.tmp) / "cv_dataset" / "upload_to_roboflow"

        d1 = self.frames_root / "2023" / "01_bahrain_ver_pole"
        d1.mkdir(parents=True)
        for i in range(1, 51):
            (d1 / f"frame_{i:05d}.jpg").touch()

        d2 = self.frames_root / "2023" / "02_saudi_nor_pole"
        d2.mkdir(parents=True)
        for i in range(1, 31):
            (d2 / f"frame_{i:05d}.jpg").touch()

        d3 = self.frames_root / "2024" / "01_bahrain_per_pole"
        d3.mkdir(parents=True)
        for i in range(1, 41):
            (d3 / f"frame_{i:05d}.jpg").touch()

        self._patches = [
            patch.object(select_training_frames, "FRAMES_ROOT", self.frames_root),
            patch.object(select_training_frames, "OUTPUT_DIR", self.output_dir),
            patch.object(select_training_frames, "SEASONS", ["2023", "2024"]),
            patch.object(select_training_frames, "FRAMES_PER_VIDEO", 11),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_track_folder_skipped(self):
        empty_dir = self.frames_root / "2023" / "03_empty_track"
        empty_dir.mkdir(parents=True)
        select_training_frames.select_frames()
        filenames = [f.name for f in self.output_dir.iterdir()]
        self.assertFalse(any("03_empty_track" in f for f in filenames))

    def test_output_dir_created_if_missing(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.assertFalse(self.output_dir.exists())
        select_training_frames.select_frames()
        self.assertTrue(self.output_dir.exists())

    def test_reproducible_selection(self):
        select_training_frames.select_frames()
        files_1 = set(f.name for f in self.output_dir.iterdir())
        for f in self.output_dir.iterdir():
            f.unlink()
        select_training_frames.select_frames()
        files_2 = set(f.name for f in self.output_dir.iterdir())
        self.assertEqual(files_1, files_2)


if __name__ == "__main__":
    unittest.main()
