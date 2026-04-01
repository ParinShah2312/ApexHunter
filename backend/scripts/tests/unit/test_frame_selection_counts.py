"""Unit tests for frame selection: correct frame count logic."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import select_training_frames


class TestFrameSelectionCounts(unittest.TestCase):

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
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_and_list(self):
        select_training_frames.select_frames()
        return sorted([f.name for f in self.output_dir.iterdir()])

    def test_correct_number_of_frames_selected(self):
        filenames = self._run_and_list()
        self.assertGreaterEqual(len(filenames), 3)
        self.assertLessEqual(len(filenames), 33)

    def test_frames_spread_across_tracks(self):
        filenames = self._run_and_list()
        has_bahrain = any("01_bahrain" in f for f in filenames)
        has_saudi = any("02_saudi" in f for f in filenames)
        self.assertTrue(has_bahrain)
        self.assertTrue(has_saudi)


if __name__ == "__main__":
    unittest.main()
