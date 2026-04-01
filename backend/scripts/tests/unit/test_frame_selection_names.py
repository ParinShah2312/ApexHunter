"""Unit tests for frame selection: filename uniqueness and prefix tests."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import select_training_frames


class TestFrameSelectionNames(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.frames_root = Path(self.tmp) / "cv_frames"
        self.output_dir = Path(self.tmp) / "cv_dataset" / "upload_to_roboflow"

        d1 = self.frames_root / "2023" / "01_bahrain_ver_pole"
        d1.mkdir(parents=True)
        for i in range(1, 51):
            (d1 / f"frame_{i:05d}.jpg").touch()

        d2 = self.frames_root / "2024" / "01_bahrain_per_pole"
        d2.mkdir(parents=True)
        for i in range(1, 41):
            (d2 / f"frame_{i:05d}.jpg").touch()

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

    def test_output_filenames_unique(self):
        filenames = self._run_and_list()
        self.assertEqual(len(set(filenames)), len(filenames))

    def test_output_filenames_include_year_prefix(self):
        filenames = self._run_and_list()
        for f in filenames:
            self.assertTrue(f.startswith("2023_") or f.startswith("2024_"),
                            f"Filename {f} does not start with year prefix")


if __name__ == "__main__":
    unittest.main()
