"""Integration tests: detect_mistakes output control (skip if exists, --force, output folder).

No mocks: runs the real pipeline on real data (2024 Bahrain qualifying, car #16, with FastF1 laps
from the project cache) into a temp folder. Skipped when the local data is not present.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from detect_mistakes import run_pipeline
from utils import DATA_LAKE_DIR

RUN_SLOW = os.environ.get("APEXHUNTER_RUN_SLOW", "0") == "1"
SESSION = DATA_LAKE_DIR / "clean_data" / "2024_1_Q.parquet"
DRIVER = "16"


@unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")
@unittest.skipUnless(SESSION.exists(), f"Needs local data: {SESSION}")
class TestPipelineSkipForce(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.tmp.name) / "nested" / "mistake_data"     # does not exist yet
        run_pipeline(str(SESSION), DRIVER, cls.output_dir, force=False)
        cls.parquet = cls.output_dir / f"2024_1_Q_{DRIVER}_mistakes.parquet"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_creates_output_folder_and_files(self):
        self.assertTrue(self.parquet.exists())
        self.assertTrue((self.output_dir / f"2024_1_Q_{DRIVER}_mistakes_meta.json").exists())

    def test_existing_output_is_skipped_without_force(self):
        with tempfile.TemporaryDirectory() as out:
            stale = Path(out) / f"2024_1_Q_{DRIVER}_mistakes.parquet"
            stale.write_bytes(b"stale")
            run_pipeline(str(SESSION), DRIVER, Path(out), force=False)
            self.assertEqual(stale.read_bytes(), b"stale")

    def test_force_overwrites_existing_output(self):
        with tempfile.TemporaryDirectory() as out:
            stale = Path(out) / f"2024_1_Q_{DRIVER}_mistakes.parquet"
            stale.write_bytes(b"stale")
            run_pipeline(str(SESSION), DRIVER, Path(out), force=True)
            self.assertNotEqual(stale.read_bytes(), b"stale")


if __name__ == "__main__":
    unittest.main()
