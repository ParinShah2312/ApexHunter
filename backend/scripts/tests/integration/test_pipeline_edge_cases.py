"""Integration tests: detect_mistakes rejects bad input cleanly (driver not in the session, a required
column missing), with the exact reason, exit code 1 and no output written.

No mocks: uses the real 2024 Bahrain qualifying file from data_lake/clean_data (copied with a column
dropped for the second case). Skipped when the local data is not present.
"""

import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from detect_mistakes import run_pipeline
from isolation_forest.mistakes_io import load_and_validate
from utils import DATA_LAKE_DIR

RUN_SLOW = os.environ.get("APEXHUNTER_RUN_SLOW", "0") == "1"
SESSION = DATA_LAKE_DIR / "clean_data" / "2024_1_Q.parquet"
LOG = logging.getLogger("test_pipeline_edge_cases")
LOG.addHandler(logging.NullHandler())
LOG.propagate = False


@unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")
@unittest.skipUnless(SESSION.exists(), f"Needs local data: {SESSION}")
class TestPipelineEdgeCases(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.output_dir = self.tmp / "output"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _exits_without_output(self, session: Path, driver: str) -> None:
        with self.assertRaises(SystemExit) as cm:
            run_pipeline(str(session), driver, self.output_dir, True)
        self.assertEqual(cm.exception.code, 1)
        self.assertFalse(self.output_dir.exists() and any(self.output_dir.iterdir()))

    def test_driver_not_in_session(self):
        with self.assertRaisesRegex(ValueError, "Driver '99' not found"):
            load_and_validate(SESSION, "99", LOG)
        self._exits_without_output(SESSION, "99")

    def test_missing_required_column(self):
        path = self.tmp / "2024_1_Q.parquet"
        pd.read_parquet(SESSION).drop(columns=["Brake"]).to_parquet(path)
        with self.assertRaisesRegex(ValueError, r"Missing required columns: \['Brake'\]"):
            load_and_validate(path, "16", LOG)
        self._exits_without_output(path, "16")


if __name__ == "__main__":
    unittest.main()
