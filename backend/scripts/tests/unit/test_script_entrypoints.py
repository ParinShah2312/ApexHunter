"""Regression tests: runnable scripts inside subpackages must import when run by file path.

Running e.g. `python backend/scripts/etl/clean_telemetry.py` puts `etl/` (not `backend/scripts/`)
on sys.path, which used to raise `ModuleNotFoundError: No module named 'utils'`. Each script is
loaded in a fresh interpreter with runpy under a non-__main__ name, so only top-level imports run.
"""

import concurrent.futures
import os
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent

ENTRY_SCRIPTS = [
    "etl/clean_telemetry.py",
    "etl/download_manual_videos.py",
    "etl/download_satellite_images.py",
    "etl/download_season_data.py",
    "etl/fix_parquet_timestamps.py",
    "bigdata/hdfs_manager.py",
    "bigdata/mongo_manager.py",
    "bigdata/spark_clean_telemetry.py",
    "cv/extract_frames.py",
    "cv/extract_turns.py",
    "cv/run_inference.py",
    "cv/select_training_frames.py",
]


def _import_by_path(rel_path: str) -> subprocess.CompletedProcess:
    """Execute a script's top level in a clean interpreter, without running main()."""
    script = SCRIPTS_DIR / rel_path
    code = f"import runpy; runpy.run_path({str(script)!r}, run_name='entrypoint_import_check')"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True, timeout=300,
    )


class TestScriptEntrypoints(unittest.TestCase):

    def test_all_entry_scripts_exist(self):
        for rel in ENTRY_SCRIPTS:
            self.assertTrue((SCRIPTS_DIR / rel).exists(), rel)

    def test_entry_scripts_import_when_run_by_path(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            results = dict(zip(ENTRY_SCRIPTS, pool.map(_import_by_path, ENTRY_SCRIPTS)))
        for rel, res in results.items():
            with self.subTest(script=rel):
                self.assertEqual(res.returncode, 0, f"{rel} failed:\n{res.stderr[-1500:]}")


if __name__ == "__main__":
    unittest.main()
