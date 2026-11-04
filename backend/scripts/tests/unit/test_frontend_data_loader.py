"""Tests for frontend/components/data_loader.py: a corrupt pit-window file is reported as such,
not as missing (the last carried-over dashboard issue, ROADMAP §3 step 7)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parents[4] / "frontend"
sys.path.insert(0, str(FRONTEND_DIR))

from components.data_loader import tyre_prediction_unreadable


class TestStrategyFileStatus(unittest.TestCase):

    def test_corrupt_missing_and_good_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corrupt, good = Path(tmp) / "bad_strategy.json", Path(tmp) / "good_strategy.json"
            corrupt.write_text('{"model": "ridge", "laps": [', encoding="utf-8")   # cut-off write
            good.write_text(json.dumps({"model": "ridge"}), encoding="utf-8")
            self.assertTrue(tyre_prediction_unreadable(str(corrupt)))
            self.assertFalse(tyre_prediction_unreadable(str(Path(tmp) / "missing_strategy.json")))
            self.assertFalse(tyre_prediction_unreadable(str(good)))


if __name__ == "__main__":
    unittest.main()
