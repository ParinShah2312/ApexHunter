"""Unit test for strategy/strategy_check.py scoring (ROADMAP §C step 6)."""

import sys
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from strategy.strategy_check import best_source, summarise


def _row(source, stint, hit, chance, sc_laps, sc_calls, under_sc=False):
    return {"source": source, "race": "R", "driver": "VER", "stint": stint, "actual_stop": 20,
            "hit": hit, "random_hit_chance": chance, "window_size": 5, "advised_laps": 20,
            "actual_under_sc_vsc": under_sc, "sc_called": under_sc and hit,
            "sc_laps": sc_laps, "sc_calls": sc_calls}


class TestSummarise(unittest.TestCase):

    def test_sources_are_scored_on_the_same_stints_and_sc_laps(self) -> None:
        rows = pd.DataFrame([
            # stint 1: both advise; the model has no advice on the early SC laps 3-4
            _row("no_change", 1, True, 0.25, [3, 4, 20, 21], [3, 4, 20], under_sc=True),
            _row("model", 1, False, 0.50, [20, 21], [21], under_sc=True),
            _row("no_change", 2, True, 0.25, [], []),
            _row("model", 2, True, 0.50, [], []),
            # stint 3: only "no change" could advise -> left out for both
            _row("no_change", 3, True, 0.25, [], []),
        ])
        s = summarise(rows)
        self.assertEqual((s["no_change"]["stints"], s["model"]["stints"]), (2, 2))
        self.assertEqual((s["no_change"]["hit_pct"], s["model"]["hit_pct"]), (100.0, 50.0))
        self.assertEqual((s["no_change"]["random_pct"], s["model"]["random_pct"]), (25.0, 50.0))
        # shared SC laps = 20, 21: "no change" called 20, the model called 21
        self.assertEqual(s["model"]["shared_sc_laps"], 2)
        self.assertEqual((s["no_change"]["sc_laps_called_pct"], s["model"]["sc_laps_called_pct"]), (50.0, 50.0))
        self.assertEqual((s["no_change"]["sc_stops_called_pct"], s["model"]["sc_stops_called_pct"]), (100.0, 0.0))
        # best = largest margin over a random window (75 vs 0 points), not the raw hit rate
        self.assertEqual(best_source(s), "no_change")
        self.assertEqual(best_source({"wide": {"hit_pct": 60.0, "random_pct": 55.0},
                                      "narrow": {"hit_pct": 40.0, "random_pct": 20.0}}), "narrow")


if __name__ == "__main__":
    unittest.main()
