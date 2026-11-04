"""Unit tests for the LSTM autoencoder mistake detector (isolation_forest/mistakes_autoencoder.py)."""

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from isolation_forest.mistakes_autoencoder import LSTMAutoencoderDetector, window_starts


def _laps(n_laps=12, per_lap=120, seed=0):
    """Smooth, repeating 'driving' on every lap: five features following the same pattern along
    the lap, plus a little noise."""
    rng = np.random.default_rng(seed)
    phase = np.linspace(0, 4 * np.pi, per_lap)
    one = np.stack([np.sin(phase + k) for k in range(5)], axis=1)
    X = np.concatenate([one + rng.normal(0, 0.05, one.shape) for _ in range(n_laps)])
    laps = np.repeat(np.arange(1, n_laps + 1), per_lap)
    return X.astype(np.float32), laps


class TestWindows(unittest.TestCase):

    def test_windows_never_cross_a_lap(self) -> None:
        laps = np.array([3] * 5 + [4] * 2 + [5] * 6)        # lap 4 too short for a 4-sample window
        starts = window_starts(laps, 4)
        self.assertEqual(starts.tolist(), [0, 1, 7, 8, 9])
        for s in starts:
            self.assertEqual(len(set(laps[s:s + 4])), 1)


class TestDetector(unittest.TestCase):

    def test_an_unusual_stretch_scores_highest(self) -> None:
        """Trained on normal laps only, the autoencoder cannot rebuild a stretch that breaks the
        usual pattern (two features swapped for ~1 s): those samples get the top scores."""
        X, laps = _laps()
        det = LSTMAutoencoderDetector(window=8, epochs=15).fit(X, laps)
        X_test, laps_test = _laps(n_laps=2, seed=1)
        X_test[150:160, [0, 3]] = X_test[150:160, [3, 0]]     # lap 2, samples 30-39
        scores = det.score(X_test, laps_test)
        self.assertEqual(scores.shape, (240,))
        threshold = np.quantile(det.score(X, laps), 0.98)       # the 2 % flag rate of the pipeline
        flagged = set(np.flatnonzero(scores > threshold).tolist())
        self.assertIn(int(np.argmax(scores)), range(150, 160))   # the top score is in the stretch
        self.assertGreaterEqual(len(flagged & set(range(150, 160))), 5, sorted(flagged))
        self.assertLess(len(flagged), 30)                        # and not much else is flagged


if __name__ == "__main__":
    unittest.main()
