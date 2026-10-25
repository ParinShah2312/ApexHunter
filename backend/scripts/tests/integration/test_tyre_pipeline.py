"""Integration tests for the full LSTM tyre cliff pipeline.
Builds synthetic race data, trains a model, and runs prediction end-to-end."""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
TESTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(TESTS_DIR))

RUN_SLOW = os.environ.get("APEXHUNTER_RUN_SLOW", "0") == "1"
slow_test = unittest.skipUnless(RUN_SLOW, "Skipped: set APEXHUNTER_RUN_SLOW=1 to run")

_FIXTURE_CACHE = {}


N_SESSIONS_ROUNDS = [1, 2, 3, 10]   # round 10 must sort last (string sorting would put it second)
LAP_SECONDS = 12.0
LAPS_PER_STINT = 15
STINT_GAP_SECONDS = 120.0


def build_synthetic_race_session(output_dir: Path, round_num: int) -> Path:
    """Create a parquet simulating one race: two 15-lap stints of degrading pace at ~10 Hz."""
    rng = np.random.RandomState(42 + round_num)
    n_per_stint = 1800
    stint_len = LAPS_PER_STINT * LAP_SECONDS

    t1 = np.linspace(0.0, stint_len, n_per_stint)
    speed1 = np.linspace(260.0, 240.0, n_per_stint) + rng.normal(0, 0.5, n_per_stint)
    t2 = np.linspace(stint_len + STINT_GAP_SECONDS, 2 * stint_len + STINT_GAP_SECONDS, n_per_stint)
    speed2 = np.linspace(262.0, 238.0, n_per_stint) + rng.normal(0, 0.5, n_per_stint)

    n = n_per_stint * 2
    df = pd.DataFrame({
        "Driver": ["44"] * n,
        "Speed": np.concatenate([speed1, speed2]).astype("float32"),
        "Throttle": (np.full(n, 72.0) + rng.normal(0, 2, n)).astype("float32"),
        "Brake": rng.randint(0, 2, n).astype("float32"),   # binary flag, as in FastF1
        "RPM": (np.full(n, 9500.0) + rng.normal(0, 100, n)).astype("float32"),
        "nGear": np.full(n, 5.0, dtype="float32"),
        "SessionTime": pd.to_timedelta(np.concatenate([t1, t2]), unit="s"),
    })
    session_path = output_dir / f"2024_{round_num}_R.parquet"
    df.to_parquet(session_path, compression="snappy")
    return session_path


def build_fake_laps() -> pd.DataFrame:
    """FastF1-style lap table aligned with the synthetic SessionTime of every session."""
    rows = []
    stint2_start = LAPS_PER_STINT * LAP_SECONDS + STINT_GAP_SECONDS
    for stint, compound, offset in [(1, "SOFT", 0.0), (2, "HARD", stint2_start)]:
        for i in range(LAPS_PER_STINT):
            rows.append({
                "DriverNumber": "44", "Stint": stint, "Compound": compound,
                "LapNumber": (stint - 1) * LAPS_PER_STINT + i + 1, "TyreLife": i + 1,
                "LapStartTime": pd.Timedelta(seconds=offset + i * LAP_SECONDS),
                "Time": pd.Timedelta(seconds=offset + (i + 1) * LAP_SECONDS),
                "LapTime": pd.Timedelta(seconds=LAP_SECONDS),
            })
    return pd.DataFrame(rows)


def run_full_pipeline_once() -> dict:
    """Run the REAL training (run_training) and prediction (predict_single_stint) code once."""
    if _FIXTURE_CACHE:
        return _FIXTURE_CACHE.copy()

    import logging
    from train_lstm import run_training
    from lstm.tyre_data import extract_stints
    from lstm.tyre_io import build_prediction_output, load_model_artifacts, save_prediction
    from lstm.tyre_model import predict_single_stint

    tmp = Path(tempfile.mkdtemp())
    sessions_dir = tmp / "data_lake" / "clean_data"   # FastF1 cache resolves to tmp/cache
    sessions_dir.mkdir(parents=True)
    models_dir = tmp / "models"
    output_dir = tmp / "tyre_predictions"
    session_paths = [build_synthetic_race_session(sessions_dir, r) for r in N_SESSIONS_ROUNDS]

    lg = logging.getLogger("test_tyre_pipeline")
    lg.handlers = [logging.NullHandler()]
    lg.setLevel(logging.CRITICAL)

    fake_laps_df = build_fake_laps()
    mock_session = mock.MagicMock()
    mock_session.laps.pick_drivers.return_value = fake_laps_df

    with mock.patch("lstm.tyre_data.fastf1.get_session", return_value=mock_session):
        config = run_training(
            sessions_dir, models_dir, ["2024"], test_split=0.25, log=lg,
            grid=[{"hidden_size": 32, "learning_rate": 0.001},
                  {"hidden_size": 64, "learning_rate": 0.001}],
            n_epochs_search=3, n_epochs_final=3,
        )

    model, scaler, config_loaded = load_model_artifacts(models_dir, lg)

    session_path = session_paths[-1]
    df = pd.read_parquet(session_path)
    df_driver = df[df["Driver"] == "44"].sort_values("SessionTime").reset_index(drop=True)
    stints_data = extract_stints(fake_laps_df, df_driver, lg)
    stint_results = [
        predict_single_stint(s, model, scaler, config_loaded["y_mean"], config_loaded["y_std"],
                             config_loaded["sequence_length"])
        for s in stints_data
    ]

    data = build_prediction_output(str(session_path), "44", stint_results)
    output_path = output_dir / "2024_10_R_44_tyre.json"
    save_prediction(data, output_path, lg)

    _FIXTURE_CACHE.update({
        "tmp_dir": tmp,
        "models_dir": models_dir,
        "output_path": output_path,
        "config": config,
        "session_path": session_path,
    })
    return _FIXTURE_CACHE.copy()


@slow_test
class TestTyrePipeline(unittest.TestCase):
    """Integration tests for the full LSTM tyre pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = run_full_pipeline_once()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.fixture["tmp_dir"], ignore_errors=True)

    def test_holds_out_whole_latest_session(self) -> None:
        """4 sessions with test_split=0.25 → 3 train sessions, 1 test session (round 10)."""
        config = self.fixture["config"]
        self.assertEqual(config["n_train_sessions"], 3)
        self.assertEqual(config["n_test_sessions"], 1)

    def test_config_records_feature_names(self) -> None:
        from lstm.tyre_data import LAP_FEATURES
        self.assertEqual(self.fixture["config"]["features"], LAP_FEATURES)

    def test_saved_model_loads_after_training(self) -> None:
        import logging
        from lstm.tyre_io import load_model_artifacts
        model, scaler, config = load_model_artifacts(self.fixture["models_dir"], logging.getLogger("t"))
        self.assertEqual(scaler.n_features_in_, config["input_size"])

    def test_training_completes(self) -> None:
        self.assertTrue((self.fixture["models_dir"] / "tyre_lstm.pt").exists())
        self.assertTrue((self.fixture["models_dir"] / "tyre_scaler.pkl").exists())
        self.assertTrue((self.fixture["models_dir"] / "tyre_config.json").exists())

    def test_config_has_required_keys(self) -> None:
        with open(self.fixture["models_dir"] / "tyre_config.json", "r") as f:
            config = json.load(f)
        for key in ["hidden_size", "learning_rate", "input_size",
                     "sequence_length", "y_mean", "y_std", "val_mse", "test_mae_kmh", "features"]:
            self.assertIn(key, config)

    def test_prediction_output_exists(self) -> None:
        self.assertTrue(self.fixture["output_path"].exists())

    def test_prediction_has_stints(self) -> None:
        with open(self.fixture["output_path"], "r") as f:
            data = json.load(f)
        self.assertGreaterEqual(len(data["stints"]), 1)

    def test_prediction_stint_arrays_equal_length(self) -> None:
        with open(self.fixture["output_path"], "r") as f:
            data = json.load(f)
        for stint in data["stints"]:
            n = stint["n_laps"]
            self.assertEqual(len(stint["actual_laps"]), n)
            self.assertEqual(len(stint["predicted_laps"]), n)
            self.assertEqual(len(stint["confidence_upper"]), n)
            self.assertEqual(len(stint["confidence_lower"]), n)

    def test_cliff_lap_is_valid_or_none(self) -> None:
        with open(self.fixture["output_path"], "r") as f:
            data = json.load(f)
        for stint in data["stints"]:
            cliff = stint["cliff_lap"]
            self.assertTrue(
                cliff is None or (isinstance(cliff, int) and 0 <= cliff < stint["n_laps"]),
                f"Invalid cliff_lap: {cliff}"
            )

    def test_actual_laps_are_floats(self) -> None:
        with open(self.fixture["output_path"], "r") as f:
            data = json.load(f)
        for stint in data["stints"]:
            for val in stint["actual_laps"]:
                self.assertIsInstance(val, (int, float))

    def test_predicted_laps_match_actual_length(self) -> None:
        with open(self.fixture["output_path"], "r") as f:
            data = json.load(f)
        for stint in data["stints"]:
            actual = stint["actual_laps"]
            preds = stint["predicted_laps"]
            self.assertEqual(len(actual), len(preds), "Predicted laps length should match actual laps")
            for i, p in enumerate(preds):
                self.assertIsNotNone(p, f"Expected float at index {i}, got None")


if __name__ == "__main__":
    unittest.main()
