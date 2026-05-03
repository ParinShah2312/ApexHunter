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


def build_synthetic_race_session(output_dir: Path) -> Path:
    """Create a parquet with 4000 rows simulating one race session.

    Two stints of ~1800 rows each separated by a 120s gap.
    Speed degrades across each stint to simulate tyre wear.
    """
    n_per_stint = 1800
    rng = np.random.RandomState(42)

    # Stint 1: 0–180s, speed 260→240 km/h
    t1 = np.linspace(0.0, 180.0, n_per_stint)
    speed1 = np.linspace(260.0, 240.0, n_per_stint).astype("float32")
    speed1 += rng.normal(0, 0.5, n_per_stint).astype("float32")

    # Gap: 120s
    gap = 120.0

    # Stint 2: continues after gap, speed 262→238 km/h
    t2 = np.linspace(t1[-1] + gap, t1[-1] + gap + 180.0, n_per_stint)
    speed2 = np.linspace(262.0, 238.0, n_per_stint).astype("float32")
    speed2 += rng.normal(0, 0.5, n_per_stint).astype("float32")

    n = n_per_stint * 2
    times = np.concatenate([t1, t2])
    speed = np.concatenate([speed1, speed2])

    df = pd.DataFrame({
        "Driver": ["44"] * n,
        "Speed": speed,
        "Throttle": np.full(n, 72.0, dtype="float32") + rng.normal(0, 2, n).astype("float32"),
        "Brake": np.full(n, 4.0, dtype="float32") + rng.normal(0, 1, n).astype("float32").clip(0),
        "RPM": np.full(n, 9500.0, dtype="float32") + rng.normal(0, 100, n).astype("float32"),
        "nGear": np.full(n, 5.0, dtype="float32"),
        "SessionTime": pd.to_timedelta(times, unit="s"),
    })

    session_path = output_dir / "2024_1_R.parquet"
    df.to_parquet(session_path, compression="snappy")
    return session_path


def run_full_pipeline_once() -> dict:
    """Run the full train → predict pipeline once, caching results."""
    if _FIXTURE_CACHE:
        return _FIXTURE_CACHE.copy()

    tmp = Path(tempfile.mkdtemp())
    sessions_dir = tmp / "clean_data"
    sessions_dir.mkdir()
    models_dir = tmp / "models"
    output_dir = tmp / "tyre_predictions"

    session_path = build_synthetic_race_session(sessions_dir)

    import logging
    lg = logging.getLogger("test_tyre_pipeline")
    lg.handlers = [logging.NullHandler()]
    lg.setLevel(logging.CRITICAL)

    from tyre_data import build_training_dataset, SEQUENCE_LENGTH, extract_stints
    from tyre_model import INPUT_SIZE, NUM_LAYERS, DROPOUT_RATE
    from tyre_io import save_model_artifacts, load_model_artifacts, build_prediction_output, save_prediction

    # ── MOCK FASTF1 LAPS ──
    # Create fake laps that align with the synthetic SessionTime (0-180s, 300-480s)
    fake_laps_data = []
    # Stint 1
    for i in range(15):
        fake_laps_data.append({
            "DriverNumber": "44", "Stint": 1, "Compound": "SOFT", "LapNumber": i+1,
            "TyreLife": i+1,
            "LapStartTime": pd.Timedelta(seconds=i*12),
            "Time": pd.Timedelta(seconds=(i+1)*12),
            "LapTime": pd.Timedelta(seconds=12)
        })
    # Stint 2
    for i in range(15):
        fake_laps_data.append({
            "DriverNumber": "44", "Stint": 2, "Compound": "HARD", "LapNumber": i+16,
            "TyreLife": i+1,
            "LapStartTime": pd.Timedelta(seconds=300 + i*12),
            "Time": pd.Timedelta(seconds=300 + (i+1)*12),
            "LapTime": pd.Timedelta(seconds=12)
        })
    fake_laps_df = pd.DataFrame(fake_laps_data)

    mock_session = mock.MagicMock()
    mock_session.laps.pick_drivers.return_value = fake_laps_df

    with mock.patch("tyre_data.fastf1.get_session", return_value=mock_session):
        X, y = build_training_dataset(sessions_dir, ["2024"], lg)

    from sklearn.preprocessing import StandardScaler
    X_2d = X.reshape(-1, INPUT_SIZE)
    scaler = StandardScaler()
    X_scaled_2d = scaler.fit_transform(X_2d)
    X_scaled = X_scaled_2d.reshape(X.shape)

    y_mean = float(y.mean())
    y_std = float(y.std())
    y_scaled = (y - y_mean) / (y_std + 1e-8)

    test_n = max(1, int(len(X_scaled) * 0.1))
    X_train, X_test = X_scaled[:-test_n], X_scaled[-test_n:]
    y_train, y_test = y_scaled[:-test_n], y_scaled[-test_n:]

    # Use only 2 configs for speed
    from tyre_model import train_one_config
    import gc

    configs = [
        {"hidden_size": 32, "learning_rate": 0.001},
        {"hidden_size": 64, "learning_rate": 0.001},
    ]

    from tyre_model import VAL_SPLIT
    split = int(len(X_train) * (1 - VAL_SPLIT))
    Xtr, Xv = X_train[:split], X_train[split:]
    ytr, yv = y_train[:split], y_train[split:]

    best_model = None
    best_mse = float("inf")
    best_h = 32
    best_lr = 0.001
    for cfg in configs:
        m, mse = train_one_config(
            Xtr, ytr, Xv, yv,
            hidden_size=cfg["hidden_size"], learning_rate=cfg["learning_rate"],
            n_epochs=5, logger=lg,
        )
        if mse < best_mse:
            best_mse = mse
            best_model = m
            best_h = cfg["hidden_size"]
            best_lr = cfg["learning_rate"]
        gc.collect()

    # Evaluate
    from tyre_model import evaluate_on_test
    mae = evaluate_on_test(best_model, X_test, y_test, y_mean, y_std)

    config = {
        "hidden_size": best_h,
        "learning_rate": best_lr,
        "input_size": INPUT_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "num_layers": NUM_LAYERS,
        "dropout_rate": DROPOUT_RATE,
        "y_mean": y_mean,
        "y_std": y_std,
        "val_mse": best_mse,
        "test_mae_kmh": mae,
        "seasons": ["2024"],
    }
    save_model_artifacts(best_model, scaler, config, models_dir, lg)

    # Predict
    from tyre_data import CLIFF_SPEED_DROP_KMH
    from tyre_model import monte_carlo_predict

    model, scaler_loaded, config_loaded = load_model_artifacts(models_dir, lg)

    df = pd.read_parquet(session_path)
    df_driver = df[df["Driver"] == "44"].copy()
    df_driver.sort_values("SessionTime", inplace=True)
    df_driver.reset_index(drop=True, inplace=True)

    stints_data = extract_stints(fake_laps_df, df_driver, lg)
    seq_len = config_loaded["sequence_length"]

    stint_results = []
    for stint_data in stints_data:
        stint_index = stint_data["stint_index"]
        lap_features = pd.DataFrame(stint_data["lap_features"])
        actual_speeds = lap_features["mean_speed"].tolist()

        # We need to build padded sequences for the test
        from tyre_data import LAP_FEATURES
        features_np = lap_features[LAP_FEATURES].values.copy()

        if len(features_np) > 1:
            features_np[0] = features_np[1]

        padded_features = np.vstack([np.tile(features_np[0], (seq_len, 1)), features_np])
        new_seqs = []
        for i in range(len(features_np)):
            new_seqs.append(padded_features[i : i + seq_len])
        sequences = np.array(new_seqs, dtype=np.float32)

        seq_2d = sequences.reshape(-1, INPUT_SIZE)
        seq_scaled = scaler_loaded.transform(seq_2d).reshape(sequences.shape)
        mean_preds, lower, upper = monte_carlo_predict(model, seq_scaled)
        mean_preds_d = (mean_preds * config_loaded["y_std"] + config_loaded["y_mean"]).tolist()
        lower_d = (lower * config_loaded["y_std"] + config_loaded["y_mean"]).tolist()
        upper_d = (upper * config_loaded["y_std"] + config_loaded["y_mean"]).tolist()
        first_pred = mean_preds_d[0] if mean_preds_d else None
        cliff_lap = None
        for i, v in enumerate(mean_preds_d):
            if first_pred and v < first_pred - CLIFF_SPEED_DROP_KMH:
                cliff_lap = i
                break
        laps_remaining = (len(actual_speeds) - cliff_lap) if cliff_lap is not None else None
        predicted_full = mean_preds_d
        lower_full = lower_d
        upper_full = upper_d
        stint_results.append({
            "stint_index": stint_index, "n_laps": len(actual_speeds),
            "actual_laps": actual_speeds, "predicted_laps": predicted_full,
            "confidence_upper": upper_full, "confidence_lower": lower_full,
            "cliff_lap": cliff_lap, "laps_remaining": laps_remaining,
        })

    data = build_prediction_output(str(session_path), "44", stint_results)
    output_path = output_dir / "2024_1_R_44_tyre.json"
    save_prediction(data, output_path, lg)
    gc.collect()

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

    def test_training_completes(self) -> None:
        self.assertTrue((self.fixture["models_dir"] / "tyre_lstm.pt").exists())
        self.assertTrue((self.fixture["models_dir"] / "tyre_scaler.pkl").exists())
        self.assertTrue((self.fixture["models_dir"] / "tyre_config.json").exists())

    def test_config_has_required_keys(self) -> None:
        with open(self.fixture["models_dir"] / "tyre_config.json", "r") as f:
            config = json.load(f)
        for key in ["hidden_size", "learning_rate", "input_size",
                     "sequence_length", "y_mean", "y_std", "val_mse", "test_mae_kmh"]:
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
