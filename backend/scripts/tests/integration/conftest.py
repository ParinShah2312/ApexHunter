"""
Shared pipeline fixture for ApexHunter integration tests.
Runs the full detect_mistakes pipeline once per test session and caches the output.
Import run_pipeline_once() in each integration test's setUpClass.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

_FIXTURE_CACHE = {}


def run_pipeline_once() -> dict:
    """
    Builds synthetic data and runs the full detect_mistakes pipeline exactly once.
    Subsequent calls return the cached result immediately.
    """
    if _FIXTURE_CACHE:
        return _FIXTURE_CACHE.copy()

    tmp = Path(tempfile.mkdtemp())

    n = 500
    rng = np.random.RandomState(42)
    speed = np.full(n, 250.0, dtype="float32")
    rpm = np.full(n, 10000.0, dtype="float32")
    throttle = np.full(n, 80.0, dtype="float32")
    brake = np.zeros(n, dtype="float32")
    ngear = np.full(n, 6.0, dtype="float32")

    # Inject anomalies at rows 200–210
    for i in range(200, 211):
        speed[i] = 20.0
        brake[i] = 80.0
        throttle[i] = 80.0
        ngear[i] = float((i % 2) + 1)

    target_df = pd.DataFrame({
        "Driver": ["44"] * n,
        "Speed": speed,
        "RPM": rpm,
        "Throttle": throttle,
        "Brake": brake,
        "X": np.linspace(0, 100, n).astype("float32"),
        "Y": np.linspace(0, 100, n).astype("float32"),
        "nGear": ngear,
        "Time": pd.to_timedelta(range(n), unit="s"),
        "SessionTime": pd.to_timedelta(range(n), unit="s"),
    })

    ref_df = pd.DataFrame({
        "Driver": ["1"] * n,
        "Speed": np.full(n, 280.0, dtype="float32"),
        "RPM": np.full(n, 12000.0, dtype="float32"),
        "Throttle": np.full(n, 90.0, dtype="float32"),
        "Brake": np.zeros(n, dtype="float32"),
        "X": np.linspace(0, 100, n).astype("float32"),
        "Y": np.linspace(0, 100, n).astype("float32"),
        "nGear": np.full(n, 6.0, dtype="float32"),
        "Time": pd.to_timedelta(range(n), unit="s"),
        "SessionTime": pd.to_timedelta(range(n), unit="s"),
    })

    combined = pd.concat([target_df, ref_df], ignore_index=True)
    session_path = tmp / "test_session.parquet"
    combined.to_parquet(session_path, compression="snappy")

    output_dir = tmp / "mistake_data"

    from detect_mistakes import run_pipeline
    run_pipeline(str(session_path), "44", None, output_dir, True)

    stem = "test_session_44"
    _FIXTURE_CACHE.update({
        "tmp_dir": tmp,
        "parquet_path": output_dir / f"{stem}_mistakes.parquet",
        "meta_path": output_dir / f"{stem}_mistakes_meta.json",
        "input_path": session_path,
        "driver": "44",
        "n_rows": n,
    })
    return _FIXTURE_CACHE.copy()


def cleanup_pipeline_fixture():
    """Call this from the last integration test's tearDownClass."""
    if "tmp_dir" in _FIXTURE_CACHE:
        shutil.rmtree(_FIXTURE_CACHE["tmp_dir"], ignore_errors=True)
        _FIXTURE_CACHE.clear()
