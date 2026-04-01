"""Feature engineering for the ApexHunter Isolation Forest pipeline.
Transforms raw telemetry columns into normalized, model-ready features.
"""

from typing import List

import pandas as pd

# ── Feature engineering constants ─────────────────────────────────────────────
MAX_SPEED_KMH: float = 380.0
MAX_RPM: float = 15000.0
MAX_THROTTLE_PCT: float = 100.0
MAX_BRAKE_PCT: float = 100.0
PEDAL_NOISE_THRESHOLD: float = 5.0      # minimum pedal % to consider meaningful
SPEED_DELTA_CLIP: float = 50.0          # max row-to-row speed change to allow
GEAR_CHANGE_CLIP: float = 4.0           # max gear skip before treating as artifact

FEATURE_COLUMNS: List[str] = [
    "speed_normalized",
    "throttle_intensity",
    "brake_intensity",
    "brake_throttle_overlap",
    "speed_delta",
    "gear_change",
    "rpm_normalized",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered feature columns from raw telemetry.

    Args:
        df: DataFrame containing at least Speed, Throttle, Brake, RPM, nGear.

    Returns:
        A **new** DataFrame with exactly 7 float32 feature columns. The input
        DataFrame is never mutated.
    """
    features = pd.DataFrame(index=df.index)

    features["speed_normalized"] = (df["Speed"] / MAX_SPEED_KMH).astype("float32")

    features["throttle_intensity"] = (df["Throttle"] / MAX_THROTTLE_PCT).astype("float32")

    features["brake_intensity"] = (df["Brake"] / MAX_BRAKE_PCT).astype("float32")

    features["brake_throttle_overlap"] = (
        (df["Brake"] > PEDAL_NOISE_THRESHOLD) & (df["Throttle"] > PEDAL_NOISE_THRESHOLD)
    ).astype("float32")

    features["speed_delta"] = (
        df["Speed"].diff().fillna(0.0).clip(-SPEED_DELTA_CLIP, SPEED_DELTA_CLIP).astype("float32")
    )

    features["gear_change"] = (
        df["nGear"].diff().abs().fillna(0.0).clip(0.0, GEAR_CHANGE_CLIP).astype("float32")
    )

    features["rpm_normalized"] = (df["RPM"] / MAX_RPM).astype("float32")

    return features
