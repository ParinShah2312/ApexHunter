"""Feature engineering for the ApexHunter Isolation Forest pipeline.

Every sample is compared with the field *at the same point on track*: samples are projected onto
a reference lap to get a track position (metres along the lap), a reference profile is built from
the fastest other drivers per 10 m of track, and the features are the driver's deviations from
that profile. Absolute values (speed, brake flag, ...) are not used: they describe where the car
is on track, which made the old model flag ordinary braking zones (ROADMAP §B).
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ── Feature engineering constants ─────────────────────────────────────────────
POSITION_SCALE_M: float = 0.1           # FastF1 X/Y are in 1/10 m (bug #18)
LINE_STEP_M: float = 1.0                # reference line is resampled to 1 m (raw spacing reaches ~28 m)
TRACK_BIN_M: float = 10.0               # reference profile resolution
ACCEL_WINDOW_S: float = 0.5             # speed change measured over 0.5 s (samples are irregular)
BRAKE_ON_THRESHOLD: float = 0.5         # FastF1 Brake is a binary flag (0 = off, 1 = on)
PEDAL_NOISE_THRESHOLD: float = 5.0      # minimum throttle % to consider meaningful
# Centred rolling mean of the features within each lap (0 = off). Speed/throttle/brake arrive at only
# ~4 Hz, so a single sample at a brake onset often deviates from the field by chance (83-93 % of
# flagged braking samples were in the first/last 0.2 s of a braking zone). 0.5 s raised the
# planted-mistake PR-AUC on 3 of 4 runs (0.156 -> 0.186, 0.161 -> 0.185, 0.190 -> 0.254) and left
# the 4th level (0.185 -> 0.183).
SMOOTH_WINDOW_S: float = 0.5

# Telemetry channel -> profile name, and the smallest spread allowed (stops a deviation being
# divided by ~0 where every reference lap is identical, e.g. full throttle on a straight)
SPREAD_CHANNELS = {"Speed": ("speed", 3.0), "Throttle": ("throttle", 5.0),
                   "nGear": ("gear", 0.5), "accel": ("accel", 5.0)}
IQR_TO_STD: float = 1.349                # IQR of a normal distribution in standard deviations

# Model inputs. Pedal overlap is computed but NOT a model input: it is a 0/1 flag that is on in
# ~1 % of samples (mostly the one sample where the brake comes on before the throttle reads 0),
# and the forest isolates any rare non-zero value in one split. With it, 90-99 % of flags were
# braking samples on 4 of 5 test runs; without it 37-54 % (base rate 15-29 %).
FEATURE_COLUMNS: List[str] = [
    "speed_deficit",
    "throttle_diff",
    "brake_mismatch",
    "gear_diff",
    "accel_diff",
]
# Not model inputs; written to the output for the event reasons: pedal overlap and the raw
# (unscaled, unsmoothed) differences from the field median in readable units.
INFO_COLUMNS: List[str] = ["brake_throttle_overlap", "speed_vs_field_kmh", "throttle_vs_field_pct",
                           "gear_vs_field", "field_brake_share"]


def build_track_line(df_lap: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Resample one lap's X/Y path to a point every LINE_STEP_M metres.

    Returns:
        (points, s): points is (n, 2) in FastF1 units, s is the distance along the lap in metres.
    """
    xy = df_lap[["X", "Y"]].to_numpy(dtype=float)
    step_m = np.linalg.norm(np.diff(xy, axis=0), axis=1) * POSITION_SCALE_M
    keep = np.concatenate([[True], step_m > 0])             # repeated points break interpolation
    xy = xy[keep]
    s_raw = np.concatenate([[0.0], np.cumsum(step_m[step_m > 0])])
    s = np.arange(0.0, s_raw[-1], LINE_STEP_M)
    points = np.column_stack([np.interp(s, s_raw, xy[:, 0]), np.interp(s, s_raw, xy[:, 1])])
    return points, s


def project_to_track(df: pd.DataFrame, line_points: np.ndarray, line_s: np.ndarray) -> np.ndarray:
    """Track position (metres along the reference lap) of every sample, by nearest line point."""
    _, nearest = cKDTree(line_points).query(df[["X", "Y"]].to_numpy(dtype=float))
    return line_s[nearest]


def compute_acceleration(df: pd.DataFrame) -> pd.Series:
    """Speed change per second over the last ACCEL_WINDOW_S seconds (km/h per s).

    `df` must be one driver's telemetry sorted by SessionTime. Using a fixed time window instead
    of the previous sample avoids the irregular sample spacing (0.003-0.34 s) turning 1 km/h
    rounding steps into huge rates.
    """
    t = df["SessionTime"].dt.total_seconds().to_numpy()
    v = df["Speed"].to_numpy(dtype=float)
    v_before = np.interp(t - ACCEL_WINDOW_S, t, v)
    return pd.Series((v - v_before) / ACCEL_WINDOW_S, index=df.index, dtype="float32")


def build_reference_profile(df_ref: pd.DataFrame, lap_length_m: float) -> pd.DataFrame:
    """Field reference per TRACK_BIN_M of track from the reference drivers' laps.

    `df_ref` needs track_pos, Speed, Throttle, Brake, nGear and accel. For speed, throttle, gear
    and acceleration each bin holds the median (`ref_<name>`) and a robust spread
    (`spread_<name>` = IQR / 1.349, at least the channel's floor); for the brake it holds the
    share of samples with the brake on (`ref_brake`). Bins without samples are filled from their
    neighbours (the lap is a loop).

    Returns:
        DataFrame indexed by bin number (0 .. n_bins-1).
    """
    n_bins = int(np.ceil(lap_length_m / TRACK_BIN_M))
    bins = np.minimum((df_ref["track_pos"].to_numpy() // TRACK_BIN_M).astype(int), n_bins - 1)
    grouped = df_ref.assign(
        bin=bins, brake_on=(df_ref["Brake"] >= BRAKE_ON_THRESHOLD).astype(float)
    ).groupby("bin")
    columns = {"ref_brake": grouped["brake_on"].mean()}
    for channel, (name, floor) in SPREAD_CHANNELS.items():
        iqr = grouped[channel].quantile(0.75) - grouped[channel].quantile(0.25)
        columns[f"ref_{name}"] = grouped[channel].median()
        columns[f"spread_{name}"] = (iqr / IQR_TO_STD).clip(lower=floor)
    profile = pd.DataFrame(columns).reindex(range(n_bins))
    # Circular fill: repeat the profile on both sides so gaps at the start/end interpolate across the line
    looped = pd.concat([profile, profile, profile], ignore_index=True).interpolate(limit_direction="both")
    return looped.iloc[n_bins:2 * n_bins].set_axis(profile.index).astype("float32")


def engineer_features(df: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """Deviation of each sample from the reference profile at the same track position.

    Speed, throttle, gear and acceleration deviations are divided by the field's spread at that
    point, so they read as "how many standard deviations from the field": a 15 km/h difference
    is ordinary at a braking point (where braking points vary by 10-20 m) but large on a
    straight. Without this, braking-zone entries still made up 73 % of the flags (2023 R1, #1).

    Args:
        df: The driver's samples with track_pos, Speed, Throttle, Brake, nGear and accel.
        profile: Output of build_reference_profile.

    Returns:
        A **new** DataFrame with FEATURE_COLUMNS + INFO_COLUMNS (float32), same index as `df`.
    """
    bins = np.minimum((df["track_pos"].to_numpy() // TRACK_BIN_M).astype(int), len(profile) - 1)
    ref = profile.iloc[bins].set_axis(df.index)
    brake_on = (df["Brake"] >= BRAKE_ON_THRESHOLD)

    features = pd.DataFrame(index=df.index)
    features["speed_deficit"] = (ref["ref_speed"] - df["Speed"]) / ref["spread_speed"]
    features["throttle_diff"] = (df["Throttle"] - ref["ref_throttle"]) / ref["spread_throttle"]
    features["brake_mismatch"] = brake_on.astype(float) - ref["ref_brake"]
    features["gear_diff"] = (df["nGear"] - ref["ref_gear"]) / ref["spread_gear"]
    features["accel_diff"] = (df["accel"] - ref["ref_accel"]) / ref["spread_accel"]
    features["brake_throttle_overlap"] = (brake_on & (df["Throttle"] > PEDAL_NOISE_THRESHOLD)).astype(float)
    features["speed_vs_field_kmh"] = df["Speed"] - ref["ref_speed"]
    features["throttle_vs_field_pct"] = df["Throttle"] - ref["ref_throttle"]
    features["gear_vs_field"] = df["nGear"] - ref["ref_gear"]
    features["field_brake_share"] = ref["ref_brake"]
    if SMOOTH_WINDOW_S > 0:
        features[FEATURE_COLUMNS] = _smooth_per_lap(features[FEATURE_COLUMNS], df)
    return features.astype("float32")


def _smooth_per_lap(features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Centred rolling mean over SMOOTH_WINDOW_S seconds within each lap (df has LapNumber, SessionTime)."""
    timed = features.set_axis(pd.Timestamp(0) + df["SessionTime"].to_numpy())
    window = pd.Timedelta(seconds=SMOOTH_WINDOW_S)
    parts = [timed[mask].rolling(window, center=True, min_periods=1).mean()
             for mask in (df["LapNumber"].to_numpy() == lap for lap in df["LapNumber"].unique())]
    out = pd.concat(parts)
    order = np.argsort(np.concatenate([np.flatnonzero(df["LapNumber"].to_numpy() == lap)
                                       for lap in df["LapNumber"].unique()]))
    return out.iloc[order].set_axis(features.index)
