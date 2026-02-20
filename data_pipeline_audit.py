"""
================================================================================
  ApexHunter - F1 Analytics Project
  Script: Data Pipeline & Audit  (data_pipeline_audit.py)
  Author: ApexHunter Team
  Date  : 2026-02-19
--------------------------------------------------------------------------------
  Academic Requirement Coverage:
    Stage 1  : Data Identification & Acquisition   (Structured + Unstructured)
    Stage 2  : Data Normalization                  (snake_case headers)
    Stage 3  : Veracity & Reliability Audit        (NaN, duplicates, statistics)
    Stage 4  : Visualization                       (Histogram, Box Plot, Heatmap)
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
#  DEPENDENCY CHECK — give the user a friendly hint if something is missing
# ──────────────────────────────────────────────────────────────────────────────
import importlib
import importlib.util
import sys

# Force UTF-8 output on Windows consoles (avoids cp1252 UnicodeEncodeError
# for any Unicode symbols that appear in print statements throughout this file)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REQUIRED = {
    "fastf1"     : "fastf1",
    "pandas"     : "pandas",
    "numpy"      : "numpy",
    "matplotlib" : "matplotlib",
    "seaborn"    : "seaborn",
    "requests"   : "requests",
    "PIL"        : "Pillow",          # Pillow ships as PIL
}

missing = []
for module, pkg in REQUIRED.items():
    if importlib.util.find_spec(module) is None:
        missing.append(pkg)

if missing:
    print("\n[ERROR] The following packages are required but not installed:")
    print("        pip install " + " ".join(missing))
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
#  STANDARD IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import re
import warnings
import requests

import fastf1
import numpy  as np
import pandas as pd
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import seaborn             as sns
from   PIL import Image
from   io  import BytesIO

warnings.filterwarnings("ignore")           # suppress minor deprecation noise

# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Output directory — all artefacts land here
OUTPUT_DIR   = "apexhunter_output"
CACHE_DIR    = os.path.join(OUTPUT_DIR, "ff1_cache")
PLOT_DIR     = os.path.join(OUTPUT_DIR, "plots")
CSV_FILENAME = os.path.join(OUTPUT_DIR, "processed_f1_data.csv")
MAP_FILENAME = os.path.join(OUTPUT_DIR, "monaco_satellite_map.png")

# Race / session selectors
YEAR     = 2023
GP_NAME  = "Monaco"
SESSION  = "Q"            # Q = Qualifying

# Drivers of interest (FastF1 abbreviations)
DRIVERS = ["VER", "ALO", "LEC"]   # Verstappen, Alonso, Leclerc

# Telemetry columns we want to extract
TELEMETRY_COLS = ["Speed", "RPM", "Throttle", "Brake", "nGear", "X", "Y"]

# ──────────────────────────────────────────────────────────────────────────────
#  HELPER — create output folders
# ──────────────────────────────────────────────────────────────────────────────

def _make_dirs():
    for d in (OUTPUT_DIR, CACHE_DIR, PLOT_DIR):
        os.makedirs(d, exist_ok=True)

_make_dirs()

# ==============================================================================
#  STAGE 1 — IDENTIFY & ACQUIRE
#  Step 1 : FastF1 telemetry download
#  Step 3 : Satellite / unstructured image download
# ==============================================================================

def stage1_acquire() -> pd.DataFrame:
    """
    Downloads Monaco 2023 Qualifying telemetry for the top-3 drivers using
    the FastF1 library, then fetches a satellite tile image of the Monaco
    track to satisfy the 'variety / unstructured data' requirement.

    Returns
    -------
    pd.DataFrame
        Raw, combined telemetry for all three drivers.
    """
    print("\n" + "="*72)
    print("  STAGE 1 — IDENTIFY & ACQUIRE")
    print("="*72)

    # ------------------------------------------------------------------
    # 1a. Configure FastF1 cache so repeated runs don't re-download data
    # ------------------------------------------------------------------
    fastf1.Cache.enable_cache(CACHE_DIR)
    print(f"[INFO] FastF1 cache directory : {os.path.abspath(CACHE_DIR)}")

    # ------------------------------------------------------------------
    # 1b. Load the Monaco 2023 Qualifying session
    # ------------------------------------------------------------------
    print(f"[INFO] Loading session  : {YEAR} {GP_NAME} Grand Prix — Qualifying")
    session = fastf1.get_session(YEAR, GP_NAME, SESSION)
    session.load(telemetry=True, laps=True)
    print("[INFO] Session loaded successfully.")

    # ------------------------------------------------------------------
    # 1c. Extract the fastest lap telemetry for each target driver
    # ------------------------------------------------------------------
    frames = []
    for drv_code in DRIVERS:
        try:
            # Fetch laps for this driver
            drv_laps = session.laps.pick_driver(drv_code)

            # Fastest qualifying lap
            fastest_lap = drv_laps.pick_fastest()

            # Pull full (car + position) telemetry
            # Note: with_distance kwarg was removed in newer FastF1 versions;
            #       use add_distance() on the merged result instead.
            telemetry = fastest_lap.get_car_data().merge_channels(
                fastest_lap.get_pos_data()
            )
            if hasattr(telemetry, 'add_distance'):
                telemetry = telemetry.add_distance()

            # Keep only the columns we need (skip missing ones gracefully)
            available_cols = [c for c in TELEMETRY_COLS if c in telemetry.columns]
            df_drv = telemetry[available_cols].copy()

            # Tag each row with its driver abbreviation
            df_drv.insert(0, "Driver", drv_code)

            frames.append(df_drv)
            print(f"  [OK] [{drv_code}] Telemetry rows: {len(df_drv):,}  |  "
                  f"Columns: {available_cols}")

        except Exception as exc:
            print(f"  [SKIP] [{drv_code}] Could not extract telemetry -- {exc}")

    if not frames:
        raise RuntimeError("No telemetry data could be retrieved. Aborting.")

    raw_df = pd.concat(frames, ignore_index=True)
    print(f"\n[INFO] Combined raw DataFrame shape : {raw_df.shape}")

    # ------------------------------------------------------------------
    # 1d. VARIETY REQUIREMENT — Download satellite / map tile image
    #     We fetch an OpenStreetMap tile that covers the Monaco circuit
    #     bounding box (roughly). This represents unstructured image data.
    # ------------------------------------------------------------------
    print("\n[INFO] Fetching satellite map tile for Monaco (unstructured data) ...")
    _download_monaco_map()

    return raw_df


def _download_monaco_map():
    """
    Downloads a single OpenStreetMap map tile that covers the Monaco circuit
    and saves it as a PNG.  Uses only the standard `requests` + `PIL` stack
    so there is no dependency on contextily / GDAL at runtime.

    Tile coordinates (zoom=14, x=8578, y=5937) cover the heart of Monaco.
    """
    # OSM tile URL — zoom 14, tile (8578, 5937) covers Monaco circuit
    tile_url = "https://tile.openstreetmap.org/14/8578/5937.png"
    headers  = {
        "User-Agent": "ApexHunterF1Analytics/1.0 (academic project)"
    }

    try:
        resp = requests.get(tile_url, headers=headers, timeout=15)
        resp.raise_for_status()

        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(MAP_FILENAME)
        print(f"  [OK] Satellite map tile saved -> {os.path.abspath(MAP_FILENAME)}")
        print(f"     Image size : {img.size[0]}x{img.size[1]} px  "
              f"(Mode: {img.mode})")

    except requests.exceptions.RequestException as exc:
        # Non-fatal -- the rest of the pipeline continues
        print(f"  [WARN] Could not download map tile: {exc}")
        print("    (Pipeline will continue without the satellite image.)")


# ==============================================================================
#  STAGE 2 — NORMALIZATION
#  Step 5 : Load into Pandas DataFrame   (already done in Stage 1)
#  Step 6 : Rename all column headers to snake_case
# ==============================================================================

def _to_snake_case(name: str) -> str:
    """
    Converts a column name to snake_case.

    Examples
    --------
    'nGear'    → 'n_gear'
    'Throttle' → 'throttle'
    'RPM'      → 'rpm'
    'X'        → 'x'
    """
    # Insert underscore before any uppercase letter that follows a lowercase
    # letter or digit  (handles camelCase: nGear → n_Gear)
    s1 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    # Insert underscore before a capital letter that precedes a lowercase
    # (handles sequences like 'RPMsensor' → 'RP_Msensor' — edge-case guard)
    s2 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s1)
    return s2.lower()


def stage2_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames all DataFrame columns to snake_case and ensures consistent dtypes.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame.
    """
    print("\n" + "="*72)
    print("  STAGE 2 — NORMALIZATION")
    print("="*72)

    old_cols = df.columns.tolist()
    rename_map = {col: _to_snake_case(col) for col in old_cols}

    df = df.rename(columns=rename_map)

    print("[INFO] Column rename mapping (original → snake_case):")
    for old, new in rename_map.items():
        marker = "  (changed)" if old != new else ""
        print(f"    {old:<15} →  {new}{marker}")

    print(f"\n[INFO] Normalized column names : {df.columns.tolist()}")
    return df


# ==============================================================================
#  STAGE 3 — VERACITY & RELIABILITY AUDIT
#  Step 7 : Missing value analysis
#  Step 8 : Duplicate record detection
#  Step 9 : Descriptive statistical summary
# ==============================================================================

def stage3_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a full data-quality audit on the normalized DataFrame:
      • Missing value counts & percentages
      • Duplicate row detection
      • Descriptive statistics (mean, std, min, max) for numeric columns

    Returns the cleaned DataFrame (duplicates dropped).
    """
    print("\n" + "="*72)
    print("  STAGE 3 — VERACITY & RELIABILITY AUDIT")
    print("="*72)

    total_rows = len(df)
    print(f"[INFO] Total records before audit : {total_rows:,}")

    # ------------------------------------------------------------------
    # Step 7 — Missing Values
    # ------------------------------------------------------------------
    print("\n── Step 7 : Missing Value Analysis ──────────────────────────────────")

    null_counts  = df.isnull().sum()
    null_pct     = (null_counts / total_rows * 100).round(2)
    missing_df   = pd.DataFrame({
        "column"         : null_counts.index,
        "null_count"     : null_counts.values,
        "null_percentage": null_pct.values,
    }).sort_values("null_count", ascending=False).reset_index(drop=True)

    # Pretty-print the table
    col_w = max(len(c) for c in missing_df["column"]) + 2
    header = f"  {'Column':<{col_w}}  {'Null Count':>12}  {'Null %':>9}"
    print(header)
    print("  " + "-" * (col_w + 26))
    for _, row in missing_df.iterrows():
        flag = "  <-- [MISSING DATA DETECTED]" if row["null_count"] > 0 else ""
        print(f"  {row['column']:<{col_w}}  {int(row['null_count']):>12,}  "
              f"{row['null_percentage']:>8.2f}%{flag}")

    total_nulls = int(null_counts.sum())
    print(f"\n  Total NaN cells  : {total_nulls:,} out of "
          f"{total_rows * len(df.columns):,} "
          f"({total_nulls / (total_rows * len(df.columns)) * 100:.2f}%)")

    # ------------------------------------------------------------------
    # Step 8 — Duplicate Records
    # ------------------------------------------------------------------
    print("\n── Step 8 : Duplicate Record Detection ──────────────────────────────")

    n_duplicates = df.duplicated().sum()
    print(f"  Duplicate rows found : {n_duplicates:,}")

    if n_duplicates > 0:
        print(f"  [ACTION] Dropping {n_duplicates} duplicate row(s) ...")
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  Records after deduplication : {len(df):,}")
    else:
        print("  [OK] No duplicate rows detected. Dataset integrity is sound.")

    # ------------------------------------------------------------------
    # Step 9 — Descriptive Statistics (outlier detection)
    # ------------------------------------------------------------------
    print("\n── Step 9 : Descriptive Statistical Summary ─────────────────────────")
    print("  (Focus: Speed, RPM, Throttle, Brake)")

    # Select only numeric columns that are likely to exist
    focus_cols = [c for c in ["speed", "rpm", "throttle", "brake"] if c in df.columns]
    stats = df[focus_cols].describe().T
    stats.index.name = "column"

    # Limit to mean, std, min, 25%, 50%, 75%, max
    stats = stats[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(4)

    print("\n" + stats.to_string())

    # Spot-check for obvious outliers in Speed
    if "speed" in df.columns:
        q1, q3 = df["speed"].quantile(0.25), df["speed"].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers_speed = df[(df["speed"] < lower_fence) | (df["speed"] > upper_fence)]
        print(f"\n  [Speed Outlier Check — IQR Method]")
        print(f"    IQR fences : [{lower_fence:.2f}, {upper_fence:.2f}] km/h")
        print(f"    Outlier rows (Speed) : {len(outliers_speed):,}")

    # Spot-check for obvious outliers in RPM
    if "rpm" in df.columns:
        q1, q3 = df["rpm"].quantile(0.25), df["rpm"].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers_rpm = df[(df["rpm"] < lower_fence) | (df["rpm"] > upper_fence)]
        print(f"\n  [RPM Outlier Check — IQR Method]")
        print(f"    IQR fences : [{lower_fence:.2f}, {upper_fence:.2f}]")
        print(f"    Outlier rows (RPM)   : {len(outliers_rpm):,}")

    print(f"\n[INFO] Clean DataFrame shape after audit : {df.shape}")
    return df


# ==============================================================================
#  STAGE 4 — VISUALIZATION
#  Step 10 : 3 plots saved as PNG
# ==============================================================================

DRIVER_PALETTE = {
    "VER": "#3671C6",   # Red Bull blue
    "ALO": "#358C75",   # Aston Martin green
    "LEC": "#E8002D",   # Ferrari red
}

DRIVER_LABELS = {
    "VER": "Verstappen (RB19)",
    "ALO": "Alonso (AMR23)",
    "LEC": "Leclerc (SF-23)",
}


def stage4_visualize(df: pd.DataFrame):
    """
    Generates and saves three publication-quality plots:
      1. Speed Distribution Histogram (per driver)
      2. RPM Box Plot (per driver, for outlier comparison)
      3. Correlation Heatmap (Speed, Throttle, Brake, RPM)
    """
    print("\n" + "="*72)
    print("  STAGE 4 — VISUALIZATION")
    print("="*72)

    # Apply a clean, modern seaborn style
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({
        "figure.dpi"        : 150,
        "font.family"       : "DejaVu Sans",
        "axes.titlesize"    : 13,
        "axes.labelsize"    : 11,
        "xtick.labelsize"   : 9,
        "ytick.labelsize"   : 9,
        "legend.fontsize"   : 9,
    })

    # ----------------------------------------------------------------
    # Plot 1 — Speed Distribution Histogram
    # ----------------------------------------------------------------
    _plot_speed_histogram(df)

    # ----------------------------------------------------------------
    # Plot 2 — RPM Box Plot per Driver
    # ----------------------------------------------------------------
    _plot_rpm_boxplot(df)

    # ----------------------------------------------------------------
    # Plot 3 — Correlation Heatmap
    # ----------------------------------------------------------------
    _plot_correlation_heatmap(df)

    print("\n[INFO] All plots saved successfully.")


def _plot_speed_histogram(df: pd.DataFrame):
    """Plot 1 — Speed distribution histogram per driver."""
    fig, ax = plt.subplots(figsize=(10, 5))

    drivers_present = [d for d in DRIVERS if d in df["driver"].unique()]

    for drv in drivers_present:
        subset = df[df["driver"] == drv]["speed"].dropna()
        color  = DRIVER_PALETTE.get(drv, "#999999")
        label  = DRIVER_LABELS.get(drv, drv)
        ax.hist(
            subset,
            bins      = 60,
            alpha     = 0.65,
            color     = color,
            label     = label,
            edgecolor = "white",
            linewidth = 0.4,
        )

    ax.set_title("Speed Distribution — Monaco 2023 Qualifying (Fastest Lap)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Frequency (telemetry samples)")
    ax.legend(loc="upper left")

    # Add vertical mean lines
    for drv in drivers_present:
        mean_spd = df[df["driver"] == drv]["speed"].mean()
        ax.axvline(mean_spd, color=DRIVER_PALETTE.get(drv, "#999"),
                   linestyle="--", linewidth=1.2, alpha=0.8)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "plot1_speed_histogram.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Plot 1 saved -> {os.path.abspath(path)}")


def _plot_rpm_boxplot(df: pd.DataFrame):
    """Plot 2 — RPM box plot per driver for outlier comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))

    drivers_present = [d for d in DRIVERS if d in df["driver"].unique()]
    plot_data  = [df[df["driver"] == d]["rpm"].dropna().values for d in drivers_present]
    tick_labels = [DRIVER_LABELS.get(d, d) for d in drivers_present]
    colors      = [DRIVER_PALETTE.get(d, "#999999")          for d in drivers_present]

    bp = ax.boxplot(
        plot_data,
        patch_artist = True,
        notch        = True,
        widths        = 0.45,
        medianprops  = dict(color="white", linewidth=2.5),
        whiskerprops = dict(linewidth=1.2),
        capprops     = dict(linewidth=1.5),
        flierprops   = dict(marker="o", markersize=2.5, alpha=0.4),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(tick_labels)
    ax.set_title("RPM Distribution & Outliers — Monaco 2023 Qualifying",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Driver")
    ax.set_ylabel("Engine RPM")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "plot2_rpm_boxplot.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Plot 2 saved -> {os.path.abspath(path)}")


def _plot_correlation_heatmap(df: pd.DataFrame):
    """Plot 3 — Correlation heatmap: Speed, Throttle, Brake, RPM."""
    corr_cols = [c for c in ["speed", "throttle", "brake", "rpm"] if c in df.columns]

    if len(corr_cols) < 2:
        print("  [WARN] Not enough numeric columns for a correlation heatmap.")
        return

    corr_matrix = df[corr_cols].corr()

    # Human-readable axis labels (Title Case)
    axis_labels = [c.replace("_", " ").title() for c in corr_cols]

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.zeros_like(corr_matrix, dtype=bool)
    np.fill_diagonal(mask, True)          # hide the trivial diagonal

    hm = sns.heatmap(
        corr_matrix,
        annot       = True,
        fmt         = ".2f",
        cmap        = "coolwarm",
        vmin        = -1,
        vmax        = 1,
        linewidths  = 0.5,
        ax          = ax,
        xticklabels = axis_labels,
        yticklabels = axis_labels,
        square      = True,
        cbar_kws    = {"label": "Pearson Correlation Coefficient"},
    )

    ax.set_title("Feature Correlation — Speed, Throttle, Brake, RPM\n"
                 "Monaco 2023 Qualifying (All Drivers Combined)",
                 fontweight="bold", pad=12)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "plot3_correlation_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Plot 3 saved -> {os.path.abspath(path)}")


# ==============================================================================
#  OUTPUT — Save cleaned DataFrame as CSV (ready for HDFS `put`)
# ==============================================================================

def save_csv(df: pd.DataFrame):
    """Saves the fully processed DataFrame as a UTF-8 CSV file."""
    print("\n" + "="*72)
    print("  OUTPUT — Saving Processed Data as CSV")
    print("="*72)

    df.to_csv(CSV_FILENAME, index=False, encoding="utf-8")
    size_kb = os.path.getsize(CSV_FILENAME) / 1024
    print(f"  [OK] CSV saved  -> {os.path.abspath(CSV_FILENAME)}")
    print(f"     Rows     : {len(df):,}")
    print(f"     Columns  : {len(df.columns)}")
    print(f"     File size: {size_kb:.1f} KB")
    print("\n  To upload to HDFS (run in your Hadoop environment):")
    print(f"    hdfs dfs -put {CSV_FILENAME} /user/apexhunter/processed_f1_data.csv")


# ==============================================================================
#  MAIN ENTRY POINT
# ==============================================================================

def main():
    print("\n" + "#"*72)
    print("#  ApexHunter — F1 Analytics   |   Data Pipeline & Audit Script   #")
    print("#" + " "*70 + "#")
    print(f"#  Race    : {YEAR} {GP_NAME} Grand Prix — Qualifying{' '*28}#")
    print(f"#  Drivers : {', '.join(DRIVERS)}{' '*56}#")
    print("#"*72)

    try:
        # ── Stage 1 ───────────────────────────────────────────────────────────
        raw_df = stage1_acquire()

        # ── Stage 2 ───────────────────────────────────────────────────────────
        norm_df = stage2_normalize(raw_df)

        # ── Stage 3 ───────────────────────────────────────────────────────────
        clean_df = stage3_audit(norm_df)

        # ── Stage 4 ───────────────────────────────────────────────────────────
        stage4_visualize(clean_df)

        # ── Output ────────────────────────────────────────────────────────────
        save_csv(clean_df)

    except Exception as exc:
        print(f"\n[FATAL] Pipeline aborted: {exc}")
        raise

    print("\n" + "="*72)
    print("  [DONE] Pipeline complete!  All artefacts written to:")
    print(f"         {os.path.abspath(OUTPUT_DIR)}")
    print("="*72 + "\n")


if __name__ == "__main__":
    main()
