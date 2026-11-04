"""
================================================================================
  ApexHunter - Big Data Integration
  Script: bigdata_tab.py
--------------------------------------------------------------------------------
  Purpose : Streamlit dashboard tab that renders comprehensive Big Data
            analytics across HDFS, Spark ETL, telemetry exploration,
            pathfinding algorithms, and MongoDB.

  Usage   : Imported by frontend/app.py — render_bigdata_tab() is the entry point
================================================================================
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from config import DRIVER_MAPPING

# ── Backend script path injection for mongo_manager ───────────────────────────
_BACKEND_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "backend" / "scripts"
if str(_BACKEND_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SCRIPTS_DIR))

try:
    from bigdata.mongo_manager import (  # type: ignore
        get_client, check_mongo_available,
        get_mistake_leaderboard,
        get_session_summary, get_mistakes_by_driver,
    )
    from utils import DATA_LAKE_DIR  # type: ignore
    MONGO_IMPORT_OK: bool = True
except ImportError:
    MONGO_IMPORT_OK = False
    from config import PROJECT_ROOT
    DATA_LAKE_DIR = PROJECT_ROOT / "data_lake"

from components.bigdata_charts import (
    build_pipeline_dag, build_hdfs_treemap, build_hdfs_bar,
    build_spark_gauge, build_spark_waterfall, build_session_file_size_bar,
    build_speed_distribution, build_correlation_heatmap, build_speed_rpm_scatter,
    build_speed_over_distance,
    build_algo_comparison_bar,
)

# ── Configuration ─────────────────────────────────────────────────────────────
HADOOP_CMD: str = "hadoop.cmd" if os.name == "nt" else "hadoop"
HDFS_STATUS_LOG: Path = Path(DATA_LAKE_DIR) / "hdfs_status.json"
SPARK_RUN_LOG: Path = Path(DATA_LAKE_DIR) / "spark_run_log.json"
MONGO_STATUS_LOG: Path = Path(DATA_LAKE_DIR) / "mongo_status.json"
TELEMETRY_SAMPLE_SIZE: int = 50000


# ── Data Loading Helpers ──────────────────────────────────────────────────────
def _load_json_log(path: Path) -> Optional[Dict[str, Any]]:
    """Load and parse a JSON log file, returning None on any failure.

    Args:
        path: Filesystem path to the JSON file.

    Returns:
        Parsed dict on success, or None if the file is missing or invalid.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=15)


def _check_hdfs_live() -> bool:
    """Live ping HDFS by running 'hadoop fs -ls /'. Cached for 15 seconds.

    Returns:
        True if HDFS responds successfully, False otherwise.
    """
    try:
        result = subprocess.run(
            [HADOOP_CMD, "fs", "-ls", "/"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


@st.cache_data(ttl=60)


def _load_sample_telemetry() -> Optional[pd.DataFrame]:
    """Load a sample session parquet for telemetry visualizations.

    Returns:
        A DataFrame sampled to at most 50,000 rows, or None if unavailable.
    """
    clean_dir = Path(DATA_LAKE_DIR) / "clean_data"
    if not clean_dir.exists():
        return None

    files = sorted(clean_dir.glob("*.parquet"))
    if not files:
        return None

    df = pd.read_parquet(files[0])
    if len(df) > TELEMETRY_SAMPLE_SIZE:
        df = df.sample(TELEMETRY_SAMPLE_SIZE, random_state=42)

    return df


@st.cache_resource(ttl=60)
def _mongo_available() -> bool:
    """Whether MongoDB answers, re-checked at most once a minute.

    Checking on every rerun cost ~3 s per click whenever MongoDB was offline (the ping timeout)
    and logged the same error each time.
    """
    return MONGO_IMPORT_OK and check_mongo_available()


@st.cache_data(ttl=30)


def _fetch_leaderboard() -> List[Dict[str, Any]]:
    """Fetch mistake leaderboard from MongoDB with 30s TTL cache.

    Returns:
        List of leaderboard dicts, or empty list if unavailable.
    """
    if not _mongo_available():
        return []
    try:
        client = get_client()
        return get_mistake_leaderboard(client["apexhunter"])
    except Exception:
        return []


# ── Section Renderers ─────────────────────────────────────────────────────────
def _render_pipeline_banner() -> None:
    """Render the Technology Stack and Pipeline Architecture banner section."""
    # The old "tech coverage radar" plotted hard-coded scores (not measured), so it was removed
    st.markdown("#### Pipeline")
    col_dag = st.container()
    with col_dag:
        st.caption("From the FastF1 download to the dashboard: storage, cleaning and the models.")
        st.plotly_chart(build_pipeline_dag(), width='stretch')


def _render_hdfs_section(hdfs_log: Optional[Dict[str, Any]]) -> None:
    """Render the HDFS Distributed Storage analytics section.

    Args:
        hdfs_log: Parsed hdfs_status.json dict, or None if unavailable.
    """
    st.markdown("#### HDFS storage")

    if hdfs_log is None:
        st.info("Run `hdfs_manager.py --stats` to populate HDFS statistics.")
        return

    hdfs_live = _check_hdfs_live()
    summary = hdfs_log.get("summary", {})

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(label="Status", value="🟢 Online" if hdfs_live else "🔴 Offline")
    with c2:
        st.metric(label="Total Files", value=f"{summary.get('total_files', 0):,}")
    with c3:
        st.metric(label="Total Size", value=f"{summary.get('total_size_gb', 0):.2f} GB")
    with c4:
        dirs = hdfs_log.get("directories", {})
        sd = dirs.get("season_data", {})
        st.metric(label="Sessions Ingested", value=f"{sd.get('file_count', 0)}")

    # Charts row
    col_tree, col_bar = st.columns([1, 1])
    with col_tree:
        st.markdown("##### Storage Treemap")
        st.plotly_chart(build_hdfs_treemap(hdfs_log), width='stretch')
    with col_bar:
        st.markdown("##### Directory Breakdown")
        st.plotly_chart(build_hdfs_bar(hdfs_log), width='stretch')

    # Session files grouped by round
    st.markdown("##### Sessions per Grand Prix Round")
    st.plotly_chart(build_session_file_size_bar(hdfs_log), width='stretch')

    st.caption(f"Last updated: {hdfs_log.get('generated_at', 'Unknown')}")


def _render_spark_section(spark_log: Optional[Dict[str, Any]]) -> None:
    """Render the Apache Spark ETL Processing analytics section.

    Args:
        spark_log: Parsed spark_run_log.json dict, or None if unavailable.
    """
    st.markdown("#### Spark cleaning (ETL)")

    if spark_log is None:
        st.info("Run `spark_clean_telemetry.py --hdfs --force` to populate Spark statistics.")
        return

    status = spark_log.get("status", "")
    if status == "success":
        st.markdown("🟢 **Last Run Successful**")
    else:
        st.markdown("🔴 **Last Run Failed**")

    # Metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(label="Rows In", value=f"{spark_log.get('rows_input', 0):,}")
    with c2:
        st.metric(label="Rows Out", value=f"{spark_log.get('rows_output', 0):,}")
    with c3:
        st.metric(label="Drop Rate", value=f"{spark_log.get('drop_rate_pct', 0.0)}%")
    with c4:
        st.metric(label="Duration", value=f"{spark_log.get('duration_seconds', 0.0):.1f}s")
    with c5:
        st.metric(label="Output Size", value=f"{spark_log.get('output_size_mb', 0.0):.0f} MB")

    # Charts
    col_gauge, col_waterfall = st.columns([1, 2])
    with col_gauge:
        st.markdown("##### Data Retention Gauge")
        st.plotly_chart(build_spark_gauge(spark_log), width='stretch')
    with col_waterfall:
        st.markdown("##### Row Processing Waterfall")
        st.plotly_chart(build_spark_waterfall(spark_log), width='stretch')

    # Spark config details
    with st.expander("🔧 Spark Configuration"):
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| **Mode** | `{spark_log.get('mode', 'N/A').upper()}` |
| **Input** | `{spark_log.get('input_path', 'N/A')}` |
| **Output** | `{spark_log.get('output_path', 'N/A')}` |
| **Spark Version** | `{spark_log.get('spark_version', 'N/A')}` |
| **Timestamp** | `{spark_log.get('run_timestamp', 'N/A')}` |
        """)

    if status == "failed":
        st.error(f"Error: {spark_log.get('error', 'Unknown error')}")


def _render_telemetry_section(df_full: Optional[pd.DataFrame] = None) -> None:
    """Render the Telemetry Data Exploration section.

    Args:
        df_full: Full session DataFrame from sidebar selection, or None.
    """
    st.markdown("#### Telemetry data exploration")

    # Use the sidebar-selected session data if available, else fall back to sample
    if df_full is not None and not df_full.empty:
        sample_df = (
            df_full.sample(min(TELEMETRY_SAMPLE_SIZE, len(df_full)), random_state=42)
            if len(df_full) > TELEMETRY_SAMPLE_SIZE
            else df_full
        )
        st.caption(f"Showing {len(sample_df):,} rows from the currently selected session")
    else:
        sample_df = _load_sample_telemetry()
        if sample_df is not None:
            st.caption(f"Sampled {len(sample_df):,} rows from a default session")

    if sample_df is None:
        st.info("No local telemetry data found for exploration charts.")
        return

    # Row 1: Speed distribution + Correlation heatmap
    col_v, col_c = st.columns([1, 1])
    with col_v:
        st.markdown("##### Speed Distribution by Driver")
        st.plotly_chart(build_speed_distribution(sample_df), width='stretch')
    with col_c:
        st.markdown("##### Feature Correlation Matrix")
        st.plotly_chart(build_correlation_heatmap(sample_df), width='stretch')

    # Row 2: Speed-RPM scatter + Speed trace
    col_s, col_h = st.columns([1, 1])
    with col_s:
        st.markdown("##### Speed vs RPM (Colored by Gear)")
        st.plotly_chart(build_speed_rpm_scatter(sample_df), width='stretch')
    with col_h:
        st.markdown("##### Speed Over Session Time")
        st.plotly_chart(build_speed_over_distance(sample_df), width='stretch')

    # Quick stats
    st.markdown("##### Session Quick Stats")
    if "Speed" in sample_df.columns:
        stats = [("Avg speed", f"{sample_df['Speed'].mean():.0f} km/h"),
                 ("Max speed", f"{sample_df['Speed'].max():.0f} km/h"),
                 ("Drivers", f"{sample_df['Driver'].nunique()}"),
                 ("Avg RPM", f"{sample_df['RPM'].mean():,.0f}"),
                 ("Max RPM", f"{sample_df['RPM'].max():,.0f}"),
                 ("Readings", f"{len(sample_df):,}")]
        for row in (stats[:3], stats[3:]):
            for col, (label, value) in zip(st.columns(3), row):
                col.metric(label=label, value=value)


def _render_pathfinding_section(preferred: Optional[str] = None) -> None:
    """Render the Pathfinding Algorithm Comparison section (preferred = the chosen session and
    driver's racing-line file; otherwise the first saved one, and the caption says which)."""
    st.markdown("#### Racing-line search: algorithm comparison")

    racing_dir = Path(DATA_LAKE_DIR) / "racing_lines"
    racing_files = sorted(racing_dir.glob("*.json")) if racing_dir.exists() else []

    if not racing_files:
        st.info("Run `optimal_line.py` to generate pathfinding data.")
        return

    chosen = next((f for f in racing_files if preferred and f.name == Path(preferred).name), racing_files[0])
    st.caption(f"Racing line: {chosen.stem.replace('_racing_line', '')} (session_round_type_driver)")
    racing_data = _load_json_log(chosen)
    if not racing_data:
        st.info("Run `optimal_line.py` to generate pathfinding data.")
        return

    algos = racing_data.get("algorithms", {})

    # Metrics
    cols = st.columns(len(algos))
    for i, (k, v) in enumerate(algos.items()):
        with cols[i]:
            st.markdown(f"**{k.upper()}**")
            st.metric(label="Cost", value=f"{v.get('cost', 0):,.1f}")
            st.metric(label="Nodes Expanded", value=f"{v.get('nodes_expanded', 0):,}")
            st.metric(label="Compute Time", value=f"{v.get('compute_time_s', 0) * 1000:.1f} ms")

    st.markdown("##### Algorithm Performance Comparison")
    st.plotly_chart(build_algo_comparison_bar(racing_data), width='stretch')


def _render_mongo_section() -> None:
    """Render the MongoDB Document Analytics section."""
    st.markdown("#### MongoDB documents")

    try:
        if not MONGO_IMPORT_OK:
            raise Exception("Module not available")
        if not _mongo_available():
            raise Exception("MongoDB offline")

        client = get_client()
        db = client["apexhunter"]

        # Summary metrics
        summary = get_session_summary(db)
        # No "mistake rate" card or ranking: every detector flags 2 % of readings by design (#41)
        c1, c2, c4 = st.columns(3)
        with c1:
            st.metric(label="Sessions in DB", value=f"{summary.get('total_sessions', 0):,}")
        with c2:
            st.metric(label="Drivers analysed", value=f"{summary.get('total_drivers_analyzed', 0):,}")
        with c4:
            q = summary.get("sessions_by_type", {}).get("Q", 0)
            r = summary.get("sessions_by_type", {}).get("R", 0)
            st.metric(label="Q / R Sessions", value=f"{q} / {r}")

        # Leaderboard
        leaderboard = _fetch_leaderboard()
        if leaderboard:
            st.markdown("##### Drivers in the database")
            df_lb = pd.DataFrame(leaderboard).rename(columns={
                "driver": "Driver", "session_file": "Session", "total_rows": "Readings"})
            keep = [c for c in ("Session", "Driver", "Readings") if c in df_lb.columns]
            st.dataframe(df_lb[keep].sort_values(keep[:2]) if keep else df_lb, width='stretch',
                         height=300, hide_index=True)

        # Per-driver drill-down
        if leaderboard:
            st.markdown("##### Per-Driver Deep Dive")
            driver_list = sorted(set(d["driver"] for d in leaderboard if "driver" in d))
            if driver_list:
                driver_labels = [f"{DRIVER_MAPPING.get(d, 'Unknown')} (#{d})" for d in driver_list]
                label_to_num = dict(zip(driver_labels, driver_list))
                sel_label = st.selectbox("Select Driver", driver_labels, key="bd_driver_select", filter_mode=None)
                sel = label_to_num.get(sel_label)
                if sel:
                    rows = get_mistakes_by_driver(db, sel)
                    if rows:
                        df_dm = pd.DataFrame(rows)
                        keep = [
                            c for c in [
                                "session_file", "mean_speed_at_mistake", "mean_anomaly_score", "total_rows",
                            ]
                            if c in df_dm.columns
                        ]
                        if keep:
                            st.dataframe(df_dm[keep], width='stretch')
        else:
            st.info("No mistake data uploaded yet.")

    except Exception as e:
        mongo_status = _load_json_log(MONGO_STATUS_LOG)
        if mongo_status:
            st.markdown(f"**Database:** `{mongo_status.get('database', 'N/A')}`")
            for coll_name, info in mongo_status.get("collections", {}).items():
                st.metric(label=f"{coll_name} docs", value=info.get("document_count", 0))
        else:
            st.warning(f"MongoDB unavailable: {e}. Start MongoDB and run backend/scripts/bigdata/mongo_manager.py --upload.")


# ── Main Entry Point ──────────────────────────────────────────────────────────
def render_bigdata_tab(df_full: Optional[pd.DataFrame] = None, racing_line_path: Optional[str] = None) -> None:
    """Render the Big Data analytics tab in the Streamlit dashboard.

    Args:
        df_full: Optional full session DataFrame from sidebar selection.
        racing_line_path: The chosen session and driver's racing-line file, if any.
    """

    _render_pipeline_banner()
    st.markdown("---")
    _render_hdfs_section(_load_json_log(HDFS_STATUS_LOG))
    st.markdown("---")
    _render_spark_section(_load_json_log(SPARK_RUN_LOG))
    st.markdown("---")
    _render_telemetry_section(df_full)
    st.markdown("---")
    _render_pathfinding_section(racing_line_path)
    st.markdown("---")
    _render_mongo_section()
