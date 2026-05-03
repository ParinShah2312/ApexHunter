"""
ApexHunter Big Data Analytics Tab
Comprehensive visualizations across HDFS, Spark, MongoDB, LSTM and Pathfinding.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from config import DRIVER_MAPPING

_BACKEND_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "backend" / "scripts"
if str(_BACKEND_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SCRIPTS_DIR))

try:
    from mongo_manager import (  # type: ignore
        get_client, check_mongo_available,
        get_mistake_leaderboard, get_anomaly_score_distribution,
        get_session_summary, get_mistakes_by_driver
    )
    from utils import DATA_LAKE_DIR  # type: ignore
    MONGO_IMPORT_OK = True
except ImportError:
    MONGO_IMPORT_OK = False
    from config import PROJECT_ROOT
    DATA_LAKE_DIR = PROJECT_ROOT / "data_lake"

from components.bigdata_charts import (
    build_pipeline_dag, build_hdfs_treemap, build_hdfs_bar,
    build_spark_gauge, build_spark_waterfall, build_session_file_size_bar,
    build_speed_distribution, build_correlation_heatmap, build_speed_rpm_scatter,
    build_speed_over_distance,
    build_tyre_degradation_line, build_lstm_confidence_area,
    build_algo_comparison_bar, build_tech_radar
)
import plotly.graph_objects as go


_HADOOP_CMD = "hadoop.cmd" if os.name == "nt" else "hadoop"

def _load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=15)
def _check_hdfs_live() -> bool:
    """Live ping HDFS by running 'hadoop fs -ls /'. Cached for 15s."""
    try:
        result = subprocess.run(
            [_HADOOP_CMD, "fs", "-ls", "/"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


@st.cache_data(ttl=60)
def _load_sample_telemetry():
    """Load a sample session parquet for telemetry visualizations."""
    clean_dir = Path(DATA_LAKE_DIR) / "clean_data"
    if not clean_dir.exists():
        return None
    files = sorted(clean_dir.glob("*.parquet"))
    if not files:
        return None
    df = pd.read_parquet(files[0])
    if len(df) > 50000:
        df = df.sample(50000, random_state=42)
    return df


@st.cache_data(ttl=30)
def _fetch_leaderboard():
    if not MONGO_IMPORT_OK:
        return []
    try:
        client = get_client()
        return get_mistake_leaderboard(client["apexhunter"])
    except Exception:
        return []


@st.cache_data(ttl=30)
def _fetch_anomaly_distribution():
    if not MONGO_IMPORT_OK:
        return {}
    try:
        client = get_client()
        return get_anomaly_score_distribution(client["apexhunter"])
    except Exception:
        return {}


def render_bigdata_tab(df_full: pd.DataFrame = None):
    st.header("🗄️ Big Data Analytics Dashboard")

    # ═══════════════════════════════════════════════════════════
    #  Section 0: Technology Radar + Pipeline DAG
    # ═══════════════════════════════════════════════════════════
    st.subheader("🔬 Technology Stack & Pipeline Architecture")
    col_radar, col_dag = st.columns([1, 2])
    with col_radar:
        st.markdown("##### Tech Coverage Radar")
        st.plotly_chart(build_tech_radar(), width='stretch')
    with col_dag:
        st.markdown("##### End-to-End Pipeline DAG")
        st.plotly_chart(build_pipeline_dag(), width='stretch')

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    #  Section 1: HDFS Storage Analytics
    # ═══════════════════════════════════════════════════════════
    st.subheader("📦 HDFS Distributed Storage")
    hdfs_stats = _load_json(Path(DATA_LAKE_DIR) / "hdfs_status.json")

    if hdfs_stats is None:
        st.info("Run `hdfs_manager.py --stats` to populate HDFS statistics.")
    else:
        hdfs_live = _check_hdfs_live()
        summary = hdfs_stats.get("summary", {})

        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Status", "🟢 Online" if hdfs_live else "🔴 Offline")
        with c2:
            st.metric("Total Files", f"{summary.get('total_files', 0):,}")
        with c3:
            st.metric("Total Size", f"{summary.get('total_size_gb', 0):.2f} GB")
        with c4:
            dirs = hdfs_stats.get("directories", {})
            sd = dirs.get("season_data", {})
            st.metric("Sessions Ingested", f"{sd.get('file_count', 0)}")

        # Charts row
        col_tree, col_bar = st.columns([1, 1])
        with col_tree:
            st.markdown("##### Storage Treemap")
            st.plotly_chart(build_hdfs_treemap(hdfs_stats), width='stretch')
        with col_bar:
            st.markdown("##### Directory Breakdown")
            st.plotly_chart(build_hdfs_bar(hdfs_stats), width='stretch')

        # Session files grouped by round
        st.markdown("##### Sessions per Grand Prix Round")
        st.plotly_chart(build_session_file_size_bar(hdfs_stats), width='stretch')

        st.caption(f"Last updated: {hdfs_stats.get('generated_at', 'Unknown')}")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    #  Section 2: Spark ETL Analytics
    # ═══════════════════════════════════════════════════════════
    st.subheader("⚡ Apache Spark ETL Processing")
    spark_stats = _load_json(Path(DATA_LAKE_DIR) / "spark_run_log.json")

    if spark_stats is None:
        st.info("Run `spark_clean_telemetry.py --hdfs --force` to populate Spark statistics.")
    else:
        status = spark_stats.get("status", "")
        if status == "success":
            st.markdown("🟢 **Last Run Successful**")
        else:
            st.markdown("🔴 **Last Run Failed**")

        # Metric cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Rows In", f"{spark_stats.get('rows_input', 0):,}")
        with c2:
            st.metric("Rows Out", f"{spark_stats.get('rows_output', 0):,}")
        with c3:
            st.metric("Drop Rate", f"{spark_stats.get('drop_rate_pct', 0.0)}%")
        with c4:
            st.metric("Duration", f"{spark_stats.get('duration_seconds', 0.0):.1f}s")
        with c5:
            st.metric("Output Size", f"{spark_stats.get('output_size_mb', 0.0):.0f} MB")

        # Charts
        col_gauge, col_waterfall = st.columns([1, 2])
        with col_gauge:
            st.markdown("##### Data Retention Gauge")
            st.plotly_chart(build_spark_gauge(spark_stats), width='stretch')
        with col_waterfall:
            st.markdown("##### Row Processing Waterfall")
            st.plotly_chart(build_spark_waterfall(spark_stats), width='stretch')

        # Spark config details
        with st.expander("🔧 Spark Configuration"):
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| **Mode** | `{spark_stats.get('mode', 'N/A').upper()}` |
| **Input** | `{spark_stats.get('input_path', 'N/A')}` |
| **Output** | `{spark_stats.get('output_path', 'N/A')}` |
| **Spark Version** | `{spark_stats.get('spark_version', 'N/A')}` |
| **Timestamp** | `{spark_stats.get('run_timestamp', 'N/A')}` |
            """)

        if status == "failed":
            st.error(f"Error: {spark_stats.get('error', 'Unknown error')}")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    #  Section 3: Telemetry Data Exploration
    # ═══════════════════════════════════════════════════════════
    st.subheader("📊 Telemetry Data Exploration")

    # Use the sidebar-selected session data if available, else fall back to sample
    if df_full is not None and not df_full.empty:
        sample_df = df_full.sample(min(50000, len(df_full)), random_state=42) if len(df_full) > 50000 else df_full
        st.caption(f"Showing {len(sample_df):,} rows from the currently selected session")
    else:
        sample_df = _load_sample_telemetry()
        if sample_df is not None:
            st.caption(f"Sampled {len(sample_df):,} rows from a default session")

    if sample_df is not None:
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
            st.markdown("##### Speed Trace Overlay")
            st.plotly_chart(build_speed_over_distance(sample_df), width='stretch')

        # Quick stats
        st.markdown("##### Session Quick Stats")
        if "Speed" in sample_df.columns:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric("Avg Speed", f"{sample_df['Speed'].mean():.1f} km/h")
            with c2:
                st.metric("Max Speed", f"{sample_df['Speed'].max():.1f} km/h")
            with c3:
                st.metric("Unique Drivers", f"{sample_df['Driver'].nunique()}")
            with c4:
                st.metric("Avg RPM", f"{sample_df['RPM'].mean():.0f}")
            with c5:
                st.metric("Max RPM", f"{sample_df['RPM'].max():.0f}")
            with c6:
                st.metric("Data Points", f"{len(sample_df):,}")
    else:
        st.info("No local telemetry data found for exploration charts.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    #  Section 4: LSTM Tyre Prediction Analytics
    # ═══════════════════════════════════════════════════════════
    st.subheader("🧠 LSTM Tyre Degradation Analytics")
    tyre_dir = Path(DATA_LAKE_DIR) / "tyre_predictions"
    tyre_files = sorted(tyre_dir.glob("*.json")) if tyre_dir.exists() else []

    if tyre_files:
        tyre_data = _load_json(tyre_files[0])
        if tyre_data:
            stints = tyre_data.get("stints", [])

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Stints", len(stints))
            with c2:
                total_laps = sum(s["n_laps"] for s in stints)
                st.metric("Total Laps", total_laps)
            with c3:
                cliffs = [s for s in stints if s.get("cliff_lap") is not None]
                st.metric("Cliff Detections", len(cliffs))
            with c4:
                drv_num = tyre_data.get("driver", "—")
                drv_name = DRIVER_MAPPING.get(str(drv_num), "Unknown")
                st.metric("Driver", f"{drv_name} (#{drv_num})")

            # Charts
            col_deg, col_conf = st.columns([1, 1])
            with col_deg:
                st.markdown("##### Multi-Stint Degradation Overlay")
                st.plotly_chart(build_tyre_degradation_line(tyre_data), width='stretch')
            with col_conf:
                st.markdown("##### LSTM Confidence Band (Stint 1)")
                st.plotly_chart(build_lstm_confidence_area(tyre_data), width='stretch')
    else:
        st.info("Run `predict_cliff.py` to generate tyre predictions.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    #  Section 5: Pathfinding Algorithm Analytics
    # ═══════════════════════════════════════════════════════════
    st.subheader("🏁 Pathfinding Algorithm Comparison")
    racing_dir = Path(DATA_LAKE_DIR) / "racing_lines"
    racing_files = sorted(racing_dir.glob("*.json")) if racing_dir.exists() else []

    if racing_files:
        racing_data = _load_json(racing_files[0])
        if racing_data:
            algos = racing_data.get("algorithms", {})
            # Metrics
            cols = st.columns(len(algos))
            pal = {"astar": "#a855f7", "dijkstra": "#00d4ff", "bfs": "#ef4444"}
            for i, (k, v) in enumerate(algos.items()):
                with cols[i]:
                    st.markdown(f"**{k.upper()}**")
                    st.metric("Cost", f"{v.get('cost', 0):,.1f}")
                    st.metric("Nodes Expanded", f"{v.get('nodes_expanded', 0):,}")
                    st.metric("Compute Time", f"{v.get('compute_time_s', 0)*1000:.1f} ms")

            st.markdown("##### Algorithm Performance Comparison")
            st.plotly_chart(build_algo_comparison_bar(racing_data), width='stretch')
    else:
        st.info("Run `optimal_line.py` to generate pathfinding data.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    #  Section 6: MongoDB Analytics
    # ═══════════════════════════════════════════════════════════
    st.subheader("🍃 MongoDB Document Analytics")
    try:
        if not MONGO_IMPORT_OK:
            raise Exception("Module not available")
        if not check_mongo_available():
            raise Exception("MongoDB offline")

        client = get_client()
        db = client["apexhunter"]

        # Summary metrics
        summary = get_session_summary(db)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Sessions in DB", f"{summary.get('total_sessions', 0):,}")
        with c2:
            st.metric("Drivers Analyzed", f"{summary.get('total_drivers_analyzed', 0):,}")
        with c3:
            st.metric("Mistake Rate", f"{summary.get('overall_mistake_rate_pct', 0.0):.1f}%")
        with c4:
            q = summary.get('sessions_by_type', {}).get('Q', 0)
            r = summary.get('sessions_by_type', {}).get('R', 0)
            st.metric("Q / R Sessions", f"{q} / {r}")

        # Leaderboard
        leaderboard = _fetch_leaderboard()
        if leaderboard:
            st.markdown("##### Driver Mistake Leaderboard")
            df_lb = pd.DataFrame(leaderboard).rename(columns={
                "driver": "Driver", "session_file": "Session",
                "mistake_rate_pct": "Mistake %", "mean_anomaly_score": "Anomaly Score",
                "total_rows": "Rows"
            })
            st.dataframe(df_lb, width='stretch', height=300)

        # Per-driver drill-down
        if leaderboard:
            st.markdown("##### Per-Driver Deep Dive")
            driver_list = sorted(set(d["driver"] for d in leaderboard if "driver" in d))
            if driver_list:
                driver_labels = [f"{DRIVER_MAPPING.get(d, 'Unknown')} (#{d})" for d in driver_list]
                label_to_num = dict(zip(driver_labels, driver_list))
                sel_label = st.selectbox("Select Driver", driver_labels, key="bd_driver_select")
                sel = label_to_num.get(sel_label)
                if sel:
                    rows = get_mistakes_by_driver(db, sel)
                    if rows:
                        df_dm = pd.DataFrame(rows)
                        keep = [c for c in ["session_file","mistake_rate_pct","mean_speed_at_mistake",
                                "best_contamination","mean_anomaly_score","total_rows"] if c in df_dm.columns]
                        if keep:
                            st.dataframe(df_dm[keep], width='stretch')
        else:
            st.info("No mistake data uploaded yet.")

    except Exception:
        mongo_status = _load_json(Path(DATA_LAKE_DIR) / "mongo_status.json")
        if mongo_status:
            st.markdown(f"**Database:** `{mongo_status.get('database', 'N/A')}`")
            for coll, info in mongo_status.get("collections", {}).items():
                st.metric(f"{coll} docs", info.get("document_count", 0))
        else:
            st.warning("MongoDB unavailable. Start MongoDB and run `mongo_manager.py --upload --hdfs --stats`.")
