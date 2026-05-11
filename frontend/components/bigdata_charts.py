"""
Chart factory for the Big Data Analytics tab.
Each function returns a Plotly figure ready for st.plotly_chart().
"""
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

_DARK_BG = "#0d1520"
_PAPER = "rgba(0,0,0,0)"
_GRID = "rgba(255,255,255,0.04)"
_FONT = dict(color="#c8d0dc", size=11)
_ACCENT = "#00d4ff"
_PURPLE = "#a855f7"
_GREEN = "#10b981"
_RED = "#ef4444"
_ORANGE = "#f97316"
_YELLOW = "#eab308"

# ──  ──────────────────────────────────────────────────────────────────────────
#  1. Pipeline DAG (network graph)
# ──  ──────────────────────────────────────────────────────────────────────────
def build_pipeline_dag() -> go.Figure:
    nodes = [
        "FastF1 API", "Local Parquets", "HDFS Raw",
        "Spark ETL", "HDFS Clean", "Isolation Forest",
        "MongoDB", "LSTM Trainer", "Tyre Predictor",
        "A* / Dijkstra / BFS", "Dashboard"
    ]
    x = [0.05, 0.20, 0.35, 0.50, 0.65, 0.50, 0.65, 0.50, 0.65, 0.35, 0.85]
    y = [0.5,  0.5,  0.5,  0.5,  0.5,  0.15, 0.15, 0.85, 0.85, 0.85, 0.5]
    colors = [
        _ACCENT, _GREEN, _ORANGE, _PURPLE, _GREEN,
        _RED, _ACCENT, _PURPLE, _YELLOW, _ORANGE, "#ffffff"
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(1,7),(7,8),(1,9),(4,10),(6,10),(8,10),(9,10)
    ]
    edge_x, edge_y = [], []
    for s, t in edges:
        edge_x += [x[s], x[t], None]
        edge_y += [y[s], y[t], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
        line=dict(color="#334155", width=2), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers+text", text=nodes,
        textposition="top center", textfont=dict(color="#e2e8f0", size=10),
        marker=dict(size=22, color=colors, line=dict(color="#1e293b", width=2)),
        hoverinfo="text", showlegend=False))
    fig.update_layout(
        paper_bgcolor=_PAPER, plot_bgcolor=_DARK_BG, height=320,
        margin=dict(t=10,b=10,l=10,r=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.02,1.02]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1,1.1]),
    )
    return fig


# ──  ──────────────────────────────────────────────────────────────────────────
#  2. HDFS file-size treemap
# ──  ──────────────────────────────────────────────────────────────────────────
def build_hdfs_treemap(hdfs_stats: dict) -> go.Figure:
    dirs = hdfs_stats.get("directories", {})
    labels, parents, values, colors = [], [], [], []
    labels.append("HDFS"); parents.append(""); values.append(0); colors.append("#1e293b")
    palette = {"season_data": _ORANGE, "clean_data": _GREEN, "mistake_data": _RED}
    for dname, info in dirs.items():
        sz = info.get("total_size_mb", 0)
        labels.append(dname); parents.append("HDFS"); values.append(sz)
        colors.append(palette.get(dname, _ACCENT))
        for f in info.get("files", []):
            labels.append(f); parents.append(dname); values.append(sz / max(len(info.get("files",[])),1))
            colors.append(palette.get(dname, _ACCENT))
    fig = go.Figure(go.Treemap(labels=labels, parents=parents, values=values,
        marker=dict(colors=colors, line=dict(color=_DARK_BG, width=1)),
        textinfo="label", textfont=dict(size=10)))
    fig.update_layout(paper_bgcolor=_PAPER, height=350, margin=dict(t=10,b=10,l=10,r=10))
    return fig


# ──  ──────────────────────────────────────────────────────────────────────────
#  3. HDFS storage bar chart (per-directory)
# ──  ──────────────────────────────────────────────────────────────────────────
def build_hdfs_bar(hdfs_stats: dict) -> go.Figure:
    dirs = hdfs_stats.get("directories", {})
    names = list(dirs.keys())
    sizes = [dirs[n].get("total_size_mb", 0) for n in names]
    counts = [dirs[n].get("file_count", 0) for n in names]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=names, y=sizes, name="Size (MB)",
        marker_color=[_ORANGE, _GREEN, _RED][:len(names)], text=[f"{s:.0f}" for s in sizes],
        textposition="auto",
        hovertemplate="%{x}<br>Size: %{y:.1f} MB<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=names, y=counts, name="File Count", mode="lines+markers",
        line=dict(color=_ACCENT, width=2), marker=dict(size=10),
        hovertemplate="%{x}<br>Files: %{y}<extra></extra>"), secondary_y=True)
    fig.update_layout(paper_bgcolor=_PAPER, plot_bgcolor=_DARK_BG, height=280,
        margin=dict(t=20,b=30,l=40,r=40), font=_FONT, legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_yaxes(title_text="Size (MB)", gridcolor=_GRID, secondary_y=False)
    fig.update_yaxes(title_text="Files", gridcolor=_GRID, secondary_y=True)
    return fig


# ──  ──────────────────────────────────────────────────────────────────────────
#  4. Spark ETL gauge
# ──  ──────────────────────────────────────────────────────────────────────────
def build_spark_gauge(spark_stats: dict) -> go.Figure:
    rows_in = spark_stats.get("rows_input", 0)
    rows_out = spark_stats.get("rows_output", 0)
    pct = (rows_out / rows_in * 100) if rows_in else 0
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=pct,
        title=dict(text="Data Retention %", font=dict(color="#e2e8f0", size=14)),
        delta=dict(reference=100, decreasing=dict(color=_RED)),
        number=dict(suffix="%", font=dict(color="#e2e8f0", size=28)),
        gauge=dict(
            axis=dict(range=[0,100], tickfont=dict(color="#6b7890")),
            bar=dict(color=_GREEN),
            bgcolor="#1e293b",
            steps=[dict(range=[0,50], color="#1e293b"), dict(range=[50,90], color="#1a2530"),
                   dict(range=[90,100], color="#0d2018")],
            threshold=dict(line=dict(color=_RED, width=2), thickness=0.8, value=95))))
    fig.update_layout(paper_bgcolor=_PAPER, height=220, margin=dict(t=40,b=10,l=30,r=30))
    return fig


# ──  ──────────────────────────────────────────────────────────────────────────
#  5. Spark processing waterfall
# ──  ──────────────────────────────────────────────────────────────────────────
def build_spark_waterfall(spark_stats: dict) -> go.Figure:
    rows_in = spark_stats.get("rows_input", 0)
    dropped = spark_stats.get("rows_dropped", 0)
    rows_out = spark_stats.get("rows_output", 0)
    fig = go.Figure(go.Waterfall(
        x=["Input Rows", "Dropped", "Output Rows"],
        y=[rows_in, -dropped, rows_out],
        measure=["absolute", "relative", "total"],
        connector=dict(line=dict(color="#334155")),
        decreasing=dict(marker=dict(color=_RED)),
        increasing=dict(marker=dict(color=_GREEN)),
        totals=dict(marker=dict(color=_ACCENT)),
        text=[f"{rows_in:,}", f"-{dropped:,}", f"{rows_out:,}"],
        textposition="outside",
        hovertemplate="%{x}<br>Rows: %{y:,}<extra></extra>"))
    fig.update_layout(paper_bgcolor=_PAPER, plot_bgcolor=_DARK_BG, height=280,
        margin=dict(t=20,b=30,l=50,r=20), font=_FONT,
        yaxis=dict(gridcolor=_GRID, title="Rows"))
    return fig


# ──  ──────────────────────────────────────────────────────────────────────────
#  6. Session heatmap (Year × Round)
# ──  ──────────────────────────────────────────────────────────────────────────
def build_session_file_size_bar(hdfs_stats: dict) -> go.Figure:
    """Bar chart showing file sizes per round, grouped by year."""
    files = hdfs_stats.get("directories", {}).get("season_data", {}).get("files", [])
    rows = []
    for f in files:
        m = re.match(r"(\d{4})_(\d+)_([A-Z])\.parquet", f)
        if m:
            rows.append({"Year": str(m.group(1)), "Round": int(m.group(2)), "Session": m.group(3), "File": f})
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)
    counts = df.groupby(["Year", "Round"]).size().reset_index(name="Files")
    fig = go.Figure()
    pal = {"2023": _ACCENT, "2024": _PURPLE}
    for yr in sorted(counts["Year"].unique()):
        sub = counts[counts["Year"] == yr]
        fig.add_trace(go.Bar(x=[f"R{r}" for r in sub["Round"]], y=sub["Files"],
            name=yr, marker_color=pal.get(yr, _GREEN),
            hovertemplate=f"{yr} · Round %{{x}}<br>Files: %{{y}}<extra></extra>"))
    fig.update_layout(paper_bgcolor=_PAPER, plot_bgcolor=_DARK_BG, height=250,
        margin=dict(t=20,b=30,l=40,r=10), font=_FONT, barmode="group",
        xaxis=dict(title="Grand Prix Round", gridcolor=_GRID),
        yaxis=dict(title="Files per Round", gridcolor=_GRID, dtick=1),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#6b7890")))
    return fig


# ──  ──────────────────────────────────────────────────────────────────────────
#  7. Telemetry speed distribution (violin)
# ──  ──────────────────────────────────────────────────────────────────────────
def build_speed_distribution(sample_df: pd.DataFrame) -> go.Figure:
    """Overlaid speed histograms for top 5 drivers — shows driving style differences."""
    if "Speed" not in sample_df.columns or "Driver" not in sample_df.columns:
        return go.Figure()
    top_drivers = sample_df["Driver"].value_counts().head(5).index.tolist()
    fig = go.Figure()
    pal = [_ACCENT, _PURPLE, _GREEN, _ORANGE, _RED]
    for i, d in enumerate(top_drivers):
        dd = sample_df[sample_df["Driver"] == d]["Speed"].dropna()
        fig.add_trace(go.Histogram(x=dd, name=f"Driver {d}", nbinsx=50,
            marker_color=pal[i % len(pal)], opacity=0.6,
            hovertemplate=f"Driver {d}<br>Speed: %{{x:.0f}} km/h<br>Count: %{{y}}<extra></extra>"))
    fig.update_layout(paper_bgcolor=_PAPER, plot_bgcolor=_DARK_BG, height=300,
        margin=dict(t=20,b=30,l=40,r=10), font=_FONT, barmode="overlay",
        xaxis=dict(title="Speed (km/h)", gridcolor=_GRID),
        yaxis=dict(title="Frequency", gridcolor=_GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#6b7890")))
    return fig


# ──  ──────────────────────────────────────────────────────────────────────────
#  8. Telemetry correlation heatmap
# ──  ──────────────────────────────────────────────────────────────────────────
def build_correlation_heatmap(sample_df: pd.DataFrame) -> go.Figure:
    num_cols = [c for c in ["Speed","RPM","Throttle","Brake","X","Y","Z","nGear"] if c in sample_df.columns]
    if len(num_cols) < 2:
        return go.Figure()
    corr = sample_df[num_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0,"#1e3a5f"],[0.5,"#0d1520"],[1,_ACCENT]],
        text=np.round(corr.values, 2), texttemplate="%{text}",
        zmin=-1, zmax=1, showscale=True,
        colorbar=dict(tickfont=dict(color="#6b7890")),
        hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>"))
    fig.update_layout(paper_bgcolor=_PAPER, plot_bgcolor=_DARK_BG, height=350,
        margin=dict(t=10,b=10,l=10,r=10), font=_FONT)
    return fig


# ──  ──────────────────────────────────────────────────────────────────────────
#  9. Speed vs RPM scatter
# ──  ──────────────────────────────────────────────────────────────────────────
def build_speed_rpm_scatter(sample_df: pd.DataFrame) -> go.Figure:
    """Speed vs RPM scatter with gear-colored markers and clear grouping."""
    if not {"Speed","RPM","nGear"}.issubset(sample_df.columns):
        return go.Figure()
    ds = sample_df.dropna(subset=["Speed","RPM"])
    if len(ds) > 8000:
        ds = ds.sample(8000, random_state=42)
    gear_colors = {1:_RED, 2:_ORANGE, 3:_YELLOW, 4:_GREEN, 5:_ACCENT, 6:_PURPLE, 7:"#ec4899", 8:"#f43f5e"}
    fig = go.Figure()
    for g in sorted(ds["nGear"].dropna().unique()):
        dg = ds[ds["nGear"]==g]
        fig.add_trace(go.Scattergl(x=dg["Speed"], y=dg["RPM"], mode="markers",
            marker=dict(size=4, color=gear_colors.get(int(g), _ACCENT), opacity=0.65,
                line=dict(width=0)),
            name=f"Gear {int(g)}",
            hovertemplate=f"Gear {int(g)}<br>Speed: %{{x:.0f}} km/h<br>RPM: %{{y:.0f}}<extra></extra>"))
    fig.update_layout(paper_bgcolor=_PAPER, plot_bgcolor=_DARK_BG, height=350,
        margin=dict(t=20,b=40,l=50,r=120), font=_FONT,
        xaxis=dict(title="Speed (km/h)", gridcolor=_GRID, showgrid=True),
        yaxis=dict(title="Engine RPM", gridcolor=_GRID, showgrid=True),
        legend=dict(bgcolor="rgba(13,21,32,0.9)", font=dict(size=11, color="#c8d0dc"),
            bordercolor="#334155", borderwidth=1, x=1.02, y=1, xanchor="left"))
    return fig


def build_speed_over_distance(sample_df: pd.DataFrame) -> go.Figure:
    """Speed trace over session time for the selected driver — like F1 TV overlay."""
    if not {"Speed"}.issubset(sample_df.columns):
        return go.Figure()
    # Use first driver in data
    drivers = sample_df["Driver"].unique()[:3]
    fig = go.Figure()
    pal = [_ACCENT, _PURPLE, _GREEN]
    for i, d in enumerate(drivers):
        dd = sample_df[sample_df["Driver"] == d].reset_index(drop=True)
        if len(dd) > 3000:
            dd = dd.iloc[::len(dd)//3000]
        fig.add_trace(go.Scatter(x=list(range(len(dd))), y=dd["Speed"],
            mode="lines", line=dict(color=pal[i % len(pal)], width=1.5),
            name=f"Driver {d}", opacity=0.8,
            hovertemplate=f"Driver {d}<br>Sample: %{{x}}<br>Speed: %{{y:.1f}} km/h<extra></extra>"))
    fig.update_layout(paper_bgcolor=_PAPER, plot_bgcolor=_DARK_BG, height=300,
        margin=dict(t=20,b=30,l=50,r=10), font=_FONT,
        xaxis=dict(title="Sample Index", gridcolor=_GRID),
        yaxis=dict(title="Speed (km/h)", gridcolor=_GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#6b7890")))
    return fig


# ──  ──────────────────────────────────────────────────────────────────────────
#  12. LSTM tyre degradation line chart (all stints overlay)
# ──  ──────────────────────────────────────────────────────────────────────────
def build_tyre_degradation_line(tyre_data: dict) -> go.Figure:
    stints = tyre_data.get("stints", [])
    if not stints:
        return go.Figure()
    pal = [_ACCENT, _PURPLE, _GREEN, _ORANGE, _RED]
    fig = go.Figure()
    for i, s in enumerate(stints):
        laps = list(range(1, s["n_laps"]+1))
        c = pal[i % len(pal)]
        fig.add_trace(go.Scatter(x=laps, y=s["actual_laps"], mode="lines+markers",
            name=f"Stint {i+1} Actual", line=dict(color=c, width=2), marker=dict(size=4),
            hovertemplate=f"Stint {i+1} Actual<br>Lap %{{x}}<br>Time: %{{y:.2f}}s<extra></extra>"))
        fig.add_trace(go.Scatter(x=laps, y=s["predicted_laps"], mode="lines",
            name=f"Stint {i+1} LSTM", line=dict(color=c, width=1.5, dash="dash"),
            hovertemplate=f"Stint {i+1} LSTM<br>Lap %{{x}}<br>Predicted: %{{y:.2f}}s<extra></extra>"))
        if s.get("cliff_lap") is not None:
            fig.add_vline(x=s["cliff_lap"]+1, line=dict(color=_RED, dash="dot", width=1),
                annotation_text=f"Cliff S{i+1}", annotation_font=dict(color=_RED, size=9))
    fig.update_layout(paper_bgcolor=_PAPER, plot_bgcolor=_DARK_BG, height=320,
        margin=dict(t=20,b=30,l=50,r=10), font=_FONT,
        xaxis=dict(title="Lap in Stint", gridcolor=_GRID),
        yaxis=dict(title="Lap Time (s)", gridcolor=_GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9, color="#6b7890")))
    return fig


# ──  ──────────────────────────────────────────────────────────────────────────
#  13. LSTM confidence band area chart
# ──  ──────────────────────────────────────────────────────────────────────────
def build_lstm_confidence_area(tyre_data: dict) -> go.Figure:
    stints = tyre_data.get("stints", [])
    if not stints:
        return go.Figure()
    s = stints[0]  # first stint
    laps = list(range(1, s["n_laps"]+1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=laps+laps[::-1],
        y=s["confidence_upper"]+s["confidence_lower"][::-1],
        fill="toself", fillcolor="rgba(168,85,247,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=laps, y=s["predicted_laps"], mode="lines",
        line=dict(color=_PURPLE, width=2), name="LSTM Predicted",
        hovertemplate="Lap %{x}<br>LSTM: %{y:.2f}s<extra></extra>"))
    fig.add_trace(go.Scatter(x=laps, y=s["actual_laps"], mode="lines+markers",
        line=dict(color=_ACCENT, width=2), marker=dict(size=4), name="Actual",
        hovertemplate="Lap %{x}<br>Actual: %{y:.2f}s<extra></extra>"))
    fig.update_layout(paper_bgcolor=_PAPER, plot_bgcolor=_DARK_BG, height=280,
        margin=dict(t=20,b=30,l=50,r=10), font=_FONT,
        xaxis=dict(title="Lap", gridcolor=_GRID), yaxis=dict(title="Lap Time (s)", gridcolor=_GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#6b7890")))
    return fig


# ──  ──────────────────────────────────────────────────────────────────────────
#  14. Racing line algorithm comparison (multi-bar)
# ──  ──────────────────────────────────────────────────────────────────────────
def build_algo_comparison_bar(racing_data: dict) -> go.Figure:
    algos = racing_data.get("algorithms", {})
    if not algos:
        return go.Figure()
    names, costs, nodes, times = [], [], [], []
    pal = {"astar": _PURPLE, "dijkstra": _ACCENT, "bfs": _RED}
    for k, v in algos.items():
        names.append(k.upper())
        costs.append(v.get("cost", 0))
        nodes.append(v.get("nodes_expanded", 0))  # correct key from JSON
        times.append(v.get("compute_time_s", 0) * 1000)  # convert seconds to ms
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Path Cost","Nodes Expanded","Time (ms)"])
    fig.add_trace(go.Bar(x=names, y=costs, marker_color=[pal.get(n.lower(),_ACCENT) for n in names],
        text=[f"{c:.1f}" for c in costs], textposition="auto", showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=nodes, marker_color=[pal.get(n.lower(),_ACCENT) for n in names],
        text=[f"{n:,}" for n in nodes], textposition="auto", showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=names, y=times, marker_color=[pal.get(n.lower(),_ACCENT) for n in names],
        text=[f"{t:.1f}" for t in times], textposition="auto", showlegend=False), row=1, col=3)
    fig.update_layout(paper_bgcolor=_PAPER, plot_bgcolor=_DARK_BG, height=280,
        margin=dict(t=30,b=20,l=30,r=10), font=_FONT)
    for i in range(1,4):
        fig.update_yaxes(gridcolor=_GRID, row=1, col=i)
    return fig


# ──  ──────────────────────────────────────────────────────────────────────────
#  15. Technology stack radar chart
# ──  ──────────────────────────────────────────────────────────────────────────
def build_tech_radar() -> go.Figure:
    cats = ["HDFS Storage", "Spark ETL", "MongoDB", "LSTM/AI", "Pathfinding", "Computer Vision"]
    vals = [92, 85, 93, 78, 88, 70]
    fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor="rgba(0,212,255,0.15)",
        line=dict(color=_ACCENT, width=2), marker=dict(size=6)))
    fig.update_layout(polar=dict(
        bgcolor=_DARK_BG,
        radialaxis=dict(visible=True, range=[0,100], gridcolor="#1e293b", tickfont=dict(color="#6b7890")),
        angularaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#c8d0dc", size=11))),
        paper_bgcolor=_PAPER, height=320, margin=dict(t=30,b=30,l=60,r=60), font=_FONT, showlegend=False)
    return fig
