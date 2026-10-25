# 🏗️ ApexHunter 2.0 — Architecture Overview

ApexHunter keeps data collection, data preparation, modelling and presentation apart. Each stage
reads files written by the previous stage, so any stage can be re-run on its own, and the dashboard
keeps working when external APIs or services are offline.

```
FastF1 API ─► season_data/ ─► clean_data/ ─┬─► Isolation Forest ─► mistake_data/ ──┐
                                           ├─► A*/Dijkstra/BFS ─► racing_lines/ ───┤
                         FastF1 laps ─────►├─► LSTM ───────────► tyre_predictions/ ┼─► Streamlit
YouTube ─► edited_videos/ ─► cv_frames/ ─► YOLOv11-Seg ─► processed_video/ + csv/ ─┘    dashboard
```

## 1. Extract
**`backend/scripts/etl/download_season_data.py`** (`python backend/scripts/etl/download_season_data.py`). It loops seasons → rounds → Q/R sessions, pulls per-driver telemetry with
FastF1, and writes one Parquet per session to `data_lake/season_data/`. Existing files are skipped
and API calls are cached in `cache/`.

Videos come from `etl/download_manual_videos.py` (yt-dlp) and are trimmed by hand into
`edited_videos/`. Satellite images come from `etl/download_satellite_images.py`.

## 2. Transform
**`etl/clean_telemetry.py`** (pandas) and its Spark counterpart
**`bigdata/spark_clean_telemetry.py`** both:
1. drop rows whose Speed, RPM, X and Y are all missing,
2. forward-fill gaps in numeric columns,
3. clip values to physical limits (Speed ≤ 380 km/h, RPM ≤ 15,000, pedals 0–100),
4. downcast to float32 and write `data_lake/clean_data/`.

For video, `cv/extract_frames.py` samples frames at 5 fps and `cv/select_training_frames.py` picks
the 506 frames used for annotation.

## 3. Model
| Stage | Entry point | Output |
|---|---|---|
| Isolation Forest (DSML) | `detect_mistakes.py` | `mistake_data/*_mistakes.parquet` + `_meta.json` |
| Racing-line search (DSML) | `optimal_line.py` | `racing_lines/*_racing_line.json` |
| LSTM tyre cliff (DL) | `train_lstm.py`, `predict_cliff.py` | `models/tyre_*`, `tyre_predictions/*_tyre.json` |
| YOLOv11-Seg apex detection (DL) | `cv/run_inference.py` (+ `cv/extract_turns.py` for FastF1 corner windows) | `processed_video/*_HUD.mp4`, `processed_csv/*_metrics.csv`, `processed_csv/*_turns.json` |

## 4. Store (optional infrastructure)
`bigdata/hdfs_manager.py` mirrors the Parquet layers to HDFS. `bigdata/mongo_manager.py` stores
mistake summaries and session metadata in MongoDB. Both write JSON status files to `data_lake/`
for the dashboard.

## 5. Serve
**`frontend/app.py`** (`streamlit run frontend/app.py`) reads only from `data_lake/` and `models/`
outputs, plus FastF1 lap metadata from the local cache. Loads are cached with `@st.cache_data`.

See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for details and current limitations.
