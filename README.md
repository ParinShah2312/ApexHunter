# 🏎️ ApexHunter 2.0 — F1 Telemetry & Onboard-Video Analytics

ApexHunter 2.0 analyses Formula 1 driving performance using official telemetry (FastF1, 2023–2024
seasons, ~57.9M rows) and onboard pole-lap video (46 laps). It combines classical machine learning,
deep learning and graph search, and shows the results in an interactive Streamlit dashboard.

**Courses (AY 2026–27):** Data Science & Machine Learning · Deep Learning
*(The project was first built in the previous semester for Artificial Intelligence and Big Data.)*

| Component | Technique | Course |
|---|---|---|
| Driver mistake detection | Isolation Forest + engineered telemetry features, K-fold contamination search | DSML |
| Racing-line search | A\*, Dijkstra, BFS on a telemetry-weighted grid | DSML |
| Tyre cliff prediction | 2-layer LSTM, grid search, Monte Carlo dropout uncertainty | Deep Learning |
| Apex detection | YOLOv11-Seg fine-tuned on 506 hand-labelled onboard frames | Deep Learning |
| Data pipeline | pandas ETL + Apache Spark, HDFS, MongoDB | Supporting infrastructure |

Full technical write-up: **[docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md)**.
Current limitations are tracked in §8 of that document.
Semester plan and scope decisions: **[docs/ROADMAP.md](docs/ROADMAP.md)**. Planned: Z-score vs
Isolation Forest vs LSTM autoencoder for mistake detection, and Ridge vs RNN vs LSTM for tyre/pace
prediction. Current progress, next steps and the rules for contributing (documentation, testing,
verification) are in §0 and §3 of the roadmap.

---

## 📁 Project Structure

```text
ApexHunter 2.0/
├── backend/
│   ├── config.json                  # seasons, CV thresholds, circuits, video URLs
│   └── scripts/
│       ├── utils.py                 # shared paths, config, logger, IST timezone
│       ├── detect_mistakes.py       # Isolation Forest orchestrator
│       ├── train_lstm.py            # LSTM training orchestrator
│       ├── predict_cliff.py         # LSTM prediction orchestrator
│       ├── optimal_line.py          # A*/Dijkstra/BFS orchestrator
│       ├── etl/                     # download_season_data, clean_telemetry, fix_parquet_timestamps,
│       │                            #   download_manual_videos, download_satellite_images
│       ├── bigdata/                 # hdfs_manager, spark_clean_telemetry, mongo_manager
│       ├── cv/                      # extract_frames, select_training_frames, extract_turns,
│       │                            #   run_inference + inference_{geometry,masking,hud,io}
│       ├── isolation_forest/        # mistakes_features, mistakes_model, mistakes_io
│       ├── lstm/                    # tyre_data, tyre_model, tyre_io
│       ├── racing_line/             # racing_line_grid, racing_line_search, racing_line_io
│       └── tests/                   # unit/ (23 files, 127 tests), integration/ (5 files, 18 tests), run_tests.py
├── frontend/
│   ├── app.py                       # Streamlit entry point
│   ├── config.py                    # paths, driver/team mappings, colours
│   └── components/                  # sidebar, data_loader, header_bar, telemetry_charts, track_map,
│                                    #   cv_feed, ai_analysis, racing_line, bigdata_tab, bigdata_charts, scrubber
├── models/                          # best.pt (YOLO), tyre_lstm.pt, tyre_scaler.pkl, tyre_config.json
├── data_lake/                       # data + model outputs (gitignored)
│   ├── season_data/  clean_data/    # raw / cleaned telemetry parquet (92 sessions each)
│   ├── downloaded_videos/ edited_videos/ cv_frames/ cv_dataset/
│   ├── raw_images/satellite_maps/
│   ├── mistake_data/ tyre_predictions/ racing_lines/
│   ├── processed_video/ processed_csv/
│   └── hdfs_status.json  spark_run_log.json  mongo_status.json
├── cache/                           # FastF1 HTTP cache (gitignored)
├── docs/                            # project documentation (+ previous-semester submission files)
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Install
```bash
git clone https://github.com/ParinShah2312/ApexHunter.git
cd "ApexHunter 2.0"
python -m venv .venv
.venv\Scripts\activate              # Windows
pip install -r requirements.txt
```

### 2. Data pipeline
All scripts can be run by file path from the project root (or as modules from `backend/scripts`,
e.g. `python -m etl.clean_telemetry`):
```bash
python backend/scripts/etl/download_season_data.py      # FastF1 → data_lake/season_data/
python backend/scripts/etl/clean_telemetry.py           # → data_lake/clean_data/
```

### 3. Models
```bash
# Isolation Forest mistake detection
python backend/scripts/detect_mistakes.py --session data_lake/clean_data/2023_1_R.parquet --driver 1

# LSTM tyre cliff (train once, then predict)
python backend/scripts/train_lstm.py --force
python backend/scripts/predict_cliff.py --session data_lake/clean_data/2023_1_R.parquet --driver 1

# Racing line search
python backend/scripts/optimal_line.py --session data_lake/clean_data/2023_1_Q.parquet --driver 1

# YOLO apex detection on an onboard video
# --video-offset = seconds into the video at which the timed lap starts (corners come from FastF1)
python backend/scripts/cv/run_inference.py --input "data_lake/edited_videos/2024/01_bahrain_ver_pole - Trim.mp4" --video-offset 4.9
```

### 4. Dashboard
```bash
streamlit run frontend/app.py
```

### 5. Tests
```bash
python backend/scripts/tests/run_tests.py          # unit tests (fast)
python backend/scripts/tests/run_tests.py --all    # unit + integration (slow)
python backend/scripts/tests/run_tests.py --file test_racing_line_pipeline.py   # one file (integration files run, not skipped)
```

### Optional: Big Data stack
Requires a local Hadoop/HDFS (`hdfs://localhost:9000`) and MongoDB (`localhost:27017`).
```bash
python backend/scripts/bigdata/hdfs_manager.py --upload all --stats
python backend/scripts/bigdata/spark_clean_telemetry.py --hdfs --force
python backend/scripts/bigdata/mongo_manager.py --upload --stats
```

---

## 🎯 Dashboard Features

| Tab | What it shows |
|---|---|
| **Race Intelligence** | Master time scrubber, YOLO HUD video with live apex status and turn report card, speed/mistake track map |
| **Telemetry** | Speed, throttle, brake, gear and RPM traces; top speed and brake/throttle overlap % (samples ≥ 50 km/h) |
| **AI Analysis** | Isolation Forest mistakes per lap; LSTM tyre-cliff chart with confidence band |
| **Racing Line** | Driver's fastest lap vs A\* / Dijkstra / BFS path, deviation per turn |
| **Big Data** | Pipeline DAG, HDFS/Spark statistics, EDA charts, MongoDB leaderboard |

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| Data | FastF1, yt-dlp, staticmap (Esri imagery), Roboflow (annotation) |
| Processing | pandas, NumPy, PyArrow/Parquet, Apache Spark (PySpark 3.5), HDFS, MongoDB (PyMongo) |
| Machine learning | scikit-learn (Isolation Forest, KFold, StandardScaler), SciPy |
| Deep learning | PyTorch (LSTM), Ultralytics YOLOv11-Seg, OpenCV |
| Frontend | Streamlit, Plotly |
| Testing | unittest, parallel runner |

---

## 👥 Team

| Name | Enrolment No. |
|---|---|
| Parin Shah | 23001091 |
| Ayush Parikh | 23000936 |
| Malav Joshi | 23001119 |
