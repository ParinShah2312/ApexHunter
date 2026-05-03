# 🏎️ ApexHunter 2.0 — F1 Telemetry Analytics Dashboard

ApexHunter 2.0 is an end-to-end Formula 1 data analytics platform that ingests official telemetry and onboard video footage, applies machine-learning and computer-vision techniques to extract actionable racing insights, and serves the results through an interactive Streamlit dashboard. The project spans three academic disciplines — **Big Data** (ETL pipeline engineering, HDFS/Spark/MongoDB integration), **Artificial Intelligence** (Isolation Forest anomaly detection, LSTM tyre cliff prediction, A\*/Dijkstra/BFS racing line optimisation), and **Computer Vision** (YOLOv11-Seg instance segmentation for apex deviation measurement).

---

## 📁 Project Structure

```text
ApexHunter 2.0/
│
├── backend/
│   ├── config.json                           # Centralized configuration (seasons, circuits, thresholds)
│   └── scripts/
│       ├── utils.py                          # Shared paths, logging, config loader, IST timezone
│       │
│       ├── download_season_data.py           # Extract: Fetch FastF1 telemetry → season_data/
│       ├── download_manual_videos.py         # Extract: Download pole lap videos via yt-dlp
│       ├── download_satellite_images.py      # Extract: Circuit satellite images (ThreadPoolExecutor)
│       ├── clean_telemetry.py                # Transform: ETL cleaning pipeline → clean_data/
│       ├── fix_parquet_timestamps.py         # Transform: Fix nanosecond timestamps for Spark 3.x
│       │
│       ├── extract_frames.py                 # CV: Extract video frames at 5fps (ProcessPoolExecutor)
│       ├── select_training_frames.py         # CV: Select ~500 diverse frames for YOLO annotation
│       ├── run_inference.py                  # CV: YOLOv11-Seg inference → HUD video + metrics CSV
│       ├── inference_geometry.py             # CV: Distance computation and apex classification
│       ├── inference_masking.py              # CV: YOLO mask processing and HUD overlay
│       ├── inference_hud.py                  # CV: Augmented reality HUD drawing
│       ├── inference_io.py                   # CV: Video I/O and CSV writing
│       │
│       ├── detect_mistakes.py                # AI: Isolation Forest orchestrator
│       ├── mistakes_features.py              # AI: 7-feature engineering pipeline
│       ├── mistakes_model.py                 # AI: Grid search + model training
│       ├── mistakes_io.py                    # AI: Data loading, validation, output writing
│       │
│       ├── train_lstm.py                     # AI: LSTM training orchestrator
│       ├── predict_cliff.py                  # AI: LSTM prediction orchestrator
│       ├── tyre_data.py                      # AI: Stint detection and lap aggregation
│       ├── tyre_model.py                     # AI: LSTM architecture and training loop
│       ├── tyre_io.py                        # AI: Model artifact I/O and prediction output
│       │
│       ├── optimal_line.py                   # AI: Racing line search orchestrator
│       ├── racing_line_grid.py               # AI: Weighted grid construction
│       ├── racing_line_search.py             # AI: A*, Dijkstra, BFS pathfinding
│       ├── racing_line_io.py                 # AI: Racing line data I/O
│       │
│       ├── hdfs_manager.py                   # Big Data: HDFS storage management
│       ├── spark_clean_telemetry.py          # Big Data: Apache Spark ETL pipeline
│       ├── mongo_manager.py                  # Big Data: MongoDB storage management
│       │
│       └── tests/
│           ├── run_tests.py                  # Parallel test runner (unit + integration)
│           ├── unit/                         # 141 unit tests across 30 test files
│           └── integration/                  # End-to-end pipeline tests
│
├── frontend/
│   ├── app.py                                # Streamlit entry point (slim orchestrator)
│   ├── config.py                             # Frontend paths, constants, driver mapping
│   └── components/
│       ├── sidebar.py                        # Year / Round / Session / Driver filters
│       ├── data_loader.py                    # Cached parquet loading + downsampling
│       ├── header_bar.py                     # Session info header with common metrics
│       ├── telemetry_charts.py               # Speed / Throttle / Brake subplots + scrubber
│       ├── track_map.py                      # WebGL scatter track map (Scattergl)
│       ├── ai_analysis.py                    # LSTM tyre cliff prediction + degradation charts
│       ├── racing_line.py                    # Racing line analysis with algorithm overlays
│       ├── cv_feed.py                        # Computer vision video feed + apex metrics
│       ├── bigdata_tab.py                    # Big Data analytics dashboard tab
│       └── bigdata_charts.py                 # Specialized big data visualizations
│
├── data_lake/                                # Central data storage layer (gitignored)
│   ├── season_data/                          # Raw ingestion layer (.parquet)
│   ├── clean_data/                           # Processed presentation layer (.parquet)
│   ├── raw_video/                            # Downloaded onboard pole lap videos
│   ├── edited_videos/                        # Trimmed videos for frame extraction
│   ├── cv_frames/                            # Extracted frames at 5fps
│   ├── cv_dataset/                           # Roboflow annotation upload folder
│   ├── satellite_images/                     # Circuit satellite imagery
│   ├── processed_video/                      # YOLO HUD overlay output videos
│   ├── processed_csv/                        # Apex deviation metric CSVs
│   ├── mistake_data/                         # Isolation Forest annotated parquets + metadata JSON
│   ├── tyre_predictions/                     # LSTM cliff-lap prediction JSON
│   └── racing_lines/                         # A*, Dijkstra, BFS path JSON
│
├── models/                                   # Trained model weights (best.pt, tyre_lstm.pt)
├── cache/                                    # FastF1 API cache (gitignored)
├── docs/                                     # Project documentation
│   ├── PROJECT_DOCUMENTATION.md              # Comprehensive academic documentation
│   ├── architecture.md                       # ETL architecture overview
│   ├── ApexHunter_Data_Architecture_Report.md
│   └── ...                                   # Academic submission documents
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone & Install
```bash
git clone https://github.com/ParinShah2312/ApexHunter.git
cd "ApexHunter 2.0"
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
pip install -r requirements.txt
```

### 2. Data Pipeline (Backend)

**Download raw telemetry** (optional — data is already included):
```bash
python backend/scripts/download_season_data.py
```

**Clean & transform the data:**
```bash
python backend/scripts/clean_telemetry.py
```

**Run Isolation Forest mistake detection:**
```bash
python backend/scripts/detect_mistakes.py --session data_lake/clean_data/2024_1_Q.parquet --driver 1
```

**Train the LSTM tyre cliff predictor:**
```bash
python backend/scripts/train_lstm.py --seasons 2024
```

**Run tyre cliff prediction on a session:**
```bash
python backend/scripts/predict_cliff.py --session data_lake/clean_data/2024_1_R.parquet --driver 1
```

**Compute optimal racing lines:**
```bash
python backend/scripts/optimal_line.py --session data_lake/clean_data/2024_1_Q.parquet --driver 1
```

**Run YOLO CV inference on a pole lap video:**
```bash
python backend/scripts/run_inference.py --input "data_lake/edited_videos/2024/01_bahrain_ver_pole - Trim.mp4"
```

### 3. Launch the Dashboard
```bash
streamlit run frontend/app.py
```

### 4. Run Tests
```bash
python backend/scripts/tests/run_tests.py          # Unit tests only (fast)
python backend/scripts/tests/run_tests.py --all     # Unit + integration (slow)
```

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| **Telemetry Playback** | Interactive time slider to scrub through Speed, Throttle, and Brake data with synchronized subplots |
| **Traffic Light Track Map** | WebGL scatter plot coloring the circuit by speed (red → yellow → green) with ghost car positioning |
| **Isolation Forest Mistake Detection** | Unsupervised anomaly detection on 7 engineered features to identify lock-ups, slides, and pedal overlap |
| **LSTM Tyre Cliff Prediction** | Recurrent neural network forecasting tyre degradation cliffs with Monte Carlo confidence bounds |
| **A\*/Dijkstra/BFS Racing Lines** | Three graph-search algorithms computing optimal paths on a telemetry-weighted 2D grid with per-corner deviation analysis |
| **YOLOv11 Apex Detection** | Instance segmentation pipeline measuring pixel-level apex deviation from onboard video |
| **Big Data Analytics** | HDFS storage management, Spark ETL pipeline, MongoDB integration, and comprehensive pipeline dashboards |
| **Perfect Corner Score** | Gamified metric showing pedal discipline (brake/throttle overlap percentage) |
| **Beginner Mode** | Toggle simplified labels ("Gas Pedal" instead of "Throttle %") for accessibility |
| **Cached Data Loading** | `@st.cache_data` ensures instant driver/session switches without disk re-reads |
| **Concurrent Processing** | ThreadPool/ProcessPool executors for downloads and frame extraction |

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | Streamlit, Plotly (WebGL), Pandas |
| **Backend** | Python 3.10+, FastF1, OpenCV, yt-dlp |
| **AI/ML** | PyTorch (LSTM), scikit-learn (Isolation Forest), Ultralytics YOLOv11-Seg |
| **Big Data** | Apache Spark (PySpark), HDFS (Hadoop), MongoDB (PyMongo), Apache Parquet |
| **Data** | FastF1 API, Roboflow (annotation), Esri World Imagery (satellite maps) |
| **Testing** | pytest, unittest, parallel test runner |
| **Timezone** | All timestamps use IST (UTC+05:30) |

---

## 📊 Test Suite

| Category | Files | Tests |
|---|---|---|
| **Unit Tests** | 30 test files | 141 tests |
| **Integration Tests** | 7 test files | End-to-end pipeline validation |
| **Coverage** | Telemetry cleaning, feature engineering, model training, I/O, grid search, pathfinding, LSTM lifecycle |

```
141 passed, 0 skipped, 0 warnings
```

---

## 👤 Author

**Parin Shah** — Student ID: 23001091
