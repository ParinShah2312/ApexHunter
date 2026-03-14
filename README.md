# 🏎️ ApexHunter F1 Analytics

ApexHunter is an interactive Big Data dashboard built to visualize, process, and analyze Formula 1 telemetry. It ingests gigabytes of raw FastF1 data, cleans it through a custom ETL pipeline, runs a YOLOv11 computer vision model for apex detection, and serves an intuitive, gamified Streamlit UI for racing fans and newcomers alike.

## 📁 Project Structure

```text
ApexHunter 2.0/
│
├── backend/
│   ├── config.json                        # Centralized configuration (seasons, circuits, thresholds)
│   └── scripts/
│       ├── utils.py                       # Shared paths, logging, and config loader
│       ├── download_season_data.py        # Stage 1: Fetches FastF1 telemetry as raw Parquet
│       ├── download_manual_videos.py      # Downloads pole lap onboard videos via yt-dlp
│       ├── download_satellite_images.py   # Downloads circuit satellite images (ThreadPoolExecutor)
│       ├── clean_telemetry.py             # Stage 2: ETL — cleans, clips outliers → clean_data/
│       ├── extract_frames.py              # Extracts video frames at 5fps (ProcessPoolExecutor)
│       ├── select_training_frames.py      # Selects ~500 diverse frames for YOLO annotation
│       └── run_inference.py               # Stage 3: Runs YOLOv11-Seg + Apex Deviation Metric
│
├── frontend/
│   ├── app.py                             # Streamlit entry point (slim orchestrator)
│   ├── config.py                          # Frontend paths, constants, driver mapping
│   └── components/
│       ├── sidebar.py                     # Year / Session / Driver filters
│       ├── data_loader.py                 # Cached parquet loading + downsampling
│       ├── telemetry_charts.py            # Speed / Throttle / Brake subplots + metrics
│       └── track_map.py                   # WebGL scatter track map (Scattergl)
│
├── data_lake/                             # Central data storage layer (gitignored)
│   ├── season_data/                       # Raw ingestion layer (.parquet)
│   ├── clean_data/                        # Processed presentation layer (.parquet)
│   ├── raw_video/                         # Downloaded onboard pole lap videos
│   ├── edited_videos/                     # Trimmed videos for frame extraction
│   ├── cv_frames/                         # Extracted frames at 5fps
│   ├── cv_dataset/                        # Roboflow annotation upload folder
│   ├── satellite_images/                  # Circuit satellite imagery
│   ├── processed_video/                   # YOLO HUD overlay output videos
│   └── processed_csv/                     # Apex deviation metric CSVs
│
├── models/                                # Trained YOLOv11-Seg weights (best.pt)
├── cache/                                 # FastF1 API cache (gitignored)
├── docs/                                  # Project documentation
│   ├── architecture.md
│   ├── Dataset Documentation Template.docx
│   ├── Dataset_Documentation_Answers.md
│   └── Studio_32_Dataset_Documentation.docx
│
├── .gitignore
├── requirements.txt
└── README.md
```

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

**Run YOLO inference on a pole lap video:**
```bash
python backend/scripts/run_inference.py --input "data_lake/edited_videos/2024/01_bahrain_ver_pole - Trim.mp4"
```

### 3. Launch the Dashboard
```bash
streamlit run frontend/app.py
```

## 🎯 Key Features

| Feature | Description |
|---|---|
| **Telemetry Playback** | Interactive time slider to scrub through Speed, Throttle, and Brake data |
| **Traffic Light Track Map** | WebGL scatter plot coloring the circuit by speed (red→yellow→green) |
| **Perfect Corner Score** | Gamified metric showing how well the driver avoids overlapping brake + throttle |
| **Beginner Mode** | Toggle simplified labels ("Gas Pedal" instead of "Throttle %") |
| **YOLOv11 Apex Detection** | Computer vision pipeline that detects curbs and calculates apex deviation |
| **Cached Data Loading** | `@st.cache_data` ensures instant driver switches without disk re-reads |
| **Concurrent Processing** | ThreadPool/ProcessPool executors for downloads and frame extraction |

## 🛠️ Tech Stack

- **Frontend:** Streamlit, Plotly (WebGL), Pandas
- **Backend:** Python 3.10+, FastF1, OpenCV, yt-dlp, Ultralytics YOLOv11
- **Data:** Parquet files, FastF1 API, Roboflow (annotation)
- **Linting:** Ruff

## 👤 Author

**Parin Shah** — Student ID: 23001091
