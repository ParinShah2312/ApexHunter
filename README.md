# 🏎️ ApexHunter F1 Analytics

ApexHunter is an interactive Big Data dashboard built to visualize, process, and demystify Formula 1 telemetry. It ingests gigabytes of raw FastF1 Parquet data, cleans it through a custom ETL pipeline, and serves an intuitive, gamified UI for racing fans and newcomers alike.

## 📁 Repository Architecture

To maintain clean separation of concerns, the project is structured as follows:

```text
ApexHunter 2.0/
│
├── backend/
│   └── scripts/
│       ├── download_season_data.py  # Stage 1: Fetches FastF1 telemetry and saves as raw Parquet.
│       ├── download_raw_video.py    # Fetches auxiliary track video data via yt-dlp.
│       └── clean_telemetry.py       # Stage 2: The ETL Script. Drops NaNs, clips outliers to `clean_data/`.
│
├── frontend/
│   └── app.py                       # Stage 3: The Streamlit Dashboard.
│
├── data_lake/                       # The central storage layer (Big Data Principle)
│   ├── season_data/                 # Raw ingestion layer (.parquet files)
│   └── clean_data/                  # Processed presentation layer (.parquet files)
│
├── docs/
│   └── architecture.md              # Detailed breakdown of the ETL pipeline
│
└── requirements.txt
```

## 🚀 Getting Started

### 1. Installation
1. Clone the repository to your local machine.
2. Ensure you have Python 3.10+ installed.
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. The Data Pipeline (Backend)
If you need to fetch brand new data or process existing raw data, navigate to the project root and run the backend scripts:

**(Optional) Download Raw Data:**
```bash
python backend/scripts/download_season_data.py
```
*Depending on the season, this ingests large amounts of telemetry directly from the official F1 timing APIs and saves them in `data_lake/season_data/`.*

**Clean the Data:**
```bash
python backend/scripts/clean_telemetry.py
```
*This acts as the Transformation step in our ETL pipeline. It strips dropped packets, clamps impossible speed/RPM outliers, and optimizes the Parquet schema saving the output to `data_lake/clean_data/`.*

### 3. Running the Dashboard (Frontend)
To launch the interactive F1 dashboard, ensure you are in the project root and run:

```bash
cd frontend
streamlit run app.py
```

### 🎯 Key Features
- **Beginner Mode:** Toggle simple terms like "Gas Pedal" instead of "Throttle %" to make telemetry accessible to non-racing fans.
- **Traffic Light Track Map:** A dynamic spatial scatter plot coloring the track based on braking (Red) and acceleration (Green) zones.
- **Telemetry Playback Scrubber:** An interactive time slider that dynamically filters the graphs so you can "scrub" through a lap.
- **Automated Data Cleaning:** A robust python backend script that ensures Streamlit never crashes due to dropped sensor packets.
