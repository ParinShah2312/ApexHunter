# Dataset Documentation — ApexHunter 2.0

Answers to the sections of the *Dataset Documentation Template*, updated to match the current data.
Team: Parin Shah (23001091), Ayush Parikh (23000936), Malav Joshi (23001119).

### 1. Basic dataset information
- **Dataset name:** ApexHunter F1 Telemetry & Onboard Video Dataset
- **Domain:** Sports analytics / motorsport
- **Type of data:** Structured (Parquet telemetry and lap metadata) and unstructured (MP4 onboard video, JPEG frames, PNG satellite images)
- **Sources:** Official F1 timing via the `fastf1` library; YouTube via `yt-dlp`; Esri World Imagery via `staticmap`
- **Reference:** https://docs.fastf1.dev/
- **License / usage:** F1 data is proprietary but accessible for non-commercial, educational analysis through FastF1. Videos are used only for academic research.

### 2. Data characteristics
**Structured**
- **Records:** 57,874,119 rows across 92 session files (2023: 22 rounds, 2024: 24 rounds; qualifying + race)
- **Columns (15 after cleaning):**
  - `Date`: absolute timestamp
  - `SessionTime`: time since session start
  - `Speed` (km/h), `RPM`, `nGear` (0–8)
  - `Throttle`: 0–100 %
  - `Brake`: binary on/off flag (0/1)
  - `X`, `Y`, `Z`: track-position coordinates
  - `Driver`: car number as string, e.g. `"44"`
  - `Round`, `Session` (Q/R), `Year`
  - `Time`: synthesised row index
- **Types:** float32 (Speed, RPM, Throttle, Brake, X, Y), float64 (Z), int (nGear, Round, Year), string (Driver, Session), datetime/timedelta (Date, SessionTime, Time)
- **Targets:** none in the raw data. Derived targets: next-lap mean speed (LSTM); segmentation masks (YOLO).

**Unstructured**
- **Files:** 46 onboard videos; 21,282 extracted frames; 506 annotated frames (1,281 train / 226 valid after augmentation); 24 satellite images
- **Formats:** MP4 (≤1080p), JPEG, PNG
- **Size:** telemetry 1.72 GB raw / 1.69 GB clean; videos 2.3 GB raw / 1.8 GB trimmed; frames 6.7 GB; satellite 76 MB
- **Annotation:** Roboflow polygon masks, 2 classes — `curb` (0), `road` (1)

### 3. Big Data perspective (5 Vs)
- **Volume:** ~58M telemetry rows (~1.7 GB compressed) plus ~10 GB of video and frames.
- **Velocity:** sampled at about 7–8 Hz per car (median interval 0.13 s); ingested in batches per session.
- **Variety:** numeric time series, categorical identifiers, lap metadata, video, images.
- **Veracity:** official source, but with packet dropouts, GPS jitter, throttle readings above 100 % and a binary-only brake channel.
- **Value:** supports anomaly detection, tyre-degradation forecasting, apex measurement and racing-line comparison.

### 4. Data quality assessment
- **Missing values:** sporadic NaN packets; forward-filled. No rows needed dropping in the 2023–24 data.
- **Duplicates:** none at file level (one file per session); samples are unique per timestamp per driver.
- **Outliers:** throttle up to 104 %; clipped to physical ranges.
- **Noise:** X/Y GPS jitter, worst at low speed.
- **Imbalance:** races contain about twice as many rows as qualifying sessions; DNFs give truncated driver data.
- **Bias:** onboard videos cover only pole sitters, which favours top teams; car performance differs strongly between teams.

### 5. Data preprocessing details
- **Cleaning:** drop rows missing Speed, RPM, X and Y simultaneously; clip to physical limits.
- **Missing values:** forward-fill numeric columns.
- **Encoding:** missing driver → `UNKNOWN`; one-hot tyre compound for the LSTM.
- **Scaling:** range normalisation (Isolation Forest), StandardScaler (LSTM).
- **Transformations:** float32 downcasting; lap aggregation into stints; frame sampling at 5 fps; segmentation masks → contours → wheel-to-curb Euclidean distance; HUD overlay blending (`addWeighted`, α = 0.5).

### 6. Exploratory Data Analysis
- **Summary statistics:** per-session top/mean speed and RPM, driver counts, brake/throttle overlap %.
- **Visualisations:** time-series subplots with a scrubber, speed-coloured track maps, speed histograms, Speed–RPM-by-gear scatter, correlation heatmap.
- **Correlations (2023 Bahrain qualifying, all drivers):** Speed–nGear 0.98, Speed–RPM 0.92, RPM–nGear 0.88, Brake–Speed −0.47, Brake–RPM −0.55. Throttle–Brake is weakly positive (+0.32), which is unexpected; it likely reflects low-speed garage and pit-lane samples and is worth investigating.
- **Observations:** braking zones line up with sharp speed and RPM drops before slow corners; the brake channel is binary.

### 7. Tools and technologies
- **Language:** Python 3.13
- **Data:** pandas, NumPy, PyArrow, FastF1, yt-dlp, staticmap
- **ML / DL:** scikit-learn, PyTorch, Ultralytics YOLOv11, OpenCV, SciPy
- **Visualisation:** Plotly, Streamlit
- **Big data:** Apache Spark (PySpark 3.5.1), HDFS, MongoDB
- **Annotation:** Roboflow

### 8. Ethical and legal considerations
- **Personal data:** none beyond public driver identities and car numbers.
- **Consent:** public broadcast and timing data, used for non-commercial education.
- **Anonymisation:** not required.
- **Sensitive attributes:** none.

### 9. Data storage and management
- **Storage:** local data lake (`data_lake/`) with layered folders, optionally mirrored to HDFS. Summaries are stored in MongoDB.
- **Formats:** snappy Parquet, MP4, JPEG, PNG, JSON, CSV.
- **Organisation:** `season_data/` (raw) → `clean_data/` (curated) → model output folders (`mistake_data/`, `tyre_predictions/`, `racing_lines/`, `processed_video/`, `processed_csv/`).
- **Backup / recovery:** everything can be regenerated from the download scripts; FastF1 responses are cached in `cache/`.
- **Scalability:** columnar Parquet plus a Spark implementation of the cleaning job.
- **Ingestion:** Python scripts calling cached APIs.
