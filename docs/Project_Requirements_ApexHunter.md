# ApexHunter 2.0 — Project Requirements

**Courses:** Data Science & Machine Learning · Deep Learning
**Academic Year:** 2026–2027

## Project Group Details

| Sr. No. | Enrolment Number | Student Name |
| :--- | :--- | :--- |
| 1 | 23001091 | Parin Shah |
| 2 | 23000936 | Ayush Parikh |
| 3 | 23001119 | Malav Joshi |

*Based on the nature of data (alphanumeric tabular, text, images or videos) you may choose appropriate techniques.*

**Nature of data used in ApexHunter 2.0:**
- **Tabular:** FastF1 telemetry, 92 sessions (2023–2024 qualifying and races), ~57.9M rows, Parquet.
- **Lap-level tabular:** FastF1 lap metadata (lap times, stints, compounds, tyre life).
- **Video / images:** 46 onboard pole-lap videos (MP4) and 21,282 extracted frames, of which 506 were
  annotated with segmentation masks.

> The previous semester's version of this document (AI & Big Data) is preserved in
> `docs/32_Ayush_Parin_AI_Prj_Specs.pdf/.docx`.

---

## 1. Problem Definition
- **Problem:** Measure Formula 1 driving quality from telemetry and onboard video. That means
  (a) automatically flagging unusual driving moments, (b) forecasting when tyre pace drops off
  within a stint, (c) measuring how close the car gets to each apex curb, and (d) comparing the
  driver's line with a cost-optimal path through their own telemetry.
- **Steps:**
  1. Download FastF1 telemetry and onboard pole-lap videos.
  2. Clean and validate the telemetry (pandas, with an equivalent Spark job).
  3. Engineer features and train an Isolation Forest for anomaly (mistake) detection.
  4. Aggregate laps into stint sequences and train an LSTM for next-lap pace prediction.
  5. Extract and annotate video frames; fine-tune YOLOv11-Seg for curb/road segmentation.
  6. Run segmentation inference to measure apex distance per frame.
  7. Run A\*, Dijkstra and BFS on a telemetry-weighted grid.
  8. Present everything in a Streamlit dashboard.

## 2. Data Collection Methods
- **Secondary data:** FastF1 API (official F1 timing feed), saved as Parquet.
- **Web scraping / crawling:** onboard videos with `yt-dlp`; satellite tiles from Esri via `staticmap` (10-thread pool).
- **Sensors / IoT:** N/A (car sensor data is consumed indirectly through FastF1).
- **Images / videos:** 46 videos trimmed to the lap, frames extracted at 5 fps, 506 selected for annotation.
- **Survey / interview:** N/A.

## 3. Data Exploration
- **Visualisation:** interactive Plotly time series (speed, throttle, brake, gear, RPM), speed-coloured track maps, per-driver speed histograms, Speed–RPM scatter by gear, correlation heatmap.
- **Shape of data:** ~420k rows per qualifying session and ~800k per race (20 drivers); median sampling interval ≈ 0.13 s; 15 columns after cleaning.
- **Missing values:** forward-filled within the session; rows missing all core channels are dropped.
- **EDA findings:** `Brake` is a binary on/off flag, not a pressure. Throttle occasionally exceeds 100 %. Speed and RPM are tied to gear in clear bands.

## 4. Data Pre-Processing
- **Cleaning:** drop empty packets, forward-fill, clip Speed/RPM/Throttle/Brake to physical limits, downcast to float32 (`etl/clean_telemetry.py`, `bigdata/spark_clean_telemetry.py`).
- **Feature engineering:**
  - Isolation Forest: 7 row-level features (normalised speed/throttle/RPM, binary brake flag, brake–throttle overlap, speed delta, gear change).
  - LSTM: 9 lap-level features (mean speed/throttle/brake/RPM, tyre age, one-hot compound, lap number), computed only on representative laps (no race start, pit, or safety-car/VSC/red-flag laps).
  - CV: inner-edge curb contour and wheel-to-curb Euclidean distance.
- **Outliers:** domain clipping; speed delta clipped to ±50 km/h and gear change to 0–4 inside the features.
- **Encodings:** one-hot encoding of tyre compound (soft/medium/hard).
- **Scaling / normalisation:** fixed physical-range normalisation for Isolation Forest features; `StandardScaler` for LSTM inputs and z-normalisation of the LSTM target.
- **Augmentation (images):** Roboflow flips, crops, colour/brightness shifts and small rotations (506 → 1,281 training images).

## 5. Data Splitting
- **Isolation Forest:** 5-fold `KFold` cross-validation for contamination selection.
- **LSTM:** chronological split by whole race sessions: the latest 10 % of sessions are the test set, and the latest 20 % of the remaining sessions are validation. Scalers are fitted on training sessions only.
- **YOLO:** Roboflow train/validation split (1,281 / 226 images).

## 6. Model Selection
- **Classification:** N/A.
- **Regression:** LSTM sequence regression (next-lap mean speed).
- **Clustering:** N/A.
- **Dimensionality reduction:** N/A.
- **Anomaly detection:** Isolation Forest (unsupervised).
- **Deep learning:**
  - **LSTM**, 2 stacked layers with dropout, for tyre-degradation forecasting.
  - **YOLOv11-Seg**, CNN instance segmentation, for curb/road masks.
- **Search / optimisation:** A\*, Dijkstra and BFS racing-line search.
- **Pre-trained with fine-tuning:** YOLOv11-Seg from COCO weights. The LSTM and Isolation Forest are trained from scratch.

## 7. Model Training
- **Optimise weights:** LSTM uses Adam + MSE loss, 20 epochs per search configuration and 50 for the final model, batch 32. YOLO was trained on a Google Colab T4 GPU.
- **Cross-validation:** 5-fold CV over Isolation Forest contamination values.
- **Overfitting:** 0.3 dropout between LSTM layers and best-validation tracking; Roboflow augmentation for YOLO.

## 8. Model Evaluation
- **Isolation Forest:** mean |decision function| on held-out folds; mistake rate; flagged points inspected on the track map.
- **LSTM:** test MAE in km/h (retraining with the corrected pipeline pending), also median AE, max AE and RMSE; Monte Carlo dropout confidence bands.
- **YOLOv11-Seg:** mAP50 / mAP50-95 (box and mask) from the Colab run; per-frame apex-status distribution on the processed lap.
- **Racing line:** path cost, nodes expanded and runtime, compared across A\*, Dijkstra and BFS.

## 9. Parameter Tuning
- **Grid search + K-fold CV:** Isolation Forest contamination ∈ {0.005, 0.01, 0.015, 0.02, 0.03, 0.05}.
- **Grid search (validation set):** LSTM hidden size × learning rate, 6 configurations.
- **YOLO:** Ultralytics default training configuration.

## 10. Model Deployment
- **Streamlit** dashboard (`frontend/app.py`) with Race Intelligence, Telemetry, AI Analysis, Racing Line and Big Data tabs.
- **Flask, Docker, MLflow, CI/CD, Kubernetes:** N/A.
- **Cloud:** Google Colab GPU for YOLO training.

See `docs/PROJECT_DOCUMENTATION.md` §8 for current limitations.
