# Artificial Intelligence Project Requirements

**Course:** AI, Cloud and Big Data  
**Academic Year:** 2023-2024 (or Current Year)  

## Project Group Details

| Sr. No. | Enrolment Number(s) | Student Name(s) |
| :--- | :--- | :--- |
| 1 | 23001091 | Parin Shah |
| 2 | | |
| 3 | | |
| 4 | | |

*Based on the nature of data (Alphanumeric Tabular, Text (labeled, unlabeled), Images or Videos (labeled, unlabeled)) in your project you may choose appropriate techniques.*

**Nature of Data used in ApexHunter 2.0:** 
- **Tabular:** FastF1 Telemetry data (Parquet formats) 
- **Images and Videos:** Onboard pole lap videos (MP4) and extracted CV frames (JPG/PNG).

---

## 1. Problem Definition
- **Explicitly define a problem:** 
  The problem is to analyze Formula 1 driver telemetry combined with visual apex detection to evaluate a driver's cornering precision. The goal is to ingest gigabytes of raw FastF1 data, process it, run a computer vision model to detect track curbs (apexes), and calculate a "Perfect Corner Score" based on how well the driver avoids overlapping brake and throttle applications.
- **Write steps involved:** 
  1. Fetch FastF1 telemetry and onboard pole lap videos.
  2. Run ETL pipeline to clean and clip telemetry data.
  3. Extract frames from onboard videos at 5fps.
  4. Annotate frames and train a YOLOv11-Seg model.
  5. Run inference on lap videos to detect apexes and deviation metrics.
  6. Serve results via an interactive Streamlit dashboard equipped with WebGL track maps and telemetry charts.

---

## 2. Data Collection Methods (Tabular, Text and Image)
- **Primary / Secondary data : CSV, JSON or SQL Dumps from repositories:** 
  Secondary Data: Fetching F1 telemetry data via the `FastF1` API (saved as Parquet files).
- **Web scrapping / Crawling:** 
  Extracting onboard lap videos via `yt-dlp` from YouTube and scraping circuit satellite images using parallel threading.
- **Sensors / IOT:** 
  N/A
- **Images or Videos:** 
  Onboard pole lap videos downloaded and trimmed, from which ~500 diverse frames were extracted to build a custom computer vision dataset.
- **Survey, Interview:** 
  N/A

---

## 3. Data Exploration (Tabular, Text and Image)
- **Visualize the dataset:** 
  Visualized speed, throttle, and brake telemetry over time using interactive Plotly subplots. Visualized spatial coordinates mapping circuit layouts using scatter plots.
- **Check shape of data:** 
  Explored rows and columns of Parquet files to understand sampling rates of telemetry (e.g., matching distance and time indices). 
- **Treating missing values:** 
  Interpolating missing telemetry data points where signal drops occur.
- **EDA (only in tabular data):** 
  Analyzed overlapping brake and throttle zones to design the gamified "Perfect Corner Score" metric. Evaluated apex timing in relation to speed telemetry.

---

## 4. Data Pre-Processing (Tabular, Text and Image)
- **Data Cleaning:** 
  Removing unwanted lap segments, filtering out outlier telemetry coordinates, and smoothing traces using custom Python routines (`clean_telemetry.py`).
- **Feature selection and engineering:** 
  Engineered features like "Apex Deviation" by matching YOLO bounding box centroids to visual references, and combining brake/throttle percentages into a unified performance score.
- **Treating Missing values and outlier detection and its treatment:** 
  Clipping outlier coordinates during extraction to prevent skewed track maps. 
- **Encodings and Embeddings:** 
  N/A
- **Scaling and Normalization:** 
  Normalized distance and telemetry parameters across different tracks for uniform dashboard rendering.
- **Data Augmentation, Data Orientation (images):** 
  Used Roboflow for YOLO dataset preparation, applying augmentation techniques (e.g., flips, cropping, color shifts) to make the model robust against various tracks and lighting conditions.

---

## 5. Data Splitting (Tabular, Text and Image)
- **Train-Test Split:** 
  N/A
- **Train, Validation, Test split:** 
  The raw extracted YOLO image dataset was uploaded to Roboflow and split into Train, Validation, and Test sets to evaluate the segmentation model's generalizability on unseen corners.

---

## 6. Model Selection (Tabular, Text and Image)
- **Classification:** N/A
- **Regression:** N/A
- **Clustering:** N/A
- **Dimensionality Reduction:** N/A
- **Deep Learning:** 
  - **YOLOv11-Seg:** Selected for real-time computer vision inference to detect track curbs and racing apexes.
  - **LSTM (Long Short-Term Memory):** Recurrent Neural Network for "Tyre Cliff" forecasting, predicting performance drops based on historic stint telemetry.
- **Model Selection (Additional):** 
  - **Isolation Forest:** An unsupervised anomaly detection algorithm used for "Driver Mistake Detection" by flagging deviations in brake and throttle telemetry compared to a reference pole lap.
- **Recommendation System:** N/A
- **Specify if pre-trained model is used with fine tuning:** 
  - **YOLOv11-Seg:** Fine-tuned on a custom F1 dataset from pre-trained COCO weights.
  - **LSTM & Isolation Forest:** Built and trained from scratch using the processed FastF1 telemetry dataset.

---

## 7. Model Training (Tabular, Text and Image)
- **Optimize weights:** 
  - Trained YOLOv11-Seg on a T4 GPU (Google Colab).
  - LSTM weights optimized using Adam optimizer and Mean Squared Error (MSE) loss function.
- **Cross Validation:** 
  - K-Fold cross-validation (if applicable) for the Isolation Forest to ensure consistent anomaly detection across different drivers.
- **Overfitting issues (Dropout, Regularization):** 
  - Added Dropout layers to the LSTM architecture to prevent overfitting on specific track layouts.
  - Used Roboflow augmentations (flips, color shifts) for the YOLO model.

---

## 8. Model Evaluation as per data and techniques used.
- **YOLOv11-Seg:** Evaluated using mAP50 and mAP50-95 for mask/box accuracy.
- **Isolation Forest:** Evaluated using Anomaly Scores; validated by cross-referencing flagged timestamps with onboard video footage of mistakes (lock-ups/slides).
- **LSTM:** Evaluated using Mean Absolute Error (MAE) between predicted lap times and actual recorded lap times.

---

## 9. Parameter Tuning Methods
- **GridSearchCV:** 
  Used to find the optimal `contamination` parameter for the Isolation Forest mistake detector.
- **RandomSearchCV:** 
  Used to tune LSTM hyperparameters (number of hidden units, learning rate).
- **Bayesian Optimization:** 
  Optional YOLO hyperparameter evolution to maximize detection mAP.

---

## 10. Model Deployment
- **Streamlit:** 
  The frontend (`app.py`) integrates all model outputs:
  - **Track Map:** Color-coded GPS points highlighting mistakes detected by the Isolation Forest.
  - **Predictive Chart:** An "Estimated Tyre Life" graph showing the LSTM-predicted "cliff" point.
  - **CV HUD:** Streamed video showing YOLO apex detection markers.
- **Flask, Docker, ML Flow:** N/A
- **CI/CD pipelines:** N/A
- **Kubernetes:** N/A
- **AWS, Azure, GCP:** 
  Google Colab (GCP) used for GPU-accelerated training of YOLO and LSTM models.
