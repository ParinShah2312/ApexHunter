# 🏗️ ApexHunter Architecture Design

ApexHunter utilizes a simplified **Big Data Pipeline Architecture** emphasizing the separation of concern between data harvesting, cleaning, and serving.

## The ETL Flow

### 1. Extract (Data Harvest)
**Component:** `backend/scripts/download_season_data.py`
**Trigger:** Manual (via Terminal)

This script interfaces with the `FastF1` public API. It operates hierarchically by looping through full F1 Seasons -> Specific Rounds -> Specific Qualifying and Race sessions. Features extracted include driver lap timings, positional metrics (X,Y,Z coordinates), and granular vehicle sensor telemetry (RPM, Speed, Throttle, Brake, Gear). 
**Output Location:** `data_lake/season_data/` (Raw Parquet Data Pool)

### 2. Transform (Data Cleaning & Normalization)
**Component:** `backend/scripts/clean_telemetry.py`
**Trigger:** Manual (via Terminal)

Because telemetry sensors attached to F1 cars occasionally drop packets going over curbs or passing grandstands, the raw `season_data` cache is inherently dirty. 
The ETL script:
1. **Filters:** Drops any dataframe rows that contain entirely empty location or speed metrics.
2. **Imputes/Interpolates:** Forward-fills (ffill) any microscopic gaps during a stint where only *one* sensor packet failed to record.
3. **Clips Extreme Outliers:** Failsafes numerical scales. Even anomalous glitches cannot register a car travelling at `800 km/h` on the dashboard because our threshold cap is set to a realistic `380 km/h`.

**Output Location:** `data_lake/clean_data/` (Processed Parquet Layer)

### 3. Load & Serve (Dashboard Frontend)
**Component:** `frontend/app.py`
**Trigger:** Interactive Web Service (via Streamlit run)

This represents the User Interface payload. 
By delegating the Extract and Transform loads to the `backend/scripts`, the visualization layer operates extremely quickly. It only consumes pre-validated, clean `.parquet` files from the central `clean_data` lake. This separation means the Streamlit frontend cannot break even if the `FastF1` APIs go down entirely.
