# ApexHunter 2.0: Data Architecture & Warehousing Analysis Report

## Executive Summary
This report provides a comprehensive breakdown of the current data structures, collection mechanisms, and storage architectures within the ApexHunter 2.0 codebase. It outlines exactly what telemetry and computer vision data is currently being ingested, identifies critical missing data points, and proposes a strategic architectural shift to transition from the current flat-file Data Lake into a highly scalable, "Massive Analytical Data Warehouse."

---

## 1. Current Data Collection & Schema Structures

The system currently extracts data using the `fastf1` library for telemetry and `ultralytics` (YOLO) for computer vision. The data is structured across multiple domain-specific pipelines.

### A. Raw & Cleaned Telemetry Pipeline
Collected per session at approximately 10Hz.
*   **Identifiers**: `Driver` (String Code), `Year` (Int), `Round` (Int), `Session` (String - Q, R).
*   **Timestamps**: `Date` (Datetime), `SessionTime` (Timedelta).
*   **Positional**: `X` (Float32), `Y` (Float32). *(Note: `Z` coordinate is collected raw but currently ignored/downcasted).*
*   **Driver Inputs**: `Throttle` (0-100%, Float32), `Brake` (0-100%, Float32), `nGear` (1-8, Int).
*   **Car State**: `Speed` (km/h, Float32 - clipped at 380), `RPM` (Float32 - clipped at 15000).

### B. Machine Learning: Tyre Degradation & Lap Features
Aggregated on a per-lap basis for Stint detection and LSTM model training.
*   **Averages**: `mean_speed`, `mean_throttle`, `mean_brake`, `mean_rpm`.
*   **Tyre Meta**: `tyre_age` (Float), `is_soft` (Binary), `is_medium` (Binary), `is_hard` (Binary).
*   **Lap Meta**: `race_lap_number` (Int), `lap_time_seconds` (Float), `cliff_lap` (Boolean target).

### C. Machine Learning: Mistake Detection (Isolation Forest)
Engineered features calculated row-by-row to detect micro-mistakes.
*   **Normalized Features**: `speed_normalized`, `throttle_intensity`, `brake_intensity`, `rpm_normalized`.
*   **Complex Interactions**: `brake_throttle_overlap` (Boolean: hitting both pedals), `speed_delta` (Row-to-row change), `gear_change`.
*   **Outputs**: `anomaly_score` (Float32), `is_mistake` (Boolean).

### D. Pathfinding: Racing Line Grid Graph
Aggregated grid nodes converting X/Y telemetry into a 2D weighted graph.
*   **Spatial**: `grid_i`, `grid_j`, `center_x`, `center_y`.
*   **Node Stats**: `mean_speed`, `mean_brake`, `point_count`.
*   **Graph Cost**: `weight` (Calculated using speed reward and brake penalty).

### E. Computer Vision: YOLO Semantic Segmentation
Extracted from pole lap videos.
*   **Geometric Distance**: Distance in pixels from the front wheel reference point to the track curb.
*   **Classification**: `Apex Status` ("Hitting Apex", "Near Apex", "Missing Apex").
*   **Context**: `Curb Presence` (Boolean), `Turn Direction`.

---

## 2. Current Storage & Warehousing Mechanisms

Currently, ApexHunter 2.0 does **NOT** use a traditional database (RDBMS, NoSQL, or Time-Series). Instead, it relies on a local **Flat-File Data Lake Architecture**:

*   **Storage Medium**: Local file system under the `data_lake/` directory.
*   **Data Formats**: 
    *   **`.parquet`**: Used for all telemetry, mistake data, and tyre features. Compressed using the `snappy` algorithm. Parquet is columnar and highly efficient for Pandas processing, but it lacks query-ability across files.
    *   **`.csv`**: Used for Computer Vision HUD metrics.
*   **Architecture Flow**: `season_data/` (Raw) -> `clean_data/` -> `mistake_data/` / `tyre_predictions/` / `racing_lines/`.
*   **Shortcomings**:
    *   **No Relational Integrity**: A driver code ("VER") in one file is not formally linked to a `Drivers` table. 
    *   **No Cross-Session Querying**: Finding "Every lap where Verstappen had a mistake at Turn 10 across 3 seasons" requires loading hundreds of Parquet files into RAM.
    *   **No Time-Series Optimization**: Telemetry is inherently time-series data, but it is currently stored as static tabular blocks.

---

## 3. The "Missing Data" (Collection Opportunities)

To achieve a true "Massive Analytical Warehouse," the following data dimensions are currently absent and represent massive growth opportunities.

### A. Environmental & Track Data (Crucial for Tyre Models)
*   **Currently Missed**: The system explicitly passes `weather=False` when calling `fastf1.get_session()`.
*   **Must Collect**: `Track Temperature`, `Air Temperature`, `Humidity`, `Rainfall`, `Wind Speed`, and `Wind Direction`.
*   **Contextual**: `Track Status` (Green, Yellow Flag, Safety Car, Virtual Safety Car). Degradation models are severely handicapped without knowing if a driver was forced to drive slowly behind a Safety Car.

### B. Car Setup & Dynamics
*   **Currently Missed**: We track speed and pedals, but miss the aerodynamic and steering profile.
*   **Must Collect**: `DRS Status` (Drag Reduction System Active/Inactive), `Steering Angle` (How much is the wheel turned? Critical for mistake detection), `Z-Coordinate (Elevation)` (Currently collected but ignored—crucial for tracks like Spa or COTA).
*   **Potential External Data**: Tyre surface temperatures, Fuel load estimation (kg), ERS (Energy Recovery System) deployment status.

### C. Advanced Computer Vision Metrics
*   **Currently Missed**: The YOLO model only looks at curbs.
*   **Must Collect**: Dynamic Track Limits (White lines), distance to the car ahead (dirty air calculation), slip angle estimation, and driver helmet movement (tracking driver eye focus).

---

## 4. Vision: Blueprint for a Massive Analytical Warehouse

To transition from local Parquet files to an enterprise-grade racing data warehouse, the following architecture must be implemented:

### Step 1: The Time-Series Core (Telemetry Data)
**Technology**: TimescaleDB (PostgreSQL extension) or InfluxDB.
**Purpose**: Telemetry is high-frequency time-series data. A TSDB will allow sub-second queries like, "Give me all telemetry for Lewis Hamilton in Sector 2, over the last 5 years, where tyre age > 15 laps."
*   **Table Structure**: A single massive hypertable partitioned by `SessionTime` and indexed by `Driver_ID` and `Circuit_ID`.

### Step 2: The Relational Metadata Layer (SQL)
**Technology**: PostgreSQL.
**Purpose**: Enforce data integrity and relationships.
*   **Tables Needed**: 
    *   `Drivers` (ID, Name, Acronym, Team)
    *   `Circuits` (ID, Name, Length, Corners, Elevation Profile)
    *   `Sessions` (ID, Circuit_ID, Date, Type, Weather_Summary)
    *   `Stints` (ID, Session_ID, Driver_ID, Compound, Start_Lap, End_Lap)

### Step 3: The Object Store Data Lake (Video & Deep Learning)
**Technology**: AWS S3 / Google Cloud Storage (or MinIO for local dev).
**Purpose**: Store the massive raw video files, YOLO frame outputs, and raw `fastf1` cache JSONs. The database will only hold the metadata URI pointers to these objects.

### Step 4: Data Correlation & Analytics Engine
**Technology**: Apache Spark or Snowflake.
**Purpose**: Once data is warehoused, you can correlate domains.
*   *Example Query*: Cross-reference the CV Database (Apex distance) with the Telemetry Database (Brake pressure) and Weather Database (Track Temp) to mathematically prove that *Max Verstappen brakes 5 meters later than the grid average in Turn 1 when Track Temps exceed 35°C on Soft Tyres.*

### Conclusion
ApexHunter 2.0 has an incredibly strong foundation in feature engineering (Isolation Forests, A* Pathfinding grids, LSTM sequences). However, its current bottleneck is the flat-file storage system. By separating data into a highly indexed Relational DB (for metadata) and a Time-Series DB (for 10Hz telemetry), the system can scale from a local script-based tool into a commercial-grade F1 Data Warehouse.
