# 🏎️ ApexHunter 2.0 — F1 Telemetry & Onboard-Video Analytics

ApexHunter analyses Formula 1 driving with official telemetry (FastF1, 2023–2024, 92 sessions,
57.9 M rows) and onboard pole-lap video, and shows the results on a Streamlit dashboard.

**Courses (AY 2026–27):** Data Science & Machine Learning · Deep Learning
*(First built last semester for Artificial Intelligence and Big Data.)*

| Part | Method | Result |
|---|---|---|
| Driver mistakes | Z-score vs Isolation Forest vs LSTM autoencoder, comparing each driver with the fastest drivers at the same point on track | F1 on planted mistakes: 0.28 / 0.34 / **0.40** |
| Tyre pace + pit window | "No change" vs Ridge vs vanilla RNN vs LSTM predicting the next lap time; pit window checked against the drivers' real stops | Lap-time error: 0.36 / 0.34 / **0.33** / 0.34 s; pit window ~2× better than chance |
| Racing line (frozen) | A\*, Dijkstra, BFS on a telemetry grid | A\* = Dijkstra, 4 % fewer nodes |
| Apex detection (frozen) | YOLOv11-Seg on 506 labelled onboard frames | 12 of 15 Bahrain corners "Hitting Apex" |
| Data pipeline | pandas; Spark, HDFS, MongoDB (frozen) | — |

**Docs:** [PROJECT_DOCUMENTATION](docs/PROJECT_DOCUMENTATION.md) (how it works, results, how to
run) · [ROADMAP](docs/ROADMAP.md) (status, scope, bug log, rules) ·
[ISOLATION_FOREST_REWORK](docs/ISOLATION_FOREST_REWORK.md) (mistake detection in full) ·
`docs/submission/` (report and presentation).

## Structure

```text
backend/scripts/   detect_mistakes.py, train_ridge.py, train_lstm.py, check_strategy.py,
                   predict_strategy.py, optimal_line.py
                   etl/ · isolation_forest/ · lstm/ · strategy/ · cv/ · racing_line/ · bigdata/ · tests/
frontend/          app.py + components/ (Streamlit dashboard)
models/            YOLO weights, tyre models (LSTM, RNN, Ridge)
data_lake/         data and outputs (gitignored) · cache/  FastF1 cache (gitignored)
docs/              documentation, submission/, ARCHIVE/, LAST_SEM/
```

## Getting started

```bash
git clone https://github.com/ParinShah2312/ApexHunter.git
cd "ApexHunter 2.0"
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

python backend/scripts/etl/download_season_data.py     # FastF1 -> data_lake/season_data/
python backend/scripts/etl/clean_telemetry.py          # -> data_lake/clean_data/

python backend/scripts/detect_mistakes.py --session data_lake/clean_data/2023_1_R.parquet --driver 1 --compare
python backend/scripts/train_ridge.py
python backend/scripts/train_lstm.py --cell lstm --force          # and --cell rnn (~16 min each)
python backend/scripts/check_strategy.py                          # scores every model's pit windows
python backend/scripts/predict_strategy.py --session data_lake/clean_data/2024_22_R.parquet --driver 1

streamlit run frontend/app.py

python backend/scripts/tests/run_tests.py          # unit tests; --all adds integration (real data)
```

Frozen parts: `optimal_line.py --session …/2023_1_Q.parquet --driver 1`;
`cv/run_inference.py --input "<video>" --video-offset 4.9`; Big Data scripts in `bigdata/` (need
local HDFS and MongoDB).

## Tech stack

FastF1 · pandas, NumPy, PyArrow · scikit-learn (Isolation Forest, Ridge) · PyTorch (RNN, LSTM, LSTM
autoencoder) · Ultralytics YOLOv11-Seg, OpenCV · Streamlit, Plotly · PySpark, HDFS, MongoDB · unittest

## Author

Parin Shah (23001091)
