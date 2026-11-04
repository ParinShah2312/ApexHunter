# ApexHunter — Project Documentation

**Project:** F1 telemetry and onboard-video analytics · **Courses:** Data Science & Machine Learning
(CMP707) · Deep Learning · **Year:** 2026–27 · **Author:** Parin Shah (23001091)

This is the short technical reference. Longer versions with every experiment and fix:
[ARCHIVE/PROJECT_DOCUMENTATION_full_2026-11-04.md](ARCHIVE/PROJECT_DOCUMENTATION_full_2026-11-04.md).
Plan, status and bug log: [ROADMAP.md](ROADMAP.md). Mistake detection in full:
[ISOLATION_FOREST_REWORK.md](ISOLATION_FOREST_REWORK.md).

---

## 1. Overview

ApexHunter downloads Formula 1 telemetry (the car's speed, throttle, brake, gear, RPM and position,
several times a second) and onboard video, runs machine-learning and deep-learning models on them
and shows the results on a Streamlit dashboard.

| Part | Method | Course | Question |
|---|---|---|---|
| **Driver mistakes** | Z-score vs Isolation Forest vs LSTM autoencoder | DSML / DL | Where did the driver do something different from the fastest drivers, in a way that cost time? |
| **Tyre pace + pit window** | "No change" vs Ridge vs vanilla RNN vs LSTM | DSML / DL | How much slower will the next lap be, and when should the driver stop for tyres? |
| Racing line | A\*, Dijkstra, BFS on a grid | DSML (search) | Is there a cheaper path through the driver's own line? |
| Apex detection | YOLOv11-Seg | DL (CNN) | How close does the car get to each apex curb? |
| Data pipeline | pandas; Spark, HDFS, MongoDB | — | Clean, store and serve the data |

**Main results:** the LSTM autoencoder finds the most planted mistakes (F1 0.40 vs 0.34 Isolation
Forest vs 0.28 Z-score). The RNN and LSTM predict the next lap best (0.33 s error vs 0.34 s Ridge vs
0.36 s "same as last lap"), but no model gives better pit windows than simple rules: a ceiling test
shows that real pit stops are decided by strategy and safety cars, not by tyre wear.

## 2. Layout

```
backend/config.json            seasons, CV thresholds, circuits, video list
backend/scripts/
  utils.py                     shared paths, config, logger, FastF1 cache
  detect_mistakes.py           mistake detection (+ --compare: all three detectors)
  train_ridge.py, train_lstm.py --cell lstm|rnn, check_strategy.py, predict_strategy.py
  optimal_line.py              racing line
  etl/                         download, clean, timestamp fix, video and satellite downloads
  isolation_forest/            mistakes_features, _io, _model, _eval, _autoencoder, _events, _timeloss
  lstm/                        tyre_data, tyre_model (TyrePaceRNN), tyre_io, tyre_ridge
  strategy/                    pit_cost, pit_window, strategy_check, strategy_output
  cv/, racing_line/, bigdata/  apex detection, racing line, Big Data
  tests/                       unit/ (31 files), integration/ (9 files), run_tests.py
frontend/app.py, components/   Streamlit dashboard
models/                        best.pt (YOLO); tyre LSTM, RNN and Ridge models + configs
data_lake/                     data and outputs (gitignored)
docs/                          this file, ROADMAP, ISOLATION_FOREST_REWORK, submission/, ARCHIVE/, LAST_SEM/
```

## 3. Data

| Data | Amount |
|---|---|
| Telemetry (FastF1): every race and qualifying of 2023 and 2024 | 92 sessions, 57.9 M rows, 1.7 GB (Parquet) |
| Lap data (FastF1): lap times, stints, tyre compound and age, pit times, track status | ~50,000 race laps |
| Onboard pole-lap videos (YouTube) | 46 videos, 21,282 frames, 506 hand-labelled |

Samples arrive about every 0.13 s but irregularly. `Brake` is on/off, not a pressure. Cleaning
(`etl/clean_telemetry.py`): drop rows with no core values, fill short gaps, clip impossible values
(speed 0–380 km/h, RPM 0–15,000, throttle 0–100 %, brake 0–1), store as float32, write via a
temporary file. The Spark version (`bigdata/`) does the same in its own folder; HDFS and MongoDB
storage are optional.

## 4. Driver mistake detection

Every sample is compared with the five fastest other drivers **at the same point on the track**.
Five features (speed, throttle, brake, gear and acceleration differences, divided by the field's
spread there) feed three detectors, each flagging its 2 % most unusual samples. Flags become
events with a corner name, a reason in plain words and the time lost.

Tested with **planted mistakes** (fake lock-ups, early lifts, late throttle, pedal overlap, missed
downshifts in laps the model has not seen) and a **time-lost check** on real laps.

| Mean of 4 sessions | Precision | Recall | F1 |
|---|---|---|---|
| Z-score | 0.55 | 0.20 | 0.28 |
| Isolation Forest | 0.51 | 0.27 | 0.34 |
| **LSTM autoencoder** | 0.54 | **0.34** | **0.40** |

Details, per-session numbers and decisions: [ISOLATION_FOREST_REWORK.md](ISOLATION_FOREST_REWORK.md).

## 5. Tyre pace and pit window

### Data
- Racing laps on dry tyres only (no lap 1, pit, safety-car or wet laps).
- Each lap → 9 numbers: lap time; time-weighted mean throttle, brake and RPM; tyre age; compound
  (soft/medium/hard); race lap. Lap time and the three means are given relative to the window's
  last lap, so the model sees the *change*, not the track.
- **Windows:** 5 consecutive laps → **target = next lap time minus last lap time** (s, + = slower).
  28,745 windows from 45 races; the change averages +0.02 s with a spread of 0.90 s.
- **Split by whole races, in date order:** the 5 latest races (2024 R19–R24, 3,006 windows) are the
  test set; scalers are fitted on training races only.

### Models
- **Ridge** (`train_ridge.py`): linear regression with a penalty (α = 30,000, chosen on the latest
  training races) on the 45 flattened inputs.
- **Vanilla RNN and LSTM** (`train_lstm.py --cell rnn|lstm`): same class (`TyrePaceRNN`), 2 layers,
  dropout 0.3, Adam, MSE; 6-setting search, then the final model with 3 seeds; best epoch kept.

| Test races | MAE (s) | RMSE (s) | R² (of the change) | MAPE (% of lap time) | MAE over 3 seeds |
|---|---|---|---|---|---|
| "No change" | 0.357 | 0.593 | 0.00 | 0.392 | – |
| Ridge | 0.344 | 0.549 | 0.14 | 0.377 | – |
| **Vanilla RNN** | **0.328** | **0.509** | **0.26** | **0.360** | 0.332 ± 0.007 |
| LSTM | 0.335 | 0.517 | 0.24 | 0.368 | 0.338 ± 0.004 |

Both networks beat Ridge; the plain RNN is level with the LSTM (with only 5 steps the gradients do
not vanish, so the LSTM's gates bring no advantage). After 10–17 epochs both start to memorise, so
the best epoch is kept.

### Pit window (`strategy/`)
- **Pit cost** from each race's own stops: in-lap + out-lap minus the cars that did not stop.
  Median 22.8 s (18.5–30 s by track); under a safety car or VSC × 0.7; no advice on a red flag.
- **No compound choice:** race lap times cannot separate soft, medium and hard (the gap flips sign
  with the lap filter), so the advice uses the compound the driver really fitted.
- **Calculation:** for every possible stop lap, cost = wear × (old-tyre ages + new-tyre ages up to
  the end of the following stint) + pit cost. Fuel, track and car are the same for every option and
  cancel. **Wear** = average lap-time change so far in the stint + 0.053 s/lap fuel gain, clipped
  at 0. **Window** = best stop lap ± 2 (5 laps). A safety-car lap gets "pit now" when the cheaper
  stop makes it the best option.
- **Check** (`check_strategy.py`, same 163 stints of the test races): is the real stop inside the
  window, compared with a random 5-lap window?

| Wear from | Real stop in window | Random window | Margin |
|---|---|---|---|
| Laps driven (real lap times) | 46 % | 24 % | **+22** |
| "No change" (fuel only) | 44 % | 30 % | +15 |
| Ridge | 37 % | 29 % | +9 |
| LSTM | 34 % | 28 % | +6 |
| Vanilla RNN | 33 % | 27 % | +5 |

- **Ceiling test:** with the *true* wear of each stint (future laps included) the window does no
  better than "no change" (39 % vs 43 % over 1,345 stints). Real stops depend on undercuts, traffic
  and safety cars, and wear is small (~0.05 s/lap), so better lap-time models cannot improve the
  pit window. The dashboard shows whichever source scores best (now "laps driven").
- **Output:** `predict_strategy.py --session … --driver … [--source all]` →
  `data_lake/tyre_predictions/{race}_{driver}_strategy.json` (laps, stints, window, safety-car
  calls, real stop, summary; with `--source all`, every wear source for the dashboard's model
  switch). Outputs exist for every driver of 8 races, all with `--source all`.

## 6. Other parts (from last semester; More tab)

- **Racing line** (`optimal_line.py`): the driver's own session positions become a grid (cost =
  slow and braking cells are expensive); A\*, Dijkstra and BFS search it between 5 points of the
  fastest lap. A\* = Dijkstra (cost 966.88, 4 % fewer nodes), BFS 2.4 % more expensive. It finds
  no faster line, by design: the grid only holds cells the driver drove.
- **Apex detection** (`cv/`): YOLOv11-Seg fine-tuned on 506 labelled frames (curb, road). For each
  video frame it measures the distance from the front wheel to the apex curb inside corner windows
  built from FastF1 corner data. 2024 Bahrain pole lap: 12 of 15 corners "Hitting Apex".

## 7. Dashboard

`streamlit run frontend/app.py`. Redesigned in November 2026 (ROADMAP §3).
- **Sidebar:** season switch → Grand Prix → Qualifying/Race switch → driver (with team); badges show
  whether mistake and pit-window results exist. Opens on 2023 Bahrain Race, Verstappen.
- **Header:** Grand Prix and session, driver in team colour (team from FastF1 for that session),
  finishing/qualifying position, fastest lap, gap to the session's fastest lap and who set it, top
  speed.
- **Tabs** (only the open one runs): **Overview** (the driver's mistakes and pit windows in short; scoreboard of the trained models:
  mean planted-mistake F1, test lap-time error), **Driver Mistakes** (detector switch
  Isolation Forest / LSTM autoencoder / Z-score; each one's mistakes on a track map and in a list;
  the chosen mistake against the field and the driver's usual laps; how the three detectors compare
  on the planted-mistake test and the real-lap check), **Tyres & Pit Window** (model switch Ridge / RNN / LSTM, opening on the best pit-window
  margin; the "no change" and "laps driven" baselines are not shown there, see §5; stint
  buttons; the stint's lap times with the window, safety-car laps, real and best stop and the
  model's predicted laps; lap-time error per model on the test races; pit window vs a random
  window),
  **Telemetry** (one lap at a time against distance, compared with another lap or another driver's
  fastest lap: lap picker, speed, time delta, throttle, brake bands, gear, RPM, track map coloured by
  speed or gear, a position slider that names the place on the lap, plain-text hover), **More** (apex detection video
  with its own player slider and corners hitting the apex, racing line with where it exists,
  Big Data).

## 8. Status

### Done this semester
Bug fixes (#1–#55) and four test audits; mistake detection rebuilt and compared with Z-score and an
LSTM autoencoder; tyre models rebuilt around lap time with Ridge, RNN and LSTM and a pit-window
check against real stops; old "tyre cliff" removed; project-wide review. Every change and its
checks: [ARCHIVE/PROJECT_DOCUMENTATION_full_2026-11-04.md](ARCHIVE/PROJECT_DOCUMENTATION_full_2026-11-04.md) §8.

### Known issues
| Area | Issue |
|---|---|
| Tyre models | Better lap times, but no better pit windows (ceiling test). |
| Mistakes | Two thirds of planted mistakes are missed; some events differ from the field without costing time; only 4 sessions scored. |
| Outputs | Mistakes for 4 sessions, pit window for 8 races, racing line and CV for one session each (#7). |
| Dashboard | Clicking map markers, list rows and lap dots is not covered by automated tests (the dropdowns doing the same are). MongoDB section not checked (MongoDB not running). |
| Racing line | Cannot find a faster line by design. |
| CV | Video list not filtered by session; YOLO training metrics not in the repo. |
| Misc | `Time` column is made from the row index; satellite images unused. |

## 9. Testing

197 tests (140 unit + 57 integration), all passing. Every test runs real project code on real data
(cached races, integration) or small inputs with a known answer (unit); no mocks. Each bug fix has a
test that fails on the old code. Run: `python backend/scripts/tests/run_tests.py` (unit, ~30 s) or
`--all` (with integration, ~3 min); lint: `python -m ruff check --select F backend frontend`.

## 10. How to run

```bash
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
python backend/scripts/etl/download_season_data.py     # FastF1 -> data_lake/season_data/
python backend/scripts/etl/clean_telemetry.py          # -> data_lake/clean_data/
python backend/scripts/detect_mistakes.py --session data_lake/clean_data/2023_1_R.parquet --driver 1 --compare
python backend/scripts/train_ridge.py
python backend/scripts/train_lstm.py --cell lstm --force   # and --cell rnn (~16 min each, CPU)
python backend/scripts/check_strategy.py
python backend/scripts/predict_strategy.py --session data_lake/clean_data/2024_22_R.parquet --driver 1 --source all
streamlit run frontend/app.py
```
