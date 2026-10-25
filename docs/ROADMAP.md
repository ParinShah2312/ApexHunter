# ApexHunter 2.0 — Semester Roadmap (AY 2026–27)

**Courses:** Data Science & Machine Learning (DSML) · Deep Learning (DL)
**Team:** Parin Shah (23001091), Ayush Parikh (23000936), Malav Joshi (23001119)

This file records **what we will and won't work on this semester**, so scope decisions don't need
to be re-discussed. Update the status column as items progress. Technical details of each bug are
in [PROJECT_DOCUMENTATION.md §8](PROJECT_DOCUMENTATION.md#8-current-status--known-limitations).

Status legend: ✅ done · 🔄 in progress · ⏳ planned · ⏸ deferred · 🤔 undecided · ✖ dropped · 🧊 frozen (no changes)

---

> **Course alignment:** which taught techniques we use, and why, is in
> [COURSE_ALIGNMENT.md](COURSE_ALIGNMENT.md).

## 1. Scope decisions

**Guiding principle (team decision, Oct 2026): few algorithms, clear story.** Each problem is
solved with a simple baseline → a machine-learning model → a deep-learning model, so the work is
easy to explain and every method maps to course material.

| Problem | Simple baseline | Machine learning (DSML) | Deep learning (DL) |
|---|---|---|---|
| Driver mistake detection | Z-score | **Isolation Forest** | **LSTM autoencoder** |
| Tyre / pace prediction | Ridge regression | — | **Vanilla RNN vs LSTM** |

In total: 3 DSML methods (Z-score, Isolation Forest, Ridge) and 3 DL models (RNN, LSTM, LSTM autoencoder).

| Area | Decision | Reason |
|---|---|---|
| YOLOv11-Seg apex detection | 🧊 Frozen. Keep as is; no retraining or generalisation to other tracks. | Already complete from last semester; DL work goes into the LSTM side instead. |
| Racing-line search (A\*/Dijkstra/BFS) | 🧊 Frozen. No new features. | Classical search, not ML/DL. Only bug fixes that other work depends on are allowed. |
| Big Data stack (Spark, HDFS, MongoDB, Big Data tab) | 🧊 Frozen. Keep as is. | Not needed for DSML/DL, but removing it is riskier than leaving it. |
| Bug fixes | ⏳ Committed. Fix one at a time. | The project must run end-to-end and match its documentation. |
| Isolation Forest rework + Z-score baseline | ⏳ Committed (DSML). | Current features and tuning are flawed (see §2B). Z-score shows whether ML is needed at all. |
| Ridge regression baseline for tyre/pace prediction | ⏳ Committed (DSML). | One classical baseline for the RNN/LSTM comparison. |
| LSTM autoencoder for mistake detection | ⏳ Committed (DL). | Deep-learning counterpart to the Isolation Forest on the same data and test. |
| Vanilla RNN vs LSTM for tyre/pace prediction | ⏳ Committed (DL). | Vanishing gradients and the RNN are taught in detail; LSTM is the taught remedy. |
| Dashboard UI redesign | ⏳ Committed (last). | Frozen components move to separate, lazily loaded tabs; new slider, sidebar and chart types. See §2E. |
| DL training-practice experiment (e.g. optimiser comparison) | 🤔 Optional — at most one, decided later. | Covers DL Units 1–2 without adding a new model. |
| Longer input window (5 vs 10 laps) for RNN vs LSTM | 🤔 Optional. | With 5-lap inputs the RNN and LSTM are expected to score closely (vanishing gradients bite on long sequences); a longer window would show the difference. |
| Driving-style clustering + PCA | ✖ Dropped (team decision). | Keep the project focused. |
| Classification task (tyre compound / lap type) | ✖ Dropped (team decision). | Keep the project focused. |
| One-Class SVM, KNN, DBSCAN, LOF anomaly detectors | ✖ Dropped (team decision). | Z-score + Isolation Forest + LSTM autoencoder tell the story with fewer methods. |
| Random Forest / Gradient Boosting regression baselines | ✖ Dropped (team decision). | One baseline (Ridge) is enough. |
| GRU / 1D-CNN / Transformer comparisons | ✖ Not planned. | RNN chosen instead of GRU (taught in detail). |

---

## 2. Work items

### A. Bug fixes (in order)

| # | Bug | Status |
|---|---|---|
| 1 | `Brake` is binary (0/1) but treated as 0–100 % (Isolation Forest features, racing-line weight, AI-tab card, chart labels) | ✅ |
| 2 | Scripts in `etl/`, `cv/`, `bigdata/` can't be run by file path (`ModuleNotFoundError: utils`) | ✅ |
| 3 | LSTM: saved model has 8 features, code has 9; scaler fitted before the split (leakage); sessions not sorted chronologically; overlapping-window split leakage; last-epoch (not best-epoch) weights saved | 🔄 Code fixed. **Retrain deliberately deferred** until other LSTM work is finished. |
| 4 | AI tab "Mistake Rate" card shows contamination instead of the measured rate | ✅ |
| 5 | Cache dir inconsistency (`data_lake/cache` vs `cache/`); `scipy` missing from `requirements.txt` (also found: racing line had no cache set at all; `pillow` missing) | ✅ |
| 6 | Dashboard sidebar footer credits only one team member | ⏸ Deferred by user — keep the footer as is; add the other members only when asked. |
| 7 | Generate model outputs for more sessions/drivers so the dashboard isn't mostly empty | ⏸ Deferred — do after all model-related fixes and changes are finished. |
| 8 | Stale integration tests: `test_pipeline_schema` (old grid / meta key) ⏳; `test_tyre_pipeline` ✅ (rewritten to run the real pipeline during bug 3). `test_pipeline_correctness` is fixed by the Isolation Forest work in §B. | 🔄 |
| 9 | Isolation Forest log line crashes the Windows console encoding (`→`) and prints 0.005 as "0.01" | 🔄 Encoding crash fixed project-wide (utils logger); 0.005 → "0.01" format remains for §B |
| 10 | `frontend/app.py` uses `st.components.v1.html`, which Streamlit has deprecated (replace with `st.iframe`) | ✅ |
| 10b | Follow-up to #10: `filter_mode=None` blocks typing but Backspace/Delete still cleared a dropdown's shown value (reported by user). Fixed with a key-lock script via the supported `st.html(..., unsafe_allow_javascript=True)`. | ✅ (needs a manual check in the browser) |
| 11 | Test runner used a bare `python`, which on Windows ran the tests under the system interpreter instead of the venv (found while fixing #2) | ✅ |
| 12 | CV turn windows were hand-typed for one Bahrain video (duplicated in the frontend) and `cv/extract_turns.py` (curvature detection) found only 11 of Bahrain's 15 corners. Now built from FastF1's official corner list + pole-lap telemetry; only a per-video `--video-offset` is manual. Validated on Bahrain 2024: 15/15 corners, apex times within 0.37 s on average of the hand annotation, direction 14/15. | ✅ |
| 13 | LSTM training data includes out-laps, in-laps (pit stops) and safety-car laps with no filtering, so slow non-representative laps pollute both inputs and targets. Fix before retraining. | ✅ |
| 14 | `predict_single_stint` silently overwrites each stint's first lap features with the second lap's (an undocumented workaround for out-laps); should be replaced by proper lap filtering (#13) | ✅ |
| 15 | **Test-suite audit:** review every unit and integration test file and confirm each test exercises real project code and checks real behaviour (not re-implemented logic, trivial assertions, or outdated expectations). Bug #3 found an integration test that re-implemented the pipeline instead of testing it, so others may do the same. Findings: [TEST_AUDIT.md](TEST_AUDIT.md). | 🔄 Step 1 done (all non-Isolation-Forest tests fixed). A3/A4/A20 are rewritten with §B. No new tests for frozen YOLO/Big Data code (user decision). |
| 16 | `cv/select_training_frames.py` never samples the end of a video (takes every `len//11`-th frame, then the first 11), contrary to the documented even spacing. Found by the test audit. | ✅ |
| 17 | `racing_line_grid.find_start_end_nodes` uses a different grid origin (min X/Y of fast samples) from `build_grid` (min of all samples), shifting start cells. Found by the test audit. | ✅ |
| — | **Pre-commit full review (Oct 2026)** — found by a fresh read of every script plus static analysis. Severity: H = wrong results/crash on real data, M = edge case/misleading, L = cleanup. All verified unless marked "(code)". | |
| 18 | **H** Racing line: metres-per-unit scale estimates laps as session duration ÷ 90 s, so qualifying (with garage time) is ~3× too large (2023 R1 Q: 0.303 vs true 0.101 m/unit). `--circuit-length` defaults to Bahrain for every circuit. | ⏳ |
| 19 | **H** Racing line silently falls back to the whole session when the fastest-lap lookup fails and the output doesn't say so. The current `2023_1_Q_1` output is such a fallback (3,071 path points instead of ~65), generated during bug #1 before the cache fix, so its documented numbers are invalid. Also dead code (unused start/end search, unreachable branch). | ⏳ |
| 20 | **M** LSTM prediction drops stints that can't produce a *training* window (short stints, or stints split by safety-car laps). (code) | ⏳ |
| 21 | **M** LSTM prediction windows bridge removed laps; the first 5 predictions of a stint use copies of lap 1 (never seen in training); lap 1's prediction sees its own target. (code) | ⏳ |
| 22 | **M** LSTM features: missing `TyreLife` falls back to the race lap number; intermediate/wet laps get an all-zero compound encoding and are mixed with slick laps. `cliff_lap`/`laps_remaining` count kept-lap positions, not race laps; unused `cliff_lap` in `extract_stints`. (code) | ⏳ |
| 23 | **M** `utils` reads `config.json` without UTF-8, so "São Paulo" becomes "SÃ£o Paulo" (the satellite image was already saved under a mangled name). | ⏳ |
| 24 | **M** `run_tests.py --file <integration test>` runs it without `APEXHUNTER_RUN_SLOW`, so every test is skipped and it reports PASS. (code) | ⏳ |
| 25 | **H** `etl/download_season_data.py` cannot download anything: `config.json` stores seasons as strings and FastF1 rejects them (`TypeError`); every session fails and is only logged. The 24-round fallback is also wrong for 2023 (22 rounds). | ⏳ |
| 26 | **H** CV: YOLO masks are stretched from the padded 384×640 network output to the frame without removing the padding, so curb/road masks are shifted/squashed vertically by up to ~23 px in every frame, which affects apex distances. Fix: `retina_masks=True` (verified); needs an inference re-run. | ⏳ |
| 27 | **M** HDFS: `hdfs_manager` counts directories and `_SUCCESS` as files (stored status shows `Year=2023`, `Year=2024`, `_SUCCESS`); `hdfs_manager` and Spark write different layouts to the same `/apexhunter/clean_data`; Spark forward-fill window partitions by driver only, mixing sessions once #8's dtype bug is fixed. (partly code) | ⏳ |
| 28 | **M** `etl/download_manual_videos.py` saves to `data_lake/raw_video/`, which nothing reads (real folder: `downloaded_videos/`). | ⏳ |
| 29 | **M** MongoDB: clients are never closed; the dashboard's uncached availability check adds ~3 s to every rerun when MongoDB is offline; mistake documents are keyed by the session path exactly as typed (duplicates for absolute vs relative paths). | ⏳ |
| 30 | **L** CV robustness: `run_inference` leaves partial outputs on a crash that later count as "already processed" (skip check also ignores `_turns.json`); fps truncated to int; `extract_frames` interval truncation (division by zero below 5 fps); skip-if-exists treats partial files as done in ETL; fallback thresholds (300/550) differ from config (130/250); unused `STRAIGHT_THRESHOLD`; misleading docstrings. (code) | ⏳ |
| 31 | **H** Dashboard: the master scrubber and the Telemetry-tab slider are out of sync (the Telemetry chart is cut at one time while the "Session Time" card and scrub line show another). | ⏳ |
| 32 | **H** Dashboard: "Proj. End Pace" delta is always red (both branches of the colour rule paint it red); faster pace should be green. (code) | ⏳ |
| 33 | **H** Dashboard: "Brake Overlap %" is dominated by garage/pit-lane samples (2024 R1 Q #11: 6.05 %, 1,447 of 1,541 overlap rows below 50 km/h). | ⏳ |
| 34 | **M** Dashboard (Big Data tab): "Speed Trace Overlay" plots randomly shuffled samples as a trace; LSTM band labelled "95% CI" is actually ±1σ. (code) | ⏳ |
| 35 | **M** Dashboard (CV panel): turn report card uses different thresholds from the per-frame classification; the status badge maps session time to video time meaninglessly (always lands on the last frame). | ⏳ |
| 36 | **M** Dashboard: session type is re-parsed from the label text (falls back to Race when the schedule is unavailable; "Sprint Shootout" → "Sprint"); FastF1 lookups cache failures with no expiry (lap selectors / event names / fastest lap stay broken until restart). | ⏳ |
| 37 | **L** Dashboard edge cases: some loaders never pick up regenerated files; `st.stop()` on empty telemetry stops all tabs; `downsample` doesn't downsample 5,001–9,999 rows; missing empty-input guards; scrubber value carries over between sessions; "waiting in the garage" note is wrong for races; CV cards count frames not corners; "Current Pace" is the stint's last (often in-) lap. (partly code) | ⏳ |
| 38 | **L** Cleanup: unused imports/variables/f-strings (ruff), unused constants (`COLOR_*`, `ALGO_LABELS`, `STRAIGHT_THRESHOLD`), dead `_fetch_anomaly_distribution` + `mongo_manager.get_anomaly_score_distribution`; caches deleted. **Kept on purpose:** the switched-off optimal-line/ghost-car traces in `track_map.py` (may be used by the UI redesign), unused function arguments (signature changes ripple into callers), `MIN_POINTS_PER_CELL` (a tunable that is currently a no-op). | ✅ |
| — | Isolation Forest items found in this review are folded into §B: `speed_delta` per *sample* is inconsistent because sample spacing varies 0.014–0.76 s; `--reference` file is not column-validated; features computed twice. | 🔗 §B |

**Bug fixes are project-wide:** "frozen" means no new features. Bugs are still fixed everywhere,
including in the YOLO/CV, racing-line and Big Data code.

### B. DSML — Isolation Forest rework + Z-score baseline (committed)
Design: [ISOLATION_FOREST_REWORK.md](ISOLATION_FOREST_REWORK.md).
- Features compared with the 5 fastest other drivers at the same track position (never the target driver).
- Only representative laps (no pit, safety-car or start laps).
- Settings chosen with planted fake mistakes; evaluated with confusion matrix, precision, recall, F1.
- Z-score baseline on the same features and test.
- Flagged samples grouped into mistake events with a plain-language reason.

### C. DSML — Ridge regression baseline (committed)
- Same task and data split as the LSTM: predict next-lap pace from the previous 5 laps of a stint.
- Metrics (taught): MAE, MSE, RMSE, R², MAPE; actual-vs-predicted plot; train vs test error.

### D. DL — LSTM autoencoder and RNN vs LSTM (committed)
- **LSTM autoencoder (mistake detection):** short (~2 s) windows of the same features as §B; the
  network compresses and rebuilds each window; trained on normal driving; rebuild error = mistake
  score. Evaluated with the same planted-mistake test → one table: Z-score vs Isolation Forest vs
  LSTM autoencoder.
- **Vanilla RNN vs LSTM (tyre/pace):** same data, split, features and metrics as §C; together with
  Ridge gives one table: Ridge vs RNN vs LSTM. The real-data LSTM retrain (bug #3) happens here.

### E. Dashboard UI redesign (committed; details to be decided)
- **Tab layout:** move the frozen components (YOLO/CV video feed, racing line, Big Data) into their
  own tab(s), away from the main view. They must **not load automatically** when the dashboard
  opens; the default view focuses on the DSML/DL work.
- **Sidebar session picker:** the single "Session" dropdown currently lists every qualifying and
  race session of the year (e.g. "Round 1: Bahrain Grand Prix - Qualifying", "… - Race", …). Split
  it into two controls: a **Grand Prix** selector (one entry per round/venue) and a **session type**
  selector (Qualifying / Race, e.g. a radio button or small dropdown).
- **Time slider:** replace or redesign the master session-time scrubber and the telemetry slider.
- **Telemetry charts:** choose a chart type suited to each channel instead of five identical line
  charts. For example, show the binary brake flag as shaded braking zones or an on/off strip rather
  than a line graph. Exact designs are still to be decided.
- New results (mistake detection: Z-score vs Isolation Forest vs LSTM autoencoder; tyre/pace:
  Ridge vs RNN vs LSTM) need their own views; plan them as part of this redesign.
- **Isolation Forest view:** the rework (§B) only adds the minimum needed to verify it (mistake
  event list with reasons, event markers). Its proper design is done here.

### F. Dropped (team decision, Oct 2026)
Driving-style clustering + PCA; classification task; One-Class SVM / KNN / DBSCAN / LOF; Random
Forest / Gradient Boosting baselines; GRU / 1D-CNN / Transformer comparisons.

---

## 3. Rules while working
- Every bug discovered along the way is added to the table in §2A immediately, even if it isn't
  fixed straight away. The project is known to have many undiscovered bugs.
- Each fix gets a regression test where practical, and is checked against the original code to
  prove the test catches the bug.
- **Tests must test real functionality.** A test must call the project's actual code and assert
  on its real behaviour. No copying logic into the test, no assertions that can't fail, no
  expectations left over from older versions. New tests follow this rule, and #15 audits the
  existing ones.
- After every code change, update the affected docs, including §8 of `PROJECT_DOCUMENTATION.md` and this file's status column.
- Last semester's `.docx` / `.pdf` files in `docs/` are kept unchanged as historical record.
