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

## 0. Current status and next steps (updated 2026-10-30)

| Step | Status | Where |
|---|---|---|
| 1. Bug fixes | ✅ Done: #1–#40 fixed, except #8/#9 (finished by step 3) and #22 (moved to step 5). | §2A |
| 2. Test audit, rounds 2–3 | ✅ Done: 252 → 145 tests, all passing; no mock tests (racing-line and LSTM integration tests now run on real data); old Isolation Forest tests deleted for replacement in step 3. | [TEST_AUDIT.md](TEST_AUDIT.md) |
| 3. Isolation Forest rework + Z-score baseline | ⏳ **Next.** Five phases with a check-in after each. Read "Before you start" in the design doc first. | §2B, [ISOLATION_FOREST_REWORK.md](ISOLATION_FOREST_REWORK.md) |
| 4. Ridge regression baseline | ⏳ | §2C |
| 5. LSTM rework: #22 fixes, vanilla RNN vs LSTM, LSTM autoencoder, real-data retrain | ⏳ | §2D |
| 6. Generate model outputs for more sessions/drivers | ⏸ After all model work (#7). | §2A |
| 7. Dashboard UI redesign | ⏳ Last. | §2E |
| CI/CD infrastructure | 🔄 Docker build, Ruff and dependency checks passed on GitHub. First run exposed stale AppTest HTML lookup (#41); repaired, CI rerun pending. Hosting remains unconfigured. | [CI_CD.md](CI_CD.md) |

**What exists today:** model outputs for one session/driver each (mistakes and tyre: 2023 R1 race,
#1; racing line: 2023 R1 qualifying, #1; CV: 2024 Bahrain pole lap). The saved LSTM is an old
8-feature model that `predict_cliff.py` rejects until the step-5 retrain; this is expected.

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
| Bug fixes | ✅ Done (Oct 2026), except #8/#9, which are finished by the Isolation Forest rework (§B), and #22, moved to the LSTM rework (§D). | The project must run end-to-end and match its documentation. |
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
| 8 | Stale integration tests: `test_pipeline_schema` (old grid / meta key) and `test_pipeline_correctness` (catches 1/11 injected anomalies) failed; `test_tyre_pipeline` ✅ (rewritten to run the real pipeline during bug 3). Both failing files were deleted in test audit round 2; the Isolation Forest rework (§B) writes their replacements. | ✅ (replacements in §B) |
| 9 | Isolation Forest log line crashes the Windows console encoding (`→`) and prints 0.005 as "0.01" | 🔗 Encoding crash fixed project-wide (utils logger); the "0.01" log format is fixed in §B (`mistakes_model.py` is rewritten there). |
| 10 | `frontend/app.py` uses `st.components.v1.html`, which Streamlit has deprecated (replace with `st.iframe`) | ✅ |
| 10b | Follow-up to #10: `filter_mode=None` blocks typing but Backspace/Delete still cleared a dropdown's shown value (reported by user). Fixed with a key-lock script via the supported `st.html(..., unsafe_allow_javascript=True)`. | ✅ (needs a manual check in the browser) |
| 11 | Test runner used a bare `python`, which on Windows ran the tests under the system interpreter instead of the venv (found while fixing #2) | ✅ |
| 12 | CV turn windows were hand-typed for one Bahrain video (duplicated in the frontend) and `cv/extract_turns.py` (curvature detection) found only 11 of Bahrain's 15 corners. Now built from FastF1's official corner list + pole-lap telemetry; only a per-video `--video-offset` is manual. Validated on Bahrain 2024: 15/15 corners, apex times within 0.37 s on average of the hand annotation, direction 14/15. | ✅ |
| 13 | LSTM training data includes out-laps, in-laps (pit stops) and safety-car laps with no filtering, so slow non-representative laps pollute both inputs and targets. Fix before retraining. | ✅ |
| 14 | `predict_single_stint` silently overwrites each stint's first lap features with the second lap's (an undocumented workaround for out-laps); should be replaced by proper lap filtering (#13) | ✅ |
| 15 | **Test-suite audit:** review every unit and integration test file and confirm each test exercises real project code and checks real behaviour (not re-implemented logic, trivial assertions, or outdated expectations). Bug #3 found an integration test that re-implemented the pipeline instead of testing it, so others may do the same. Findings: [TEST_AUDIT.md](TEST_AUDIT.md). | ✅ Round 1: all non-Isolation-Forest tests fixed. Rounds 2–3: only essential tests kept and no mock tests (252 → 145, all passing); old Isolation Forest tests (A3/A4/A20) deleted, replacements written in §B. No new tests for frozen YOLO/Big Data code (user decision). |
| 16 | `cv/select_training_frames.py` never samples the end of a video (takes every `len//11`-th frame, then the first 11), contrary to the documented even spacing. Found by the test audit. | ✅ |
| 17 | `racing_line_grid.find_start_end_nodes` uses a different grid origin (min X/Y of fast samples) from `build_grid` (min of all samples), shifting start cells. Found by the test audit. | ✅ |
| — | **Pre-commit full review (Oct 2026)** — found by a fresh read of every script plus static analysis. Severity: H = wrong results/crash on real data, M = edge case/misleading, L = cleanup. All verified unless marked "(code)". | |
| 18 | **H** Racing line: metres-per-unit scale estimates laps as session duration ÷ 90 s, so qualifying (with garage time) is ~3× too large (2023 R1 Q: 0.303 vs true 0.101 m/unit). `--circuit-length` defaults to Bahrain for every circuit. **Fix:** scale is FastF1's documented position unit (1/10 m, checked on 10 sessions at 4 circuits: 0.0996–0.1006); scale estimator and `--circuit-length` removed; output records the measured lap length. | ✅ |
| 19 | **H** Racing line silently falls back to the whole session when the fastest-lap lookup fails and the output doesn't say so. The current `2023_1_Q_1` output is such a fallback (3,071 path points instead of ~65), generated during bug #1 before the cache fix, so its documented numbers are invalid. Also dead code (unused start/end search, unreachable branch). **Fix:** the lap lookup returns nothing on failure (incl. FastF1's `pick_fastest()` returning `None`) and the run stops with an error; the output records the FastF1 lap used (`fastest_lap`: number, time, length); `find_start_end_nodes` and the unreachable branch removed (so #17's code is gone). Regenerated: lap 14, 89.708 s, 66 path points. | ✅ |
| 20 | **M** LSTM prediction drops stints that can't produce a *training* window (short stints, or stints split by safety-car laps). (code) **Measured:** 51 of 391 race stints (13 %, 2023–24 R1–R4). **Fix (with #21, same function):** `extract_stints` keeps stints without windows (empty sequences, so training is unchanged); prediction reports them with actual laps and `null` predictions. | ✅ |
| 21 | **M** LSTM prediction windows bridge removed laps; the first 5 predictions of a stint use copies of lap 1 (never seen in training); lap 1's prediction sees its own target. (code) **Fix:** shared `tyre_data.window_positions` rule: a lap is predicted only from the 5 consecutive kept laps before it (identical to training windows); other laps `null`; cliff search runs over predicted laps only; Big Data confidence band skips `null`s. Fixed together with #20 because letting short stints through with the old padded windows would have made their predictions pure padding. Not runnable on real data until the LSTM is retrained (old model rejected); verified by unit tests and the synthetic train → predict integration test. | ✅ |
| 22 | **M** LSTM features: missing `TyreLife` falls back to the race lap number; intermediate/wet laps get an all-zero compound encoding and are mixed with slick laps. `cliff_lap`/`laps_remaining` count kept-lap positions, not race laps; unused `cliff_lap` in `extract_stints`. (code) | 🔗 Moved to §D (user decision, Oct 2026): the LSTM rework rewrites these features and needs a retrain anyway, so it is fixed there. |
| 23 | **M** `utils` reads `config.json` without UTF-8, so "São Paulo" becomes "SÃ£o Paulo". (Correction: the existing satellite image has the correct name; it was saved by an earlier run, but a re-run would have saved a second, mangled copy.) **Fix:** `encoding="utf-8"` on the config read and on all 17 other text-mode `open()` calls (latent: they only handled ASCII so far); guard test fails on any new text `open()` without an encoding. All 24 satellite names now match the files on disk. | ✅ |
| 24 | **M** `run_tests.py --file <integration test>` runs it without `APEXHUNTER_RUN_SLOW`, so every test is skipped and it reports PASS. (code) **Fix:** integration files named with `--file` run with the slow flag; the runner parses unittest's run/skipped counts, prints them, and labels fully skipped files SKIP ("checked nothing") instead of PASS. Reproduced: `--file test_racing_line_pipeline.py` said PASS with 13/13 skipped; now runs 13. Tests: `test_test_runner.py`. | ✅ |
| 25 | **H** `etl/download_season_data.py` cannot download anything: `config.json` stores seasons as strings and FastF1 rejects them (`TypeError`); every session fails and is only logged. The 24-round fallback is also wrong for 2023 (22 rounds). **Fix:** years converted to int; no round-count guess (season skipped and reported); per-driver failures counted; end-of-run summary and exit code 1 on failed/partial sessions. Verified by a real re-download of 2024 R1 Q (identical to the stored file). No new unit test (frozen Big Data code, user decision). | ✅ |
| 26 | **H** CV: YOLO masks are stretched from the padded 384×640 network output to the frame without removing the padding, so curb/road masks are shifted/squashed vertically by up to ~23 px in every frame, which affects apex distances. Fix: `retina_masks=True` (verified); needs an inference re-run. **Done:** on 6 frames the old masks were off by up to 29 px (curb IoU down to 0.41) and the retina masks match YOLO's polygons exactly (IoU 1.00). Bahrain 2024 re-processed (offset 4.9 s, ~19 min CPU): 66/4,917 frames changed status, median distance change 1 cm, turn report unchanged (12 Hitting, 3 Near). No new unit test (frozen CV code, user decision). | ✅ |
| 27 | **M** HDFS: `hdfs_manager` counts directories and `_SUCCESS` as files (stored status shows `Year=2023`, `Year=2024`, `_SUCCESS`); `hdfs_manager` and Spark write different layouts to the same `/apexhunter/clean_data`; Spark forward-fill window partitions by driver only, mixing sessions once #8's dtype bug is fixed. (partly code) **Fix:** recursive file listing (no folders/`_SUCCESS`/`.crc`); Spark writes to its own `clean_data_spark` (HDFS and local; local `--force` could have deleted the pandas `clean_data`); fill window per Year/Round/Session/Driver (+ #40) and pandas fill per driver (latent: raw data has no NaNs). Also fixed: local Spark mode never worked (Hadoop default FS → `file:///` URIs), failed runs exited 0, `files_processed` always 0. Verified by a local Spark run (92 files, 57,874,119 rows, pandas folder unchanged). HDFS not re-run (not running here; user: don't get stuck on Hadoop) — stored `hdfs_status.json` is from May. No new tests (frozen Big Data). | ✅ |
| 28 | **M** `etl/download_manual_videos.py` saves to `data_lake/raw_video/`, which nothing reads (real folder: `downloaded_videos/`). **Fix:** saves to `downloaded_videos/{year}/`, with the year set next to the batch in `config.json` (`manual_downloads: {year, videos}`; the user downloaded videos in batches of six and the config holds the last one). Also found and fixed: it deleted the existing video *before* downloading, so a failed download lost the file; now downloads to `{name}.download.mp4` and replaces only on success. Checked with a faked yt-dlp (success replaces, failure keeps the old file); no real download, no new tests (network script). | ✅ |
| 29 | **M** MongoDB: clients are never closed; the dashboard's uncached availability check adds ~3 s to every rerun when MongoDB is offline; mistake documents are keyed by the session path exactly as typed (duplicates for absolute vs relative paths). **Fix:** dashboard caches the availability check for 60 s (`st.cache_resource`): offline rerun 3.08 s → 0.01 s; one shared `MongoClient` per process (created after a successful ping); `session_file` keyed by file name and older full-path copies deleted on upload (regex checked on path variants). MongoDB not running here, so uploads were not re-run; no new tests (frozen Big Data). | ✅ |
| 30 | **L** CV robustness: `run_inference` leaves partial outputs on a crash that later count as "already processed" (skip check also ignores `_turns.json`); fps truncated to int; `extract_frames` interval truncation (division by zero below 5 fps); skip-if-exists treats partial files as done in ETL; fallback thresholds (300/550) differ from config (130/250); unused `STRAIGHT_THRESHOLD`; misleading docstrings. (code) **Fix:** run_inference removes partial outputs on any crash/interrupt and skips only if video, CSV and `_turns.json` all exist (checked: simulated crash on frame 3 left no files; a 5-frame run wrote all three and the re-run skipped); fps kept as float (videos are 50.00–50.01 fps, effect ≤ 0.01 s); frame interval `max(1, round(fps/5))` (unchanged for every existing video); ETL parquet written to `.tmp` then renamed; fallback thresholds = config; "hardcoded" turn-direction docstrings corrected; `STRAIGHT_THRESHOLD` was already removed by #38. No outputs changed; no new tests (frozen CV/ETL). | ✅ |
| 31 | **H** Dashboard: the master scrubber and the Telemetry-tab slider are out of sync (the Telemetry chart is cut at one time while the "Session Time" card and scrub line show another). **Fix:** one shared position in `components/scrubber.py`; each slider's `on_change` writes it and the other slider (Streamlit's linked-widget pattern), and the script body only writes a widget's value when missing or out of range. Also fixed: the "Session Time" card truncated milliseconds (1622.599 s → 27:02.598). Reproduced and verified on the real dashboard with AppTest; regression test `test_frontend_scrubber.py` (3 of its tests fail on the old behaviour). | ✅ |
| 32 | **H** Dashboard: "Proj. End Pace" delta is always red (both branches of the colour rule paint it red); faster pace should be green. (code) **Fix:** always `delta_color="inverse"` (lower lap time = green). Confirmed via the Metric proto colour (old: RED for both); AppTest regression in `test_frontend_ai_analysis.py`. | ✅ |
| 33 | **H** Dashboard: "Brake Overlap %" is dominated by garage/pit-lane samples (2024 R1 Q #11: 6.05 %, 1,447 of 1,541 overlap rows below 50 km/h). **Fix:** only samples ≥ 50 km/h count (garage, pit box and grid starts excluded): 2024 R1 Q #11 6.05 → 0.67 %, 2023 R1 Q #1 21.27 → 0.41 %. Flying-lap filtering was considered and not needed (0 to 80 km/h thresholds give almost the same value; pit lane has no overlaps). Regression tests in `test_frontend_telemetry.py`. | ✅ |
| 34 | **M** Dashboard (Big Data tab): "Speed Trace Overlay" plots randomly shuffled samples as a trace; LSTM band labelled "95% CI" is actually ±1σ. (code) **Fix:** samples sorted by `SessionTime` per driver, x = session time (min), line broken at gaps > 10 s, retitled "Speed Over Session Time" (2023 R1 Q: median jump between plotted points 93 → 2.6 km/h); band relabelled "±1σ (MC dropout)". No new tests (frozen Big Data). | ✅ |
| 35 | **M** Dashboard (CV panel): turn report card uses different thresholds from the per-frame classification; the status badge maps session time to video time meaninglessly (always lands on the last frame). **Fix:** `_turns.json` now records the video's lap (FastF1: year/round/session/driver/number, lap start/end in session s); `cv_feed.video_time_for` maps the scrubber into the video only for that session+driver and inside the lap, else badge "Not in this lap". Report card takes the status of each corner's closest-approach frame (same thresholds as per-frame): T9/T15 Near → Missing. Bahrain `_turns.json` regenerated with FastF1 only (windows identical). Regression test `test_frontend_cv_feed.py`. | ✅ |
| 36 | **M** Dashboard: session type is re-parsed from the label text (falls back to Race when the schedule is unavailable; "Sprint Shootout" → "Sprint"); FastF1 lookups cache failures with no expiry (lap selectors / event names / fastest lap stay broken until restart). **Fix:** round and session code come from the session file name (`header_bar.session_from_file`); the three FastF1 lookups expire after 10 min so failures are retried. Checked with the full app in AppTest. | ✅ |
| 37 | **L** Dashboard edge cases: some loaders never pick up regenerated files; `st.stop()` on empty telemetry stops all tabs; `downsample` doesn't downsample 5,001–9,999 rows; missing empty-input guards; scrubber value carries over between sessions; "waiting in the garage" note is wrong for races; CV cards count frames not corners; "Current Pace" is the stint's last (often in-) lap. (partly code) **Fixed now:** CV metrics loader keyed on file mtime (the other loaders are uncached); `st.stop()` on empty telemetry → return; `downsample` uses a ceiling step (5,001–9,999 rows now reduced). **Moved to §E (UI redesign, user decision: the dashboard is being rebuilt):** scrubber carry-over between sessions (already clamped since #31), the "waiting in the garage" note shown for races, CV cards counting frames, "Current Pace" = last lap of the stint. | ✅ (rest in §E) |
| 38 | **L** Cleanup: unused imports/variables/f-strings (ruff), unused constants (`COLOR_*`, `ALGO_LABELS`, `STRAIGHT_THRESHOLD`), dead `_fetch_anomaly_distribution` + `mongo_manager.get_anomaly_score_distribution`; caches deleted. **Kept on purpose:** the switched-off optimal-line/ghost-car traces in `track_map.py` (may be used by the UI redesign), unused function arguments (signature changes ripple into callers), `MIN_POINTS_PER_CELL` (a tunable that is currently a no-op). | ✅ |
| 39 | **H** Racing line "time saved" is not meaningful: path time = length ÷ mean speed of the cells it passes, with cell speeds averaged over the whole session (out-laps included), so A\* gains just by passing through fast cells. 2023 R1 Q #1: driver's lap estimated at 110.5 s (real 89.7 s), A\* "saves" ~9 s although its path is longer; clamped to 5.0 for all three algorithms. Found while fixing #18/#19 (the old clamp hid it). **Fix:** time = Σ segment length ÷ speed of the nearest fastest-lap sample (scipy cKDTree), applied identically to the driver's full-resolution lap and to each search path; clamp removed. Validated: driver's lap 88.76 s vs real 89.708 s (1 %). New result: A\*/Dijkstra −0.079 s, BFS −0.104 s (no faster line, within error). Rejected alternative: per-cell fastest-lap speeds (half the cells have no sample at 5 m resolution). | ✅ |
| 40 | **M** Spark's numeric-column check compared `str(dtype)` with names like `"DoubleType"`, but PySpark 3.5 prints `"DoubleType()"`, so the forward-fill never ran (previously listed only in PROJECT_DOCUMENTATION §8). **Fix:** `isinstance` check; fixed together with #27's fill window. No data impact (raw data has no gaps). | ✅ |
| 41 | CI's first full unit run found that the dashboard keyboard-lock test assumed HTML elements were Streamlit `UnknownElement` instances. Use public `at.get("html")` and retain the script/JavaScript assertions. | 🔄 Fixed in code; CI rerun pending. |
| — | Isolation Forest items found in this review are folded into §B: `speed_delta` per *sample* is inconsistent because sample spacing varies 0.014–0.76 s; `--reference` file is not column-validated; features computed twice. | 🔗 §B |

**Bug fixes are project-wide:** "frozen" means no new features. Bugs are still fixed everywhere,
including in the YOLO/CV, racing-line and Big Data code.

### B. DSML — Isolation Forest rework + Z-score baseline (committed)
Design: [ISOLATION_FOREST_REWORK.md](ISOLATION_FOREST_REWORK.md).
- Features compared with the 5 fastest other drivers at the same track position (never the target driver).
- Only representative laps (no pit, safety-car or start laps). Evidence from #33: stationary garage
  samples hold brake + throttle together (up to 94 % of a qualifying session's overlap samples), so
  the current `brake_throttle_overlap` feature is dominated by them.
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
- **Required LSTM fixes (from bug #22), done before the retrain:** missing `TyreLife` must not fall
  back to the race lap number; intermediate/wet laps need their own handling (currently an all-zero
  compound encoding mixed with slick laps); `cliff_lap`/`laps_remaining` must count race laps, not
  kept-lap positions; remove the unused `cliff_lap` in `extract_stints`. Earlier LSTM fixes (#3,
  #13, #14, #20, #21) stay and are built on.

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
  than a line graph. The user finds the current tab too bland and too simple: it doesn't show the
  details needed to read a lap, so the redesign should add more of them, not just restyle the
  charts. Exact designs are still to be decided.
- New results (mistake detection: Z-score vs Isolation Forest vs LSTM autoencoder; tyre/pace:
  Ridge vs RNN vs LSTM) need their own views; plan them as part of this redesign.
- **Carried over from bug #37 (fix while redesigning):** the session-time position carries over
  between sessions (it is clamped to the new range, but not reset); the "waiting in the garage"
  note also appears for races; the CV stat cards count frames, not corners; the AI tab's "Current
  Pace" is the stint's last lap, which is often the in-lap.
- **Isolation Forest view:** the rework (§B) only adds the minimum needed to verify it (mistake
  event list with reasons, event markers). Its proper design is done here.

### F. Dropped (team decision, Oct 2026)
Driving-style clustering + PCA; classification task; One-Class SVM / KNN / DBSCAN / LOF; Random
Forest / Gradient Boosting baselines; GRU / 1D-CNN / Transformer comparisons.

---

## 3. Rules while working

These apply to everyone (and to AI assistants) working on the project.

**Changes and documentation**
- Work on one item at a time and explain each change: what was wrong, what changed, how it was
  verified, and what was *not* verified.
- After every code change, update the affected docs in the same piece of work: this file's status
  column, `PROJECT_DOCUMENTATION.md` (the method sections and §8 "Fixed" / "Known issues"),
  `README.md`, `architecture.md` and any other doc that describes the changed behaviour.
- Every bug discovered along the way is added to the table in §2A immediately (next number: #42),
  even if it isn't fixed straight away. The project is known to have undiscovered bugs.
- Last semester's `.docx` / `.pdf` files in `docs/` are kept unchanged as historical record.

**Verification**
- Prove fixes on real data where possible (run the pipeline on a real session and compare before
  and after), and record the numbers in the docs. This counts for more than a unit test.
- Where a regression test is added, check that it fails on the original code.
- Frozen code (YOLO/CV, racing line, Big Data) and the dashboard get only quick fixes: the
  dashboard is being redesigned (§E), so presentation issues go there. Infrastructure that is not
  running locally (HDFS, MongoDB) is not set up just to verify a fix; say what is unverified.

**Tests: only essential, real ones**
- A test must call the project's actual code and assert on its real behaviour: no copied logic,
  no assertions that can't fail, no expectations left over from older versions.
- **No mock tests.** Do not replace project code, FastF1 or models with mocks. Use small real
  inputs, or real data from `data_lake/` and the FastF1 `cache/` in integration tests (skipped when
  the data is absent). Redirecting a path to a temp folder is fine; faking behaviour is not.
- Add at most one regression test per fix (plus a close edge case if it matters). Do not add tests
  for test tooling, tests that test other tests, tests for behaviour that didn't change, or
  near-duplicates. The test count should not grow for its own sake.
- No new tests for frozen code (existing CV tests are kept).
- Tests for code that is about to be rewritten (the old Isolation Forest) are replaced as part of
  the rewrite, not patched beforehand.

**Scope**
- The method list in §1 is final: no extra algorithms without a team decision.
- The real LSTM model is not retrained until the LSTM rework (§D).
