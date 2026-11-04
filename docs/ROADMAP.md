# ApexHunter 2.0 — Roadmap (AY 2026–27)

**Courses:** Data Science & Machine Learning (DSML, CMP707) · Deep Learning (DL)
**Student:** Parin Shah (23001091)

What we work on, what is done, what is left, and every bug found. The full history (long
descriptions, dated decisions, intermediate numbers) is in
[ARCHIVE/ROADMAP_full_2026-11-04.md](ARCHIVE/ROADMAP_full_2026-11-04.md). Section references in code
comments (§B mistakes, §C tyre/pit strategy, §D RNN/LSTM/autoencoder, §E dashboard) point there.

Status: ✅ done · 🔄 in progress · ⏳ to do · ⏸ deferred · 🧊 frozen

---

## 1. Status (2026-11-05)

| Step | Status |
|---|---|
| Bug fixes and test audits (4 rounds, every test checks real code; no mocks) | ✅ |
| Mistake detection: Z-score vs Isolation Forest vs LSTM autoencoder | ✅ |
| Tyre pace: Ridge vs vanilla RNN vs LSTM, pit window checked against real stops | ✅ |
| Old "tyre cliff" predictor removed | ✅ |
| Project-wide review (bugs #48–#54) | ✅ |
| Docs condensed; team guide (`TEAM_GUIDE.md`, gitignored) | ✅ |
| **Dashboard redesign (§3)** | ⏳ next |
| Report (DSML) + presentation (DSML + DL): drafts in `docs/submission/`, regenerated once after the redesign (planted mistakes kept, reframed as a controlled test on synthetic mistakes), then reviewed by the user | 🔄 |
| Outputs for every race and driver (#7) | ⏸ not confirmed |

## 2. Scope (fixed)

Each problem is answered with a simple baseline → a machine-learning model → a deep-learning model.

| Problem | Baseline | Machine learning (DSML) | Deep learning (DL) |
|---|---|---|---|
| Driver mistakes | Z-score | Isolation Forest | LSTM autoencoder |
| Tyre pace / pit window | "No change" (next lap = last lap) | Ridge regression | Vanilla RNN vs LSTM |

Course mapping: anomaly detection (DSML 4.3); Ridge and regression metrics (DSML 3.1,
`LinearReg.ipynb`); RNN, vanishing gradients and LSTM (DL NN deck p61–85); precision/recall/F1 and
MAE/RMSE/R²/MAPE, train vs test error (DSML 3.1). Full survey: `ARCHIVE/COURSE_ALIGNMENT.md`.

**Frozen (kept, not developed):** YOLOv11-Seg apex detection, racing-line search (A\*/Dijkstra/BFS),
Big Data stack (Spark, HDFS, MongoDB). **Dropped:** other anomaly detectors, Random Forest /
boosting, clustering + PCA, classification, GRU, choosing a tyre compound (race data cannot rank the
compounds).

## 3. Dashboard redesign (next; details to agree with the user)

- **Layout:** frozen parts (CV video, racing line, Big Data) in their own tabs that do not load at
  start; the main view is the DSML/DL work.
- **Sidebar:** Grand Prix + session type instead of one long session list.
- **Charts:** a fitting chart per telemetry channel (e.g. braking as shaded zones), more detail per
  lap; new views for the three mistake detectors and the tyre models.
- **Carried over:** session time is not reset between sessions; "waiting in the garage" note shown
  for races; CV cards count frames not corners; the Race Intelligence mistakes map draws one marker
  per flagged sample instead of per event; unused racing-line overlay code; unused
  `render_ai_analysis` arguments; hard-coded scores on the Big Data radar; pathfinding section
  always shows the first file; corrupt strategy file shown as "missing"; check selectors reset on a
  driver change; MongoDB leaderboard ranks by a ~2 % rate (#41); cut-off card label (#43).
- **Minor, deferred:** `utils.py` continues with defaults on a broken `config.json`;
  `get_directory_size` defined twice; satellite downloader counts failures as skips;
  `requirements.txt` unpinned (CLAUDE.md lists tested versions); 2024 R21 labelled "in the
  training data"; unused module loggers in `lstm/`; `mean_speed` computed but unused.

## 4. Bug log

All fixed except #7 (⏸), #41 and #43 (⏳, redesign). Each fix has a test that fails on the old code
or a recorded real-data check. Details: archived roadmap.

| # | Bug |
|---|---|
| 1 | Brake (on/off flag) treated as 0–100 % |
| 2 | Scripts in subfolders could not run by path |
| 3 | LSTM: feature/model mismatch, scaler leakage, unsorted sessions, leaky split, last-epoch weights |
| 4 | "Mistake Rate" card showed the contamination setting |
| 5 | Two FastF1 cache folders; scipy missing from requirements |
| 6 | Sidebar credits |
| 7 | Outputs for more sessions/drivers ⏸ |
| 8 | Stale Isolation Forest tests; contamination chosen by a score that always picked the smallest |
| 9 | Log line crashed the Windows console; wrong value printed |
| 10 | Deprecated Streamlit component |
| 11 | Test runner used the system Python |
| 12 | CV turn windows hand-typed for one video |
| 13 | LSTM trained on pit, safety-car and first laps |
| 14 | Prediction overwrote each stint's first lap |
| 15 | Test audit (mocks, untested code) |
| 16 | Frame selection never reached the end of a video |
| 17 | Racing-line start/end on a different grid |
| 18 | Racing-line metres-per-unit scale estimated from session length |
| 19 | Racing line silently used the whole session |
| 20 | Short stints dropped from tyre predictions |
| 21 | Prediction windows bridged removed laps; padded first laps |
| 22 | Wet laps mixed with dry; missing tyre age replaced by lap number |
| 23 | `config.json` read without UTF-8 |
| 24 | `run_tests.py --file` skipped integration tests |
| 25 | Season downloader rejected string years |
| 26 | YOLO masks stretched (padding not removed) |
| 27 | HDFS counted folders as files |
| 28 | Video downloader wrote to an unused folder |
| 29 | MongoDB clients never closed; slow availability check |
| 30 | Partial CV outputs counted as done |
| 31 | Scrubber and Telemetry slider out of sync |
| 32 | Pace delta coloured red both ways |
| 33 | Brake-overlap % dominated by garage samples |
| 34 | Big Data charts: shuffled "trace"; ±1σ labelled 95 % CI |
| 35 | CV report card used other thresholds; badge did not follow the scrubber |
| 36 | Session type parsed from label text |
| 37 | Dashboard edge cases (stale caches, `st.stop()` on empty data) |
| 38 | Unused imports, constants and dead code |
| 39 | Racing-line "time saved" not meaningful |
| 40 | Spark numeric-column check never matched |
| 41 | MongoDB leaderboard ranks by a rate that is ~2 % for everyone ⏳ |
| 42 | Streamlit static-folder warning |
| 43 | Cut-off card label ⏳ |
| 44 | Lap means not weighted by time (irregular samples) |
| 45 | Tyre target was absolute speed (mostly encodes the track) |
| 46 | Config recorded the search run's validation score |
| 47 | Big Data tab crashed on new strategy files |
| 48 | Planted-mistake test recomputed acceleration across traffic gaps (results re-run) |
| 49 | Timestamp fixer rewrote raw files in place |
| 50 | Cleaner exited 0 on failures |
| 51 | RNN training log named the LSTM's files |
| 52 | Small future leak: no prediction for the lap after a run of clean laps |
| 53 | Dashboard re-assigned the backend's lap numbers |
| 54 | Wrong dashboard hints and script paths |

Next free number: **#55**.

## 5. Working rules

- The user commits; never commit for them.
- Update README, PROJECT_DOCUMENTATION (Fixed/Known issues), this roadmap and
  ISOLATION_FOREST_REWORK (mistake detection) with every change. Keep them short.
- Explain every change: what was wrong, what changed, how it was checked, what was not.
- Tests: only essential ones, on real code with real data or small real inputs; no mocks; one
  regression test per fix that fails on the old code.
- Log every new bug here straight away.
- Keep the scope (§2); no new algorithms.
- Never edit `docs/LAST_SEM/`; never delete `docs/ARCHIVE/`.
