# ApexHunter — Roadmap (AY 2026–27)

**Courses:** Data Science & Machine Learning (DSML, CMP707) · Deep Learning (DL)
**Student:** Parin Shah (23001091)

What we work on, what is done, what is left, and every bug found. The full history (long
descriptions, dated decisions, intermediate numbers) is in
[ARCHIVE/ROADMAP_full_2026-11-04.md](ARCHIVE/ROADMAP_full_2026-11-04.md). Section references in code
comments (§B mistakes, §C tyre/pit strategy, §D RNN/LSTM/autoencoder, §E dashboard) point there.

Status: ✅ done · 🔄 in progress · ⏳ to do · ⏸ deferred

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
| **Dashboard redesign (§3)** | ✅ done (2026-11-05) |
| Report (DSML, 21 pages) + presentation (DSML + DL, 25 slides) in `docs/submission/`: rebuilt 2026-11-05 (final numbers, autoencoder, synthetic-mistake test with its limit and the real-lap check, dashboard section, one baseline row per comparison) | 🔄 user review |
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

**From last semester (in the More tab):** YOLOv11-Seg apex detection, racing-line search (A\*/Dijkstra/BFS),
Big Data stack (Spark, HDFS, MongoDB). **Dropped:** other anomaly detectors, Random Forest /
boosting, clustering + PCA, classification, GRU, choosing a tyre compound (race data cannot rank the
compounds).

## 3. Dashboard redesign (done 2026-11-05)

**Look:** dark theme, one accent colour, few emojis; one fixed colour per model everywhere; one-line
caption per chart ("what this shows"); hints on hover; units everywhere; tables collapsed.

Build order:
1. ✅ **Shell.** Sidebar: season switch (2023 | 2024) → Grand Prix dropdown → session switch
   (Qualifying | Race) → driver (with team); badges "Mistakes ✓/✗", "Pit window ✓/✗". Opens on
   2023 Bahrain Race, Verstappen. Slim top bar: GP · session, driver, position, fastest lap, gap to
   the session's fastest (and who set it), top speed. Tabs: Overview · Driver Mistakes · Tyres & Pit
   Window · Telemetry · More; only the open tab runs.
2. ✅ **Telemetry.** Lap picker (lap-time dots; pit grey, SC yellow, fastest purple; click to pick);
   compare with none / another lap / another driver's fastest (default: session fastest). Stacked
   charts vs distance in the lap (m), corners marked T1…: speed (line), time delta (line, red/green
   fill), throttle (filled area), brake (shaded bands, it is on/off), gear (step line); RPM in a
   closed "More channels" panel. Track map coloured by speed/gear. Shared hover line + map dot; lap
   position slider (the scrubber, this tab only); zoom one chart = zoom all. Cards: lap time, gap,
   top speed, full-throttle %. Distance = speed × time summed per lap; delta = time at the same
   fraction of the lap (ends exactly at the lap-time gap). Slider and hover name the place ("Exit
   of Turn 4", "Straight: Turn 4 → Turn 5"; slider labels also carry the metres, as Streamlit needs
   unique labels); hover text is plain sentences. The session-time
   scrubber, old track map and "brake overlap" card were removed. Fixed: a tab-header style rule
   greyed out every switch inside a tab (selected option not visible).
3. ✅ **Driver Mistakes.** Map with one marker per event; event list; click an event → driver vs field
   (speed, throttle, brake) at that corner. Detector switch Z-score / Isolation Forest / autoencoder
   (backend also saves the Z-score and autoencoder flags; re-run the 4 sessions, numbers unchanged).
   Detector comparison chart (precision/recall/F1) + real-lap check. Done: re-run of the 4 sessions
   left every meta file and the Isolation Forest outputs identical; each mistake's detail also shows
   the driver's usual laps (median); cards: mistakes, time lost, share that cost time vs random
   spots, readings flagged.
4. ✅ **Tyres & Pit Window.** Stint buttons; lap-time chart with window band, SC shading, real stop;
   model switch Best / Ridge / RNN / LSTM / No change; bars: lap-time error per model, window hit
   rate vs random.
   Done: `predict_strategy.py --source all` stores every source in each file (best on top, rest
   under "sources"); all 159 files regenerated (top part identical to before, 30 MB in total).
   Cards: real stop, normal window, in the window, lap-time error this race. Error bars start at 0.
   The tab shows only the three trained models (user, 2026-11-05); "no change" and "laps
   driven" stay in the outputs, check_strategy.py and these docs (§2 scope unchanged).
5. ✅ **Overview.** Mistake summary (count, time lost, top 3), pit-window summary per stint, model
   scoreboard (mistake F1, lap-time error).
   Done: mistakes card (Isolation Forest: count, time lost, share that cost time, top 3) and pit
   card (stops in window, per-stint line; follows the Tyres tab's model); scoreboard = trained
   models only: mean planted F1 over the scored sessions, test lap-time error.
6. ✅ **More.** Switch inside: Apex detection / Racing line / Big Data (moved as they are; only the
   chosen one loads).
7. ✅ **Finish.** Style pass, carried-over fixes below, real-data AppTest checks, docs.

- **Carried over:** all fixed (step 7: a corrupt pit-window file now says so instead of "missing";
  step 6: CV cards now count
  corners, 12 of 15 on the Bahrain pole lap; made-up radar removed; pathfinding shows the chosen
  session's file and names it; #41, #43. Gone with the old views: racing-line overlay code, old
  check selectors.)
- **Minor, deferred:** `utils.py` continues with defaults on a broken `config.json`;
  `get_directory_size` defined twice; satellite downloader counts failures as skips;
  `requirements.txt` unpinned (CLAUDE.md lists tested versions); 2024 R21 labelled "in the
  training data"; unused module loggers in `lstm/`; `mean_speed` computed but unused.

## 4. Bug log

All fixed except #7 (⏸). Each fix has a test that fails on the old code
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
| 41 | MongoDB leaderboard ranked by a rate that is ~2 % for everyone (now a plain list; not run: MongoDB offline) |
| 42 | Streamlit static-folder warning |
| 43 | Cut-off card label (old mistakes cards, replaced in step 3) |
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
| 55 | Header showed one fixed team per car number (2023 drivers got 2024 teams) |

Next free number: **#56**.

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
