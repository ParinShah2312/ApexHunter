# Isolation Forest Rework — Design (roadmap §B)

**Course:** Data Science & Machine Learning · **Status:** agreed, ready to implement
(starts after the test audit, ROADMAP §0)
**Also closes:** bugs #8 and #9 (remaining part), test-audit items A3, A4, A20.

---

## 0. Before you start (handoff notes)

Read this section, then ROADMAP §3 ("Rules while working"), before writing any code.

### 0.1 What the code does today (to be replaced)
- `detect_mistakes.py` orchestrates `isolation_forest/mistakes_features.py` (7 absolute features
  per sample: `speed_normalized`, `throttle_intensity`, `brake_intensity`, `brake_throttle_overlap`,
  `speed_delta`, `gear_change`, `rpm_normalized`), `mistakes_model.py` (5-fold "grid search" over
  contamination 0.005–0.05 with a criterion that always picks 0.005; 200 trees) and
  `mistakes_io.py` (load/validate, reference driver = fastest by mean speed, outputs).
- Training data = target driver + one reference driver, **all samples** (garage, pit and
  safety-car laps included). Every problem listed in §1 below is real and measured.

### 0.2 Data facts that have caused bugs before
- `Brake` is **binary** (0/1), not a percentage (bug #1). Throttle is 0–100.
- X/Y are in **1/10 m** (FastF1 docs; checked on 4 circuits, bug #18). Use 0.1 m per unit.
- Samples are **irregularly spaced** (median 0.13 s, range 0.014–0.76 s), so a per-sample
  difference (e.g. `speed_delta`) is not a rate. Use time-normalised differences.
- Raw telemetry has **no NaNs**; garage time is recorded as 0 km/h samples, often with brake and
  throttle on together (up to 94 % of a qualifying session's overlap samples, bug #33).
- Each session parquet holds all drivers; `Driver` is the car number as a string ("1", "44").
- `SessionTime` is the real time axis; the `Time` column is synthesised from the row index and
  meaningless.
- FastF1's `pick_fastest()` can return `None` (driver without a timed lap, bug #19).

### 0.3 Reuse, don't re-implement
| Need | Use |
|---|---|
| Representative laps (no lap 1, pit in/out, SC/VSC/red flag, deleted, inaccurate) | `lstm.tyre_data.representative_lap_mask(laps)` |
| FastF1 session/laps | `utils.enable_fastf1_cache()` then `fastf1.get_session(year, round, "Q"/"R")`; `laps.pick_drivers(driver)` |
| Nearest-neighbour projection onto a lap | `scipy.spatial.cKDTree` (as in `racing_line_io.estimate_path_time`) |
| Slicing telemetry to one lap | `SessionTime` between `LapStartTime` and `Time` (as in `racing_line_io.fetch_fastest_lap_bounds`) |
| Logging | `utils.setup_logger` (handles Windows console encoding) |

### 0.4 Who reads the outputs (keep these working, or update them in the same change)
- File names: `data_lake/mistake_data/{session_stem}_{driver}_mistakes.parquet` and
  `…_mistakes_meta.json` (the sidebar derives these paths and shows a status dot).
- Parquet columns used by the dashboard: `X`, `Y`, `Speed`, `SessionTime`, `is_mistake`,
  `anomaly_score` (AI tab map and track-map "Mistakes" mode), `brake_intensity` (AI tab "Under
  Braking" card — replace this card if the feature disappears). The dashboard assigns `LapNumber`
  itself from FastF1 lap times.
- Meta keys used by the dashboard: `best_contamination`, `reference_driver` (will become a list of
  5 drivers; update `ai_analysis.py`). Used by `bigdata/mongo_manager.py`: `session_file`,
  `driver`, `best_contamination`, plus parquet `anomaly_score`, `is_mistake`, `Speed`.
- New outputs (`*_mistakes_events.json`, evaluation metrics in the meta file) are additive.

### 0.5 Tests for the rework
- Replace the mock-only model tests with a few tests that run the real code (audit item A3): e.g.
  the injected-mistake evaluation finds planted mistakes on a small real or realistic lap; the
  deviation features are zero when the driver equals the reference profile; events merge
  consecutive flags and give the expected reason.
- The old Isolation Forest tests were deleted in test audit round 2 (they pinned the old method).
  Write their replacements here: a pipeline output/schema test for the new outputs (old A20) and a
  correctness test that planted mistakes are found (old `test_pipeline_correctness`). The kept
  `integration/test_pipeline_edge_cases.py` and `test_pipeline_skip_force.py` must keep passing.
- Follow the test rules in ROADMAP §3: real code, one test per behaviour that matters, no padding,
  **no mocks** (use small real inputs, or real sessions from `data_lake/` in integration tests,
  as `integration/test_racing_line_pipeline.py` and `test_tyre_pipeline.py` do).
- The main proof of each phase is a **real-data run** (2023 R1 race, driver 1, plus at least one
  other session), with the numbers written into PROJECT_DOCUMENTATION §5.2.

### 0.6 How to work through it
- One phase at a time (§4), check in with the user after each phase, update the docs at the end
  of each phase (PROJECT_DOCUMENTATION §5.2 and §8, ROADMAP §0/§2B, README if commands change).
- Log any bug found on the way in ROADMAP §2A (next number #41).
- Keep the dashboard change minimal (decision 4 in §5); the real UI comes in ROADMAP §E.

## 1. What is wrong today

| Problem | Evidence |
|---|---|
| **The model flags braking, not mistakes.** Features are absolute values per sample (speed/380, brake flag, …). Braking happens in ~20 % of samples, so to the model "braking" is simply rarer than "not braking". | 2023 Bahrain race, Verstappen: 201 of 212 flagged samples are braking samples. |
| **No notion of *where* on track.** Braking hard before Turn 1 is normal; braking hard in the middle of a straight is a mistake. The features can't tell these apart. | Features contain no track position. |
| **Contamination is chosen by a criterion that can't work.** Mean \|decision_function\| always shrinks as contamination grows, so the smallest value always wins. | CV scores decrease monotonically; 0.005 is always selected. |
| **The baseline can be the driver being analysed.** The reference is the fastest driver by mean speed; when that's the target driver, the model compares the driver with themselves. | In the only existing output the reference *is* the target (#1). |
| **Non-racing laps pollute the data.** Pit laps, safety-car laps and the race start look nothing like racing laps and are flagged trivially. | No lap filtering (unlike the fixed LSTM pipeline). |
| **Output is 212 scattered dots.** Consecutive flagged samples belonging to the same incident are reported separately, with no explanation of *why* each was flagged. | Mistakes map in the AI tab. |
| **Tests can't detect any of this.** All model tests mock the model. | Test audit A3/A4/A20. |

## 2. Core idea: compare the driver with the field *at the same point on track*

Instead of asking "is this sample unusual in general?", ask **"is what the driver is doing here
unusual compared with what fast drivers do at this exact point of the lap?"**

```
session telemetry ──► representative laps only (no pit / SC / lap 1)
                  ──► track position s (metres along a reference lap) for every sample
                  ──► field reference profile per 10 m of track
                        (median speed / throttle / brake / gear of the fastest other drivers)
                  ──► per-sample deviation features vs that profile
                  ──► Isolation Forest (tuned with injected-mistake validation)
                  ──► flagged samples ──► merged into mistake *events* with a reason label
```

### 2.1 Track position
Take one clean reference lap (e.g. the session's fastest representative lap) as the track line.
Every sample is projected onto it with a nearest-neighbour search (`scipy.spatial.cKDTree`),
giving `s` = metres along the lap. This needs no extra data and works for any session.

### 2.2 Field reference profile
For each 10 m bin of `s`, the **median** speed, throttle, brake-on rate and gear across the
representative laps of the **N fastest drivers, excluding the target driver** (N = 5, agreed).
Using a median across several drivers and laps is robust to one driver's off lap, and excluding
the target fixes the "compared with themselves" bug for good.

### 2.3 Features (per sample)

| Feature | Meaning | Mistake it captures |
|---|---|---|
| `speed_deficit` | (reference speed − driver speed) at this `s`, km/h | slow corner, lock-up, off-track moment |
| `throttle_diff` | driver throttle − reference throttle at `s` | early lift / late throttle |
| `brake_mismatch` | driver brake flag − reference brake-on rate at `s` | braking where the field doesn't (or not braking where it does) |
| `gear_diff` | driver gear − reference gear at `s` | wrong gear / missed downshift |
| `speed_delta` | change in speed since the previous sample (kept) | sudden deceleration |
| `brake_throttle_overlap` | brake on and throttle > 5 % (kept) | pedal overlap |

Absolute speed/RPM/throttle are dropped: they describe *where* the car is, not *how well* it is
driven, and they are what made braking look anomalous.

### 2.4 Model selection that actually measures something
There are no labelled mistakes in F1 telemetry, so we create a test we *can* score: take real laps,
plant fake mistakes in them at known places, and check whether the model finds them. This is a
standard way to evaluate unsupervised anomaly detectors.

**Step by step**
1. **Hold out laps.** Split the driver's representative laps into groups (GroupKFold by lap). The
   model is trained on some laps and tested on others it has never seen.
2. **Plant fake mistakes** in the held-out laps. Each one edits a short stretch of real telemetry
   (about 0.3–1 s) at a randomly chosen suitable spot, and the edited samples are marked
   "injected". The types:

   | Fake mistake | Where it is planted | What is changed |
   |---|---|---|
   | Lock-up / over-braking | inside a braking zone | speed drops an extra 15–30 km/h, then recovers gradually |
   | Early lift | on a full-throttle stretch just before a braking zone | throttle set to 0 earlier than the field does; speed a few km/h lower |
   | Late throttle | just after a corner apex | throttle kept at 0 for an extra 0.5–1 s; exit speed lower |
   | Pedal overlap | in a braking zone | brake on while throttle is held above 5 % |
   | Missed downshift | through a slow corner | gear held one higher than the field uses |

   Each test lap gets a few of these, so injected samples are a small minority (roughly 1–3 %),
   like real mistakes. Mistake sizes are moderate, not extreme, so the test isn't trivially easy.
3. **Score.** The model gives every sample an anomaly score. We measure how well the scores rank
   the injected samples above the normal ones using **average precision (PR-AUC)**: 1.0 means
   every planted mistake is ranked above every normal sample; random guessing scores about the
   share of injected samples (≈ 0.02). We also report **precision** (of what was flagged, how much
   was really planted) and **recall** (of what was planted, how much was found) at the chosen
   contamination.
4. **Choose settings.** Repeat over all folds and several random seeds, try each combination of
   `contamination`, `max_samples` and `max_features`, and keep the one with the best average score.

**Limitations (stated in the report):** the fake mistakes are simplified versions of real ones,
so the scores measure "can the model find these patterns", not exact real-world accuracy.
Flagged events on real laps are also reviewed by hand as a sanity check.

### 2.5 Mistake events and reasons
Consecutive flagged samples (gap < 0.5 s) become one **event** with: lap, start/end time, track
position, duration, peak anomaly score, and a **reason**, which is the feature with the largest
standardised deviation, mapped to plain language ("braked where the field didn't",
"speed deficit 18 km/h", "throttle–brake overlap", …). The dashboard shows events instead of
hundreds of dots.

### 2.6 Comparison: simple baseline → ML → DL
Team decision (Oct 2026): keep the comparison small and easy to explain. Three detectors, all on the
same features and the same planted-mistake test, all on the DSML anomaly-detection slide:

| Detector | Idea, in plain terms | Role |
|---|---|---|
| **Z-score** | No machine learning: for each feature, how many standard deviations from normal; flag if any is extreme (e.g. > 3). | Simple baseline — shows whether ML is needed at all. |
| **Isolation Forest** | Randomly splits the data again and again; unusual samples get separated from the rest in very few splits. | Main DSML model. |
| **LSTM autoencoder** | Neural network that compresses ~2 s windows of telemetry and rebuilds them; trained on normal driving, so mistakes rebuild badly. Rebuild error = mistake score. | Deep Learning counterpart (roadmap §D). |

Reported per detector: confusion matrix, precision, recall, F1 (taught), PR-AUC (extra), and how
many flagged events overlap. One-Class SVM, KNN, DBSCAN and LOF were considered and dropped to keep
the project focused.

## 3. What changes in the code

| File | Change |
|---|---|
| `isolation_forest/mistakes_features.py` | Track projection, field reference profile, deviation features (replaces the absolute features). |
| `isolation_forest/mistakes_model.py` | Grid search driven by injected-mistake PR-AUC with lap-grouped folds; fixes the log format (#9). |
| `isolation_forest/mistakes_eval.py` *(new)* | Mistake injection, metrics (confusion matrix, precision, recall, F1, PR-AUC), Z-score baseline, and the comparison harness shared with the LSTM autoencoder. |
| `isolation_forest/mistakes_events.py` *(new)* | Sample flags → events with reason labels. |
| `isolation_forest/mistakes_io.py` | Lap filtering via FastF1 (reusing `lstm.tyre_data.representative_lap_mask`), reference-driver selection (N fastest, never the target), new outputs. |
| `detect_mistakes.py` | Orchestrates the new steps; adds `--compare` to run the Z-score vs Isolation Forest comparison (the autoencoder joins in roadmap §D). |
| Outputs | `*_mistakes.parquet` (+ `track_pos`, lap, new features), `*_mistakes_events.json` *(new)*, `*_mistakes_meta.json` (+ evaluation metrics, reference drivers). Keys used by MongoDB and the Big Data tab are kept. |
| `frontend/components/ai_analysis.py` | Only enough to verify the rework: event list with reasons and map markers per event. The full redesign is done later with the rest of the UI (roadmap §E). |
| Tests | Old tests already deleted (test audit round 2). New: real-model tests (old A3), output/schema test (old A20), planted-mistake correctness test; kept edge-case and skip/force tests must pass. |
| Docs | PROJECT_DOCUMENTATION §5.2 rewritten with the new method, evaluation and comparison results. |

## 4. Plan (phases)

| Phase | Work | Checkpoint |
|---|---|---|
| 1 | Lap filtering, track projection, field reference, deviation features | On real data: braking samples no longer dominate the flags |
| 2 | Injection + evaluation harness; contamination/hyperparameter search | Selection no longer always picks the smallest contamination; PR-AUC reported |
| 3 | Events + reasons; outputs; minimal dashboard update | Bahrain 2023 race shows a short, readable event list |
| 4 | Z-score baseline in the same evaluation (autoencoder added in roadmap §D) | Comparison table in the docs |
| 5 | Tests, regenerate outputs, docs | Full test suite green, including the new Isolation Forest tests |

Each phase ends with a check-in before the next.

## 5. Decisions

| # | Question | Decision |
|---|---|---|
| 1 | Reference for "normal driving" | **Median of the 5 fastest other drivers** (agreed). |
| 2 | Choose settings using planted fake mistakes (§2.4) | **Agreed.** |
| 3 | Comparison models | **Z-score baseline + Isolation Forest + LSTM autoencoder only** (§2.6). One-Class SVM, KNN, DBSCAN, LOF dropped. |
| 4 | Dashboard changes | **Only the minimum needed to check the rework worked** (event list and markers in the AI tab). The full redesign happens with the rest of the UI work (roadmap §E), at the end (agreed). |
