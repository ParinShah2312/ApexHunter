# Isolation Forest Rework — Design (roadmap §B)

**Course:** Data Science & Machine Learning · **Status:** **done** (all 5 phases, Oct 2026; progress
log in §6). The final method and results are written up in PROJECT_DOCUMENTATION §5.2. Sections 0–5
below are the original design and handoff notes, kept for the record (§0.1 describes the *old*
code that was replaced).
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
  `anomaly_score` (AI tab map and track-map "Mistakes" mode), `LapNumber` (the dashboard
  re-assigns it from FastF1 lap times). *(Phase 3: `brake_intensity` and the "Under Braking" card
  were removed; the AI tab reads `*_mistakes_events.json`, derived from the meta path in app.py.)*
- Meta keys used by the dashboard: `best_contamination`, `reference_drivers` (list of 5, since
  phase 1). Used by `bigdata/mongo_manager.py`: `session_file`,
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

## 6. Progress log

### Phase 1: done (October 2026)
Implemented in `mistakes_features.py` (track line, projection, acceleration, reference profile,
features), `mistakes_io.py` (`load_session_laps`, `select_laps`, `select_reference_drivers`,
`slice_laps`) and `detect_mistakes.py`. Model selection is still the old criterion until phase 2.
Results: PROJECT_DOCUMENTATION §5.2 (brake-on share of flags 95 % → 37–54 % on 5 runs).

Changes to the design above, each decided on real-data measurements:
- **Deviations are divided by the field's spread** at that point (IQR / 1.349, with floors of
  3 km/h, 5 % throttle, 0.5 gear, 5 km/h/s). Raw differences still gave 73 % braking flags
  (2023 R1 #1), because braking points vary 10–20 m, so speed differences are always large at
  braking-zone entries. `brake_mismatch` stays unscaled (it is already a 0–1 share).
- **`speed_delta` → `accel_diff`:** speed change over a fixed 0.5 s (irregular sample spacing),
  compared with the field's acceleration at that point. A plain speed delta is large in every
  braking zone, which is the original problem.
- **`brake_throttle_overlap` is not a model input** (still in the output, for event reasons in
  phase 3). It is a 0/1 flag on in ~1 % of samples, mostly the transition sample; the forest
  isolates it in one split, giving 90–99 % braking flags on 4 of 5 runs. Comparing it with the
  field's overlap rate did not help (67–93 %).
- **Laps are also cut at 107 % of the driver's best kept lap** (qualifying cool-down laps are not
  caught by `representative_lap_mask`). Reference drivers are ranked by **median** kept-lap time
  (at least 3 laps); the track line is the session's fastest kept lap.
- **The model is trained on the target driver's kept laps only** (not target + reference): a
  slower driver would otherwise look anomalous everywhere. Phase 2 can revisit this.
- **`--reference <file>` removed** (track positions from another session don't line up).
- Measured: samples lie a median 0.3 m from the track line (99th percentile < 1 m), so plain
  nearest-point projection is accurate.

Tests (no mocks): new `integration/test_mistakes_pipeline.py` (2023 R1 #44: 5 other reference
drivers, no lap 1 or pit laps, braking < 60 % of flags; with the overlap flag as an input it is
98 %, so the test catches that regression); `test_pipeline_skip_force.py` rewritten on real data
(2024 R1 Q #16); `test_pipeline_edge_cases.py` keeps the missing-driver and missing-column tests
(the single-row test was removed: a one-row driver now has no laps and exits). The unused
`integration/conftest.py` fixture was deleted.

### Phase 2: done (October 2026)
`isolation_forest/mistakes_eval.py` (new: planting, metrics, `planted_test_set`) and a rewritten
`mistakes_model.py` (`evaluate_settings`: GroupKFold by lap, 3 seeds, max_samples × max_features
by PR-AUC, then contamination by F1; #9 log format fixed). Results in PROJECT_DOCUMENTATION §5.2:
PR-AUC 0.219 on 2023 R1 #1 after the max_samples sweep below (chance 0.022; 0.183 before);
contamination 0.02 selected for the saved output. Parameters are not final (user).

Changes and findings:
- **Brake-onset check (asked by the user):** 83–93 % of flagged braking samples were in the first
  or last 0.2 s of a braking zone (base 20–28 %): chance deviations of single samples, since
  speed/throttle/brake arrive at ~4 Hz. Other spread definitions did not help; a **0.5 s centred
  rolling mean of the features** (`SMOOTH_WINDOW_S`) did (PR-AUC up on 3 of 4 runs, edge share
  47–66 %). Adopted.
- **Early lift** is planted just before the driver's own lift point (drivers already lift a
  moment before braking, so "full throttle up to the braking point" never existed).
- **Contamination by F1 depends on the planted share (~2 %)**, so it is partly a property of the
  test. F1 peaks at 0.03 here (0.184 at 0.005, 0.260 at 0.03, 0.247 at 0.05). Open question for
  the user: keep F1, or choose a fixed flag rate.
- **max_samples sweep (user request, Oct 2026): how much data each tree uses.** 4096 was the
  largest first tried and won every time, so larger values were tested. Mean PR-AUC
  (max_features 1.0; each setting's search ~15 s, the same at every size):

  | Run (training rows per fold) | 4096 | 8192 | 16384 | 32768 | 65536 |
  |---|---|---|---|---|---|
  | 2023 R1 #1 (~29k) | 0.183 | 0.197 | 0.209 | **0.219** | 0.219 |
  | 2023 R1 #44 (~30k) | 0.185 | **0.192** | 0.191 | 0.191 | 0.191 |
  | 2024 R8 #16 (~32k) | 0.186 | 0.197 | 0.204 | 0.212 | **0.214** |
  | 2024 R1 Q #16 (~3.3k) | 0.254 | 0.254 | 0.254 | 0.254 | 0.254 |

  The ceiling is the training rows: above it the value is capped to "all rows", so results stop
  changing (qualifying is capped from 4096). No overfitting showed (PR-AUC on held-out laps never
  dropped by more than 0.001). **Decision (user): max_samples = all rows for every session**
  (`MAX_SAMPLES_VALUES = [1.0]`, still tunable); the meta file records `effective_max_samples`
  and `n_train_rows`. Contamination was later fixed at 0.02 for all sessions (phase 3
  decision); both remain tunable (one constant each in `mistakes_model.py`). With 32768 on 2023 R1 #1, F1
  selects contamination **0.02** (precision 0.256, recall 0.383, F1 0.306; 3 %: 0.212 / 0.444 /
  0.286; 4 %: 0.186 / 0.494 / 0.270).
- **Precision is low (~0.2)**: natural deviations score as high as moderate planted mistakes.
  Early lifts are found 95 % of the time, the other types 24–31 % (2023 R1 #1).

Test: `test_mistakes_pipeline.py` also checks PR-AUC > 4× the planted share (measured ~8×).

### Phase 3: done (October 2026)
`isolation_forest/mistakes_events.py` (new: `assign_events`, `build_events`), events JSON written
by `mistakes_io.save_outputs`, `event_id` and raw field-difference columns in the parquet, AI tab
updated (one marker per event, top-15 table with reasons, counts per lap; `brake_intensity` and
the "Under Braking" card removed). Results and the hand review: PROJECT_DOCUMENTATION §5.2.

Design choices:
- **Merge rule:** flags of the same lap less than 0.5 s apart (§2.5).
- **Ranking by severity = how unusual × how long**, not peak score: a single noisy sample often
  has a high peak (brake-onset noise, phase 2), so peak-only ranking would put noise first.
- **Reason = the feature with the largest mean deviation in units of its own spread** across the
  driver's samples (features have different scales); text uses the raw differences in km/h, %,
  gears and the field's brake-on share. Pedal overlap is the reason when it is on for at least
  half the event.

Findings for the user (2023 R1 #1 and #44, 2024 R8 Monaco #16): 273 / 333 / 283 events per race,
33 / 59 / 35 of them at least 0.5 s long. The top events are readable, but many are unusual
rather than mistakes: the winners' last-lap easing off is the #1 event for both #1 and #16;
slipstream/DRS (27 km/h faster than the field on the main straight); lift-and-coast; repeated
slow exits at one corner (traffic or a different line); Hamilton's/Leclerc's pedal overlap
(19–20 % of their events); a possible gear glitch.

**Decisions (user, Oct 2026):** (1) each driver's last race lap is excluded (`select_laps(...,
is_race=True)`; also removed from the reference drivers' laps); (2) **contamination fixed at 0.02
for every session** (`CONTAMINATION` in `mistakes_model.py`; other values still scored in the
meta `search` table). Result: 262 / 232 / 291 events, 30 / 35 / 34 of them ≥ 0.5 s; PR-AUC 0.217
/ 0.185 / 0.229; no last-lap events in the top lists.

Tests: `unit/test_mistakes_events.py` (merge rule; sustained deviation ranks above a single
sample and both get the expected reason); `test_mistakes_pipeline.py` checks every flagged sample
is in exactly one event, events are sorted by severity, and the last lap (57) is excluded.

### Phase 4: done (October 2026)
`mistakes_eval.py` gained the shared harness: `build_folds` (lap-grouped folds + planted test
sets, built once), `fold_metrics` / `summarise`, `ZScoreDetector` and `compare_detectors` (any
detector with `fit(X)` / `score(X)`, higher = more anomalous; the LSTM autoencoder plugs in here
in §D). `mistakes_model.py` has `ForestDetector` and its search now uses the same helpers.
`detect_mistakes.py --compare` adds a `comparison` key to the meta file (planted metrics for both
detectors + agreement on the real laps); `mistakes_events.event_overlap` measures the agreement.
Also fixed: the injection wrote float64 values into float32 columns (pandas FutureWarning; would
fail in a future pandas).

Results (PROJECT_DOCUMENTATION §5.2): mean over 4 sessions, Z-score vs Isolation Forest: PR-AUC
0.238 vs 0.221, F1 at 2 % 0.249 vs 0.283. IF better on lock-ups and missed downshifts in all four
sessions (several moderate deviations at once); Z-score as good or better on early lifts (one
extreme feature). IF events containing a Z-score flag: 54–72 %. Conclusion: ML helps mainly on
mistakes that combine features; the features do most of the work.

Test: `test_mistakes_pipeline.py` now runs with `compare=True` and checks both detectors use the
same planted share and flag rate and both score above 4x chance.

### Phase 5: done (October 2026)
- Tests: output-schema test added to `integration/test_mistakes_pipeline.py` (replaces old A20:
  the parquet columns, meta keys and event fields read by `ai_analysis`, `track_map`, `app.py`
  and `mongo_manager`). Rework tests in total: `test_mistakes_pipeline.py` (7, real 2023 Bahrain
  race), `test_pipeline_skip_force.py` (3, real 2024 Bahrain qualifying),
  `test_pipeline_edge_cases.py` (2), `unit/test_mistakes_events.py` (2), plus the updated AI-tab
  test. No mocks. Suite: 153 tests, all passing.
- Outputs: `data_lake/mistake_data/2023_1_R_1_*` regenerated with `--compare` (more sessions are
  ROADMAP #7, after all model work).
- Docs: PROJECT_DOCUMENTATION §5.2 rewritten as the final description (method, design choices
  with measured evidence, evaluation, results, Z-score comparison, hand review, outputs).
- Logged #41: the MongoDB leaderboard ranks by mistake rate, now ~2 % for every driver.

## 5. Decisions

| # | Question | Decision |
|---|---|---|
| 1 | Reference for "normal driving" | **Median of the 5 fastest other drivers** (agreed). |
| 2 | Choose settings using planted fake mistakes (§2.4) | **Agreed.** |
| 3 | Comparison models | **Z-score baseline + Isolation Forest + LSTM autoencoder only** (§2.6). One-Class SVM, KNN, DBSCAN, LOF dropped. |
| 4 | Dashboard changes | **Only the minimum needed to check the rework worked** (event list and markers in the AI tab). The full redesign happens with the rest of the UI work (roadmap §E), at the end (agreed). |
