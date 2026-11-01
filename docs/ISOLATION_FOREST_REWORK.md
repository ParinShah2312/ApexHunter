# Driver Mistake Detection — Isolation Forest vs Z-score (roadmap §B)

**Course:** Data Science & Machine Learning · **Status:** **done** (reworked in all 5 phases, Oct
2026; review changes 2026-10-31: traffic filter, events of at least 0.5 s, corner names, §11
phase 6; per-mistake scoring and time lost 2026-10-31, phase 7). This file holds everything about the mistake detector: method, evaluation, results,
outputs, tests, decisions and the development history. The LSTM autoencoder joins the same
comparison in the LSTM rework (ROADMAP §D).
**Also closed:** bugs #8 and #9, test-audit items A3, A4, A20.

**Files:** `backend/scripts/detect_mistakes.py` (orchestrator); `isolation_forest/`
`mistakes_features.py` (track position, field profile, features), `mistakes_io.py` (FastF1
session, laps, reference drivers, traffic filter, corner positions, outputs), `mistakes_model.py` (Isolation Forest, model selection),
`mistakes_eval.py` (planted-mistake test, Z-score baseline, comparison), `mistakes_events.py`
(events and reasons), `mistakes_timeloss.py` (time lost, saving label, real-data check).

*(The original design notes written before implementation are in
`ARCHIVE/ISOLATION_FOREST_REWORK_before_cleanup.md`.)*

---

## 1. Why it was reworked

The first version (previous semester) used 7 absolute features per sample (`speed_normalized`,
`throttle_intensity`, `brake_intensity`, `brake_throttle_overlap`, `speed_delta`, `gear_change`,
`rpm_normalized`), trained on the target driver + one reference driver over all samples.

| Problem | Evidence |
|---|---|
| **The model flagged braking, not mistakes.** Features were absolute values, and braking happens in ~20 % of samples, so to the model "braking" was simply rarer than "not braking". | 2023 Bahrain race, Verstappen: 201 of 212 flagged samples were braking samples. |
| **No notion of *where* on track.** Braking hard before Turn 1 is normal; braking hard mid-straight is a mistake. | Features contained no track position. |
| **Contamination chosen by a criterion that can't work** (#8). Mean \|decision_function\| always shrinks as contamination grows, so the smallest value always won; the log printed 0.005 as "0.01" (#9). | CV scores decreased monotonically; 0.005 always selected. |
| **The reference could be the driver being analysed.** Reference = fastest driver by mean speed. | In the only existing output the reference *was* the target (#1). |
| **Non-racing laps polluted the data.** Pit, safety-car and start laps are flagged trivially. | No lap filtering. |
| **Output was 212 scattered dots** with no explanation. | AI tab mistakes map. |
| **Tests could not detect any of this.** All model tests mocked the model. | Test audit A3/A4/A20. |

## 2. Method

### Idea
Isolation Forest is an unsupervised anomaly detector: it builds random trees that split on random
features at random thresholds, and points isolated after few splits are anomalous. The question
asked of every telemetry sample is **"is the driver doing something unusual compared with what the
fastest other drivers do at this exact point on track?"**, not "is this sample unusual in
general?". The old method asked the second question, so ordinary braking made up 95 % of its flags.

### Pipeline
1. **Load** the clean session (`Driver, Speed, Throttle, Brake, X, Y, SessionTime, nGear, Year,
   Round, Session`).
2. **Laps** from FastF1. Kept: representative laps (`lstm.tyre_data.representative_lap_mask`: not
   lap 1, pit in/out laps, SC/VSC/red-flag laps, deleted or inaccurate laps), within 107 % of the
   driver's best such lap (drops qualifying cool-down laps), and in a race not the driver's last
   lap (drivers ease off on it). Samples belong to a lap by `SessionTime`.
   **Traffic filter (races only):** samples with DRS open or less than 1 s behind another car are
   left out, for the driver and the reference drivers. DRS and the gap to the car ahead are not in
   the cleaned parquet; they come from FastF1's telemetry for the same samples (`DRS` ≥ 10 = open;
   gap = FastF1's `DistanceToDriverAhead` / speed; matched on `SessionTime`). Qualifying is not
   filtered, because every driver uses DRS there.
3. **Reference drivers:** the 5 fastest *other* drivers by median kept-lap time (at least 3 laps).
4. **Track position:** the session's fastest kept lap is resampled every 1 m; each sample is
   projected onto it (nearest point, `cKDTree`) → `track_pos` in metres. Samples lie a median
   0.3 m from this line (99th percentile < 1 m).
5. **Field profile** per 10 m of track from the reference drivers' laps: median speed, throttle,
   gear and acceleration, their robust spread (IQR / 1.349, with floors of 3 km/h, 5 % throttle,
   0.5 gear, 5 km/h/s), and the share of samples with the brake on.
6. **Features** (below), smoothed with a centred 0.5 s rolling mean within each lap.
7. **Model selection** on planted fake mistakes (§3), then the final Isolation Forest
   (200 trees, all rows per tree) fitted on the driver's own kept laps and used to score them.
   Training on the reference drivers too would make a slower driver anomalous everywhere.
8. **Flags:** the 2 % most anomalous samples (contamination 0.02, fixed for every session so
   counts compare across drivers and races).
9. **Events:** flags of the same lap less than 1 s apart form one flag run; runs of at least
   0.5 s are **events** (shorter runs keep their flags in the parquet but are not events). Each
   event gets a plain-language **reason** from the feature that deviates most (in units of its
   own spread; pedal overlap is the reason when it is on for at least half the event), a
   **location** from FastF1's corner markers projected onto the track line ("Turn N" within 50 m
   of a marker, "Approaching Turn N" up to 200 m before it, otherwise "After Turn N"), and its
   **time lost** vs the driver's own laps (§3). Events are ranked by time lost; likely
   lift-and-coast / tyre saving is labelled and ranked last (§6).

### Features (model inputs)

| Feature | Formula | Catches |
|---|---|---|
| `speed_deficit` | (field median speed − speed) / field spread | slow corner, lock-up, off-track moment |
| `throttle_diff` | (throttle − field median) / field spread | early lift, late throttle |
| `brake_mismatch` | brake flag − field brake-on share (unscaled, already 0–1) | braking where the field doesn't, or not braking where it does |
| `gear_diff` | (gear − field median) / field spread | wrong gear, missed downshift |
| `accel_diff` | (acceleration − field median) / field spread; acceleration = speed change over the last 0.5 s (km/h per s) | sudden deceleration |

Info columns (output only, used for reasons): `brake_throttle_overlap`, `speed_vs_field_kmh`,
`throttle_vs_field_pct`, `gear_vs_field`, `field_brake_share`.

### Why each design choice (measured on real sessions)

| Choice | Without it | With it |
|---|---|---|
| Deviations from the field at the same track position, not absolute values | 95 % of flags are braking samples (2023 Bahrain #1) | 37–54 % (5 runs; base rate 15–29 %) |
| Divide by the field's spread at that point | 73 % braking flags on #1 (braking points vary by 10–20 m, so raw speed differences are always large there) | 39 % |
| Pedal overlap (brake + throttle > 5 %) **not** a model input (kept for reasons) | 90–99 % braking flags on 4 of 5 runs: a 0/1 flag on in ~1 % of samples is isolated in one split. Comparing it with the field's overlap rate did not help (67–93 %). | 37–54 % |
| Acceleration over a fixed 0.5 s, not per sample | samples are 0.003–0.34 s apart, so a per-sample difference is not a rate | — |
| 0.5 s smoothing of the features | 83–93 % of flagged braking samples sit in the first/last 0.2 s of a braking zone (telemetry arrives at ~4 Hz, so the brake onset is only known to ~0.25 s) | 47–66 %; planted-mistake PR-AUC up on 3 of 4 runs |
| Laps cut at 107 % of the driver's best | qualifying cool-down laps are not caught by `representative_lap_mask` | — |
| Last race lap excluded | easing off on the last lap was the #1 event for both race winners checked | no last-lap events in the top lists |
| DRS-open and < 1 s-behind samples left out (races) | slipstream/DRS and dirty air among the top events (#44: 27 km/h faster than the field on the main straight) | per-sample F1 up on 2 of 3 races (§11 phase 6) |
| Only flag runs ≥ 0.5 s are events | 232–291 events per race, most of them 1–2 samples | 27–31 events per race (32–40 with the 1 s merge gap) |
| Flags < 1 s apart joined into one event (was 0.5 s) | one planted mistake was often flagged in pieces, each too short to be an event | planted-mistake recall 0.22 → 0.27, F1 0.29 → 0.34 (§11 phase 8) |
| Time lost per event, measured on a 300 m stretch | full sectors (~30 s) are too noisy to show a 0.2 s mistake | events ranked by real cost, e.g. +0.64 s |

## 3. Evaluation
Two checks, used for every detector (Z-score, Isolation Forest, later the LSTM autoencoder).

**1. Planted mistakes, scored per mistake (main metric).** F1 telemetry has no labelled
mistakes, so fake ones are planted at known places in real laps the model has not been trained on
(`mistakes_eval.py`):
- **Folds:** 5-fold GroupKFold by lap; the detector is fitted on the clean training laps.
- **Planting:** 3 mistakes per held-out lap, random types, 3 random seeds:

  | Fake mistake | Where | What is changed |
  |---|---|---|
  | Lock-up | inside a braking zone | an extra 15–30 km/h speed drop, then recovery |
  | Early lift | just before the driver's own lift point | throttle off 0.5–1 s earlier, speed 3–8 km/h lower |
  | Late throttle | after the driver re-applied throttle | throttle held at 0 for 0.5–1 s, speed 5–10 km/h lower |
  | Pedal overlap | in a braking zone | throttle 20–40 % while braking |
  | Missed downshift | through a corner below 150 km/h | one gear too high |

- **Scoring per mistake:** the detector's flags (2 % threshold from the training laps) become
  events with the same rule as the real output (flags < 1 s apart joined, runs ≥ 0.5 s). **Recall** = planted mistakes with
  at least one sample inside an event ("did it notice the lock-up at all?"); **precision** =
  events that touch a planted mistake ("when it says mistake, is there one?"); F1. Settings
  (max_features ∈ {1.0, 0.6}) are chosen by this F1.
- **Limitation:** the mistakes are fake, so this measures "can it find these patterns".

**2. Time lost on real laps (real-data check).** For each event, the time over a stretch from
100 m before to 200 m after it is compared with the median over the same stretch on the driver's
nearby laps of the same stint (±5 laps; in qualifying, all other kept laps). The event is
**costly** if it lost more than 2× the normal lap-to-lap spread there and at least 0.05 s
(`mistakes_timeloss.py`). The check: the share of a detector's events that are costly, compared
with 200 random spots on the same laps (the chance level). A detector that points at real costly
moments beats the random spots.

## 4. Results (final settings, Isolation Forest)

| Session | Laps | Samples (traffic left out) | Events | Precision | Recall | F1 | Costly events | Costly random spots |
|---|---|---|---|---|---|---|---|---|
| 2023 Bahrain race, #1 Verstappen | 49 | 35,860 (36) | 40 | 0.545 | 0.307 | 0.381 | 15 % | 10 % |
| 2023 Bahrain race, #44 Hamilton | 49 | 33,126 (3,800) | 32 | 0.520 | 0.247 | 0.324 | 19 % | 12 % |
| 2024 Monaco race, #16 Leclerc | 67 | 33,330 (6,610) | 35 | 0.551 | 0.238 | 0.329 | 6 % | 10 % |
| 2024 Bahrain qualifying, #16 Leclerc | 6 | 4,168 (no filter) | 3 | 0.433 | 0.300 | 0.338 | 1 of 3 | 19 % |

- **Planted mistakes:** about half of the events are on a planted mistake, and about 1 in 4
  planted mistakes is found. Many planted mistakes are flagged only for a moment (< 0.5 s), which
  is not an event, so recall is low (if every flagged sample counted, recall would be ~0.56 but
  precision ~0.24).
- **Real laps:** in both Bahrain races the events cost time more often than random spots
  (15–19 % vs 10–12 %). At Monaco they do not (6 % vs 10 %): there the events are mostly different from the field
  without costing Leclerc time against his own laps.

## 5. Comparison: Z-score baseline vs Isolation Forest
The Z-score baseline has no machine learning: it learns each feature's mean and standard
deviation on the training laps and flags a sample when any one feature is extreme (score = the
largest |z|). Same features, folds, planted mistakes and 2 % flag rate (`--compare`).

| Session | Precision Z / IF | Recall Z / IF | F1 Z / IF | Costly events Z / IF (random) |
|---|---|---|---|---|
| 2023 Bahrain race, #1 | 0.536 / 0.545 | 0.236 / **0.307** | 0.318 / **0.381** | 18 % / 15 % (10 %) |
| 2023 Bahrain race, #44 | 0.540 / 0.520 | 0.230 / 0.247 | 0.320 / 0.324 | 16 % / 19 % (12 %) |
| 2024 Monaco race, #16 | **0.760** / 0.551 | 0.187 / **0.238** | 0.298 / **0.329** | 0 of 12 / 6 % (10 %) |
| 2024 Bahrain qualifying, #16 | 0.367 / 0.433 | 0.144 / **0.300** | 0.199 / **0.338** | 0 of 3 / 1 of 3 (19 %) |
| **Mean** | **0.551** / 0.512 | 0.199 / **0.273** | 0.284 / **0.343** | |

Recall by planted mistake type (Z / IF):

| Session | Lock-up | Early lift | Late throttle | Pedal overlap | Missed downshift |
|---|---|---|---|---|---|
| 2023 Bahrain #1 | 0.17 / **0.39** | **0.81** / 0.61 | 0.10 / **0.20** | 0.20 / 0.28 | 0.13 / **0.21** |
| 2023 Bahrain #44 | 0.07 / **0.19** | 0.60 / 0.59 | 0.38 / 0.29 | 0.18 / 0.20 | 0.02 / **0.09** |
| 2024 Monaco #16 | 0.36 / **0.51** | 0.52 / 0.51 | 0.04 / 0.06 | 0.15 / 0.10 | 0.09 / **0.16** |
| 2024 Bahrain Q #16 | 0.29 / **0.64** | **0.71** / 0.00 | 0.00 / 0.00 | **1.00** / 0.00 | 0.00 / **0.33** |

- **Isolation Forest finds more mistakes** (higher recall and F1 in all four sessions; nearly
  equal on #44); Z-score is more often right at Monaco (higher precision).
- **Different strengths:** Isolation Forest is better on lock-ups and missed downshifts in all four
  sessions (several moderate deviations at once, which a "one feature is extreme" rule misses);
  Z-score is better on early lifts (throttle alone is extreme).
- **Real laps:** both detectors' events cost time about equally often (15–19 %, above the
  random spots in Bahrain; neither beats chance at Monaco).
- **Conclusion:** machine learning is a modest win, coming from mistakes that combine several
  smaller deviations. That motivates the LSTM autoencoder (ROADMAP §D), which also sees how the
  deviations develop over ~2 s.

## 6. What the events look like
Events are ranked by time lost. Top events, 2023 Bahrain #44: Turn 10 on lap 38 (+0.64 s, 16 km/h
slower than his own laps), approaching Turn 12 on lap 34 (+0.52 s), Turn 4 on laps 37 and 47
(+0.42 s, +0.39 s). #1: Turn 15 on lap 51 (+0.25 s), Turn 2 on lap 9 (+0.20 s).

**Lift-and-coast / tyre saving** is labelled, not removed: if the event coasts (throttle < 10 %,
brake off, ≥ 150 km/h, ≥ 0.3 s) and the driver coasts at the same spot (±50 m) on at least half
of the other laps of the stint, the event gets "Coasts here on 13 of 16 other laps of the stint:
likely lift-and-coast or tyre saving" and is ranked last. #1 has 4 such events (all approaching
Turn 14, leading comfortably); the others have none. The label says "likely": the data cannot
tell why a driver saves.

Still flagged but not mistakes: driving style and a different line; the detector finds
"different from the field", and time lost only shows which of those differences cost time.

## 7. Outputs
- `data_lake/mistake_data/{session}_{driver}_mistakes.parquet`: the driver's kept-lap samples
  with the telemetry plus `LapNumber`, `track_pos`, `anomaly_score` (decision_function; negative =
  anomalous), `is_mistake`, `event_id` (−1 = not flagged or in a flag run shorter than 0.5 s),
  the five features and the info columns. In races, traffic samples (DRS open, < 1 s behind) are
  not in the file.
- `..._mistakes_events.json`: events, most costly first: `rank`, `event_id`, `lap`, `start_s`,
  `end_s`, `duration_s`, `n_samples`, `track_pos_m`, `start_pos_m`, `end_pos_m`, `location`,
  `x`/`y`/`speed_kmh` at the peak, `peak_score`, `severity`, `main_feature`, `reason`,
  `time_lost_s`, `costly`, `min_speed_vs_own_kmh`, `likely_saving`, `saving_note`.
- `..._mistakes_meta.json`: `session_file`, `driver`, `reference_drivers`, `n_laps`, chosen
  settings (`best_contamination`, `best_max_samples`, `best_max_features`, `effective_max_samples`,
  `n_train_rows`), `evaluation` (per-mistake precision/recall/F1, recall by type, counts),
  `search` (every setting's scores), `time_lost_check` (costly share of events and of random
  spots), `comparison` (with `--compare`), `n_events`,
  `traffic_samples_removed`, `flags_outside_events`, totals, mistake rate, feature list, timestamp.
- **Who reads them:** the sidebar status dot (file names); the AI tab (`X`, `Y`, `Speed`,
  `SessionTime`, `LapNumber`, `is_mistake`, `anomaly_score`, events JSON, `best_contamination`,
  `reference_drivers`); the Race Intelligence map's "Mistakes" mode (flagged samples);
  `bigdata/mongo_manager.py` (`session_file`, `driver`, `best_contamination`, `anomaly_score`,
  `is_mistake`, `Speed`).
- **Dashboard (minimal until the UI redesign, ROADMAP §E):** the AI tab shows one marker per event
  of the selected lap and a table of all events, most costly first, with location, time lost and reason.
- Saved outputs exist for 2023 Bahrain race, #1 only (more sessions: ROADMAP #7).

## 8. Usage
```bash
python backend/scripts/detect_mistakes.py --session data_lake/clean_data/2023_1_R.parquet --driver 1 [--output-dir <dir>] [--force] [--compare]
```
~40 s per race on CPU with `--compare` once the FastF1 cache holds the session's telemetry (the
first load of a session's telemetry from the cache takes ~30 s more). The old `--reference <file>` option was
removed: track positions from another session would not line up.

To add another detector to the comparison (the LSTM autoencoder in §D): give it `fit(X)` and
`score(X)` (higher = more anomalous) and pass it to `mistakes_eval.compare_detectors` with the
folds from `build_folds`.

## 9. Tests (all real data, no mocks)
- `integration/test_mistakes_pipeline.py` (9 tests, real 2023 Bahrain race, #44, `compare=True`):
  output schema read by the dashboard and MongoDB; 5 other reference drivers; braking < 60 % of
  flags (with the overlap flag as an input it is 98 %, so this catches that regression);
  planted mistakes found (per-mistake precision > 0.3, recall > 0.1); events are flag runs
  ≥ 0.5 s, ranked by time lost, each with a corner location; traffic samples left out; events
  cost time more often than random spots; Z-score and IF on the same planted test and flag rate;
  lap 1, pit laps and the last lap excluded.
- `integration/test_pipeline_skip_force.py` (3, real 2024 Bahrain qualifying): skip / force /
  output folder.
- `integration/test_pipeline_edge_cases.py` (2): missing driver, missing columns.
- `unit/test_mistakes_events.py` (3): merge rule and runs < 0.5 s dropped; a longer deviation
  ranks above a shorter one with a higher peak, both with the expected (capitalised) reason;
  corner names, including across the finish line.
- `unit/test_mistakes_timeloss.py` (3): a slow stretch costs its extra time and a normal lap is
  not costly; repeated coasting is labelled saving and ranked last; a planted mistake counts once
  however many of its samples are flagged.
- `unit/test_frontend_ai_analysis.py`: the AI tab renders events.

## 10. Decisions

| # | Question | Decision |
|---|---|---|
| 1 | Reference for "normal driving" | Median of the **5 fastest other drivers** at the same track position. |
| 2 | How to evaluate and choose settings | Planted fake mistakes in lap-grouped folds, scored **per mistake** (precision, recall, F1); per-sample scores dropped (user, 2026-10-31). Time lost on real laps as the real-data check. |
| 3 | Comparison models | **Z-score baseline + Isolation Forest + LSTM autoencoder only.** One-Class SVM, KNN, DBSCAN, LOF dropped. |
| 4 | Dashboard changes | Only the minimum needed to check the rework (event list and markers in the AI tab); the full redesign is ROADMAP §E. |
| 5 | max_samples | **All rows** for every session (user, Oct 2026; still tunable: `MAX_SAMPLES_VALUES` in `mistakes_model.py`). |
| 6 | Contamination | **Fixed at 0.02** for every session, so counts compare across drivers and races (user, Oct 2026; `CONTAMINATION`). |
| 7 | Last race lap | **Excluded** (user, Oct 2026; `select_laps(..., is_race=True)`, also for the reference drivers). |
| 8 | Slipstream / DRS | **Left out in races:** samples with DRS open or < 1 s behind another car, for the driver and the reference drivers (user, 2026-10-31; `mistakes_io.traffic_mask`). |
| 9 | Short flag runs | **Not events:** only runs of at least 0.5 s are events (user, 2026-10-31; `MIN_EVENT_S`). Flags stay in the parquet. |
| 10 | Flag rate after the review | **Stays at 2 %** (user, 2026-10-31): 1 % would raise precision but lower recall. |
| 11 | Combining Z-score and IF (flag only where both agree) | **Not done** (user, 2026-10-31): the LSTM autoencoder joins in §D, and comparing the three detectors separately is clearer than combined outputs. |
| 12 | Event ranking | **By time lost** vs the driver's own nearby laps; likely lift-and-coast / tyre saving labelled and ranked last, not removed (user, 2026-10-31). |

---

## 11. Development history (Oct 2026)

### Phase 1: lap filtering, track position, field reference, deviation features
Implemented in `mistakes_features.py` (track line, projection, acceleration, reference profile,
features), `mistakes_io.py` (`load_session_laps`, now `load_fastf1_session`; `select_laps`, `select_reference_drivers`,
`slice_laps`) and `detect_mistakes.py`. Brake-on share of flags 95 % → 37–54 % on 5 runs.

Changes to the original design, each decided on real-data measurements:
- **Deviations are divided by the field's spread** at that point. Raw differences still gave 73 %
  braking flags (2023 R1 #1), because braking points vary 10–20 m.
- **`speed_delta` → `accel_diff`:** speed change over a fixed 0.5 s (irregular sample spacing),
  compared with the field's acceleration at that point. A plain speed delta is large in every
  braking zone, which is the original problem.
- **`brake_throttle_overlap` is not a model input** (see §2).
- **Laps are also cut at 107 %** of the driver's best kept lap. Reference drivers are ranked by
  **median** kept-lap time (at least 3 laps); the track line is the session's fastest kept lap.
- **The model is trained on the target driver's kept laps only** (not target + reference).
- **`--reference <file>` removed.**

Tests: new `integration/test_mistakes_pipeline.py`; `test_pipeline_skip_force.py` rewritten on
real data (2024 R1 Q #16); `test_pipeline_edge_cases.py` keeps the missing-driver and
missing-column tests (the single-row test was removed: a one-row driver now has no laps and
exits). The unused `integration/conftest.py` fixture was deleted.

### Phase 2: planted-mistake evaluation and model selection
`mistakes_eval.py` (new: planting, metrics, `planted_test_set`) and a rewritten
`mistakes_model.py` (`evaluate_settings`: GroupKFold by lap, 3 seeds, max_samples × max_features
by PR-AUC; #9 log format fixed). PR-AUC 0.219 on 2023 R1 #1 after the max_samples sweep below
(chance 0.022; 0.183 before).

Changes and findings:
- **Brake-onset check (asked by the user):** 83–93 % of flagged braking samples were in the first
  or last 0.2 s of a braking zone (base 20–28 %): chance deviations of single samples, since
  speed/throttle/brake arrive at ~4 Hz. Other spread definitions did not help; a **0.5 s centred
  rolling mean of the features** (`SMOOTH_WINDOW_S`) did. Adopted.
- **Early lift** is planted just before the driver's own lift point (drivers already lift a
  moment before braking, so "full throttle up to the braking point" never existed).
- **Contamination by F1 depends on the planted share (~2 %)**, so it is partly a property of the
  test. In the first phase-2 run F1 peaked at 0.03 (0.184 at 0.005, 0.260 at 0.03, 0.247 at 0.05). This led to
  decision 6 (fixed rate).
- **max_samples sweep (user request): how much data each tree uses.** 4096 was the largest first
  tried and won every time, so larger values were tested. Mean PR-AUC (max_features 1.0; each
  setting's search ~15 s, the same at every size):

  | Run (training rows per fold) | 4096 | 8192 | 16384 | 32768 | 65536 |
  |---|---|---|---|---|---|
  | 2023 R1 #1 (~29k) | 0.183 | 0.197 | 0.209 | **0.219** | 0.219 |
  | 2023 R1 #44 (~30k) | 0.185 | **0.192** | 0.191 | 0.191 | 0.191 |
  | 2024 R8 #16 (~32k) | 0.186 | 0.197 | 0.204 | 0.212 | **0.214** |
  | 2024 R1 Q #16 (~3.3k) | 0.254 | 0.254 | 0.254 | 0.254 | 0.254 |

  The ceiling is the training rows: above it the value is capped to "all rows", so results stop
  changing (qualifying is capped from 4096). No overfitting showed (PR-AUC on held-out laps never
  dropped by more than 0.001). This led to decision 5 (all rows); the meta file records
  `effective_max_samples` and `n_train_rows`.

Test: `test_mistakes_pipeline.py` also checks PR-AUC > 4× the planted share (measured ~8×).

### Phase 3: events, reasons, minimal AI-tab update
`mistakes_events.py` (new: `assign_events`, `build_events`), events JSON written by
`mistakes_io.save_outputs`, `event_id` and raw field-difference columns in the parquet, AI tab
updated (one marker per event, top-15 table with reasons, counts per lap; `brake_intensity` and
the "Under Braking" card removed).

Design choices:
- **Merge rule:** flags of the same lap less than 0.5 s apart (1 s since phase 8).
- **Ranking by severity = how unusual × how long**, not peak score: a single noisy sample often
  has a high peak (brake-onset noise, phase 2), so peak-only ranking would put noise first.
- **Reason = the feature with the largest mean deviation in units of its own spread** across the
  driver's samples (features have different scales); the text uses the raw differences in km/h,
  %, gears and the field's brake-on share.

First run (contamination chosen by F1, last lap included):

| Run | Flag rate (F1 choice) | Flagged samples | Events | Events ≥ 0.5 s |
|---|---|---|---|---|
| 2023 Bahrain race, #1 | 2 % | 733 | 273 | 33 |
| 2023 Bahrain race, #44 | 3 % | 1,131 | 333 | 59 |
| 2024 Monaco race, #16 | 2 % | 811 | 283 | 35 |

The top events were readable, but many were unusual rather than mistakes: the winners' last-lap
easing off was the #1 event for both #1 and #16; slipstream/DRS; lift-and-coast; repeated slow
exits at one corner; Hamilton's/Leclerc's pedal overlap (19–20 % of their events); a possible
gear glitch. This led to decisions 6 and 7. After both:

| Run | Laps | Flagged samples | Events | Events ≥ 0.5 s | PR-AUC | Precision / recall |
|---|---|---|---|---|---|---|
| 2023 Bahrain race, #1 | 49 | 718 | 262 | 30 | 0.217 | 0.254 / 0.380 |
| 2023 Bahrain race, #44 | 50 | 738 | 232 | 35 | 0.185 | 0.245 / 0.331 |
| 2024 Monaco race, #16 | 67 | 799 | 291 | 34 | 0.229 | 0.276 / 0.347 |

Tests: `unit/test_mistakes_events.py`; `test_mistakes_pipeline.py` checks every flagged sample is
in exactly one event, events are sorted by severity, and the last lap (57) is excluded.

### Phase 4: Z-score baseline
`mistakes_eval.py` gained the shared harness: `build_folds` (lap-grouped folds + planted test
sets, built once), `fold_metrics` / `summarise`, `ZScoreDetector` and `compare_detectors`.
`mistakes_model.py` has `ForestDetector` and its search uses the same helpers.
`detect_mistakes.py --compare` adds a `comparison` key to the meta file (planted metrics for both
detectors + agreement on the real laps); `mistakes_events.event_overlap` measures the agreement.
Also fixed: the injection wrote float64 values into float32 columns (pandas FutureWarning; would
fail in a future pandas). Results in §5.

Test: `test_mistakes_pipeline.py` runs with `compare=True` and checks both detectors use the same
planted share and flag rate and both score above 4× chance.

### Phase 5: final tests, outputs, write-up
Output-schema test added to `test_mistakes_pipeline.py` (replaces old A20). Saved
`data_lake/mistake_data/2023_1_R_1_*` regenerated with `--compare`. Logged #41: the MongoDB
leaderboard ranks by mistake rate, now ~2 % for every driver (to fix in the UI redesign).
Suite: 153 tests, all passing.

### Phase 6: review changes (2026-10-31)
The user's review asked for fewer, more believable events and readable locations:
- **Traffic filter** (decision 8). FastF1 has a `DRS` channel and `DriverAhead` /
  `DistanceToDriverAhead` per sample; the download kept neither, so they are read from the
  FastF1 cache at run time (the cleaned parquet's samples match FastF1's telemetry 100 % on
  `SessionTime`; no ETL change). When nobody is ahead, `DriverAhead` is empty and the distance is
  meaningless, so only samples with a car ahead count. FastF1's gap is an estimate (good enough
  for a 1 s rule).
- **Events of at least 0.5 s only** (decision 9; `mistakes_events.MIN_EVENT_S`).
- **Corner names** for every event (`location`; `mistakes_io.corner_positions` +
  `mistakes_events.location_label`). Our corner positions agree with FastF1's own corner
  distances to ~1 % (2023 Bahrain: Turn 1 at 709 m vs 717 m, Turn 15 at 4,928 m vs 4,975 m;
  ours run slightly short because they are measured along the X/Y line).
- **Reasons in sentence case** ("Off the throttle: …"); the AI-tab table shows Location instead
  of Position (m).

Before → after (Isolation Forest, planted-mistake test; qualifying is not filtered, so unchanged):

| Run | Samples left out | PR-AUC | Precision | Recall | F1 | Events |
|---|---|---|---|---|---|---|
| 2023 Bahrain race, #1 | 36 (0.1 %) | 0.217 → 0.245 | 0.254 → 0.269 | 0.380 → 0.406 | 0.303 → 0.319 | 262 → 31 |
| 2023 Bahrain race, #44 | 3,800 (10 %) | 0.185 → 0.194 | 0.245 → 0.257 | 0.331 → 0.368 | 0.276 → 0.299 | 232 → 27 |
| 2024 Monaco race, #16 | 6,610 (17 %) | 0.229 → 0.227 | 0.276 → 0.255 | 0.347 → 0.338 | 0.306 → 0.290 | 291 → 31 |
| 2024 Bahrain qualifying, #16 | — | 0.254 | 0.222 | 0.290 | 0.246 | 31 → 3 |

- The scores improved on the two Bahrain races and dropped slightly at Monaco; mean IF F1 0.283
  → 0.289, mean precision 0.249 → 0.251. On #1 almost nothing was left out (he led), so most of
  its gain comes from the cleaner field profile (the reference drivers' DRS/tow samples are left
  out too).
- The event count fell ~9× almost entirely because of the 0.5 s rule (before the filter there
  were already only 30 / 35 / 34 runs of ≥ 0.5 s).
- The Z-score baseline improved too (mean F1 0.249 → 0.265); the comparison in §5 uses the new
  numbers.
- Not verified: whether the remaining events are real mistakes (no labels; hand review in §6).

Tests: `unit/test_mistakes_events.py` (merge test extended to short runs, reasons capitalised,
new corner-name test); `test_mistakes_pipeline.py` (event test rewritten, new traffic test,
`location` in the schema test). `data_lake/mistake_data/2023_1_R_1_*` regenerated with `--compare`.
(The scores in this phase are the old per-sample ones; §4–§5 use per-mistake scores.)

### Phase 7: per-mistake scoring and time lost (2026-10-31)
- **Per-mistake scoring replaced the per-sample scores** (PR-AUC and per-sample precision/recall
  were the scikit-learn defaults, not a deliberate choice; a mistake is an event, so it should be
  counted once). Settings are now chosen by per-mistake F1; the selection did not change
  (max_features 1.0 in all four sessions).
- **Time lost** per event (`mistakes_timeloss.py`), events ranked by it, the saving label, and
  the real-data check against random spots. Two problems found while building it and fixed: a
  lap whose first sample sat just before the finish line was shifted by a whole lap (no time lost
  for most events), and the first coasting rule (any no-pedal sample) labelled ordinary slow-corner
  phases as saving (16 of 31 events on #1); coasting now needs ≥ 150 km/h for ≥ 0.3 s (5 events).
- Qualifying compares all kept laps with each other (every run is its own stint).
- Dashboard table: Location, Time lost ("+0.18 s (costly)"), reason with the saving note.

Tests: new `unit/test_mistakes_timeloss.py` (3); `test_mistakes_pipeline.py` updated (per-mistake
scores, ranking by time lost, new real-data check test). `data_lake/mistake_data/2023_1_R_1_*`
regenerated with `--compare`.
(The numbers in this phase used the 0.5 s merge gap; §4–§6 use the 1 s gap from phase 8.)

### Phase 8: wider merge gap, simpler table (2026-11-01)
- **Merge gap 0.5 s → 1 s** (`EVENT_GAP_S`), to raise recall: one planted mistake was often
  flagged in pieces, each shorter than the 0.5 s event rule. Tried on the 4 sessions (mean, IF):

  | Merge gap | Precision | Recall | F1 | Events per session | Median event length |
  |---|---|---|---|---|---|
  | 0.5 s (before) | 0.490 | 0.224 | 0.291 | 23 | 0.75 s |
  | **1.0 s (chosen)** | 0.512 | 0.273 | 0.343 | 28 | 0.81 s |
  | 1.5 s | 0.512 | 0.294 | 0.360 | 31 | 0.94 s |
  | 2.0 s | 0.517 | 0.318 | 0.381 | 35 | 1.21 s |
  | 3.0 s | 0.504 | 0.371 | 0.410 | 40 | 1.83 s |

  Recall keeps rising with the gap, but a wider gap starts joining separate moments (1 s is
  ~50 m at 200 km/h; 3 s is a whole corner sequence) into one "event", which a planted mistake
  then touches more easily. 1 s is the widest gap that still reads as one action. Z-score also
  improved (mean F1 0.240 → 0.284). The real-data costly share fell slightly (#1 19 % → 15 %,
  #44 22 % → 19 %) and is still above the random spots in Bahrain.
- **Dashboard:** the "(costly)" text was removed from the Time lost column (the threshold
  depends on each spot's lap-to-lap spread, so 0.09 s could be costly and 0.18 s not, which read
  as inconsistent; the flag stays in the JSON for the real-data check), and the table lists all
  events instead of the top 15.

Tests: the merge test in `unit/test_mistakes_events.py` now checks the 1 s rule (fails with
0.5 s). `data_lake/mistake_data/2023_1_R_1_*` regenerated with `--compare` (40 events).
