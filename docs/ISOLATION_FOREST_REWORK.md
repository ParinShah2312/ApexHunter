# Isolation Forest Rework — Design (roadmap §B)

**Course:** Data Science & Machine Learning · **Status:** agreed, ready to implement
**Also closes:** bugs #8 and #9 (remaining part), test-audit items A3, A4, A20.

---

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
| Tests | Mock-only tests replaced by real-model tests (A3); log test fixed (A4); schema test rewritten (A20); `test_pipeline_correctness` should pass. |
| Docs | PROJECT_DOCUMENTATION §5.2 rewritten with the new method, evaluation and comparison results. |

## 4. Plan (phases)

| Phase | Work | Checkpoint |
|---|---|---|
| 1 | Lap filtering, track projection, field reference, deviation features | On real data: braking samples no longer dominate the flags |
| 2 | Injection + evaluation harness; contamination/hyperparameter search | Selection no longer always picks the smallest contamination; PR-AUC reported |
| 3 | Events + reasons; outputs; minimal dashboard update | Bahrain 2023 race shows a short, readable event list |
| 4 | Z-score baseline in the same evaluation (autoencoder added in roadmap §D) | Comparison table in the docs |
| 5 | Tests, regenerate outputs, docs | Full test suite green (incl. the two currently failing integration files) |

Each phase ends with a check-in before the next.

## 5. Decisions

| # | Question | Decision |
|---|---|---|
| 1 | Reference for "normal driving" | **Median of the 5 fastest other drivers** (agreed). |
| 2 | Choose settings using planted fake mistakes (§2.4) | **Agreed.** |
| 3 | Comparison models | **Z-score baseline + Isolation Forest + LSTM autoencoder only** (§2.6). One-Class SVM, KNN, DBSCAN, LOF dropped. |
| 4 | Dashboard changes | **Only the minimum needed to check the rework worked** (event list and markers in the AI tab). The full redesign happens with the rest of the UI work (roadmap §E), at the end (agreed). |
