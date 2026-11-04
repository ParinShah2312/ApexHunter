# Driver Mistake Detection — Z-score vs Isolation Forest vs LSTM Autoencoder

**Courses:** DSML (Z-score, Isolation Forest) · Deep Learning (LSTM autoencoder) · **Status:** done
(Nov 2026). The full write-up with every design experiment and the dated history is in
[ARCHIVE/ISOLATION_FOREST_REWORK_full_2026-11-04.md](ARCHIVE/ISOLATION_FOREST_REWORK_full_2026-11-04.md).

**Files:** `backend/scripts/detect_mistakes.py`; `isolation_forest/` `mistakes_features.py`,
`mistakes_io.py`, `mistakes_model.py`, `mistakes_eval.py`, `mistakes_autoencoder.py`,
`mistakes_events.py`, `mistakes_timeloss.py`.

## 1. The idea

Last semester's detector compared absolute values, so ordinary braking looked unusual: **95 % of its
"mistakes" were just braking**. The new question for every telemetry sample is: *is the driver doing
something different from the fastest other drivers at this exact point on the track?*

## 2. Method

1. **Laps:** racing laps only (no lap 1, pit or safety-car laps, nothing slower than 107 % of the
   driver's best, not the last race lap). In races, samples with DRS open or < 1 s behind another
   car are left out (slipstream and dirty air are not mistakes).
2. **Reference:** the 5 fastest *other* drivers. Every sample is placed on the track (distance from
   the start line, by projection onto the fastest lap), and the field gives a profile per 10 m:
   typical speed, throttle, gear, acceleration, their spread, and how many brake there.
3. **Five features** (deviation from the field, divided by the field's spread at that point,
   smoothed over 0.5 s): speed deficit, throttle difference, brake mismatch, gear difference,
   acceleration difference.
4. **Detectors** (each trained on the driver's own clean laps; each flags its 2 % most unusual samples):
   - **Z-score:** flags a sample when any one feature is extreme. No machine learning.
   - **Isolation Forest:** 200 random trees; points isolated in few splits are unusual. Settings
     chosen with the planted-mistake test (§3).
   - **LSTM autoencoder:** reads ~2 s windows (16 samples); an encoder LSTM compresses the window,
     a decoder LSTM rebuilds it; what it cannot rebuild well is unusual. Windows never cross a lap.
5. **Events:** flags less than 1 s apart are joined; runs of at least 0.5 s become events, each with
   a corner name ("Approaching Turn 14"), a plain-language reason ("Off the throttle: 45 % less than
   the field") and the time lost against the driver's own nearby laps. Likely lift-and-coast / tyre
   saving is labelled and ranked last.

## 3. How we test it

There are no labelled mistakes in F1 data, so we use two checks:

1. **Planted mistakes (main score).** Laps are split into 5 folds (by lap). In each held-out lap we
   plant 3 fake mistakes of random types — lock-up, early lift, late throttle, pedal overlap,
   missed downshift — with 3 random seeds. **Recall** = planted mistakes found; **precision** =
   events that are on a planted mistake; **F1** combines them.
2. **Time lost on real laps.** Does an event really cost time against the driver's own laps,
   more often than 200 random spots on the same laps?

## 4. Results (4 sessions)

| Session | F1: Z-score / Isolation Forest / LSTM AE | Costly real events Z / IF / AE (random spots) |
|---|---|---|
| 2023 Bahrain race, Verstappen | 0.318 / 0.381 / **0.405** | 18 % / 15 % / 9 % (10 %) |
| 2023 Bahrain race, Hamilton | 0.320 / 0.324 / **0.436** | 16 % / 19 % / 19 % (12 %) |
| 2024 Monaco race, Leclerc | 0.285 / 0.317 / **0.433** | 0 % / 6 % / 2 % (10 %) |
| 2024 Bahrain qualifying, Leclerc | 0.199 / **0.338** / 0.337 | 0 of 3 / 1 of 3 / 1 of 4 (19 %) |
| **Mean** (precision / recall) | **0.280 / 0.340 / 0.403** (0.55/0.20, 0.51/0.27, 0.54/0.34) | |

Recall by mistake type (mean; Z / IF / AE): lock-up 0.22 / **0.43** / 0.41 · early lift 0.66 / 0.43 /
**0.79** · late throttle 0.13 / 0.13 / **0.23** · pedal overlap 0.37 / 0.14 / **0.49** · missed
downshift 0.06 / **0.20** / 0.10.

- **Machine learning helps:** the Isolation Forest beats the Z-score rule, mainly on mistakes that
  combine several small deviations (lock-ups, missed downshifts).
- **Deep learning helps more:** the LSTM autoencoder finds the most planted mistakes (best in 3 of
  4 sessions), especially mistakes that unfold over time (early lift, late throttle, pedal overlap).
- **Real laps are mixed:** events cost time more often than random spots in Bahrain, not at Monaco.
  "Different from the field" is not always "slower" (driving style, a different line).
- Changing the flags-to-events rule from a 0.5 s to a 1 s merge gap raised the Isolation Forest's
  recall from 0.22 to 0.27; removing the braking-dominated features cut braking flags from 95 % to
  37–54 % (braking is 15–29 % of samples).

Example (2023 Bahrain, Hamilton): Turn 10 on lap 38, +0.64 s, 16 km/h slower than his own laps.

## 5. Outputs and usage

```bash
python backend/scripts/detect_mistakes.py --session data_lake/clean_data/2023_1_R.parquet --driver 1 [--compare] [--force]
```

`data_lake/mistake_data/{session}_{driver}_mistakes.parquet` (samples with features, scores, flags,
event ids), `_events.json` (events, most costly first) and `_meta.json` (settings, scores; with
`--compare`, all three detectors). About 2 minutes per race with `--compare`. Outputs exist for the
four sessions above. The AI tab shows the events on a track map and in a table.

## 6. Decisions

- Contamination fixed at 2 % for every session, so counts compare across drivers.
- Isolation Forest uses all rows per tree; settings chosen by planted-mistake F1.
- The last race lap is left out (drivers ease off).
- The three detectors are compared side by side; there is no combined "both agree" output.
- Autoencoder settings (window 16, hidden 16, 20 epochs) are fixed, not searched.

## 7. Limitations

- Planted mistakes are fake: they show the detector finds these patterns, not that every real flag
  is a mistake.
- Only 4 sessions are scored, because each needs a full cross-validation run.
- About two thirds of planted mistakes are still missed, many because they are flagged for less
  than 0.5 s, which is not an event.
