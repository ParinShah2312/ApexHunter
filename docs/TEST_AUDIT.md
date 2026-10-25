# Test-Suite Audit (roadmap #15)

Audit of every test file against one rule: **a test must call the project's real code and assert on
its real behaviour.** Checked for: tests that re-implement logic instead of calling it, assertions
that cannot fail, expectations left over from older code, mocks that replace the thing under test,
and important behaviour with no test at all.

Started October 2026. Status legend: ✅ fixed · ⏳ to do · 🔗 handled by another roadmap item.

## 1. Summary

- 39 test files (31 unit, 8 integration), 249 tests. The full run has 4 failures, all known
  (`test_pipeline_schema` ×2 stale, `test_pipeline_correctness` ×2 catching the real Isolation
  Forest tuning bug).
- Line coverage of `backend/scripts` + `frontend`: **56 %** (coverage.py, unit + integration).
  Coverage under-reports the dashboard files, because AppTest executes them in a way coverage can't
  attribute, but is accurate for the backend.
- 10 files written or rewritten during the October 2026 bug fixes already follow the rule:
  `test_script_entrypoints`, `test_tyre_training`, `test_tyre_lap_filtering`,
  `test_frontend_ai_analysis`, `test_project_config`, `test_frontend_selectboxes`,
  `test_turn_windows`, `test_geometry_classification`, `integration/test_tyre_pipeline`,
  `integration/test_turn_windows_bahrain`.

## 2. Findings in existing tests

| # | File | Problem | Kind | Status |
|---|---|---|---|---|
| A1 | `unit/test_frame_selection_counts` | Asserts 3–33 frames selected; any broken selector passes. Correct answer is exactly 11 per video (33). | Can't fail | ✅ |
| A2 | `unit/test_features_deltas` | Speed-delta clipping tested at +50 only, not −50. | Gap | ✅ |
| A3 | `unit/test_model_grid_search`, `test_model_training`, `test_model_inference` | Every test mocks `IsolationForest`; they only check call counts and kwargs. No test runs a real model, so detection quality and contamination choice are never tested (the "always smallest contamination" bug is invisible). | Mocks the subject | 🔗 §B |
| A4 | `unit/test_model_grid_search::test_grid_search_logs_each_contamination` | Asserts the 2-decimal log format that is bug #9 (0.005 printed as "0.01"); would fail if the bug were fixed. | Encodes a bug | 🔗 #9 / §B |
| A5 | `unit/test_io_saving` | Checks files exist but never reads the saved parquet back; `mistake_rate_pct` value never checked. | Weak | ✅ |
| A6 | `unit/test_racing_line_search` | Every search graph has uniform costs, so A\*/Dijkstra vs BFS behave identically; nothing shows cost-aware search avoiding expensive (braking) cells. | Gap | ✅ |
| A7 | `unit/test_racing_line_io::test_compute_time_saved_returns_float_or_none` | The function always returns float or None. | Can't fail | ✅ |
| A8 | `unit/test_racing_line_grid::test_compute_coordinate_scale_returns_float` | Checks 0.001 ≤ scale ≤ 1, which the code clamps to anyway. No known-geometry check. | Can't fail | ✅ (estimator later removed by #18; replaced by a known-circle lap-length test) |
| A9 | `unit/test_racing_line_grid::test_get_nearest_node_finds_closest` | Queries a cell that is itself a node, so the answer is trivially itself; the off-grid case (the real use) is untested. | Trivial | ✅ |
| A10 | `unit/test_racing_line_grid`, `integration/test_racing_line_pipeline` | Random speeds/brakes with no seed → non-deterministic tests. | Flaky risk | ✅ |
| A11 | `unit/test_racing_line_search` | `run_full_lap` (stitching the 4 lap segments) untested; deviation only tested for identical paths. | Gap | ✅ |
| A12 | `integration/test_racing_line_pipeline::test_bfs_nodes_expanded_gte_astar` | Asserts BFS expands ≥ A\* nodes. Not a true property: on the real Bahrain run BFS expanded 2,201 vs A\* 2,587. Passes on random synthetic data by luck. | Wrong claim | ✅ |
| A13 | `unit/test_tyre_data::test_insufficient_laps_returns_empty`, `test_telemetry_too_sparse_skips_lap` | Since lap filtering (#13) removes lap 1, both return `[]` regardless of the rule they claim to test. | Stale / passes for wrong reason | ✅ |
| A14 | `unit/test_tyre_data::TestConstants` (4 tests) | Assert that constants are positive / lists of strings. Cannot catch a real bug. | Trivial | ✅ |
| A15 | `unit/test_tyre_data` | `_build_synthetic_df()` (2,000-row fixture) is defined but never used; no test checks the per-lap aggregated values (e.g. `mean_speed`). | Dead code / gap | ✅ |
| A16 | `unit/test_tyre_io::TestPredictionOutput` | Builds a dict, wraps it with `build_prediction_output`, then checks it has the keys it was given. | Circular | ✅ |
| A17 | `unit/test_tyre_model` | Only checks types/shapes; no test shows the LSTM can learn anything. | Gap | ✅ |
| A18 | `integration/test_pipeline_edge_cases::test_single_row_driver` | Wrapped in `try/except SystemExit: pass` with no assertion — passes whatever happens. | Can't fail | ✅ |
| A19 | `integration/test_pipeline_edge_cases::test_empty_driver_data_after_filter` | Exact duplicate of `test_driver_not_in_session`. | Duplicate | ✅ |
| A20 | `integration/test_pipeline_schema` | Expects meta keys that never existed (`session`, `model_type`) and the old contamination grid. | Stale | 🔗 #8 / §B |

Files reviewed with no significant problems: `test_features_contract`, `test_features_flags`,
`test_features_normalization`, `test_frame_selection_edge`, `test_frame_selection_names`,
`test_geometry_distance`, `test_geometry_wheels`, `test_io_loading`, `test_telemetry_cleaning`,
`test_telemetry_dtypes`, `test_telemetry_synthesis`, `integration/conftest`,
`integration/test_pipeline_outputs`, `integration/test_pipeline_skip_force`,
`integration/test_pipeline_correctness` (it fails because it correctly catches a real bug).

### Step 1 result (October 2026)

All non-Isolation-Forest findings (A1, A2, A5–A19) fixed. Overlapping files were merged
(3 frame-selection files → `test_frame_selection.py`; 4 feature files → `test_features.py`;
2 Isolation Forest I/O files → `test_mistakes_io.py`) and trivial/duplicate tests removed. The
suite went from 249 to 230 tests, every one checking real behaviour, and coverage rose from 56 % to
57 %. Mutation checks (deliberately breaking the code) confirmed the new tests fail where the old
ones did not, e.g. removing negative speed-delta clipping passed the old feature tests but fails
the new ones. Remaining open: A3, A4, A20 (Isolation Forest, rewritten in roadmap §B).

### Round 2 result (October 2026): keep only essential tests

Rule applied (now in ROADMAP §3): every test must guard real behaviour of a main feature and would
fail on a realistic regression. Padding (trivial shape/type/file-exists checks, unchanged
behaviour, near-duplicates, tests of test tooling) was removed. Bug fixes from the October review
are proven mainly by real-data runs recorded in PROJECT_DOCUMENTATION §8, so removing a unit test
does not remove that evidence.

| Removed | Tests | Why |
|---|---|---|
| Old Isolation Forest unit tests: `test_features`, `test_model_grid_search`, `test_model_inference`, `test_model_training`, `test_mistakes_io` | 38 | Pin the old, flawed method that the rework replaces (19 are mock-only, A3; one asserts the "target can be its own reference" bug). The rework writes new real-model tests (ISOLATION_FOREST_REWORK §0.5). |
| Old Isolation Forest integration tests: `test_pipeline_correctness`, `test_pipeline_schema`, `test_pipeline_outputs` | 21 | Two files failed (stale schema A20; correctness catches the real tuning bug); all check the old features/meta. Replaced in the rework. Kept: `test_pipeline_edge_cases`, `test_pipeline_skip_force` (behaviour the rework keeps). |
| Telemetry dtype checks (5 of 6 merged into one), `.snappy`/directory-size helpers (3) | 8 | One test now checks all six columns; helpers are trivia. |
| Dashboard: "slower pace is red" (unchanged behaviour), empty-frame rate, card millisecond rounding, rendered-sidebar duplicate of the static selectbox scan, empty overlap input | 5 | Edge cases or duplicates; the dashboard is being redesigned (§E). |
| Test runner "fully skipped file" output; LSTM output shape; stint index / empty laps; JSON save round trip; wheel ints | 6 | Tooling or trivial. |
| Racing-line integration "file created" / "JSON valid"; tyre integration "training completes", "config keys", "output exists", "floats" | 6 | Implied by the remaining tests (the fixture fails if the file is missing). |

Result: **168 tests** (24 unit files / 140 tests, 5 integration files / 28 tests), all passing,
none skipped, down from 252 tests with 2 failing files. A3, A4 and A20 are closed by deletion;
their replacements are part of the Isolation Forest rework.

### Round 3 result (October 2026): no mock tests

Of the 168 tests, 12 replaced real behaviour with mocks, and the racing-line integration fixture
faked FastF1 for its whole file. Decision (user): no mock tests.

| Change | Tests | Detail |
|---|---|---|
| Racing-line integration rewritten on real data | 11 → 4 | Runs the real pipeline on `2023_1_Q.parquet`, car #1, with the real FastF1 cache; checks lap 14 / 89.708 s, 66 path points, scale 0.1 and ~5.34 km, time estimate within 2 % of the real lap, A\* = Dijkstra < BFS. |
| LSTM integration rewritten on real data | 7 → 4 | Trains a small model (temp folder, never `models/`) on real 2024 R1–R3 races, holds out R4, predicts car #1 in R4 with real FastF1 laps; checks the session split, saved feature list, prediction windows, and predicted lap times within 10 % of the real ones. |
| Deleted (mocked FastF1, MC dropout, file processing or test-runner internals) | 10 | `test_racing_line_io` fastest-lap lookup (4, now covered by the real integration test), `test_project_config` cache and log-encoding (3), `test_test_runner` (1), LSTM cliff rule (1, revisited in ROADMAP §D), chronological processing (1, `session_sort_key` test kept). |
| Deleted (unnecessary) | 3 | Index-split fallback (dead path), duplicate empty-contour test, turn-label suffix. |

Kept on purpose: `test_frame_selection.py` patches module *constants* (the frames/output folders
and season list) so the real selection code runs on real files in a temp folder. No behaviour is
faked; only paths are redirected.

Result: **145 tests** (23 unit files / 127 tests, 5 integration files / 18 tests), all passing,
none skipped. The integration tests need the local `data_lake/` and `cache/` and are skipped
(not failed) without them.

## 3. Untested code (coverage 0 %)

Decision (user, Oct 2026): YOLO/CV inference and Big Data code are last semester's frozen work;
**no new tests will be written for them.** The list is kept for reference only.

| Code | Notes | Status |
|---|---|---|
| `cv/inference_masking.py`, `cv/inference_hud.py`, `cv/inference_io.py`, `cv/run_inference.py` | The whole YOLO inference loop. Testable with synthetic masks/frames and a stubbed YOLO. | Not planned |
| `predict_cliff.py` | LSTM prediction orchestrator (end to end, after the retrain the model loading path is exercised for real). | Not planned |
| `cv/extract_frames.py` | Testable with a tiny generated video. | Not planned |
| `etl/fix_parquet_timestamps.py` | Small, testable. | Not planned |
| `bigdata/hdfs_manager.py`, `bigdata/mongo_manager.py` (18 %), `bigdata/spark_clean_telemetry.py` | Need Hadoop/MongoDB/Spark. Pure helpers (parsing `hadoop fs -ls` output, status-log writing, summary statistics) can be tested with mocked subprocess/collections. | Not planned |
| `etl/download_season_data.py`, `etl/download_manual_videos.py`, `etl/download_satellite_images.py` | Network downloads; testable only with mocks. Low priority. | Not planned |

## 4. Bugs found by the audit (and while fixing it)

| Roadmap # | Bug |
|---|---|
| 16 ✅ | `cv/select_training_frames.py` takes every `len // 11`-th frame then the first 11, so the end of each video is never sampled (~last 10 s of a 460-frame video; videos under 22 frames get only their first 11 frames). The docs promise evenly spaced frames. |
| 17 ✅ | `racing_line/racing_line_grid.find_start_end_nodes` maps rows to grid cells using the min X/Y of samples with Speed > 10, but `build_grid` uses the min of all samples, so start cells can be shifted. (The function was later removed as unused by #19.) |
| #9 (part) ✅ | Log lines containing `→` were dropped with a "Logging error" when stdout is redirected on Windows (cp1252); fixed project-wide in `utils.setup_logger`. |
| — ✅ | `racing_line_io.fetch_fastest_lap_bounds` used FastF1's deprecated `pick_driver` (to be removed); switched to `pick_drivers`. |
