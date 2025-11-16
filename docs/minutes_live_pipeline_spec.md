% Minutes Live Pipeline – Design Spec

## 1. Context & Goals

- Serve pre-game projections for the current slate using the existing Minutes V1 LightGBM bundle.
- Reuse as much of the historical “gold” feature pipeline as possible so that live outputs stay schema-compatible with the trained model.
- Guarantee fast iteration **and** reproducibility: every run has an explicit `run_as_of_ts` (default = “now”) so we can reason about the information state and audit regressions.
- Provide deterministic, versioned disk artifacts under `data/live/` plus publishing to `artifacts/minutes_v1/daily/`, both keyed by run ID instead of silent overwrites.

## 2. Requirements

| Category | Requirement |
| --- | --- |
| Run identity / as-of semantics | CLI accepts `--run-as-of-ts` (ISO, default now). All snapshot rows must satisfy `snapshot_ts ≤ min(run_as_of_ts, tip_ts)` to avoid leakage. Directories, summaries, and downstream consumers include the run ID. |
| Data inputs | Need fresh schedule, injuries snapshot, odds snapshot, roster snapshot, optional coach static. For each snapshot we enforce both (a) membership in a configurable date window and (b) maximum age relative to `run_as_of_ts`. |
| Live label stubs | Historical labels provide trailing context. For target games we synthesize **live label stubs** (minutes=NA, starters inferred from roster) purely to keep the feature builder contract; label-based features only touch historical rows. |
| Feature schema | Must match `FEATURES_MINUTES_V1_SCHEMA` exactly so LightGBM bundles remain compatible. Live rows have `minutes = NA`. |
| Output location | `data/live/features_minutes_v1/<date>/run=<run_as_of_ts>/` with `{features.parquet, ids.csv, summary.json, coverage.csv}` plus a `latest` marker. |
| Scoring | `projections.cli.score_minutes_v1 --mode live --features-path .../run=<ts>/features.parquet` and writes scored outputs to `artifacts/minutes_v1/daily/<date>/run=<ts>/`. |
| Validation & parity | Enforce per-game anti-leak, dedupe keys, compute per-game freshness metrics, emit team-minutes sanity checks, and provide a parity-test mode that compares live builds against historical gold partitions for past dates.

## 3. High-Level Architecture

1. **Snapshot Acquisition**
   - Schedule: call existing NBA schedule scraper or read pre-built silver partitions keyed by `season`.
   - Injuries: fetch pdf snapshots (midnight window + intraday) and run `_build_injury_snapshot`.
   - Odds: call Oddstrader scraper for target date.
   - Roster: use last nightly export or re-run roster ETL near noon ET.
   - All snapshot fetchers should support “skip if already materialized” to avoid needless API calls when orchestrated.

2. **Historical Context Assembly**
   - Load frozen labels (`data/labels/season=YYYY/boxscore_labels.parquet`) for games < target day (optionally limit to N historical days for speed).
   - Deduplicate to last copy per `game/player/team`.

3. **Live Label Scaffolding**
   - Filter roster_nightly to target date. If empty, allow fallback to the most recent snapshot that satisfies both `--roster-fallback-days` and `--roster-max-age-hours`. Log source metadata.
   - Create rows with `minutes = NA`, `starter_flag = starter_flag_label = roster starter flag or 0`, `source = live_inference_roster`.
   - Align to `BOX_SCORE_LABELS_SCHEMA` and append to the historical labels frame.

4. **Feature Build**
   - Instantiate `MinutesFeatureBuilder` with schedule/injuries/odds/roster slices limited to the union of (historical + live) game_ids.
   - Filter each snapshot per-game to `snapshot_ts ≤ min(run_as_of_ts, tip_ts)` (no-leak rule).
   - Run `build()` and reuse the existing `_finalize`/schema enforcement path.
   - Keep the full schema (including `minutes`) so parity tests remain trivial; live rows will have `minutes = NA`.

5. **Live Output Writer**
   - Filter feature frame to `game_date == target_day`.
   - Write parquet + ids + summary to `data/live/features_minutes_v1/<date>/run=<run_as_of_ts>/`.
   - Summary captures `{date, run_as_of_ts, rows, roster_source_date, roster_snapshot_age_minutes, per-game injury/odds freshness stats, warnings}`. Coverage CSV captures team minutes sanity metrics.

6. **Scoring Path**
   - `score_minutes_v1 --mode live --features-path data/live/features_minutes_v1/<date>/run=<run_as_of_ts>/features.parquet`.
   - Skip starter derivation, skip historical-specific cleaning, but retain filtering of OUT players.
   - Write results to `artifacts/minutes_v1/daily/<date>/minutes.parquet` and `summary.json` as usual. Include a flag in the summary indicating `mode = "live"` + `features_source`.

## 4. CLI Contracts

### `projections.cli.build_minutes_live`

```
uv run python -m projections.cli.build_minutes_live \
  --date 2025-11-13 \
  --roster-fallback-days 1 \
  --history-days 30 \
  --out-root data/live/features_minutes_v1 \
  --data-root data
```

- Options for overriding each snapshot path (`--schedule-path`, `--injuries-path`, etc.).
- `--history-days` trims labels to reduce runtime.
- Exits non-zero with descriptive errors if snapshot coverage is missing or violates anti-leak constraints.

### `projections.cli.score_minutes_v1`

```
uv run python -m projections.cli.score_minutes_v1 \
  --date 2025-11-13 \
  --mode live \
  --features-path data/live/features_minutes_v1/2025-11-13/features.parquet \
  --artifact-root artifacts/minutes_v1/daily \
  --bundle-config config/minutes_current_run.json
```

- `--features-root` ignored in live mode; `--features-path` required.
- `--mode historical` remains default for backward compatibility.

## 5. Data & Storage Conventions

| Path | Contents |
| --- | --- |
| `data/live/features_minutes_v1/<date>/run=<run_as_of_ts>/features.parquet` | Full Minutes V1 schema; live rows have `minutes = NA`. |
| `data/live/features_minutes_v1/<date>/run=<run_as_of_ts>/ids.csv` | Unique key triples for quick diffing. |
| `data/live/features_minutes_v1/<date>/run=<run_as_of_ts>/summary.json` | `{date, run_as_of_ts, rows, snapshot_ages, fallback flags, mode}`. |
| `data/live/features_minutes_v1/<date>/run=<run_as_of_ts>/coverage.csv` | Team minutes sanity metrics, per-game snapshot coverage, etc. |
| `data/live/features_minutes_v1/<date>/latest` | Symlink or pointer to the run the dashboard should serve. |
| `data/live/snapshots/<kind>/<date>/run=<run_as_of_ts>/...` | Raw extracts for schedule/injuries/odds/roster. |
| `artifacts/minutes_v1/daily/<date>/run=<run_as_of_ts>/minutes.parquet` | Scored live projections consumed by dashboard/JSON appenders. |
| `artifacts/minutes_v1/daily/<date>/latest` | Pointer to the active run. |

## 6. Fallback & Guardrails

- **Roster fallback**: default 0 days (require same-day). CLI offers `--roster-fallback-days` and `--roster-max-age-hours` to bound both date and freshness. When fallback is used we record `roster_source_date`, `roster_snapshot_age_minutes`, and add `roster_game_date_source` column to the features.
- **Injury/Odds freshness**: enforce `snapshot_ts ≤ min(run_as_of_ts, tip_ts)` per game. Optional flags (`--allow-missing-injuries`, etc.) control whether we fail or insert sentinel rows when snapshot coverage is absent.
- **Schedule gaps**: raise error if any target-game lacks schedule coverage or `tip_ts`.
- **Per-game freshness metrics**: compute `as_of_age_minutes = run_as_of_ts - snapshot_ts` and `snapshot_to_tip_minutes = tip_ts - snapshot_ts` for injuries, odds, roster. Warn/fail when thresholds are exceeded and expose those metrics in the summary/coverage.
- **Parity mode**: `--parity-check-date YYYY-MM-DD --parity-run-as-of-ts ...` rebuilds a historical date using the live pipeline and diffs features against the canonical gold partition (ignoring columns known to differ). CI can run this nightly to ensure schema parity.

## 7. Observability & Monitoring

- Emit CLI logs summarizing counts per dataset, snapshot ages, fallback usage, parity results, and per-game freshness statistics.
- Append coverage CSVs under `reports/minutes_live/<date>/run=<run_as_of_ts>/` with:
  - Team minutes sanity metrics (sum of p50, counts above thresholds, flags outside [230, 260]).
  - Injury/odds snapshot coverage and freshness per game.
  - Derived metrics like `feature_as_of_ts_max` and `run_as_of_ts - feature_as_of_ts_max`.
- Persist raw snapshots for audit and for reproducing older runs.
- Run “next-day validation” jobs that join live scores to box-score labels to measure projection vs actual minutes so we can monitor calibration drift in production.

## 8. Automation Plan

1. **Morning job (~11:00 ET)**  
   - Refresh/pull schedule, injury, roster, odds snapshots; persist them under `data/live/snapshots/.../run=<run_as_of_ts>/`.  
   - Run `build_minutes_live --date today --run-as-of-ts 2025-11-13T16:00:00Z --roster-fallback-days 1 --roster-max-age-hours 18`.  
   - Score features and mark that run as `latest` for dashboards.

2. **Afternoon refresh (~16:00 ET)**  
   - Re-run snapshot fetch (delta mode).  
   - Rebuild features + score with a later `run_as_of_ts`.  
   - Dashboard always points to the newest run unless an operator pins a specific run ID.

3. **Overnight batch (~03:00 ET)**  
   - Ingest box scores, freeze labels, rebuild historical gold features.  
   - Optionally run the live builder in parity mode (run_as_of_ts = nightly freeze) to ensure schema parity vs gold partitions.

Scheduling can be handled via the existing orchestration layer (e.g., cron + bash runner or Prefect). Each job writes a small registry entry summarizing `{date, run_as_of_ts, dataset paths, warnings}`.

## 9. Open Questions

- Snapshot retention: how long do we keep `data/live/snapshots/...` before archiving?
- Dashboard UX: how do we surface run metadata (run_as_of_ts, fallback usage, snapshot ages) so operators know how “fresh” the data is? Badge? Banner? Date picker?
- Multi-slate handling: do we eventually need per-slate outputs (main/late) or is “daily superset” sufficient?
- Live calibration risk: do we need model adjustments when running earlier in the day? Answer via experiments described below.

## 10. Next Steps

1. Promote `run_as_of_ts` to a required CLI concept (argument parsing, artifact paths, summary metadata, per-game snapshot filtering).
2. Implement snapshot freshness guards (`--*-max-age-hours`), persist raw extracts, and document the NBA.com roster scrape (warn by default, optional enforcement flag).
3. Add parity-test mode + regression coverage comparing live builds vs gold partitions for one or more historical dates.
4. Extend coverage reporting with team-minutes sanity metrics and per-game snapshot freshness; wire warnings into CLI exit codes.
5. Update dashboard + automation scripts to read/write versioned run directories (and maintain `latest` pointers).
6. Run historical “what-if” experiments (e.g., run_as_of = tip-2h vs tip-30m) to quantify projection deltas and validate that the existing LightGBM bundle behaves when fed live features.

Once these pieces are in place, we can resume coding against a clear target and avoid ad-hoc adjustments.

---

### Implementation Checklist

1. [ ] **Run identity plumbing**
   - [ ] Add `--run-as-of-ts` argument to the live builder and scorer CLIs (default = current UTC).
   - [ ] Store artifacts under `data/live/.../<date>/run=<run_as_of_ts>/` and `artifacts/minutes_v1/.../<date>/run=<run_as_of_ts>/` with `latest` pointers.
2. [ ] **Snapshot freshness & persistence**
   - [ ] Enforce per-game snapshot filtration (`snapshot_ts ≤ min(run_as_of_ts, tip_ts)` and `run_as_of_ts - snapshot_ts ≤ max_age`).
   - [ ] Persist raw snapshot extracts under `data/live/snapshots/<kind>/<date>/run=<run_as_of_ts>/`.
   - [ ] Record snapshot ages/fallback metadata (including NBA.com roster scrape status and enforcement setting) in summaries and coverage CSVs.
3. [ ] **Live label stubs & historical context**
   - [ ] Ensure live label stubs (minutes=NA) are only used for schema compatibility and label-derived features use historical rows only.
   - [ ] Validate the impact of history window truncation (speed vs fidelity experiment).
4. [ ] **Parity & validation tooling**
   - [ ] Implement parity-test mode comparing live builds vs gold partitions for past dates.
   - [ ] Extend coverage reports with team-minutes sanity metrics and per-game freshness; wire thresholds into CLI warnings/errors.
5. [ ] **Scoring & automation integration**
   - [ ] Update scoring CLI, dashboards, and appenders to read/write versioned run directories and manage `latest` pointers.
   - [ ] Update automation scripts to emit per-run registry entries (date, run_as_of_ts, snapshot ages, warnings).
6. [ ] **Evaluation experiments**
   - [ ] Run historical “tip-2h / tip-30m” experiments to quantify projection deltas and confirm LightGBM robustness with live features.
