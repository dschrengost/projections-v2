# Medallion + Builders Leakage Audit (Draft)

Status: in progress. This document captures findings as they are confirmed.

## Findings (draft)

### 1) minutes_for_rates can leak post-lock minutes (High)
- Paths: `scripts/minutes/build_minutes_for_rates.py`, `gold/projections_minutes_v1/game_date=YYYY-MM-DD/minutes.parquet`
- Failure mode: `minutes_for_rates` is built from a flat gold minutes file that the live pipeline overwrites in-place, so historical training inputs can change after lock and are non-idempotent.
- Manifestation: re-running rates training yields different features for the same dates; post-lock minutes can leak into rates/fpts training.
- Fix: read run-scoped minutes (`gold/projections_minutes_v1/game_date=.../run=<id>/minutes.parquet`), stamp `minutes_run_id` in `minutes_for_rates`, and require a BuildPackage/snapshot id for training jobs.

### 2) Name-based joins remain a primary integrity risk (High)
- Paths: `projections/optimizer/player_pool_loader.py`, `projections/cli/score_ownership_live.py`, `projections/cli/finalize_projections.py`, `projections/api/optimizer_service.py`, `scripts/ownership/build_ownership_training_base.py`
- Failure mode: joins on normalized name + team can drop or mis-map players (diacritics, trades, duplicates), silently reducing projection coverage.
- Manifestation: missing high-salary players in optimizer pool, ownership/fpts mismatch, unstable join counts across runs.
- Fix: introduce a DK↔NBA id map and enforce minimum join coverage with hard failures; only use name joins as explicit fallback with warnings and thresholds.

### 3) Silver snapshot tables are overwrite/merge-in-place (Medium)
- Paths: `projections/etl/injuries.py`, `projections/etl/odds.py`, `silver/injuries_snapshot/.../injuries_snapshot.parquet`, `silver/odds_snapshot/.../odds_snapshot.parquet`
- Failure mode: monthly snapshots are merged in-place without run_id or immutable partitions, so historical rows can be replaced by later runs.
- Manifestation: repeated runs for the same date window can change training features without code changes; breaks lineage and reproducibility.
- Fix: move to append-only silver partitions (e.g., `run_ts=`) and materialize gold lock/pretip snapshots with explicit `as_of_ts` and `lock_ts`.

### 4) FPTS-only sim path removed; rates-only enforced (Info)
- Paths: `config/sim_v2_profiles.json`, `scripts/sim_v2/generate_worlds_fpts_v2.py`, `projections/sim_v2/config.py`
- Facts: production profiles (`baseline`, `sim_v3`, `sim_v3_novacancy`) set `mean_source: "rates"`, so sim uses minutes + rates. `run_sim_live` does not pass any FPTS run id.
- Fix: `mean_source` defaults to `"rates"` and non-rates values now hard-fail. The legacy FPTS-only branch is removed from sim generation.

### 5) Mixed partition naming + legacy fallbacks can select the wrong run (Medium)
- Paths: `artifacts/sim_v2/projections` (mix of `date=` and `game_date=`), `gold/projections_minutes_v1` (mix of `game_date=` and raw date folders), `projections/api/minutes_api.py`, `projections/api/optimizer_service.py`
- Failure mode: loaders scan multiple directory patterns and fall back to newest `run=` or flat files; if multiple producers write different layouts, the “latest” can diverge from minutes/rates run_id.
- Manifestation: optimizer/API shows a different sim run than the minutes run (or old FPTS gold artifacts), especially after a backfill or manual run.
- Fix: standardize partitions (`game_date=` + `run=`), deprecate legacy layouts, and require explicit run_id in all loaders.

### 6) gold/projections_minutes_v1 has no run_id in rows (Low/Med)
- Paths: `gold/projections_minutes_v1/game_date=YYYY-MM-DD/minutes.parquet`, `scripts/run_live_score.sh`
- Failure mode: run scoping exists in the path but rows only carry `model_run_id`; downstream joins can’t verify run alignment inside the data.
- Manifestation: mismatched minutes/sim merges not detectable when paths are copied or combined.
- Fix: add `minutes_run_id` and `run_as_of_ts` to minutes parquet rows on write.

### 7) Legacy FPTS gold outputs still referenced by some consumers (Medium)
- Paths: `projections/api/minutes_api.py`, `projections/optimizer/player_pool_loader.py`, `gold/projections_fpts_v1/*`
- Failure mode: API/optimizer can still pull `gold/projections_fpts_v1` when unified/sim outputs are missing, but this path is no longer produced by the live sim pipeline.
- Manifestation: stale or missing FPTS in API/UI/optimizer when sim outputs are present but in new artifact paths.
- Fix: remove gold FPTS fallbacks or replace with unified projections + sim_v2 artifacts only.

### 8) Live-score manifests drift from actual outputs (Low/Med)
- Paths: `manifests/YYYY-MM-DD/live-score/*.json`
- Failure mode: manifests list outputs like `gold/sim_v2/game_date=...` that are not written by the current sim pipeline; manifests don’t record run_id or run_as_of_ts.
- Manifestation: BuildPackage lineage reports are misleading; difficult to reproduce runs from manifests alone.
- Fix: update manifest writer to record actual artifact paths (artifacts/sim_v2/worlds_fpts_v2, artifacts/projections) + run_id + run_as_of_ts.

## Medallion Inventory (draft)

### Bronze
- `bronze/injuries_raw/season=YYYY/date=YYYY-MM-DD/hour=HH/injuries.parquet`: grain (game_id, player_id, as_of_ts), columns include status, report_date, as_of_ts, ingested_ts; stable ids present.
- `bronze/odds_raw/season=YYYY/date=YYYY-MM-DD/run_ts=.../odds.parquet`: grain (game_id, as_of_ts); contains odds + as_of_ts/ingested_ts. Also a date-level `odds.parquet` exists (latest overwrite).
- `bronze/roster_nightly_raw/season=YYYY/date=YYYY-MM-DD/roster.parquet`: grain (game_id, player_id, as_of_ts); no run partition.
- `bronze/daily_lineups/season=YYYY/date=YYYY-MM-DD/daily_lineups_raw.parquet`: grain unclear, minimal columns (date + ingested_ts) → relies on downstream silver normalization.
- `bronze/boxscores_raw/season=YYYY/date=YYYY-MM-DD/boxscores_raw.parquet`: grain (player/game) but only `tip_ts` timestamp; no as_of.
- `bronze/dk/draftables/*.json`: flat JSON snapshots (no partition metadata).

### Silver
- `silver/injuries_snapshot/season=YYYY/month=MM/injuries_snapshot.parquet`: grain (game_id, player_id, as_of_ts); monthly overwrite, no run_id.
- `silver/odds_snapshot/season=YYYY/month=MM/odds_snapshot.parquet`: grain (game_id, as_of_ts); monthly overwrite.
- `silver/roster_nightly/season=YYYY/month=MM/roster.parquet`: grain (game_id, player_id, as_of_ts); includes lineup_timestamp; monthly overwrite.
- `silver/nba_daily_lineups/season=YYYY/date=YYYY-MM-DD/lineups.parquet`: grain (game_id, player_id, lineup_timestamp); daily partition.
- `silver/schedule/season=YYYY/month=MM/schedule.parquet`: grain (game_id); includes tip_ts/tip_local_ts.
- `silver/espn_injuries/date=YYYY-MM-DD/injuries.parquet`: grain (player_id, as_of_ts) for ESPN outs.
- `silver/ownership_predictions/YYYY-MM-DD/<draft_group_id>.parquet`: includes run_id + model_run in rows but stored in flat date folder (run-scoped dirs not yet present in data).

### Gold
- `gold/features_minutes_v1/season=YYYY/month=MM/features.parquet`: grain (game_id, player_id); includes injury_as_of_ts, odds_as_of_ts, roster_as_of_ts, lineup_timestamp, feature_as_of_ts.
- `gold/labels_minutes_v1/season=YYYY/game_date=YYYY-MM-DD/labels.parquet`: grain (game_id, player_id); includes label_frozen_ts.
- `gold/projections_minutes_v1/game_date=YYYY-MM-DD/minutes.parquet`: grain (game_id, player_id); includes model_run_id + odds_as_of_ts but no minutes_run_id/run_as_of_ts.
- `gold/minutes_for_rates/season=YYYY/game_date=YYYY-MM-DD/minutes_for_rates.parquet`: grain (game_id, player_id); no as_of/run metadata.
- `gold/rates_training_base/season=YYYY/game_date=YYYY-MM-DD/rates_training_base.parquet`: grain (game_id, player_id); no as_of/run metadata.
- `gold/rates_v1_live/YYYY-MM-DD/run=.../rates.parquet`: grain (game_id, player_id); run id only in path.
- `gold/ownership_*_base/*.parquet`: single-file training bases (no run_id/as_of fields).
- `gold/slates/season=YYYY/game_date=YYYY-MM-DD/game_id=*/{lock,pretip}.parquet`: includes snapshot_ts, tip_ts, injury/odds/roster as_of fields; immutable snapshots.
- `gold/projections_fpts_v1/*`: legacy FPTS outputs; used by some consumers but not produced by current sim pipeline.

### Artifacts / Live
- `live/features_minutes_v1/YYYY-MM-DD/run=.../features.parquet` + `latest_run.json`: run-scoped live feature slices with feature_as_of_ts.
- `live/features_rates_v1/YYYY-MM-DD/run=.../features.parquet` + `latest_run.json`: run-scoped rates features (uses minutes run id).
- `artifacts/minutes_v1/daily/YYYY-MM-DD/run=.../minutes.parquet` + `latest_run.json`: scored minutes outputs.
- `artifacts/sim_v2/worlds_fpts_v2/game_date=YYYY-MM-DD/run=.../projections.parquet`: sim outputs include minutes_run_id + rates_run_id in rows.
- `artifacts/sim_v2/projections/{date=...,game_date=...}/run=...`: legacy aggregator output; mixed partition naming.
- `artifacts/projections/YYYY-MM-DD/run=.../projections.parquet`: unified outputs (minutes+sim+ownership) with run in path only.
- `builds/optimizer/YYYY-MM-DD/*.json`: build packages missing run_id in config (often null).

## FPTS-only Model Audit (draft)

### Is FPTS feeding production sim -> projections?
No. The live path uses `scripts/sim_v2/run_sim_live.py` with profiles from `config/sim_v2_profiles.json`, which all specify `mean_source: "rates"`. In that mode, `scripts/sim_v2/generate_worlds_fpts_v2.py` builds means from minutes + rates and does not invoke the old FPTS model or `gold/fpts_training_base`.

### Where the FPTS-only path existed (now removed)
- The legacy branch in `scripts/sim_v2/generate_worlds_fpts_v2.py` has been removed, and `projections/sim_v2/config.py` now rejects `mean_source != "rates"`.

### Legacy FPTS consumers still present
- `projections/api/minutes_api.py` and `projections/optimizer/player_pool_loader.py` still read from `gold/projections_fpts_v1` when unified or sim outputs are missing.
- The live sim path now writes `artifacts/sim_v2/worlds_fpts_v2` + `artifacts/projections`, not `gold/projections_fpts_v1`.

## Lineage (draft)

- Bronze: `injuries_raw`, `odds_raw`, `daily_lineups`, `roster_nightly_raw`, `boxscores_raw`
- Silver: `injuries_snapshot`, `odds_snapshot`, `roster_nightly`, `nba_daily_lineups`, `schedule`, `espn_injuries`
- Gold (training): `labels_minutes_v1`, `features_minutes_v1`, `rates_training_base`, `fpts_training_base`, `minutes_for_rates`
- Live: `live/features_minutes_v1` -> `artifacts/minutes_v1/daily` -> `gold/projections_minutes_v1` -> `gold/rates_v1_live` -> `artifacts/sim_v2/projections` -> `artifacts/projections` (unified)

## Builder / Model Dataflow (draft)

### Minutes model
- Training: `live/features_minutes_v1/*/run=*/features.parquet` + `gold/labels_minutes_v1/*/labels.parquet` -> `training/snapshots_minutes_v1/run=*/features.parquet` (`feature_as_of_ts <= tip_ts` by default) -> `projections/cli/train_minutes_dual.py` -> `artifacts/minutes_lgbm/<run_id>`.
- Live inference: `projections/cli/build_minutes_live.py` reads `silver/{injuries_snapshot,odds_snapshot,roster_nightly}` + `silver/schedule`, filters by run_as_of_ts and tip_ts, writes `live/features_minutes_v1/YYYY-MM-DD/run=<id>` + `latest_run.json` (run_as_of_ts recorded). `projections/cli/score_minutes_v1.py` scores features, writes `artifacts/minutes_v1/daily/YYYY-MM-DD/run=<id>` + `latest_run.json`; `scripts/run_live_score.sh` mirrors to `gold/projections_minutes_v1`.
- As-of enforcement: build_minutes_live filters snapshots by as_of_ts <= run_as_of_ts and <= tip_ts; score_minutes_v1 filters ESPN injuries by run_as_of_ts when provided.
- Gaps: `gold/projections_minutes_v1` rows lack `minutes_run_id`/`run_as_of_ts` (path-only).

### Rates model
- Training: `scripts/minutes/build_minutes_for_rates.py` reads `gold/projections_minutes_v1` (flat) -> `gold/minutes_for_rates` (no run_id). `scripts/rates/build_training_base.py` joins boxscores + labels + odds_snapshot (tip-aware) + injuries_snapshot (as_of <= tip) + roster_nightly (latest by as_of, not tip-aware) + minutes_for_rates -> `gold/rates_training_base`. `scripts/rates/train_rates_v1.py` trains models and writes `artifacts/rates_v1/runs/<run_id>`; `config/rates_current_run.json` points to production.
- Live inference: `projections/cli/build_rates_features_live.py` loads `live/features_minutes_v1` (run-aligned via latest_run.json), adds season aggregates/tracking/vacancy from `gold/rates_training_base`, writes `live/features_rates_v1/YYYY-MM-DD/run=<id>`. `projections/cli/score_rates_live.py` scores model and writes `gold/rates_v1_live/YYYY-MM-DD/run=<id>` + `latest_run.json`.
- Gaps: rates live outputs do not carry run_id inside rows; training base lacks as_of/run metadata.

### Sim / worlds
- Live sim: `scripts/sim_v2/run_sim_live.py` -> `scripts/sim_v2/generate_worlds_fpts_v2.py` uses minutes + rates runs, outputs `artifacts/sim_v2/worlds_fpts_v2/game_date=.../run=<id>` with `minutes_run_id` + `rates_run_id` columns.
- Legacy aggregation: `scripts/sim_v2/aggregate_worlds_to_projections.py` writes `artifacts/sim_v2/projections/date=.../run=<id>`; mixed `date=` vs `game_date=` layouts still exist.

### Ownership model
- Training: `scripts/ownership/build_ownership_training_base.py` uses Linestar CSVs and injury name joins; writes `gold/ownership_training_base` (single file; no run_id/as_of).
- Live inference: `projections/cli/score_ownership_live.py` loads sim projections for run_id (or latest/cutoff), maps DK names to NBA ids using minutes run, filters injuries by cutoff_ts <= lock, writes `silver/ownership_predictions/YYYY-MM-DD/<draft_group_id>.parquet` and locked versions; run_id stored in rows.
- Gaps: name-based joins persist; on-disk outputs not yet run-scoped in data.

### Unified projections + optimizer/API
- `projections/cli/finalize_projections.py` merges minutes + sim + ownership by run_id and writes `artifacts/projections/YYYY-MM-DD/run=<id>/projections.parquet` (no run_id columns).
- Optimizer/API loads unified projections when available; otherwise falls back to sim + minutes; both use `latest_run.json` when run_id missing. Legacy `gold/projections_fpts_v1` is still referenced in some loaders.

## Per-Model Leakage/Lineage Notes (draft)

### Minutes
- Inputs + joins: minutes features join on (game_id, player_id, team_id); labels join on same keys.
- As-of enforcement: build_minutes_live uses snapshot as_of_ts and tip_ts cutoffs; training snapshots drop post-tip features by default.
- Run_id propagation: run_id in path + latest_run.json only; no `minutes_run_id` in rows.
- Non-determinism: deterministic given seedless pipeline; variability comes from upstream snapshot timing.

### Rates
- Inputs + joins: rates_training_base joins boxscores + labels + odds_snapshot (tip-aware) + injuries_snapshot (as_of<=tip) + roster_nightly (latest by as_of, not tip-aware) + minutes_for_rates.
- As-of enforcement: odds/injuries tip-aware in training base; roster not tip-aware; live features use minutes snapshots but do not record as_of_ts.
- Run_id propagation: training base has no run_id; live outputs have run_id only in path + latest_run.json.
- Non-determinism: training deterministic given inputs; leakage risk from mutable minutes_for_rates + roster snapshots.

### Sim/worlds
- Inputs + joins: sim_v2 joins minutes + rates on (game_date, game_id, team_id, player_id); outputs carry minutes_run_id + rates_run_id columns.
- As-of enforcement: inherits minutes/rates run selection; no explicit as_of_ts in sim outputs.
- Run_id propagation: run_id in path + latest_run.json, plus run_id columns for minutes/rates in rows.
- Non-determinism: stochastic sampling but deterministic with fixed seed; run_id does not affect seed by default.

### Ownership
- Inputs + joins: sim projections (player_id) + DK salaries (player_name) mapped via minutes name join; injuries_snapshot filtered by lock cutoff.
- As-of enforcement: injuries filtered by as_of_ts <= lock cutoff; ownership history filtered by game_date < current date; no run_as_of_ts for salary feed.
- Run_id propagation: run_id stored in rows but not used in storage layout (flat date dir).
- Leakage risk: if lock cache missing, locked slates can be rescored with post-lock data.

### Unified / Optimizer
- Inputs + joins: minutes + sim join on ids; ownership join by normalized player_name; DK salaries joined by normalized names.
- As-of enforcement: relies on upstream run_id selection; no in-row run_id to validate.
- Leakage risk: `latest_run.json` fallbacks can mix runs across minutes/rates/sim/ownership.

## Training vs Live Divergence (draft)
- Minutes: training snapshots sourced from live feature runs with optional post-tip drop; live features are built from silver snapshots. Alignment is good, but training depends on feature_as_of_ts <= tip_ts while live runs can be repeated with different run_as_of_ts (feature drift across runs).
- Rates: training uses `minutes_for_rates` built from `gold/projections_minutes_v1` (flat/overwrite). Live uses minutes feature runs (`live/features_minutes_v1`) for rates features. This mismatch + mutable minutes_for_rates can leak post-lock minutes into training.
- Ownership: training features derived from Linestar CSVs + injury name joins; live features derive from sim projections + salaries + minutes mapping. Feature distribution mismatch can lead to weak calibration and leakage if lock cache missing.
- Sim: no training data per se; stochastic generation depends on minutes/rates runs and calibrated noise params (rates_noise/minutes_noise) that are not run-scoped to a slate.

## Open Questions
- What is the authoritative BuildPackage root (e.g., `data_root/builds` vs `data_root/gold/slates`)?
