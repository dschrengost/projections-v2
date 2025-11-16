# 2025-11-14 Live Pipeline Refresh
- Added CLI ETLs for odds (`projections.etl.odds`), roster nightly (`projections.etl.roster_nightly` w/ NBA.com fallback + bronze writer), and box scores (`projections.etl.boxscores`).
- Schedule loader now deduplicates `game_id`s, so repeated silver parquets no longer break roster merging.
- Ran all three ETLs for 2025-11-14: odds produced an empty snapshot (Oddstrader had no data yet), roster fallback scraped 106 rows into `data/bronze|silver/roster_nightly/season=2025/month=11/`, and box scores froze 6,187 rows to `data/labels/season=2025/boxscore_labels.parquet`.
- Rebuilt the live pipeline for 2025-11-14 (`run=20251114T193000Z`) and scored it so downstream consumers can point at the latest run artifacts.

## 2025-11-14T18:45 run
- Injuries/odds/roster ETLs refreshed for Nov 14 (bronze+silver partitions).
- Built live features via `projections.cli.build_minutes_live --date 2025-11-14 --run-as-of-ts 2025-11-14T18:45:00 --history-days 30 --roster-fallback-days 1` → `data/live/features_minutes_v1/2025-11-14/run=20251114T184500Z/`.
- Scored with `projections.cli.score_minutes_v1 --mode live --run-id 20251114T184500Z` producing `artifacts/minutes_v1/daily/2025-11-14/run=20251114T184500Z/minutes.parquet` and updating `latest_run.json`.
