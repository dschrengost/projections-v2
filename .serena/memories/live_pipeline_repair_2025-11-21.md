Summary of repairs on 2025-11-21:

- Fixed NBA boxscore ETL to zero-pad game IDs (already in repo), serialize raw payloads as JSON strings, write per-day bronze partitions, and rebuild legacy labels. Game date derivation now uses local tip timestamps; label coverage is 100% for 2025-10-21 through 2025-11-20.
- Updated build_minutes_live to load gold daily labels (gold/labels_minutes_v1) with legacy fallback, skip label_frozen gating for historical rows, and re-merge starter flags from roster. Trend features now populate because labels are no longer NaN.
- Updated roster_nightly to append to the monthly silver file instead of overwriting, preserving snapshots across runs.
- Updated build_minutes_labels to prefer local tip timestamps when deriving game_date and avoid UTC date shifts.
- Data rebuilds executed against PROJECTIONS_DATA_ROOT=/home/daniel/projections-data:
  - Re-scraped boxscores for 2025-10-21–2025-11-20 (bronze/boxscores_raw, labels/season=2025 refreshed).
  - Regenerated gold daily labels for the same window (gold/labels_minutes_v1/season=2025/*).
  - Built live features for 2025-11-21 at run_as_of_ts 2025-11-21T18:10Z → /home/daniel/projections-data/live/features_minutes_v1/2025-11-21/run=20251121T181000Z/.
  - Scored model bundle artifacts/minutes_v1/v1_full_calibration → artifacts/minutes_v1/daily/2025-11-21/run=20251121T181000Z/minutes.parquet (247 rows, 9 games).
- Known: roster silver still only contains 2025-11-21 snapshot; backfilling historical rosters would need another pipeline run with broader start/end if required.