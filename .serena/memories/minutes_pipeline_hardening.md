Implemented minutes_v1 pipeline hardening:
- Added CLI `projections.cli.build_minutes_labels` reading bronze boxscores partitions and writing gold labels to `data/gold/labels_minutes_v1/season=YYYY/game_date=YYYY-MM-DD/labels.parquet` with starter label derivation and schema validation.
- `build_features_minutes_v1` now accepts `--start-date/--end-date`, reads new gold labels partitions (falls back to legacy path), validates required feature columns, and still writes gold features under season/month.
- `build_starter_priors` gained `--cutoff-date` (default yesterday UTC) to limit features; filters partitions and feature rows before building priors/history.
- `score_minutes_v1` now hard-fails with clear errors when bundle dir, features, or starter priors are missing; exits on empty feature slices.
- Added live QC CLI `projections.cli.check_minutes_live` checking projections vs salaries and team-minute sanity, writing summaries to `artifacts/minutes_v1/live_qc/game_date=.../summary.json`.
- Added docs/pipeline/minutes_live_systemd.md with systemd service/timer templates (boxscore-labels, features-priors, live QC) and relevant env vars.
- New tests `tests/test_check_minutes_live.py` cover healthcheck CLI happy path and missing-player failure.