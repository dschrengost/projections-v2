# Archetype Feature Rollout & Training (2025-11-15)

## Artifacts built
- Generated minutes roles for seasons 2022-2025 via `projections.cli.build_minutes_roles`.
- Backfilled `data/gold/injury_features/season=*` to derive availability snapshots.
- Built archetype delta artifacts for 2022-2025 with `projections.cli.build_archetype_deltas`.
- Rebuilt all historical gold feature partitions (`data/gold/features_minutes_v1/season=*/month=*`) using the new archetype roles/deltas. Added `data/gold/features_minutes_v1/all_features.parquet` by concatenating every monthly parquet (≈131k rows).
- Live slice for 2025-11-14 located at `data/live/features_minutes_v1/2025-11-14/run=20251115T043815Z/`.

## Training runs
1. **minutes_archetype_20251115** (Dec-2024 only)
   - Command: `uv run python -m projections.models.minutes_lgbm --run-id minutes_archetype_20251115 --season 2024 --month 12 --data-root data --allow-guard-failure`
   - Guardrail issues: conditional P90 over-coverage for `bench|8-18`. Metrics stored at `artifacts/minutes_v1/minutes_archetype_20251115/metrics.json`.

2. **minutes_archetype_full_20251115** (full history 2022-10 through 2025-04 train/cal, 2025-10→2025-11-14 val)
   - Used consolidated `all_features.parquet`.
   - Command: `uv run python -m projections.models.minutes_lgbm --run-id minutes_archetype_full_20251115 --train-start 2022-10-01T00:00:00 --train-end 2025-02-28T23:59:59 --cal-start 2025-03-01T00:00:00 --cal-end 2025-04-30T23:59:59 --val-start 2025-10-01T00:00:00 --val-end 2025-11-14T23:59:59 --features data/gold/features_minutes_v1/all_features.parquet --data-root data --allow-guard-failure`
   - Resulting artifacts: `artifacts/minutes_v1/minutes_archetype_full_20251115/`.
   - **Poor metrics:** Validation MAE ≈ 6.55; playable Winkler 23.98; multiple conditional coverage failures (bench buckets and overall P10/P90). Guardrails flagged: `bench|8-18` P10=0.065 & P90=0.955, `bench|<8` P10=0.030, `bench|>18` P10=0.068 & P90=0.995, `starter|>18` P90=0.938, and overall validation P10=0.077 / P90=0.937.

## Scoring
- Re-scored 2025-11-14 live slate twice (old and full-history bundles) using `projections.cli.score_minutes_v1 --mode live` with `--features-path data/live/features_minutes_v1/2025-11-14/run=20251115T043815Z`.
- Latest scoring (full-history bundle) emitted to `artifacts/minutes_v1/daily/2025-11-14/run=20251115T043815Z/minutes.parquet`.

## Notes
- Rebuild run logs show schedule gaps for some 2024-2025 months (coach_tenure CSV missing). Guardrail failures mean further calibration/analysis is needed before promoting the full-history model.
