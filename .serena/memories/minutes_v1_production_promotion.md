# Minutes V1 Production Promotion (Nov 22 2025)

## Summary
- Promoted LightGBM run `lgbm_full_v1_vol` (features with volatility/regime signals) to the production minutes bundle shared by LightGBM and NN experiments.

## Key Changes
1. Added `config/minutes_current_run.json` pointing at `artifacts/minutes_lgbm/lgbm_full_v1_vol`.
2. Introduced `projections/minutes_v1/production.py` with `resolve_production_run_dir` + `load_production_minutes_bundle`, supporting env overrides.
3. Updated `projections/cli/score_minutes_v1.py` to default to the production bundle via the new resolver, for both historical and live modes.
4. Adjusted scripts `run_live_score.sh` / `run_live_pipeline.sh` so `--bundle-dir` overrides are optional; default path derived from production config.
5. Added `scripts/minutes_debug/show_production_sample.py` for quick prod-bundle smoke testing.
6. Added `tests/test_minutes_production_model.py` verifying the resolver + bundle load.
7. Confirmed systemd services (`live-score`, `live-pipeline`, etc.) invoke the updated scripts, so live scoring & API now use `lgbm_full_v1_vol` by default.

## Verification
- `uv run pytest tests/test_minutes_production_model.py` passes.
- Debug CLI `uv run python -m scripts.minutes_debug.show_production_sample --date 2025-11-10 --rows 5` prints calibrated predictions, showing prod bundle loads successfully.
