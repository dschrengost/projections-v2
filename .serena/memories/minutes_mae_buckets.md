## Summary
- Added `projections/metrics/minutes.py` with `compute_mae_by_actual_minutes_bucket`, which reports MAE per actual-minutes bucket plus an overall value.
- `projections/models/minutes_lgbm` now logs `val_mae_*` entries sourced from that helper so bucket quality is visible in `metrics.json`.
- `projections/evaluate.regression_metrics` also merges the helper output, and tests live in `tests/test_metrics_minutes.py`.