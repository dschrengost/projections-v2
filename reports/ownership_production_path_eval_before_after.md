# Ownership Production-Path Eval — Before/After (Fixed Local Window)

- Window: `2025-11-30..2025-12-15` (DK), “main slate” selected per date via max `slate_entries`
- Data root: `/home/daniel/projections-data`
- This report evaluates the **production scoring path** (DK salaries + sim_v2 `proj_fpts`) and is separate from the fixed DK↔LineStar slice in `reports/ownership_eval_before_after.md`.

## Reproduce

### 1) Score baseline model on the window
`uv run python -m projections.cli.score_ownership_live --date <DATE> --run-id <RUN_ID> --data-root /home/daniel/projections-data --model-run dk_only_v6_logit_chalk5_cleanbase_seed1337 --ignore-lock-cache --no-write-lock-cache`

### 2) Evaluate production-path predictions vs DK actuals
`uv run python scripts/ownership/evaluate_ownership_production_path.py --start-date 2025-11-30 --end-date 2025-12-15 --data-root /home/daniel/projections-data --slate-selector largest_entries --pred-snapshot latest --out-md /tmp/prod_eval.md --out-parquet /tmp/prod_joined.parquet`

### 3) (Optional) Fit power calibrator on the joined dataset
`uv run python scripts/ownership/fit_power_calibrator.py --in-parquet /tmp/prod_joined.parquet --out-json /tmp/power_calibrator.json --out-md /tmp/power_calibrator.md`

## Baseline (dk_only_v6_logit_chalk5_cleanbase_seed1337)

- n_slates: `13`, n_rows: `1765`

### Raw Model (`pred_own_pct_raw`)
- MAE/RMSE: `5.5666 / 11.1918`
- Spearman pooled: `0.6043`
- Spearman top10/top20: `0.0583 / 0.0165`
- Recall@10/20: `0.3077 / 0.4808`
- ECE: `3.0254`

### Scaled-to-Sum (scale `pred_own_pct_raw` → `800%`)
- MAE/RMSE: `5.2842 / 9.9684`
- Spearman pooled: `0.5587`
- Spearman top10/top20: `0.0583 / 0.0165`
- Recall@10/20: `0.3077 / 0.4808`
- ECE: `1.4832`

### Production Output (`pred_own_pct`)
- MAE/RMSE: `5.2295 / 9.9676`
- Spearman pooled: `0.5628`
- Spearman top10/top20: `0.0583 / 0.0165`
- Recall@10/20: `0.3077 / 0.4808`
- ECE: `1.5786`

## After (dk_prod_v5_logit_locksafe_1130_1215_seed1337)

- n_slates: `13`, n_rows: `1765`

### Raw Model (`pred_own_pct_raw`)
- MAE/RMSE: `10.0592 / 17.4342`
- Spearman pooled: `0.6658`
- Spearman top10/top20: `0.1786 / -0.0222`
- Recall@10/20: `0.2385 / 0.4692`
- ECE: `7.5735`

### Scaled-to-Sum (scale `pred_own_pct_raw` → `800%`)
- MAE/RMSE: `5.0678 / 9.6114`
- Spearman pooled: `0.6919`
- Spearman top10/top20: `0.1786 / -0.0222`
- Recall@10/20: `0.2385 / 0.4692`
- ECE: `1.6355`

### Production Output (`pred_own_pct`)
- MAE/RMSE: `4.9774 / 9.5944`
- Spearman pooled: `0.6896`
- Spearman top10/top20: `0.1786 / -0.0222`
- Recall@10/20: `0.2385 / 0.4692`
- ECE: `1.7656`

## After + Power Calibration (gamma ≈ 0.75)

- Fit result: `gamma=0.7500`, `eps=0.001`, `R=8.0`, `cap_pct=100.0`
- Eval (calibrated output `pred_own_pct_power`):
  - MAE/RMSE: `4.9699 / 9.1518`
  - Spearman pooled: `0.6906`
  - Spearman top10/top20: `0.1786 / -0.0222`
  - Recall@10/20: `0.2385 / 0.4692`
  - ECE: `0.8880`

