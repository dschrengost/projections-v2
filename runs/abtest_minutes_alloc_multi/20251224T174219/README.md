# Minutes Allocator A/B/C/D/E/F Test Results

Generated: 2025-12-24T17:42:23.350838

## Allocators

| Allocator | Description |
|-----------|-------------|
| **A** | SCALE_SHARES: Share model predictions scaled to 240 per team |
| **B** | ROTALLOC: Production allocator with rotation classifier + conditional means |
| **C** | SHARE_WITH_ROTALLOC_ELIGIBILITY: Share predictions scaled within RotAlloc's eligible set |
| **D** | BLEND_WITHIN_ELIGIBLE: Blends share weights with RotAlloc proxy weights within eligible set |
| **E** | FRINGE_ONLY_ALPHA: Two-tier blend - core players (top k by w_rot) use alpha_core, fringe use alpha_fringe |
| **F** | POWER_POSTPROCESS: Power transform on RotAlloc minutes to fix flat-top (increases concentration) |

### Allocator D Details

D blends two weight vectors within RotAlloc's eligible set:
- w_share = share_pred ^ gamma (realism from share model)
- w_rot = (p_rot ^ a) * (mu_cond ^ mu_power) (bench ordering from RotAlloc)
- Final: w = alpha * w_share + (1-alpha) * w_rot

Default alpha = 0.7 (70% share, 30% RotAlloc proxy). Grid search tests alpha in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0].

### Allocator E Details

E improves on D by using different alpha values for core vs fringe players:
- Core players (top k_core by w_rot): use alpha_core (default 0.8 = lean shares)
- Fringe players (remaining eligible): use alpha_fringe (default 0.3 = lean RotAlloc proxy)

This targets improved MAE in the 10-30 minute buckets where fringe rotation players benefit from RotAlloc ordering.

### Allocator F Details

F applies a power transform to RotAlloc (B) minutes to fix the flat-top pathology:
- m_raw = m_B ** p (where p >= 1.0)
- m_F = 240 * m_raw / sum(m_raw)

Higher p increases top-end concentration while preserving B's bench ordering. Default p = 1.2.

## Quality Tiers

| Tier | Criteria |
|------|----------|
| **clean** | Passes integrity + missing_feature_frac ≤ 2% |
| **degraded** | Passes integrity + 2% < missing_feature_frac ≤ 10% |
| **skipped** | Failed integrity or missing_feature_frac > 10% |

## Coverage

| Metric | Count |
|--------|-------|
| Total slates | 36 |
| Processed | 15 |
| Clean | 15 |
| Degraded | 0 |
| Skipped | 21 |
| With labels | 8 |

### Skip Reasons
  - too_many_missing_features (9/58): 21

## Summary (Clean Slates Only)

| Metric | A | B | C | D | E | Winner |
|--------|---|---|---|---|---|--------|
| MAE (mean) | 5.11 | 4.95 | 4.72 | 4.69 | 4.71 | D |
| RMSE (mean) | 7.15 | 7.52 | 7.34 | 7.17 | 7.21 | A |
| Top-5 sum | 150.8 | 146.6 | 162.8 | 162.8 | 161.9 | C |
| Max minutes | 35.7 | 31.7 | 38.5 | 37.8 | 37.8 | – |
| Gini | 0.408 | 0.178 | 0.284 | 0.276 | 0.272 | – |
| Roster size | 15.6 | 10.9 | 10.9 | 10.9 | 10.9 | – |
| D best alpha (mode) | – | – | – | 0.7 | – | – |
| D best MAE | – | – | – | 4.61 | – | – |

## Consistency (Clean Slates)

| Metric | A vs B | C vs B | D vs B | E vs B |
|--------|--------|--------|--------|--------|
| % slates wins MAE | 50.0% | 75.0% | 75.0% | 75.0% |
| % slates wins top5 | 75.0% | 100.0% | 100.0% | 100.0% |
| % slates wins both | 50.0% | 75.0% | 75.0% | 75.0% |
| MAE delta mean±std | +0.16 ± 0.50 | -0.23 ± 0.47 | -0.26 ± 0.35 | -0.25 ± 0.35 |

## Per-Bucket Analysis (Clean Slates)

| Bucket | MAE (A) | MAE (B) | MAE (C) | MAE (D) | MAE (E) | Winner |
|--------|---------|---------|---------|---------|---------|--------|
| 0-10 min | 3.73 | 4.05 | 2.92 | 2.97 | 3.09 | C |
| 10-20 min | 6.21 | 4.56 | 6.77 | 6.53 | 6.40 | B |
| 20-30 min | 7.10 | 6.25 | 7.41 | 7.64 | 7.49 | B |
| 30+ min | 5.85 | 6.15 | 5.02 | 4.72 | 4.77 | D |

## Pathology Checks (Clean Slates)

| Metric | A | B | C | D | E |
|--------|---|---|---|---|---|
| % slates with top5 < 150 | 66.7% | 93.3% | 0.0% | 0.0% | 0.0% |
| % slates with max < 30 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Bench crush (mean) | 9.2 | 14.6 | 9.7 | 10.1 | 10.2 |
| Bench crush (p90) | 10.2 | 14.9 | 10.7 | 11.1 | 11.1 |

## Recommendation

**Use Allocator C**: SHARE_WITH_ROTALLOC_ELIGIBILITY wins on both accuracy and realism.

## Next Steps

1. Examine C vs B bucket-level losses to understand where RotAlloc's conditional means help
2. Test different sharpen_exponent values for the share model
3. Consider hybrid: use share model weights within RotAlloc's probability framework
4. Investigate skipped slates for data pipeline fixes
5. Fine-tune E's alpha_core and alpha_fringe parameters for optimal 10-30 bucket performance

## Files

- `aggregate_summary.json` - All aggregated metrics (all processed)
- `aggregate_summary_clean.json` - Aggregated metrics (clean only)
- `aggregate_by_bucket.csv` - Per-bucket MAE breakdown
- `aggregate_by_slate.csv` - Per-slate comparison table
- `skips.csv` - Skipped slates with reasons
