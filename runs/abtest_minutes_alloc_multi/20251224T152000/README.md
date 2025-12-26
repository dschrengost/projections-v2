# Minutes Allocator A/B/C Test Results

Generated: 2025-12-24T15:20:04.198277

## Allocators

| Allocator | Description |
|-----------|-------------|
| **A** | SCALE_SHARES: Share model predictions scaled to 240 per team |
| **B** | ROTALLOC: Production allocator with rotation classifier + conditional means |
| **C** | SHARE_WITH_ROTALLOC_ELIGIBILITY: Share predictions scaled within RotAlloc's eligible set |

## Quality Tiers

| Tier | Criteria |
|------|----------|
| **clean** | Passes integrity + missing_feature_frac ≤ 2% |
| **degraded** | Passes integrity + 2% < missing_feature_frac ≤ 10% |
| **skipped** | Failed integrity or missing_feature_frac > 10% |

## Coverage

| Metric | Count |
|--------|-------|
| Total slates | 61 |
| Processed | 32 |
| Clean | 32 |
| Degraded | 0 |
| Skipped | 29 |
| With labels | 12 |

### Skip Reasons
  - too_many_missing_features (9/58): 28
  - features_not_found: 1

## Summary (Clean Slates Only)

| Metric | A | B | C | Winner |
|--------|---|---|---|--------|
| MAE (mean) | 5.43 | 5.29 | 5.12 | C |
| RMSE (mean) | 7.44 | 7.89 | 7.81 | A |
| Top-5 sum | 148.7 | 146.1 | 162.4 | A |
| Max minutes | 35.3 | 31.7 | 38.5 | - |
| Gini | 0.431 | 0.177 | 0.284 | - |
| Roster size | 16.5 | 11.0 | 11.0 | - |

## Consistency (Clean Slates)

| Metric | A vs B | C vs B |
|--------|--------|--------|
| % slates wins MAE | 41.7% | 66.7% |
| % slates wins top5 | 58.3% | 100.0% |
| % slates wins both | 33.3% | 66.7% |
| MAE delta mean±std | +0.14 ± 0.43 | -0.16 ± 0.46 |

## Per-Bucket Analysis (Clean Slates)

| Bucket | MAE (A) | MAE (B) | MAE (C) | A vs B | C vs B |
|--------|---------|---------|---------|--------|--------|
| 0-10 min | 4.39 | 4.59 | 3.53 | A | C |
| 10-20 min | 6.35 | 5.32 | 7.25 | B | B |
| 20-30 min | 6.98 | 6.38 | 7.44 | B | B |
| 30+ min | 5.89 | 5.82 | 5.17 | B | C |

## Pathology Checks (Clean Slates)

| Metric | A | B | C |
|--------|---|---|---|
| % slates with top5 < 150 | 62.5% | 96.9% | 0.0% |
| % slates with max < 30 | 0.0% | 0.0% | 0.0% |
| Bench crush (mean) | 9.0 | 14.6 | 9.6 |
| Bench crush (p90) | 10.2 | 14.9 | 10.9 |

## Recommendation

**Use Allocator C**: SHARE_WITH_ROTALLOC_ELIGIBILITY wins on both accuracy and realism.

## Next Steps

1. Examine C vs B bucket-level losses to understand where RotAlloc's conditional means help
2. Test different sharpen_exponent values for the share model
3. Consider hybrid: use share model weights within RotAlloc's probability framework
4. Investigate skipped slates for data pipeline fixes

## Files

- `aggregate_summary.json` - All aggregated metrics (all processed)
- `aggregate_summary_clean.json` - Aggregated metrics (clean only)
- `aggregate_by_bucket.csv` - Per-bucket MAE breakdown
- `aggregate_by_slate.csv` - Per-slate comparison table
- `skips.csv` - Skipped slates with reasons
