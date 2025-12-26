# Minutes Allocator A/B/C Test Results

Generated: 2025-12-24T13:01:33.855906

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
| Total slates | 8 |
| Processed | 3 |
| Clean | 3 |
| Degraded | 0 |
| Skipped | 5 |
| With labels | 3 |

### Skip Reasons
  - missing_summary: 5

## Summary (Clean Slates Only)

| Metric | A | B | C | Winner |
|--------|---|---|---|--------|
| MAE (mean) | 3.98 | 4.30 | 3.54 | A |
| RMSE (mean) | 5.72 | 6.71 | 5.63 | A |
| Top-5 sum | 162.9 | 149.0 | 169.8 | A |
| Max minutes | 37.4 | 31.5 | 39.0 | - |
| Gini | 0.379 | 0.181 | 0.313 | - |
| Roster size | 13.5 | 10.9 | 10.9 | - |

## Consistency (Clean Slates)

| Metric | A vs B | C vs B |
|--------|--------|--------|
| % slates wins MAE | 100.0% | 100.0% |
| % slates wins top5 | 100.0% | 100.0% |
| % slates wins both | 100.0% | 100.0% |
| MAE delta mean±std | -0.32 ± 0.12 | -0.76 ± 0.27 |

## Per-Bucket Analysis (Clean Slates)

| Bucket | MAE (A) | MAE (B) | MAE (C) | A vs B | C vs B |
|--------|---------|---------|---------|--------|--------|
| 0-10 min | 2.28 | 3.25 | 1.39 | A | C |
| 10-20 min | 6.21 | 3.28 | 6.18 | B | B |
| 20-30 min | 6.56 | 6.08 | 6.80 | B | B |
| 30+ min | 4.20 | 5.60 | 3.89 | A | C |

## Pathology Checks (Clean Slates)

| Metric | A | B | C |
|--------|---|---|---|
| % slates with top5 < 150 | 0.0% | 66.7% | 0.0% |
| % slates with max < 30 | 0.0% | 0.0% | 0.0% |
| Bench crush (mean) | 7.2 | 14.8 | 7.3 |
| Bench crush (p90) | 9.0 | 14.9 | 9.0 |

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
