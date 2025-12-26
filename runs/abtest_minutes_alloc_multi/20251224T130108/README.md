# Minutes Allocator A/B/C Test Results

Generated: 2025-12-24T13:01:08.449062

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
| Total slates | 3 |
| Processed | 3 |
| Clean | 3 |
| Degraded | 0 |
| Skipped | 0 |
| With labels | 3 |

### Skip Reasons
  (none)

## Summary (Clean Slates Only)

| Metric | A | B | C | Winner |
|--------|---|---|---|--------|
| MAE (mean) | 6.59 | 6.46 | 6.60 | B |
| RMSE (mean) | 8.80 | 9.33 | 9.69 | A |
| Top-5 sum | 142.5 | 145.7 | 158.4 | C |
| Max minutes | 34.0 | 32.2 | 37.8 | - |
| Gini | 0.430 | 0.172 | 0.260 | - |
| Roster size | 17.3 | 10.9 | 10.9 | - |

## Consistency (Clean Slates)

| Metric | A vs B | C vs B |
|--------|--------|--------|
| % slates wins MAE | 33.3% | 33.3% |
| % slates wins top5 | 0.0% | 100.0% |
| % slates wins both | 0.0% | 33.3% |
| MAE delta mean±std | +0.12 ± 0.33 | +0.14 ± 0.40 |

## Per-Bucket Analysis (Clean Slates)

| Bucket | MAE (A) | MAE (B) | MAE (C) | A vs B | C vs B |
|--------|---------|---------|---------|--------|--------|
| 0-10 min | 6.33 | 6.25 | 5.69 | B | C |
| 10-20 min | 6.66 | 7.20 | 8.48 | A | B |
| 20-30 min | 7.17 | 6.99 | 8.31 | B | B |
| 30+ min | 6.65 | 5.63 | 5.52 | B | C |

## Pathology Checks (Clean Slates)

| Metric | A | B | C |
|--------|---|---|---|
| % slates with top5 < 150 | 100.0% | 100.0% | 0.0% |
| % slates with max < 30 | 0.0% | 0.0% | 0.0% |
| Bench crush (mean) | 9.8 | 14.8 | 10.3 |
| Bench crush (p90) | 10.5 | 14.9 | 11.1 |

## Recommendation

**Hybrid approach**: C has better realism but B has better accuracy. Consider tuning C's caps or weights.

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
