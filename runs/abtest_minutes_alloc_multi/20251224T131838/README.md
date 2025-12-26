# Minutes Allocator A/B/C Test Results

Generated: 2025-12-24T13:18:38.766216

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
| Total slates | 2 |
| Processed | 2 |
| Clean | 2 |
| Degraded | 0 |
| Skipped | 0 |
| With labels | 2 |

### Skip Reasons
  (none)

## Summary (Clean Slates Only)

| Metric | A | B | C | Winner |
|--------|---|---|---|--------|
| MAE (mean) | 6.31 | 6.40 | 6.29 | A |
| RMSE (mean) | 8.38 | 9.27 | 9.28 | A |
| Top-5 sum | 141.8 | 146.4 | 157.6 | C |
| Max minutes | 33.6 | 32.2 | 37.3 | - |
| Gini | 0.425 | 0.172 | 0.254 | - |
| Roster size | 17.3 | 10.8 | 10.8 | - |

## Consistency (Clean Slates)

| Metric | A vs B | C vs B |
|--------|--------|--------|
| % slates wins MAE | 50.0% | 50.0% |
| % slates wins top5 | 0.0% | 100.0% |
| % slates wins both | 0.0% | 50.0% |
| MAE delta mean±std | -0.09 ± 0.15 | -0.11 ± 0.23 |

## Per-Bucket Analysis (Clean Slates)

| Bucket | MAE (A) | MAE (B) | MAE (C) | A vs B | C vs B |
|--------|---------|---------|---------|--------|--------|
| 0-10 min | 6.09 | 6.03 | 5.37 | B | C |
| 10-20 min | 6.18 | 6.76 | 7.81 | A | B |
| 20-30 min | 6.81 | 7.25 | 8.22 | A | B |
| 30+ min | 6.69 | 5.89 | 5.41 | B | C |

## Pathology Checks (Clean Slates)

| Metric | A | B | C |
|--------|---|---|---|
| % slates with top5 < 150 | 100.0% | 100.0% | 0.0% |
| % slates with max < 30 | 0.0% | 0.0% | 0.0% |
| Bench crush (mean) | 10.0 | 14.8 | 10.5 |
| Bench crush (p90) | 10.6 | 14.9 | 11.2 |

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
