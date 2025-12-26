# Minutes Allocator A/B/C Test Results

Generated: 2025-12-24T13:03:32.597084

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
| Processed | 9 |
| Clean | 9 |
| Degraded | 0 |
| Skipped | 52 |
| With labels | 9 |

### Skip Reasons
  - too_many_missing_features (9/58): 43
  - features_not_found: 8
  - integrity_failed: incomplete_rosters (min 1 players): 1

## Summary (Clean Slates Only)

| Metric | A | B | C | Winner |
|--------|---|---|---|--------|
| MAE (mean) | 5.34 | 5.27 | 5.10 | C |
| RMSE (mean) | 7.44 | 7.86 | 7.77 | A |
| Top-5 sum | 152.8 | 147.1 | 163.9 | A |
| Max minutes | 35.8 | 31.9 | 38.4 | - |
| Gini | 0.393 | 0.176 | 0.286 | - |
| Roster size | 15.0 | 10.9 | 10.9 | - |

## Consistency (Clean Slates)

| Metric | A vs B | C vs B |
|--------|--------|--------|
| % slates wins MAE | 44.4% | 66.7% |
| % slates wins top5 | 55.6% | 100.0% |
| % slates wins both | 33.3% | 66.7% |
| MAE delta mean±std | +0.07 ± 0.40 | -0.17 ± 0.52 |

## Per-Bucket Analysis (Clean Slates)

| Bucket | MAE (A) | MAE (B) | MAE (C) | A vs B | C vs B |
|--------|---------|---------|---------|--------|--------|
| 0-10 min | 4.21 | 4.53 | 3.46 | A | C |
| 10-20 min | 6.48 | 5.28 | 7.37 | B | B |
| 20-30 min | 7.14 | 6.48 | 7.69 | B | B |
| 30+ min | 5.62 | 5.72 | 4.89 | A | C |

## Pathology Checks (Clean Slates)

| Metric | A | B | C |
|--------|---|---|---|
| % slates with top5 < 150 | 44.4% | 88.9% | 0.0% |
| % slates with max < 30 | 0.0% | 0.0% | 0.0% |
| Bench crush (mean) | 8.8 | 14.8 | 9.1 |
| Bench crush (p90) | 9.8 | 14.9 | 10.4 |

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
