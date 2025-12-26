# Minutes Allocator A/B/C Test Results

Generated: 2025-12-24T13:03:12.199116

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
| Total slates | 4 |
| Processed | 2 |
| Clean | 2 |
| Degraded | 0 |
| Skipped | 2 |
| With labels | 2 |

### Skip Reasons
  - too_many_missing_features (9/58): 1
  - integrity_failed: incomplete_rosters (min 1 players): 1

## Summary (Clean Slates Only)

| Metric | A | B | C | Winner |
|--------|---|---|---|--------|
| MAE (mean) | 4.38 | 4.74 | 3.88 | A |
| RMSE (mean) | 6.25 | 7.29 | 6.22 | A |
| Top-5 sum | 163.8 | 147.9 | 172.5 | A |
| Max minutes | 37.4 | 31.3 | 39.4 | - |
| Gini | 0.416 | 0.182 | 0.325 | - |
| Roster size | 14.4 | 11.0 | 11.0 | - |

## Consistency (Clean Slates)

| Metric | A vs B | C vs B |
|--------|--------|--------|
| % slates wins MAE | 100.0% | 100.0% |
| % slates wins top5 | 100.0% | 100.0% |
| % slates wins both | 100.0% | 100.0% |
| MAE delta mean±std | -0.36 ± 0.13 | -0.85 ± 0.28 |

## Per-Bucket Analysis (Clean Slates)

| Bucket | MAE (A) | MAE (B) | MAE (C) | A vs B | C vs B |
|--------|---------|---------|---------|--------|--------|
| 0-10 min | 2.75 | 3.77 | 1.66 | A | C |
| 10-20 min | 6.17 | 2.95 | 6.12 | B | B |
| 20-30 min | 7.19 | 5.88 | 7.61 | B | B |
| 30+ min | 4.47 | 6.60 | 4.10 | A | C |

## Pathology Checks (Clean Slates)

| Metric | A | B | C |
|--------|---|---|---|
| % slates with top5 < 150 | 0.0% | 100.0% | 0.0% |
| % slates with max < 30 | 0.0% | 0.0% | 0.0% |
| Bench crush (mean) | 6.6 | 14.8 | 6.7 |
| Bench crush (p90) | 8.6 | 14.8 | 8.7 |

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
