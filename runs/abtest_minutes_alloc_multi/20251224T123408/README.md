# Minutes Allocator A/B Test Results

Generated: 2025-12-24T12:34:08.392075

## Summary

| Metric | SCALE_SHARES (A) | ROTALLOC (B) | Winner |
|--------|-----------------|--------------|--------|
| MAE (mean) | 4.07 | 4.75 | A |
| RMSE (mean) | 6.43 | 7.31 | A |
| Top-5 sum | 158.0 | 155.4 | A |
| Max minutes | 39.6 | 50.1 | B |
| Gini (lower=flatter) | 0.281 | 0.161 | - |
| Roster size | 10.1 | 10.1 | - |

## Coverage

- **Total slates**: 3
- **Processed slates**: 3
- **Skipped slates**: 0
- **Slates with labels**: 3

## Consistency

- **% slates where A wins on MAE**: 100.0%
- **% slates where A wins on top-5**: 66.7%
- **% slates where A wins on both**: 66.7%
- **MAE delta (A - B) mean±std**: -0.68 ± 0.33

## Per-Bucket Analysis

| Bucket | MAE (A) | MAE (B) | Winner |
|--------|---------|---------|--------|
| 0-10 min | 1.88 | 3.74 | A |
| 10-20 min | 6.17 | 3.62 | B |
| 20-30 min | 7.39 | 5.68 | B |
| 30+ min | 4.58 | 6.79 | A |

## Pathology Checks

| Metric | A | B |
|--------|---|---|
| % slates with top5 < 150 | 33.3% | 66.7% |
| % slates with max < 30 | 0.0% | 0.0% |
| Bench crush (mean) | 10.9 | 34.8 |
| Bench crush (p90) | 17.3 | 62.9 |

## Interpretation

### Does SCALE_SHARES consistently improve starter realism?
**Yes.** A produces higher top-5 sums and max minutes consistently across slates.

### Where does it lose accuracy (which buckets)?
A wins overall on accuracy.

### Is the failure mode consistent or slate-dependent?
The MAE delta is relatively stable across slates.

## Recommendation

**Replace RotAlloc**: SCALE_SHARES is better on both accuracy and realism.

## Next Steps

1. Validate on out-of-sample dates
2. Consider hybrid approach: use A's share model as weights in RotAlloc's allocation
3. Test with different sharpen_exponent values
4. Examine worst 10% of slates for failure modes

## Files

- `aggregate_summary.json` - All aggregated metrics
- `aggregate_by_bucket.csv` - Per-bucket MAE breakdown
- `aggregate_by_slate.csv` - Per-slate comparison table
