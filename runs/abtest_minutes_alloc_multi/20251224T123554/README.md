# Minutes Allocator A/B Test Results

Generated: 2025-12-24T12:35:55.063198

## Summary

| Metric | SCALE_SHARES (A) | ROTALLOC (B) | Winner |
|--------|-----------------|--------------|--------|
| MAE (mean) | 6.60 | 6.46 | B |
| RMSE (mean) | 9.69 | 9.33 | B |
| Top-5 sum | 158.4 | 145.7 | A |
| Max minutes | 37.8 | 32.2 | A |
| Gini (lower=flatter) | 0.260 | 0.172 | - |
| Roster size | 10.9 | 10.9 | - |

## Coverage

- **Total slates**: 3
- **Processed slates**: 3
- **Skipped slates**: 0
- **Slates with labels**: 3

## Consistency

- **% slates where A wins on MAE**: 33.3%
- **% slates where A wins on top-5**: 100.0%
- **% slates where A wins on both**: 33.3%
- **MAE delta (A - B) mean±std**: +0.14 ± 0.40

## Per-Bucket Analysis

| Bucket | MAE (A) | MAE (B) | Winner |
|--------|---------|---------|--------|
| 0-10 min | 5.69 | 6.25 | A |
| 10-20 min | 8.48 | 7.20 | B |
| 20-30 min | 8.31 | 6.99 | B |
| 30+ min | 5.52 | 5.63 | A |

## Pathology Checks

| Metric | A | B |
|--------|---|---|
| % slates with top5 < 150 | 0.0% | 100.0% |
| % slates with max < 30 | 0.0% | 0.0% |
| Bench crush (mean) | 10.3 | 14.8 |
| Bench crush (p90) | 11.1 | 14.9 |

## Interpretation

### Does SCALE_SHARES consistently improve starter realism?
**Yes.** A produces higher top-5 sums and max minutes consistently across slates.

### Where does it lose accuracy (which buckets)?
A loses primarily in buckets where RotAlloc's adaptive depth helps: ['10-20', '20-30']

### Is the failure mode consistent or slate-dependent?
The MAE delta is relatively stable across slates.

## Recommendation

**Hybrid approach**: Consider using A's share model with B's eligibility pruning. A has better realism but B has better accuracy.

## Next Steps

1. Investigate bucket-specific failures
2. Consider hybrid approach: use A's share model as weights in RotAlloc's allocation
3. Test with different sharpen_exponent values
4. Examine worst 10% of slates for failure modes

## Files

- `aggregate_summary.json` - All aggregated metrics
- `aggregate_by_bucket.csv` - Per-bucket MAE breakdown
- `aggregate_by_slate.csv` - Per-slate comparison table
