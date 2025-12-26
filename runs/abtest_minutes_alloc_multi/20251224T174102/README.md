# Minutes Allocator A/B/C/D/E/F Test Results

Generated: 2025-12-24T17:41:02.770586

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
| Total slates | 1 |
| Processed | 0 |
| Clean | 0 |
| Degraded | 0 |
| Skipped | 1 |
| With labels | 0 |

### Skip Reasons
  - 'minutes_mean': 1

## Summary (Clean Slates Only)

| Metric | A | B | C | D | E | Winner |
|--------|---|---|---|---|---|--------|
| MAE (mean) | N/A | N/A | N/A | N/A | N/A | – |
| RMSE (mean) | N/A | N/A | N/A | N/A | N/A | – |
| Top-5 sum | N/A | N/A | N/A | N/A | N/A | – |
| Max minutes | N/A | N/A | N/A | N/A | N/A | – |
| Gini | N/A | N/A | N/A | N/A | N/A | – |
| Roster size | N/A | N/A | N/A | N/A | N/A | – |
| D best alpha (mode) | – | – | – | – | – | – |
| D best MAE | – | – | – | N/A | – | – |

## Consistency (Clean Slates)

| Metric | A vs B | C vs B | D vs B | E vs B |
|--------|--------|--------|--------|--------|
| % slates wins MAE | N/A% | N/A% | N/A% | N/A% |
| % slates wins top5 | N/A% | N/A% | N/A% | N/A% |
| % slates wins both | N/A% | N/A% | N/A% | N/A% |
| MAE delta mean±std | N/A ± N/A | N/A ± N/A | N/A ± N/A | N/A ± N/A |

## Per-Bucket Analysis (Clean Slates)

| Bucket | MAE (A) | MAE (B) | MAE (C) | MAE (D) | MAE (E) | Winner |
|--------|---------|---------|---------|---------|---------|--------|
| 0-10 min | N/A | N/A | N/A | N/A | N/A | – |
| 10-20 min | N/A | N/A | N/A | N/A | N/A | – |
| 20-30 min | N/A | N/A | N/A | N/A | N/A | – |
| 30+ min | N/A | N/A | N/A | N/A | N/A | – |

## Pathology Checks (Clean Slates)

| Metric | A | B | C | D | E |
|--------|---|---|---|---|---|
| % slates with top5 < 150 | N/A% | N/A% | N/A% | N/A% | N/A% |
| % slates with max < 30 | N/A% | N/A% | N/A% | N/A% | N/A% |
| Bench crush (mean) | N/A | N/A | N/A | N/A | N/A |
| Bench crush (p90) | N/A | N/A | N/A | N/A | N/A |

## Recommendation

**No data**: No slates were successfully processed.

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
