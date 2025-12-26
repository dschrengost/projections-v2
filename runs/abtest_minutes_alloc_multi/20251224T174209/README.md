# Minutes Allocator A/B/C/D/E/F Test Results

Generated: 2025-12-24T17:42:09.527467

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
| Processed | 1 |
| Clean | 1 |
| Degraded | 0 |
| Skipped | 0 |
| With labels | 1 |

### Skip Reasons
  (none)

## Summary (Clean Slates Only)

| Metric | A | B | C | D | E | Winner |
|--------|---|---|---|---|---|--------|
| MAE (mean) | 7.47 | 6.65 | 7.08 | 6.90 | 6.90 | F |
| RMSE (mean) | 9.94 | 9.63 | 10.65 | 10.23 | 10.27 | F |
| Top-5 sum | 141.8 | 144.3 | 159.1 | 159.6 | 159.1 | D |
| Max minutes | 33.3 | 32.2 | 37.4 | 37.6 | 37.4 | – |
| Gini | 0.429 | 0.175 | 0.271 | 0.269 | 0.265 | – |
| Roster size | 17.6 | 11.0 | 11.0 | 11.0 | 11.0 | – |
| D best alpha (mode) | – | – | – | – | – | – |
| D best MAE | – | – | – | 6.74 | – | – |

## Consistency (Clean Slates)

| Metric | A vs B | C vs B | D vs B | E vs B |
|--------|--------|--------|--------|--------|
| % slates wins MAE | 0.0% | 0.0% | 0.0% | 0.0% |
| % slates wins top5 | 0.0% | 100.0% | 100.0% | 100.0% |
| % slates wins both | 0.0% | 0.0% | 0.0% | 0.0% |
| MAE delta mean±std | +0.82 ± 0.00 | +0.43 ± 0.00 | +0.25 ± 0.00 | +0.25 ± 0.00 |

## Per-Bucket Analysis (Clean Slates)

| Bucket | MAE (A) | MAE (B) | MAE (C) | MAE (D) | MAE (E) | Winner |
|--------|---------|---------|---------|---------|---------|--------|
| 0-10 min | 7.09 | 6.07 | 6.08 | 5.80 | 5.84 | F |
| 10-20 min | 7.72 | 7.77 | 9.51 | 9.38 | 9.26 | A |
| 20-30 min | 8.54 | 7.68 | 9.26 | 9.41 | 9.27 | B |
| 30+ min | 7.18 | 6.21 | 5.52 | 5.25 | 5.36 | D |

## Pathology Checks (Clean Slates)

| Metric | A | B | C | D | E |
|--------|---|---|---|---|---|
| % slates with top5 < 150 | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| % slates with max < 30 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Bench crush (mean) | 9.2 | 14.6 | 10.2 | 10.3 | 10.3 |
| Bench crush (p90) | 9.2 | 14.6 | 10.2 | 10.3 | 10.3 |

## Recommendation

**Hybrid approach**: C has better realism but B has better accuracy. Consider tuning C's caps or weights.

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
