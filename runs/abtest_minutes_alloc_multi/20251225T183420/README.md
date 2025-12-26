# Minutes Allocator A/B/C/D/E/F/M Test Results

Generated: 2025-12-25T18:34:22.622773

## Allocators

| Allocator | Description |
|-----------|-------------|
| **A** | SCALE_SHARES: Share model predictions scaled to 240 per team |
| **B** | ROTALLOC: Production allocator with rotation classifier + conditional means |
| **C** | SHARE_WITH_ROTALLOC_ELIGIBILITY: Share predictions scaled within RotAlloc's eligible set |
| **D** | BLEND_WITHIN_ELIGIBLE: Blends share weights with RotAlloc proxy weights within eligible set |
| **E** | FRINGE_ONLY_ALPHA: Two-tier blend - core players (top k by w_rot) use alpha_core, fringe use alpha_fringe |
| **F** | POWER_POSTPROCESS: Power transform on RotAlloc minutes to fix flat-top (increases concentration) |
| **M** | MIXTURE_SHARES_SCALED: Mixture model expected minutes → shares → scaled to 240 |

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

### Allocator M Details

M uses a trained mixture model to predict expected minutes directly:
- Predict expected minutes from features using mixture distribution
- Mask inactive players (OUT, OFS, NWT status)
- Convert to shares within team
- Scale to 240 per team with iterative cap/redistribution

Requires --mixture-bundle path to enable.

## Quality Tiers

| Tier | Criteria |
|------|----------|
| **clean** | Passes integrity + missing_feature_frac ≤ 2% |
| **degraded** | Passes integrity + 2% < missing_feature_frac ≤ 10% |
| **skipped** | Failed integrity or missing_feature_frac > 10% |

## Coverage

| Metric | Count |
|--------|-------|
| Total slates | 10 |
| Processed | 10 |
| Clean | 10 |
| Degraded | 0 |
| Skipped | 0 |
| With labels | 7 |

### Skip Reasons
  (none)

## Summary (Clean Slates Only)

| Metric | A | B | C | D | E | M | Winner |
|--------|---|---|---|---|---|---|--------|
| MAE (mean) | 4.87 | 4.25 | 4.81 | 4.45 | 4.57 | N/A | B |
| RMSE (mean) | 7.38 | 6.52 | 7.57 | 6.90 | 7.04 | N/A | B |
| Top-5 sum | 147.3 | 153.0 | 155.2 | 158.4 | 159.4 | N/A | F |
| Ordering 10-30 | 0.684 | 0.742 | 0.688 | 0.724 | 0.708 | N/A | – |
| Max minutes | 34.5 | 33.9 | 36.3 | 37.3 | 37.5 | N/A | – |
| Gini | 0.324 | 0.207 | 0.226 | 0.262 | 0.269 | N/A | – |
| Roster size | 12.8 | 10.9 | 10.5 | 10.8 | 10.9 | N/A | – |
| D best alpha (mode) | – | – | – | 0.3 | – | – | – |
| D best MAE | – | – | – | 4.33 | – | – | – |

## Consistency (Clean Slates)

| Metric | A vs B | C vs B | D vs B | E vs B | M vs B |
|--------|--------|--------|--------|--------|--------|
| % slates wins MAE | 0.0% | 0.0% | 0.0% | 0.0% | N/A% |
| % slates wins top5 | 57.1% | 85.7% | 100.0% | 100.0% | N/A% |
| % slates wins both | 0.0% | 0.0% | 0.0% | 0.0% | N/A% |
| MAE delta mean±std | +0.63 ± 0.26 | +0.56 ± 0.26 | +0.21 ± 0.25 | +0.32 ± 0.25 | N/A ± N/A |

## Per-Bucket Analysis (Clean Slates)

| Bucket | MAE (A) | MAE (B) | MAE (C) | MAE (D) | MAE (E) | MAE (M) | Winner |
|--------|---------|---------|---------|---------|---------|---------|--------|
| 0-10 min | 2.91 | 3.19 | 2.71 | 2.52 | 2.53 | N/A | D |
| 10-20 min | 6.52 | 4.82 | 7.37 | 6.68 | 7.11 | N/A | B |
| 20-30 min | 7.03 | 6.07 | 7.15 | 6.61 | 6.85 | N/A | B |
| 30+ min | 6.58 | 4.60 | 5.96 | 5.48 | 5.56 | N/A | F |

## Pathology Checks (Clean Slates)

| Metric | A | B | C | D | E | M |
|--------|---|---|---|---|---|---|
| % slates with top5 < 150 | 30.0% | 30.0% | 30.0% | 30.0% | 30.0% | N/A% |
| % slates with max < 30 | 30.0% | 0.0% | 0.0% | 0.0% | 0.0% | N/A% |
| Bench crush (mean) | 10.5 | 13.1 | 10.8 | 10.4 | 9.4 | N/A |
| Bench crush (p90) | 13.3 | 14.6 | 14.8 | 13.9 | 12.6 | N/A |

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
