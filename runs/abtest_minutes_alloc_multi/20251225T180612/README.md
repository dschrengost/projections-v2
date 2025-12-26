# Minutes Allocator A/B/C/D/E/F/M Test Results

Generated: 2025-12-25T18:06:16.497330

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
| Total slates | 14 |
| Processed | 14 |
| Clean | 14 |
| Degraded | 0 |
| Skipped | 0 |
| With labels | 11 |

### Skip Reasons
  (none)

## Summary (Clean Slates Only)

| Metric | A | B | C | D | E | M | Winner |
|--------|---|---|---|---|---|---|--------|
| MAE (mean) | 4.76 | 4.10 | 4.73 | 4.39 | 4.51 | N/A | B |
| RMSE (mean) | 7.39 | 6.35 | 7.56 | 6.88 | 7.05 | N/A | B |
| Top-5 sum | 151.5 | 154.1 | 157.9 | 160.4 | 161.4 | N/A | F |
| Ordering 10-30 | 0.672 | 0.720 | 0.675 | 0.708 | 0.699 | N/A | – |
| Max minutes | 35.5 | 34.2 | 37.0 | 38.0 | 38.1 | N/A | – |
| Gini | 0.327 | 0.211 | 0.240 | 0.272 | 0.278 | N/A | – |
| Roster size | 12.5 | 10.8 | 10.4 | 10.8 | 10.8 | N/A | – |
| D best alpha (mode) | – | – | – | 0.5 | – | – | – |
| D best MAE | – | – | – | 4.30 | – | – | – |

## Consistency (Clean Slates)

| Metric | A vs B | C vs B | D vs B | E vs B | M vs B |
|--------|--------|--------|--------|--------|--------|
| % slates wins MAE | 0.0% | 0.0% | 0.0% | 0.0% | N/A% |
| % slates wins top5 | 72.7% | 90.9% | 100.0% | 100.0% | N/A% |
| % slates wins both | 0.0% | 0.0% | 0.0% | 0.0% | N/A% |
| MAE delta mean±std | +0.66 ± 0.26 | +0.63 ± 0.28 | +0.29 ± 0.25 | +0.41 ± 0.25 | N/A ± N/A |

## Per-Bucket Analysis (Clean Slates)

| Bucket | MAE (A) | MAE (B) | MAE (C) | MAE (D) | MAE (E) | MAE (M) | Winner |
|--------|---------|---------|---------|---------|---------|---------|--------|
| 0-10 min | 2.57 | 2.91 | 2.40 | 2.22 | 2.21 | N/A | E |
| 10-20 min | 6.71 | 4.96 | 7.47 | 6.81 | 7.28 | N/A | B |
| 20-30 min | 7.10 | 6.01 | 7.21 | 6.68 | 6.90 | N/A | B |
| 30+ min | 6.51 | 4.44 | 6.09 | 5.69 | 5.78 | N/A | B |

## Pathology Checks (Clean Slates)

| Metric | A | B | C | D | E | M |
|--------|---|---|---|---|---|---|
| % slates with top5 < 150 | 21.4% | 21.4% | 21.4% | 21.4% | 21.4% | N/A% |
| % slates with max < 30 | 21.4% | 0.0% | 0.0% | 0.0% | 0.0% | N/A% |
| Bench crush (mean) | 10.0 | 13.0 | 10.2 | 9.9 | 8.9 | N/A |
| Bench crush (p90) | 13.2 | 14.4 | 14.7 | 13.8 | 12.5 | N/A |

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
