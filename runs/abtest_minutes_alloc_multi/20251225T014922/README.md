# Minutes Allocator A/B/C/D/E/F/M Test Results

Generated: 2025-12-25T01:49:34.859406

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
| Total slates | 61 |
| Processed | 32 |
| Clean | 32 |
| Degraded | 0 |
| Skipped | 29 |
| With labels | 12 |

### Skip Reasons
  - too_many_missing_features (9/58): 28
  - features_not_found: 1

## Summary (Clean Slates Only)

| Metric | A | B | C | D | E | M | Winner |
|--------|---|---|---|---|---|---|--------|
| MAE (mean) | 5.43 | 5.29 | 5.12 | 5.07 | 5.10 | 5.65 | D |
| RMSE (mean) | 7.44 | 7.89 | 7.81 | 7.60 | 7.66 | 7.47 | A |
| Top-5 sum | 148.7 | 146.1 | 162.4 | 162.0 | 161.2 | 122.6 | C |
| Ordering 10-30 | 0.717 | 0.677 | 0.702 | 0.693 | 0.694 | 0.728 | – |
| Max minutes | 35.3 | 31.7 | 38.5 | 37.8 | 37.9 | 26.6 | – |
| Gini | 0.431 | 0.177 | 0.284 | 0.277 | 0.271 | 0.330 | – |
| Roster size | 16.5 | 11.0 | 11.0 | 11.0 | 11.0 | 16.2 | – |
| D best alpha (mode) | – | – | – | 0.7 | – | – | – |
| D best MAE | – | – | – | 4.98 | – | – | – |

## Consistency (Clean Slates)

| Metric | A vs B | C vs B | D vs B | E vs B | M vs B |
|--------|--------|--------|--------|--------|--------|
| % slates wins MAE | 41.7% | 66.7% | 66.7% | 66.7% | 41.7% |
| % slates wins top5 | 58.3% | 100.0% | 100.0% | 100.0% | 0.0% |
| % slates wins both | 33.3% | 66.7% | 66.7% | 66.7% | 0.0% |
| MAE delta mean±std | +0.14 ± 0.43 | -0.16 ± 0.46 | -0.21 ± 0.35 | -0.19 ± 0.37 | +0.36 ± 0.76 |

## Per-Bucket Analysis (Clean Slates)

| Bucket | MAE (A) | MAE (B) | MAE (C) | MAE (D) | MAE (E) | MAE (M) | Winner |
|--------|---------|---------|---------|---------|---------|---------|--------|
| 0-10 min | 4.39 | 4.59 | 3.53 | 3.53 | 3.66 | 5.14 | C |
| 10-20 min | 6.35 | 5.32 | 7.25 | 7.03 | 6.92 | 3.62 | M |
| 20-30 min | 6.98 | 6.38 | 7.44 | 7.67 | 7.54 | 5.33 | M |
| 30+ min | 5.89 | 5.82 | 5.17 | 4.83 | 4.88 | 8.91 | D |

## Pathology Checks (Clean Slates)

| Metric | A | B | C | D | E | M |
|--------|---|---|---|---|---|---|
| % slates with top5 < 150 | 62.5% | 96.9% | 0.0% | 0.0% | 0.0% | 100.0% |
| % slates with max < 30 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 84.4% |
| Bench crush (mean) | 9.0 | 14.6 | 9.6 | 10.0 | 10.2 | 13.2 |
| Bench crush (p90) | 10.2 | 14.9 | 10.9 | 11.1 | 11.1 | 14.1 |

## Recommendation

**Use Allocator C**: SHARE_WITH_ROTALLOC_ELIGIBILITY wins on both accuracy and realism.

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
