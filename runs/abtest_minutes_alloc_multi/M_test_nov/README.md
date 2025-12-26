# Minutes Allocator A/B/C/D/E/F/M Test Results

Generated: 2025-12-25T01:03:00.813072

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
| Total slates | 30 |
| Processed | 8 |
| Clean | 8 |
| Degraded | 0 |
| Skipped | 22 |
| With labels | 2 |

### Skip Reasons
  - too_many_missing_features (9/58): 22

## Summary (Clean Slates Only)

| Metric | A | B | C | D | E | M | Winner |
|--------|---|---|---|---|---|---|--------|
| MAE (mean) | 6.16 | 5.55 | 5.43 | 5.37 | 5.42 | 5.70 | D |
| RMSE (mean) | 7.84 | 8.16 | 8.25 | 7.93 | 8.02 | 7.61 | M |
| Top-5 sum | 151.6 | 147.9 | 167.2 | 165.8 | 164.4 | 127.3 | C |
| Ordering 10-30 | 0.777 | 0.736 | 0.764 | 0.762 | 0.765 | 0.796 | – |
| Max minutes | 36.4 | 32.1 | 40.1 | 39.1 | 39.1 | 27.9 | – |
| Gini | 0.464 | 0.186 | 0.312 | 0.301 | 0.292 | 0.392 | – |
| Roster size | 17.5 | 11.0 | 11.0 | 11.0 | 11.0 | 16.0 | – |
| D best alpha (mode) | – | – | – | 1.0 | – | – | – |
| D best MAE | – | – | – | 5.27 | – | – | – |

## Consistency (Clean Slates)

| Metric | A vs B | C vs B | D vs B | E vs B | M vs B |
|--------|--------|--------|--------|--------|--------|
| % slates wins MAE | 0.0% | 50.0% | 50.0% | 50.0% | 50.0% |
| % slates wins top5 | 50.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| % slates wins both | 0.0% | 50.0% | 50.0% | 50.0% | 0.0% |
| MAE delta mean±std | +0.61 ± 0.42 | -0.12 ± 0.36 | -0.18 ± 0.24 | -0.13 ± 0.28 | +0.15 ± 0.96 |

## Per-Bucket Analysis (Clean Slates)

| Bucket | MAE (A) | MAE (B) | MAE (C) | MAE (D) | MAE (E) | MAE (M) | Winner |
|--------|---------|---------|---------|---------|---------|---------|--------|
| 0-10 min | 5.67 | 5.14 | 4.18 | 4.13 | 4.33 | 5.49 | D |
| 10-20 min | 6.58 | 6.23 | 7.61 | 7.59 | 7.58 | 3.11 | M |
| 20-30 min | 6.77 | 6.31 | 6.46 | 6.83 | 6.60 | 4.97 | M |
| 30+ min | 6.56 | 5.38 | 5.84 | 5.33 | 5.32 | 9.01 | F |

## Pathology Checks (Clean Slates)

| Metric | A | B | C | D | E | M |
|--------|---|---|---|---|---|---|
| % slates with top5 < 150 | 12.5% | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| % slates with max < 30 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| Bench crush (mean) | 7.9 | 14.5 | 8.3 | 8.9 | 9.4 | 13.0 |
| Bench crush (p90) | 8.8 | 14.7 | 9.3 | 9.7 | 10.1 | 13.5 |

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
