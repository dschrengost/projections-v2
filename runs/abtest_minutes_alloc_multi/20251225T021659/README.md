# Minutes Allocator A/B/C/D/E/F/M Test Results

Generated: 2025-12-25T02:17:09.399936

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
| Processed | 31 |
| Clean | 31 |
| Degraded | 0 |
| Skipped | 30 |
| With labels | 11 |

### Skip Reasons
  - too_many_missing_features (9/58): 28
  - integrity_failed: incomplete_rosters (min 1 players): 1
  - features_not_found: 1

## Summary (Clean Slates Only)

| Metric | A | B | C | D | E | M | Winner |
|--------|---|---|---|---|---|---|--------|
| MAE (mean) | 5.49 | 5.33 | 5.16 | 5.12 | 5.15 | N/A | D |
| RMSE (mean) | 7.51 | 7.92 | 7.85 | 7.64 | 7.70 | N/A | A |
| Top-5 sum | 149.1 | 145.8 | 162.9 | 162.3 | 161.4 | N/A | C |
| Ordering 10-30 | 0.718 | 0.675 | 0.702 | 0.691 | 0.691 | N/A | – |
| Max minutes | 35.5 | 31.8 | 38.8 | 38.0 | 38.0 | N/A | – |
| Gini | 0.436 | 0.176 | 0.287 | 0.278 | 0.273 | N/A | – |
| Roster size | 16.6 | 10.9 | 10.9 | 10.9 | 10.9 | N/A | – |
| D best alpha (mode) | – | – | – | 0.7 | – | – | – |
| D best MAE | – | – | – | 5.02 | – | – | – |

## Consistency (Clean Slates)

| Metric | A vs B | C vs B | D vs B | E vs B | M vs B |
|--------|--------|--------|--------|--------|--------|
| % slates wins MAE | 36.4% | 63.6% | 63.6% | 63.6% | N/A% |
| % slates wins top5 | 54.5% | 100.0% | 100.0% | 100.0% | N/A% |
| % slates wins both | 27.3% | 63.6% | 63.6% | 63.6% | N/A% |
| MAE delta mean±std | +0.16 ± 0.45 | -0.17 ± 0.48 | -0.21 ± 0.37 | -0.18 ± 0.39 | N/A ± N/A |

## Per-Bucket Analysis (Clean Slates)

| Bucket | MAE (A) | MAE (B) | MAE (C) | MAE (D) | MAE (E) | MAE (M) | Winner |
|--------|---------|---------|---------|---------|---------|---------|--------|
| 0-10 min | 4.48 | 4.66 | 3.59 | 3.60 | 3.73 | N/A | C |
| 10-20 min | 6.49 | 5.46 | 7.42 | 7.21 | 7.12 | N/A | B |
| 20-30 min | 7.07 | 6.44 | 7.46 | 7.70 | 7.57 | N/A | B |
| 30+ min | 5.81 | 5.68 | 5.08 | 4.73 | 4.78 | N/A | D |

## Pathology Checks (Clean Slates)

| Metric | A | B | C | D | E | M |
|--------|---|---|---|---|---|---|
| % slates with top5 < 150 | 58.1% | 96.8% | 0.0% | 0.0% | 0.0% | N/A% |
| % slates with max < 30 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | N/A% |
| Bench crush (mean) | 9.0 | 14.7 | 9.5 | 10.0 | 10.2 | N/A |
| Bench crush (p90) | 10.0 | 14.9 | 10.7 | 11.1 | 11.1 | N/A |

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
