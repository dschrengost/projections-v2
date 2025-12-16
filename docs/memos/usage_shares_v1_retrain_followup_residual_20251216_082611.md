# Usage Shares v1 Retrain Follow-up

**Run ID:** `residual_20251216_082611`  
**Date:** 2025-12-16  
**Date Range:** 2024-10-22 to 2025-11-28 (val: 2025-10-29 to 2025-11-28)

---

## Summary

**Recommendation: 🟡 PROCEED TO WIRE (FGA only, behind flag)**

The residual-on-baseline approach beats the rate-weighted baseline for FGA, including in high-vacancy games. TOV improvement is marginal and not worth the complexity.

---

## Key Findings

### 1. Metric Bug Confirmed: "0% Starters" Was Wrong Question

The previous report showed "0% of top beneficiaries are starters", which alarmed us. Investigation revealed:

**Root cause:** The metric used `argmax(share_change)` (who gained most vs baseline), not `argmax(share_pred)` (who has highest predicted share).

**Fixed metric now reports BOTH:**

| Metric Type | FGA % Starter | FGA % Top2 Min | FGA Avg Rank |
|-------------|---------------|----------------|--------------|
| **ABS** (predicted top share) | 0%* | **82.2%** | 1.7 |
| **DELTA** (biggest gain vs baseline) | 0%* | 31.1% | 4.9 |

*\*0% starters is a **data quality issue** — only 4.8% of 2025-10+ rows have `is_starter=1` correctly populated.*

**Conclusion:** The model IS putting the highest predicted share on top-minutes players (rank 1.7). The DELTA statistic is noisy because bench players have more room to increase.

---

### 2. Residual Training Works

Instead of predicting raw log-weights, we train to predict:
```
delta = y_true_logw - y_baseline_logw
```

At inference:
```
logw_pred = y_baseline_logw + shrink * delta_pred
```

This leverages the strong baseline while learning targeted corrections.

**Global Results (Val):**

| Target | Baseline MAE | Residual MAE | Improvement | Best Shrink |
|--------|--------------|--------------|-------------|-------------|
| FGA | 0.0338 | 0.0331 | **+2.1%** ✅ | 0.75 |
| TOV | 0.0673 | 0.0668 | **+0.7%** | 0.5 |

---

### 3. High-Vacancy Performance

**FGA in Q4 (top 25% vacancy):**

| Metric | Baseline | Residual | Δ |
|--------|----------|----------|---|
| MAE | 0.0346 | 0.0341 | **+1.4%** ✅ |
| KL | 0.135 | 0.114 | **+15.5%** ✅ |
| Top1 | 55.8% | 55.8% | — |

**This meets acceptance criteria:** ≥1-2% relative improvement in high-vacancy.

**TOV in Q4:**

| Metric | Baseline | Residual | Δ |
|--------|----------|----------|---|
| MAE | 0.0673 | 0.0681 | **-1.2%** ❌ |

TOV is slightly worse in high-vacancy but better overall (+0.7%). The baseline is harder to beat for TOV because it's sparser and noisier.

---

## Changes Made

### Code Changes

1. **`scripts/usage_shares_v1/model_report.py`**
   - Fixed `ReallocationSanity` dataclass to track BOTH ABS and DELTA beneficiaries
   - Updated `compute_reallocation_sanity()` to compute both metrics
   - Updated `print_reallocation_sanity()` to display both perspectives

2. **`projections/usage_shares_v1/features.py`**
   - Added interaction features:
     - `minutes_pred_team_rank` (rank within team, 1=highest)
     - `vac_min_szn_x_is_starter` (vacancy × starter interaction)
     - `vac_min_szn_x_minutes_rank` (vacancy × inverted rank)
     - `vac_fga_szn_x_is_starter` (FGA vacancy × starter)

3. **`scripts/usage_shares_v1/train_lgbm_residual.py`** (NEW)
   - Trains LGBM to predict residual from baseline
   - Grid search over shrink values {0.25, 0.5, 0.75, 1.0}
   - Saves config with selected shrink per target

### Artifacts

| Artifact | Path |
|----------|------|
| Residual models | `runs/residual_20251216_082611/lgbm_residual/` |
| Metrics | `runs/residual_20251216_082611/metrics_residual.json` |
| Config (incl. shrink) | `runs/residual_20251216_082611/lgbm_residual/config.json` |

---

## Recommendation

### Wire: FGA Only, Behind Flag

1. **Wire FGA residual model** with shrink=0.75 behind a profile flag
2. **Keep TOV on baseline** — improvement is marginal and regresses in high-vacancy
3. **Fix is_starter data** upstream for 2025-10+ before relying on starter-based metrics
4. **Monitor** high-vacancy games in production to validate improvement

### Why Partial Wire?

- FGA beats baseline globally (+2.1%) AND in high-vacancy (+1.4%)
- TOV beats baseline globally (+0.7%) but REGRESSES in high-vacancy (-1.2%)
- Conservative approach: start with FGA, expand to TOV after more data/tuning

### Future Work

1. Fix `is_starter` population in training base for 2025 season
2. Investigate why TOV regresses in high-vacancy (may need TOV-specific features)
3. Test with same-season-only training (less distribution shift)
4. Consider per-position vacancy features (guard vacancy → PG/SG benefit)

---

## Appendix: Experiment Details

### Shrink Grid Search Results

**FGA:**
| Shrink | Val MAE | vs Baseline |
|--------|---------|-------------|
| 0.25 | 0.0334 | +1.3% |
| 0.50 | 0.0331 | +2.0% |
| **0.75** | **0.0331** | **+2.1%** |
| 1.00 | 0.0337 | +0.4% |

**TOV:**
| Shrink | Val MAE | vs Baseline |
|--------|---------|-------------|
| 0.25 | 0.0668 | +0.7% |
| **0.50** | **0.0668** | **+0.7%** |
| 0.75 | 0.0673 | -0.0% |
| 1.00 | N/A | N/A |

### Feature Importance (Top 5)

(From residual model, showing what CORRECTIONS the model learns)

FGA:
1. `minutes_pred_p50` (primary signal)
2. `season_fga_per_min` (historical usage rate)
3. `vac_min_szn_x_minutes_rank` (vacancy interaction)
4. `minutes_pred_p50_team_scaled`
5. `vac_min_szn`

---

*Generated: 2025-12-16*
