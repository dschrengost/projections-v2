# Usage Shares v1 Retrain Decision Memo

**Run ID:** `20251216_080827`  
**Date Range:** 2024-10-22 to 2025-11-28 (9,113 rows, 98.3% coverage)  
**Train/Val Split:** tail-30-days (Train: 4,489 rows, Val: 4,624 rows)

---

## Summary

**Recommendation: 🔴 DON'T WIRE YET**

The LGBM models do not beat the rate-weighted baseline, especially in high-vacancy regimes where injury behavior matters most.

---

## Coverage After Backfill

| Metric | Value |
|--------|-------|
| Total rows | 9,113 |
| minutes_pred_p50 coverage | 98.3% |
| Date range | 2024-10-22 to 2025-11-28 |
| Unique dates | 204 |

**Improvement:** Coverage improved from ~84% to 98.3% after backfill. Gaps remain in:
- Early Feb 2025 (All-Star break?)
- Recent Dec 2025 dates (pipeline lag)

---

## Global Metrics

| Target | Backend | MAE | vs Baseline | KL | Top1 |
|--------|---------|-----|-------------|-----|------|
| FGA | baseline | 0.0338 | — | 0.129 | 56.7% |
| FGA | lgbm | 0.0339 | **-0.3%** | 0.115 | 50.4% |
| TOV | baseline | 0.0673 | — | 0.442 | 35.1% |
| TOV | lgbm | 0.0695 | **-3.4%** | 0.469 | 26.0% |

**Observation:** Models overfit to training data (beat baseline on train, lose on val).

---

## High-Vacancy Performance (Q4 = Top Decile)

| Target | Backend | vac_min Range | MAE | vs Baseline |
|--------|---------|---------------|-----|-------------|
| FGA | baseline | 444-1915 | 0.0346 | — |
| FGA | lgbm | 444-1915 | 0.0352 | **-1.9%** |
| TOV | baseline | 444-1915 | 0.0673 | — |
| TOV | lgbm | 444-1915 | 0.0715 | **-6.1%** |

**Critical:** Models perform WORST in the high-vacancy regime—exactly where we need them most.

---

## Reallocation Sanity (High-Vacancy Games)

| Target | Backend | % Starter | % Top2 Minutes | Avg Minutes Rank |
|--------|---------|-----------|----------------|------------------|
| FGA | lgbm | **0%** | 31.1% | 4.9 |
| TOV | lgbm | **0%** | 26.7% | 4.6 |

**Problem:** When vacancy is high, the model shifts usage to NON-starters and players ranked ~5th in minutes. This is implausible—in real injury scenarios, usage typically flows to remaining starters or top rotation players.

---

## Failure Mode Analysis

### Pattern 1: Overfitting
- Train MAE beats baseline by ~3-5%
- Val MAE loses to baseline by 0.3-3.4%
- Signal: Not enough data or features too noisy

### Pattern 2: Vacancy Signal Not Learned
- High-vacancy buckets show WORSE performance
- Model doesn't correctly associate vacancy features with reallocation

### Pattern 3: Wrong Beneficiaries
- 0% of top beneficiaries are starters
- Model may be learning spurious correlations (e.g., bench player roles)

### Worst-Case Examples (High KL)
- Many worst cases have high vacancy (588, 1211, 1221 min)
- Cleveland (game 22500225) appears in both baseline and LGBM worst cases
- Some worst cases have 0 vacancy but still fail—general instability

---

## Root Causes

1. **Train/Val Distribution Shift**
   - Train: 2024-10 to 2025-10-28 (historical)
   - Val: 2025-10-29 to 2025-11-28 (current season start)
   - Rosters changed significantly between seasons

2. **Validity Flags Missing for 2025-10+ Data**
   - Had to assume NaN validity = valid
   - May be including pathological rows

3. **Team-Scaled Feature Not Helping**
   - Added `minutes_pred_p50_team_scaled` but no improvement
   - May need more targeted vacancy-interaction features

4. **Baseline is Hard to Beat**
   - Rate-weighted baseline already uses `season_{stat}_per_min * minutes_pred_p50`
   - This captures both role and playing time well
   - Need much better signal to improve

---

## Recommendations

### Immediate (Before Wiring)
1. **Fix training base validity flags** for 2025-10+ data
2. **Investigate seasonal distribution shift**—consider training on 2025 data only
3. **Add explicit vacancy-interaction features:**
   - `vacancy * is_starter` interaction
   - `vacancy_same_position` (starter out → same position benefits)

### Short-term
4. **Increase regularization** to reduce overfitting
5. **Try ensemble:** average LGBM + baseline predictions

### Medium-term
6. **Collect more data** with proper validity flags
7. **Add player-level features:** historical share-when-teammate-out

---

## Decision

**DO NOT WIRE** the current models into simulation.

The rate-weighted baseline outperforms LGBM in:
- Overall MAE
- High-vacancy MAE  
- Top-1 accuracy
- Reallocation plausibility

**Next Steps:**
1. Fix 2025-10+ validity flags in training base builder
2. Train on same-season data only (2025-10 to 2025-11)
3. Add vacancy-position interaction features
4. Re-evaluate after those fixes

---

*Generated: 2025-12-16*
