# Usage Shares v1 Wire Decision Report

**Run ID:** `decision_20251216_083525`  
**Date:** 2025-12-16  
**Model:** FGA Residual-on-Baseline (Starterless)

---

## TL;DR

**🟢 RECOMMENDATION: WIRE FGA behind feature flag**

| Gate | Criteria | Result | Status |
|------|----------|--------|--------|
| 1. Overall improvement | MAE better than baseline | +2.20% | ✓ |
| 2. Bootstrap significance | >80% bootstrap samples positive | 100.0% | ✓ |
| 3. High-vacancy improvement | >50% bootstrap samples positive in top decile | 90.1% | ✓ |
| 4. Sanity check | ABS top share goes to top-2 minute players ≥70% | 80.0% | ✓ |

**Configuration:**
- Shrink: 0.75
- Features: 25 (starterless variant)
- Model: `decision_20251216_083525/model_fga_starterless.txt`

---

## Section A: Data Quality Audit

### is_starter Column Status

| Week | % with is_starter=1 |
|------|---------------------|
| 2025-10-20/2025-10-26 | 47.9% |
| 2025-10-27/2025-11-02 | 35.2% |
| 2025-11-03/2025-11-09 | **0.0%** ❌ |
| 2025-11-10/2025-11-16 | **0.0%** ❌ |
| 2025-11-17/2025-11-23 | **0.0%** ❌ |
| 2025-11-24/2025-11-30 | 8.2% |

**Root Cause:** The `is_starter` field in `rates_training_base` has complete dropout for 3 weeks in November 2025. This originates upstream (likely roster snapshot ETL failure).

**Statistics:**
- % of team-games with 5 starters: 28.9%
- % of team-games with 0 starters: **70.6%**

### Remediation Chosen: Option A (Starterless Model)

We REMOVED the following features:
- `is_starter`
- `vac_min_szn_x_is_starter`
- `vac_fga_szn_x_is_starter`

This unblocks the wire decision without waiting for upstream fix.

### Remediation Option B (Future: Fix Upstream)

To fix `is_starter`:
1. File: `projections/cli/build_rates_features_live.py` or upstream ETL
2. Join key: `(game_id, player_id)` with roster/lineup data
3. Data source: Likely needs `roster_nightly` or lineup confirmations
4. Validation: Every team-game should have exactly 5 starters

---

## Section B: Statistical Significance

### Bootstrap Test (1,000 resamples by team-game)

#### Overall (450 games)

| Metric | Delta (baseline - model) | 95% CI | % Positive |
|--------|--------------------------|--------|------------|
| MAE | 0.745e-3 | [0.412e-3, 1.101e-3] | **100.0%** |
| KL | 0.0161 | [0.0108, 0.0215] | **100.0%** |

**Interpretation:** The model beats baseline in 100% of bootstrap samples. The improvement is statistically significant with a narrow confidence interval that excludes zero.

#### High-Vacancy Top Decile (45 games)

| Metric | Delta (baseline - model) | 95% CI | % Positive |
|--------|--------------------------|--------|------------|
| MAE | 0.621e-3 | [-0.364e-3, 1.600e-3] | **90.1%** |
| KL | 0.0227 | [0.0069, 0.0431] | **99.9%** |

**Interpretation:** In high-vacancy games, the MAE CI includes zero but 90% of samples favor the model. KL improvement is significant (99.9% positive).

---

## Section C: Injury Behavior Analysis

### Reallocation Sanity (Top Decile by vac_min_szn, 45 games)

| Metric Type | % in Top-2 Minutes | Avg Minutes Rank |
|-------------|-------------------|------------------|
| **ABS** (predicted top share) | **80.0%** | 1.7 |
| **DELTA** (biggest gain vs baseline) | 48.9% | 4.6 |

**Interpretation:**
- ABS: The player with highest predicted share is in the top-2 by minutes 80% of the time (avg rank 1.7). This is good — the model correctly assigns usage to high-minutes players.
- DELTA: The player with biggest improvement from baseline is often mid-rotation (rank 4.6). This is expected — bench players have more room to gain.

### Worst Examples (Top 5 by KL in High-Vacancy)

| Game | Team | Date | vac_min | KL | Issue |
|------|------|------|---------|----|----|
| 22500277 | ORL | 2025-11-23 | 1211 | 0.48 | Low-min player (7.2) had 23% true share |
| 22500278 | LAC | 2025-11-23 | 725 | 0.34 | 0.7-min player had 17.6% true share |
| 22500276 | ATL | 2025-11-23 | 781 | 0.34 | Low-min player (5.3) had 19.8% true share |
| 22500199 | DET | 2025-11-10 | 833 | 0.30 | Top player's true share (39.5%) exceeded prediction |
| 22500280 | POR | 2025-11-23 | 1076 | 0.27 | 1.2-min player had 12% true share |

**Pattern:** Worst cases are games where low-minutes players had unexpectedly high usage (likely garbage time or unusual rotations).

---

## Section D: Retrain & Eval Matrix

### Variants Tested

| Variant | MAE | KL | Shrink |
|---------|-----|-----|--------|
| **Baseline** | 0.0338 | 0.129 | — |
| **Original Residual** | 0.0331 | 0.112 | 0.75 |
| **Starterless Residual** | **0.0331** | **0.113** | **0.75** |

### Shrink Grid Search (Starterless)

| Shrink | Val MAE | Δ vs Baseline |
|--------|---------|---------------|
| 0.25 | 0.0334 | +1.3% |
| 0.50 | 0.0331 | +2.0% |
| **0.75** | **0.0331** | **+2.2%** |
| 1.00 | 0.0334 | +1.4% |

---

## Section E: Final Recommendation

### Decision: 🟢 WIRE FGA

**Configuration:**
- backend: `lgbm_residual`
- target: `fga`
- shrink: `0.75`
- feature_set: starterless (25 features)

**Wiring plan:**
1. Add behind profile flag `use_learned_usage_shares`
2. Default: keep baseline
3. Enable for internal testing
4. Monitor high-vacancy performance in production

**For TOV:** Keep on baseline (improvement marginal, regresses in high-vacancy).

---

## Appendix: Commands to Reproduce

```bash
# Run decision report
uv run python -m scripts.usage_shares_v1.decision_report \
    --data-root /home/daniel/projections-data \
    --start-date 2024-10-22 \
    --end-date 2025-11-28 \
    --seed 1337
```

*Generated: 2025-12-16*
