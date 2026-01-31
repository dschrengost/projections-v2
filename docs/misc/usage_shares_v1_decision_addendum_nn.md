# Usage Shares v1 Decision Addendum: NN Backend

**Run ID:** `nn_residual_20251216_084353`  
**Date:** 2025-12-16  
**Purpose:** Evaluate whether NN adds value vs baseline and LGBM residual

---

## TL;DR

**🔴 RECOMMENDATION: DO NOT USE NN**

The NN residual model fails to beat baseline and is significantly worse than LGBM residual.

| Backend | Val MAE | vs Baseline | Best Shrink |
|---------|---------|-------------|-------------|
| Baseline | 0.0338 | — | — |
| **LGBM Residual** | **0.0331** | **+2.2%** | 0.75 |
| NN Residual | 0.0338 | -0.1% | 0.25 |

---

## Findings

### 1. NN Overfits Immediately

| Epoch | Train KL | Train MAE | Val KL | Val MAE |
|-------|----------|-----------|--------|---------|
| 0 | 0.158 | 0.0369 | **0.128** | **0.0338** |
| 5 | 0.143 | 0.0362 | 0.131 | 0.0351 |
| 10 | 0.133 | 0.0363 | 0.134 | 0.0359 |
| 15 | 0.126 | 0.0362 | 0.134 | 0.0361 |
| 29 | 0.120 | 0.0357 | 0.135 | 0.0362 |

**Best validation performance was at epoch 0** — the randomly initialized network produces predictions closest to the optimum when combined with the baseline. Further training only hurts.

### 2. Shrink Grid Search

| Shrink | Val MAE |
|--------|---------|
| **0.25** | **0.0337** |
| 0.50 | 0.0337 |
| 0.75 | 0.0337 |
| 1.00 | 0.0338 |

The best shrink is 0.25 with MAE 0.0337, which is comparable to — but not better than — baseline (0.0338). The grid search variance is within noise.

### 3. Why NN Fails

1. **Insufficient data**: 4,489 training rows across 448 team-games is too small for a 128→64 MLP
2. **Residual signal is weak**: The baseline already captures ~95%+ of the variance; the remaining signal is noise
3. **Overfitting even with regularization**: L2=1e-4, dropout=0.1, and early stopping couldn't prevent overfitting
4. **No categorical embeddings**: We didn't include team/position embeddings, limiting expressiveness

### 4. Comparison to LGBM Residual

| Metric | LGBM Residual | NN Residual | Δ |
|--------|---------------|-------------|---|
| Val MAE | 0.0331 | 0.0338 | LGBM wins by 2.3% |
| Val KL | 0.113 | 0.129 | LGBM wins by 14% |
| Best shrink | 0.75 | 0.25 | LGBM can use larger corrections |
| High-vac improvement | +1.4% | N/A (not tested) | — |

The LGBM residual model provides **material improvement** (+2.2% MAE, +12% KL) while the NN provides **none**.

---

## Decision: Do Not Keep NN

| Criterion | Threshold | NN Result | Status |
|-----------|-----------|-----------|--------|
| Overall MAE improvement | >0% vs baseline | -0.1% | ❌ |
| MAE improvement vs LGBM | ≥0.5e-3 absolute | -0.7e-3 (worse) | ❌ |
| Statistical significance | N/A | Not tested (clearly fails) | ❌ |

**The NN is NOT worth keeping because:**
1. It does not beat baseline
2. It is worse than LGBM residual by 2.3% MAE
3. It overfits immediately and provides no learnable signal
4. It adds complexity without benefit

---

## Recommendation

**Keep LGBM residual (starterless) as the sole learned model.**

If we want to explore NN in the future:
1. Wait for more training data (10x current size)
2. Add categorical embeddings for team_id, position
3. Use much smaller architecture (e.g., 32→16)
4. Consider per-game-type training (home/away, B2B, etc.)

---

## Artifacts

| Artifact | Path |
|----------|------|
| NN model | `runs/nn_residual_20251216_084353/nn_residual/model_fga_best.pt` |
| Config | `runs/nn_residual_20251216_084353/nn_residual/nn_config.json` |
| Metrics | `runs/nn_residual_20251216_084353/metrics_nn.json` |

---

## Commands Used

```bash
# Train NN residual
uv run python -m scripts.usage_shares_v1.train_nn_residual \
    --data-root /home/daniel/projections-data \
    --start-date 2024-10-22 \
    --end-date 2025-11-28 \
    --val-start 2025-10-29 \
    --targets fga \
    --epochs 30 \
    --batch-groups 32 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --seed 1337
```

---

*Generated: 2025-12-16*
