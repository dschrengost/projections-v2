# Minutes Sim vs Pred Audit Report

**Date Range**: 2025-01-20 to 2025-01-26
**Profile**: baseline
**Samples per game**: 100
**Total rows**: 201
**Generated**: 2025-12-16T07:55:29.182292

---

## 1. Overall Delta Distribution

| Metric | All Players | Rotation (p50≥12) |
|--------|-------------|-------------------|
| Delta mean | -0.50 | - 0.39 |
| Delta p50 | -0.59 | - 0.24 |
| Delta p90 | 3.95 | - 9.53 |
| Abs delta mean | 3.55 | - 3.77 |
| Abs delta p95 | 11.38 | - 11.92 |

---

## 2. Correlation Analysis

- **corr(pred_p50, sim_mean)**: 0.9375
- **corr(pred_p50 × team_scale, sim_mean)**: 0.9704

---

## 3. Scale-Only Fit Quality

- **Residual mean**: -0.000 min
- **Residual std**: 3.895 min
- **Residual abs p95**: 7.887 min

> If residuals are near zero, enforce_240 is mostly scaling.
> Large residuals indicate reshuffling beyond pure scaling.

---

## 4. Per-Stage Team Sum Diagnostics

> This section tracks where minutes are lost/gained in the pipeline.

### 4.1 Stage-by-Stage Team Sums (mean across worlds)

| Stage | Mean | P05 | P95 |
|-------|------|-----|-----|
| pre_zeroing | 257.3 | 220.9 | 286.7 |
| pre_enforce | 235.4 | 198.7 | 270.5 |
| post_enforce | 240.0 | 240.0 | 240.0 |
| final | 240.0 | 240.0 | 240.0 |

### 4.2 Enforcement Quality Flags

- **Fraction of worlds where post_enforce ≠ 240**: 0.2%
- **Fraction of worlds where final < 240**: 0.2%
- **Fraction of worlds where rotation cap applied (>10 active)**: 72.6%

> ✅ enforce_240 is working as expected for most worlds.

### 4.3 Active Player Counts

- **Active players per team (mean)**: 11.4
- **Active players range**: [8, 16]

- **Max rotation size**: 10
> ⚠️ Average active players (11.4) exceeds rotation cap (10).
> This explains why team sums < 240 after rotation capping.

---

## 5. Breakdown by Starter/Bench

| Role | Count | Delta Mean | Delta Abs Mean | Residual Abs P95 |
|------|-------|------------|----------------|------------------|
| Starter | 79 | +2.47 | 3.68 | 4.60 |
| Bench | 122 | -2.42 | 3.47 | 9.74 |

---

## 6. Breakdown by Minutes Bucket

| Bucket | Count | Delta Mean | Residual Mean | Residual P95 |
|--------|-------|------------|---------------|--------------|
| 0-10 | 74 | -1.89 | -2.31 | 6.90 |
| 10-20 | 30 | -4.51 | -3.03 | 12.15 |
| 20-30 | 67 | +1.26 | +2.67 | 4.29 |
| 30-48 | 30 | +3.02 | +2.77 | 4.83 |

---

## 7. Top 20 Largest Absolute Deltas

| Player | Team | Pred P50 | Sim Mean | Delta |
|--------|------|----------|----------|-------|
| Nicolas Batum | 1610612746 | 16.5 | 0.9 | -15.6 |
| Karlo Matković | 1610612740 | 16.0 | 1.4 | -14.7 |
| Jabari Walker | 1610612757 | 14.6 | 0.0 | -14.6 |
| Yuki Kawamura | 1610612763 | 16.1 | 2.0 | -14.1 |
| Bones Hyland | 1610612746 | 13.5 | 0.0 | -13.5 |
| Jamal Cain | 1610612740 | 16.1 | 3.1 | -13.0 |
| Josh Giddey | 1610612741 | 30.7 | 42.6 | +11.9 |
| Zyon Pullin | 1610612763 | 16.1 | 4.2 | -11.9 |
| Patrick Williams | 1610612741 | 30.1 | 41.7 | +11.5 |
| Brandon Boston | 1610612740 | 16.1 | 4.6 | -11.5 |
| Isaiah Hartenstein | 1610612760 | 29.2 | 40.6 | +11.4 |
| Nikola Vučević | 1610612741 | 31.3 | 42.6 | +11.3 |
| Cason Wallace | 1610612760 | 30.1 | 41.4 | +11.2 |
| Shai Gilgeous-Alexan | 1610612760 | 29.9 | 40.9 | +11.0 |
| Jalen Williams | 1610612760 | 29.0 | 39.9 | +10.9 |
| Jaylen Brown | 1610612738 | 33.7 | 44.6 | +10.9 |
| Derrick White | 1610612738 | 33.7 | 44.5 | +10.9 |
| Jayson Tatum | 1610612738 | 33.7 | 44.2 | +10.5 |
| Giannis Antetokounmp | 1610612749 | 35.0 | 44.8 | +9.8 |
| Damian Lillard | 1610612749 | 34.9 | 44.7 | +9.8 |

---

## 8. Top 20 Largest Scale Residuals

| Player | Team | Pred×Scale | Sim Mean | Residual |
|--------|------|------------|----------|----------|
| Nicolas Batum | 1610612746 | 13.8 | 0.9 | -12.8 |
| Karlo Matković | 1610612740 | 13.6 | 1.4 | -12.3 |
| Yuki Kawamura | 1610612763 | 14.0 | 2.0 | -12.0 |
| Jabari Walker | 1610612757 | 11.4 | 0.0 | -11.4 |
| Bones Hyland | 1610612746 | 11.3 | 0.0 | -11.3 |
| Jamal Cain | 1610612740 | 13.7 | 3.1 | -10.6 |
| Zyon Pullin | 1610612763 | 14.0 | 4.2 | -9.8 |
| Brandon Boston | 1610612740 | 13.7 | 4.6 | -9.0 |
| Jordan Miller | 1610612746 | 8.2 | 0.0 | -8.2 |
| Bones Hyland | 1610612746 | 7.9 | 0.0 | -7.9 |
| Kai Jones | 1610612746 | 7.9 | 0.0 | -7.9 |
| Mo Bamba | 1610612746 | 10.1 | 2.3 | -7.8 |
| Shake Milton | 1610612747 | 9.3 | 1.7 | -7.6 |
| Duop Reath | 1610612757 | 7.2 | 0.0 | -7.2 |
| Dillon Jones | 1610612760 | 7.5 | 0.8 | -6.7 |
| Adam Flagler | 1610612760 | 7.5 | 1.0 | -6.5 |
| Branden Carlson | 1610612760 | 7.5 | 1.0 | -6.5 |
| Keion Brooks Jr. | 1610612740 | 14.0 | 7.6 | -6.4 |
| Anton Watson | 1610612738 | 7.7 | 1.3 | -6.4 |
| Trentyn Flowers | 1610612746 | 6.5 | 0.3 | -6.2 |

---

## 9. Conclusions

1. **Correlation**: pred_p50 → sim_mean correlation is **0.937**
   - High correlation, but some reshuffling occurs within teams.

2. **Residual P95**: 7.89 min after scaling
   - enforce_240 involves **significant reshuffling**.
   - Consider training usage on simulated minutes or adding noise.

3. **Team Sum Accuracy**: mean = 240.0 (target 240)
   - ✅ Team totals are correctly enforced to ~240.
