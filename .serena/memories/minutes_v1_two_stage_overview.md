## Minutes v1 (Jan 2025) — Two-Stage Conformal Pipeline

### Data prep
- Filter OUT-at-tip rows before train/cal/val; keep them only for reporting/ops awareness.
- Train play-probability head (`PlayProbabilityArtifacts`): LightGBM classifier + SimpleImputer + isotonic calibration.
- Train conditional LightGBM quantile models on minutes>0 rows only; calibration windows remain exclusive.

### Serving
1. Predict `q̂ = P(minutes>0)`.
2. Predict conditional quantiles (p10/p50/p90) >0 using per-bucket two-sided offsets (starter×p50 or starter×p50×injury).
3. Apply physical clamp [0,48] after conformal adjustments.
4. Mix unconditional quantiles via the formula:
   - if (1 − q̂) ≥ α ⇒ pα = 0,
   - else pα = Q⁺((α − (1 − q̂)) / q̂).
5. Final quantiles are re-clamped and re-enforced for monotonicity.

### Metrics / Guardrails
- Track play-prob quality: `val_play_prob_brier`, `val_play_prob_ece`, `val_play_prob_mean`.
- Report `val_floor_p10/p90` (atomic mass at 0) and `val_excess_p10/p90` (actual minus floor).
- Conditional coverage metrics (`val_p10_conditional`, `val_p90_conditional`) drive guardrails; unconditional coverage is informational.
- Per-bucket payload now includes `cond_n`, `cond_p10_cov`, `cond_p90_cov`, etc.

### Calibration-prod config stubs
- `play_probability` block: toggles mixture head, records artifact ID, and sets Brier/ECE warning thresholds.
- `minutes_guardrails` block: defines playable filter (e.g., min p50, min proj points) and conditional tail tolerances, plus optional monitoring band for p90 and flag to log floor metrics.

### Latest walk (wf_2024-12_fold2_twoS_10d_k600_mixture2)
- Validation rows (after OUT filter): 1,675.
- `floor_p10 ≈ 0.27` (atom at 0 is sizable); unconditional p10 ≈ 0.247 reflects that.
- Conditional tails: p10_cond ≈ 0.093 (close), p90_cond ≈ 0.940 (slightly conservative, driven by >18 buckets).
- Play-prob head calibrated (Brier≈0.099, ECE≈0.044, mean q̂≈0.764).
- Right tails for high-minutes buckets still overshoot target (bench|>18 cond_p90≈0.997), but within “monitor” band.

### Outstanding work (if we revisit minutes)
- Optionally tighten right-tail scale caps for >18 buckets or add richer bucket keys.
- Add ops dashboards for team-minute sums and rotation sanity.
- Consider yellow/red alerting for conditional p90 misses >0.03 but <0.06; only block if >0.06.
- Future tuning should focus on playable subset defined in config (p50>=10, etc.).