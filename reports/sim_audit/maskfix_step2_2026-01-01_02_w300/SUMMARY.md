## sim_v3 audit (mask consistency + diagnostics)

**Run**: `2026-01-01` → `2026-01-02`, `300` worlds, seed `0`  
**Output**: `reports/sim_audit/maskfix_step2_2026-01-01_02_w300/audit/`

### What changed vs prior audit

- **Fixed active mask consistency** in `scripts/sim_v2/generate_worlds_fpts_v2.py`: the stored per-world `active_mask_samples` used for conditional aggregation is now snapshotted *after* any in-place mutations (notably `bench_zero_mixture` drops). This prevents corrupt conditional moments like `E[minutes | plays]` / `E[stat | plays]` when the "plays" mask changes after sampling.
- **Added DEV-only asserts** (guarded by `PROJECTIONS_SIM_DEV_ASSERTS=1`) to fail fast on:
  - `count_negative_minutes == 0`
  - `max_abs(team_world_minutes_sum - 240) < 1e-4`
- **Extended `scripts/sim_v2/audit_sim_v3.py` outputs** (no new model knobs):
  - `E[min | plays]` and `P(min == 0)` by `minutes_p50` bucket
  - `dk_fpts` tail coverage (`q90/q95/q99`) by `minutes_p50` bucket with **starter vs bench** split
  - Same-team / cross-team **residual** correlation (`actual_fpts - sim_mean_fpts`)

### Integrity checks (this run)

- Minutes invariant passed: `count_negative_minutes = 0`, `max_abs_team_world_sum_err ≈ 8.6e-06`.
- DEV asserts were enabled and did not fire.

### Conditional minutes + tail coverage

- Conditional minutes + zero-mass by `minutes_p50` bucket are written to:
  - `reports/sim_audit/maskfix_step2_2026-01-01_02_w300/audit/minutes_bucket_stats_per_date.csv`
- Tail coverage (starter/bench split) is written to:
  - `reports/sim_audit/maskfix_step2_2026-01-01_02_w300/audit/fpts_tail_coverage_by_bucket_starter.csv`

Compared to the prior `bz_on2_2026-01-01_02_w300_p075_s020` audit on the same window:
- `dk_fpts` **q90** coverage (ALL): `0.8674 → 0.8712`
- `dk_fpts` **q90** coverage `[16,24)` bucket: `0.7403 → 0.7792`
- `dk_fpts` **q95** coverage `[16,24)` bucket: `0.8312 → 0.8571`

This is directionally consistent with fixing the conditional mask used in aggregation (expected to affect conditional moments/quantiles when `bench_zero_mixture` drops occur). Sample sizes are small in some buckets/splits, so treat bucket-level deltas as noisy on this 2-day window.

### Correlations

New residual correlation diagnostics are in:
- `reports/sim_audit/maskfix_step2_2026-01-01_02_w300/audit/correlation_summary.csv`

On this window, same-team residual correlation is generally higher than cross-team on `2026-01-01`, and near-zero on `2026-01-02`. This is consistent with a remaining (date-dependent) shared-game component, but no model changes were made in this step.

