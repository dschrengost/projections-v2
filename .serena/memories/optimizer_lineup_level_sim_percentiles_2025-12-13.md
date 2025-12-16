## Lineup-level sim percentiles (post-optimization)

### What changed
- Added true lineup distribution stats computed from sim_v2 world outputs (sum 8 players per world, then take percentiles).
- Stats are attached to optimizer build results (API + saved build JSON) and the dashboard now prefers lineup-level `p90` over summing player `p90`.

### Key files
- `projections/optimizer/lineup_sim_stats.py`
  - `load_world_fpts_matrix(...)`: reads `world=*.parquet` under `<DATA_ROOT>/artifacts/sim_v2/worlds_fpts_v2/game_date=YYYY-MM-DD/` via `pyarrow.dataset`, returning `(world_ids, player_ids, fpts_by_world)`.
  - `compute_lineup_distribution_stats(...)`: for each lineup, computes totals per world and returns `mean, p10, p50, p75, p90, stdev, ceiling_upside`.

- `projections/api/optimizer_service.py`
  - `run_quick_build(...)` now attempts to compute `lineup_stats` after QuickBuild finishes, using sim_v2 worlds, then stores them on the job and includes them in `save_build(...)` output.

- `projections/api/optimizer_api.py`
  - `LineupRow` response model includes optional `mean/p10/p50/p75/p90/stdev/ceiling_upside`.
  - `/build/{job_id}/lineups` returns these per-lineup fields when available.

- `web/minutes-dashboard/src/pages/OptimizerPage.tsx`
  - Uses `lu.p90` when provided (fallback to old sum-of-player-p90 if missing), for filtering/sorting/display.

### Output fields (per lineup)
- `mean`, `p10`, `p50`, `p75`, `p90`, `stdev`, `ceiling_upside` (= `p90 - mean`).

### Notes
- If a date partition only has `projections.parquet` (no `world=*.parquet`), lineup stats won’t compute and will be omitted/null.

### Tests
- `tests/optimizer/test_lineup_sim_stats.py` covers matrix loading + percentile correctness and demonstrates why summing player p90 is wrong.