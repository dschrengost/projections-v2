# sim_v2 live pipeline update (2025-12-05)

## What changed
- **Worlds now carry boxscore totals**  
  Each sim_v2 world row includes `pts_world`, `reb_world` (plus `oreb_world`, `dreb_world`), `ast_world`, `stl_world`, `blk_world`, `tov_world`, alongside `dk_fpts_world` and `minutes_sim`.

- **Minutes sampling + 240 enforcement in sims**  
  Minutes are sampled (when minutes_noise is enabled) around the projected minutes column with play_prob dropout, then enforced to 240 per team per world. Worlds expose `minutes_sim`; aggregates expose `minutes_sim_mean`.

- **Aggregation outputs**  
  Aggregated projections now include DK FPTS quantiles p05/p10/p25/p50/p75/p95, mean/std, per-stat means (pts/reb/ast/stl/blk/tov), and `minutes_sim_mean`. Written to `artifacts/sim_v2/projections/date=YYYY-MM-DD/run=<id>/sim_v2_projections.parquet` with `latest_run.json`.

- **Live orchestration**  
  Default live sim profile is `rates_v0` (live minutes + live rates). Live scoring script calls `scripts.sim_v2.run_sim_live` after FPTS scoring, passing the live minutes run_id; latest sim artifacts are under `PROJECTIONS_DATA_ROOT/artifacts/sim_v2/...`.

- **API + dashboard**  
  Minutes API merges sim projections, prefixing fields with `sim_` (e.g., `sim_dk_fpts_p05`, `sim_pts_mean`, `sim_minutes_sim_mean`). Dashboard has a “Show sim_v2 columns” toggle to display sim quantiles/stat means and Sim MIN.

## Current limitations
- Scoring still treats attempts as makes (no shooting pct/makes-per-minute head in rates), so PTS/FPTS are inflated. AST/REB/STL/BLK/TOV are accurate given per-minute rates. Fix requires adding efficiency outputs to rates and retraining, or applying an efficiency factor in sim scoring.

## Latest live run (example)
- Date: 2025-12-05, profile: rates_v0, worlds: 1000
- Worlds: `/home/daniel/projections-data/artifacts/sim_v2/worlds_fpts_v2/game_date=2025-12-05/`
- Projections: `/home/daniel/projections-data/artifacts/sim_v2/projections/date=2025-12-05/run=20251205T190329Z/sim_v2_projections.parquet`
