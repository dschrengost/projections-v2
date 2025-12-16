## sim_v2 live wiring (2025-01)
- Added per-stat outputs to sim_v2 worlds (`pts_world`, `reb_world`, `ast_world`, `stl_world`, `blk_world`, `tov_world`, oreb/dreb) and tagged worlds with `sim_profile`. Derived DK FPTS now computed alongside box-score stats.
- Aggregator now writes sim_v2 projections with DK FPTS quantiles (p05/p10/p25/p50/p75/p95), mean/std, and per-stat means to `artifacts/sim_v2/projections/date=YYYY-MM-DD/run=<id>/sim_v2_projections.parquet` with `latest_run.json` pointer.
- New Typer CLI `scripts.sim_v2.run_sim_live` runs worlds + aggregation for a date (defaults: profile=baseline, 1000 worlds). Live scoring script now calls it after FPTS scoring (env knobs: `LIVE_RUN_SIM`, `LIVE_SIM_PROFILE`, `LIVE_SIM_WORLDS`).
- Minutes API loads sim_v2 projections (prefixed `sim_`) and exposes them to the dashboard; added `sim_available` meta flag. Dashboard shows sim quantiles + stat means with toggle.
