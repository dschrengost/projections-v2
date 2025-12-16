# sim_v2 minutes sampling fix (2025-12-12)

## Problem
Dashboard showed sim minutes == minutes model p50 because sim outputs only `minutes_mean` (base minutes center) and API mapped it to `sim_minutes_sim_mean`. Also, in `mean_source="rates"` path worlds could fall back to fixed p50 minutes (flattening projections) when game-script inputs (e.g. spread column) were missing.

## Fix (implemented)
- `scripts/sim_v2/generate_worlds_fpts_v2.py` (rates mean_source path):
  - Always samples per-world minutes (no fixed p50 fallback).
  - Uses game scripts when enabled; otherwise samples from a per-player distribution derived from minutes p10/p50/p90.
  - Applies `play_prob` as a per-world active mask.
  - Enforces team 240 minutes per world (`enforce_team_240_minutes`).
  - Writes sim minutes summary stats into projections output: `minutes_sim_mean`, `minutes_sim_std`, `minutes_sim_p10`, `minutes_sim_p50`, `minutes_sim_p90`.
  - Attempts to load minutes-noise params (if enabled); if missing, logs warning and falls back to quantile-based sampling.

- `projections/api/minutes_api.py`:
  - Exposes `sim_minutes_sim_p10/p50/p90/std`.
  - Prefers `minutes_sim_*` when present and avoids renaming collisions with legacy `minutes_mean`.

- `web/minutes-dashboard`:
  - Adds columns for `Sim MIN p50` and `Sim MIN mean`.
  - Uses sim p50 in the minutes distribution card + expanded details.

## Tests
- `tests/test_sim_v2_minutes_sampling.py`:
  - Ensures minutes variance exists in rates path and persists even when minutes projections have no `spread_home` column.

## Notes
- Baseline profile currently enables minutes_noise but the expected params file was missing in `/home/daniel/projections-data/artifacts/sim_v2/minutes_noise/<run>_minutes_noise.json`; sim now warns and continues with quantile-based sampling.
