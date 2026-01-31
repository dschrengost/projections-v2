# sim_v2 Team Correlation Baselines

This document tracks reference baselines for same-team DK FPTS correlation in sim_v2 worlds. Use these numbers as targets when tuning noise/correlation knobs.

## Baseline v1

- Profile: `baseline`
- Date range: `2025-11-01` to `2025-12-04` (inclusive)
- Noise scales:
  - `team_sigma_scale = 1.0`
  - `player_sigma_scale = 1.0`
  - `rates_sigma_scale` as configured in `config/sim_v2_profiles.json` for profile `baseline`
- Worlds source: `artifacts/sim_v2/worlds_fpts_v2` via `scripts/sim_v2/eval_team_corr.py`

### Summary (same-team DK FPTS correlation, sim_v2)

Derived via `scripts/sim_v2/eval_team_corr.py` on sim_v2 DK FPTS worlds (`dk_fpts_world`):

| metric |  value |
|-------:|-------:|
| mean   | 0.1735 |
| median | 0.1782 |
| p10    | 0.1262 |
| p25    | 0.1564 |
| p75    | 0.1972 |
| p90    | 0.2157 |

Future baselines (e.g., different noise settings or seasons) can be appended as Baseline v2, Baseline v3, etc., with corresponding config details and summary tables.

## Baseline v2 (in-flight)

- Profile: `baseline` (updated)
- Date range: `2025-11-01` to `2025-12-04` (inclusive)
- Noise scales:
  - `team_sigma_scale = 0.7`
  - `player_sigma_scale = 1.0`
  - `rates_sigma_scale` as configured in `config/sim_v2_profiles.json` for profile `baseline`
- Worlds source: `artifacts/sim_v2/worlds_fpts_v2`

### Summary (same-team DK FPTS correlation, sim_v2)

Derived via `scripts/sim_v2/eval_team_corr.py` on sim_v2 DK FPTS worlds (`dk_fpts_world`) after applying the updated scales:

| metric |  value |
|-------:|-------:|
| mean   | 0.0861 |
| median | 0.0867 |
| p10    | 0.0577 |
| p25    | 0.0718 |
| p75    | 0.1012 |
| p90    | 0.1180 |
