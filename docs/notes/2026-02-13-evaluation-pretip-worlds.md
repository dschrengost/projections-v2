# Evaluation Semantics Update: Pre-Tip Snapshots + Worlds MAE

This note documents the evaluation behavior used by `scripts/analyze_accuracy.py` and shown in the dashboard.

## What changed

- **Snapshot selection per game** now uses the **closest pre-tip run**.
  - For each `game_id`, evaluation picks the run with the latest `run_ts` such that `run_ts <= tip_ts`.
  - `tip_ts` is read from `silver/schedule/.../schedule.parquet`.
- **MAE point estimates** are now world-derived by default:
  - FPTS MAE source preference: `dk_fpts_p50` → `fpts_sim_cond_p50` → `fpts_sim_uncond_p50` → `dk_fpts_mean`
  - Minutes MAE source preference: `minutes_sim_p50` → `minutes_p50_cond` → `minutes_p50` → `minutes_mean`

## Why

- Prevents post-tip leakage from later snapshots.
- Aligns reported MAE with what downstream world-based workflows consume.

## Added evaluation diagnostics

Per-day metrics now include:

- `snapshot_selection_mode`
- `games_with_tip`
- `games_with_pre_tip_snapshot`
- `games_missing_pre_tip_snapshot`
- `pretip_snapshot_coverage`
- `selected_runs_count`
- `selected_game_run_map` (game-to-run audit map)
- `fpts_point_source`
- `minutes_point_source`

The dashboard summary aggregates these fields to show pre-tip coverage and source columns used for MAE.
