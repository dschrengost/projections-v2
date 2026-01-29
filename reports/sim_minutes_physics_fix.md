# Sim Minutes Physics Fix (2026-01-28)

This report captures before/after diagnostics for the minutes-worlds “physics” fix:
prevent sparse-team availability draws from forcing the team-240 allocator to create
unrealistic “bench sponge” cap worlds.

## Repro Commands

Diagnostics (1000 worlds, all teams on slate):

```bash
# BEFORE (no feasibility gate, no absorption caps)
uv run python -m scripts.diagnostics.world_sparsity_stats \
  --date 2026-01-28 --n-worlds 1000 --data-root /home/daniel/projections-data --no-physics

# AFTER (sim_v3 profile with physics enabled)
uv run python -m scripts.diagnostics.world_sparsity_stats \
  --date 2026-01-28 --n-worlds 1000 --data-root /home/daniel/projections-data
```

Acceptance-mode (fails non-zero if thresholds are violated):

```bash
uv run python -m scripts.diagnostics.world_sparsity_stats \
  --date 2026-01-28 --n-worlds 1000 --data-root /home/daniel/projections-data --assert
```

## Summary Metrics

These metrics are from `scripts/diagnostics/world_sparsity_stats.py` with `--hard-cap 48`:

Before (`--no-physics`):
- Teams: 18
- `max P(n_active < 8)`: 0.545
- `p95 P(n_active < 8)`: 0.453
- `max P(any cap48)`: 0.234
- `p95 P(any cap48)`: 0.223

After (physics enabled):
- Teams: 18
- `max P(n_active < 8)`: 0.000
- `max P(any cap48)`: 0.000
- `allocator infeasible rate`: 0.0 for all teams
- Resampling diagnostics:
  - `mean frac_worlds_infeasible_pre_resample`: 0.741
  - `mean avg_resample_attempts`: 4.064 (max team avg: 8.208)

## Worst-Offending Teams (Before)

Top teams by `P(n_active < 8)` and cap hits (all values are per-team across 1000 worlds):

| game_id | team_id | P(n_active<8) | P(any cap48) | n_active p50 | sum_demand_active p50 |
|---:|---:|---:|---:|---:|---:|
| 22500677 | 1610612742 | 0.545 | 0.221 | 7 | 182.7 |
| 22500675 | 1610612752 | 0.437 | 0.170 | 8 | 198.9 |
| 22500674 | 1610612748 | 0.306 | 0.234 | 8 | 182.8 |
| 22500679 | 1610612759 | 0.282 | 0.215 | 8 | 176.3 |
| 22500676 | 1610612763 | 0.261 | 0.189 | 8 | 194.1 |

After the fix, all teams on 2026-01-28 show:
- `P(n_active < 8) == 0`
- `P(any cap48) == 0`

