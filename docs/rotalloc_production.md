# RotAlloc production mode (minutes allocation)

## Enable / disable

- Config (default): `config/minutes_current_run.json`
  - `minutes_alloc_mode`: `"legacy"`, `"rotalloc_expk"`, or `"rotalloc_fringe_alpha"`
  - `rotalloc_bundle_dir`: path to a directory containing `promote_config.json` and `models/`
- Kill switch (overrides config):
  - `PROJECTIONS_MINUTES_ALLOC_MODE=legacy` (force legacy)
  - `PROJECTIONS_MINUTES_ALLOC_MODE=rotalloc_expk` (force RotAlloc)
  - `PROJECTIONS_MINUTES_ALLOC_MODE=rotalloc_fringe_alpha` (force RotAlloc + core/fringe blend shape layer)

## What changes in outputs

- `projections/cli/score_minutes_v1.py` writes extra columns when RotAlloc is active:
  - `minutes_mean`, `minutes_alloc_mode`, `eligible_flag`, `p_rot`, `mu_cond`, `team_minutes_sum`
- When `minutes_alloc_mode=rotalloc_expk`, `score_minutes_v1`:
  - disables the minutes upside adjustment (sim_v2 already provides variance)
  - derives `minutes_p10`/`minutes_p90` by re-centering legacy tail deltas around the RotAlloc `minutes_p50`,
    clamped to a feasible per-team cap (prevents unrealistic `minutes_p90 > 50` starters).
- sim_v2 (`scripts/sim_v2/generate_worlds_fpts_v2.py`) masks availability/minutes by `eligible_flag` and writes `metrics.json` per run dir.
- `promote_config.json` should set `allocator.p_cutoff` to define the rotation eligibility cutoff.
  - If missing, RotAlloc defaults to `0.15` and **fails in CI** to force explicit config.
- `promote_config.json` can set `allocator.mu_power` to control how aggressively minutes concentrate into high-`mu_cond` players.
  - Default is `1.5` (higher → starters/6th man get more, 8th/9th man get less; helps avoid flattened bench minutes).
- RotAlloc will **fail in CI** if the rotation classifier outputs only a handful of discrete probabilities (guardrail against a collapsed classifier).

## rotalloc_fringe_alpha knobs (Allocator E)

`rotalloc_fringe_alpha` is a deterministic “shape layer” that blends RotAlloc proxy weights with a historical-minutes proxy inside the eligible set (fixes the bench spreading too wide when proxy weights are diffuse).

- Config defaults: `config/rotalloc_production.json` → `fringe_alpha_blend`
- Env overrides:
  - `ROTALLOC_BLEND_K_CORE` (default 8)
  - `ROTALLOC_BLEND_ALPHA_CORE` (default 0.8)
  - `ROTALLOC_BLEND_ALPHA_FRINGE` (default 0.3)
  - `ROTALLOC_BLEND_SHARE_GAMMA` (default 1.0)
  - `ROTALLOC_BLEND_SHARE_COL` (default auto-select from `roll_mean_5`, `min_last5`, `min_last3`, `roll_mean_10`)

## Local commands

Run Prefect flow locally (one date):
```bash
PROJECTIONS_MINUTES_ALLOC_MODE=rotalloc_expk uv run python prefect_flows/live_pipeline.py score --date 2025-01-01
```

Run the promotion smoke test:
```bash
uv run pytest -q -k rotalloc_promote_smoke
```

Force legacy allocation immediately (kill switch):
```bash
PROJECTIONS_MINUTES_ALLOC_MODE=legacy uv run python prefect_flows/live_pipeline.py score --date 2025-01-01
```

Verify run pointers:
- Minutes: `$PROJECTIONS_DATA_ROOT/artifacts/minutes_v1/daily/<YYYY-MM-DD>/latest_run.json`
- sim worlds: `$PROJECTIONS_DATA_ROOT/artifacts/sim_v2/worlds_fpts_v2/game_date=<YYYY-MM-DD>/latest_run.json`
- Unified projections: `$PROJECTIONS_DATA_ROOT/artifacts/projections/<YYYY-MM-DD>/latest_run.json`
- Optional pin (API prefers pinned over latest): `$PROJECTIONS_DATA_ROOT/artifacts/projections/<YYYY-MM-DD>/pinned_run.json`

Notes:
- Sim source of truth is `worlds_fpts_v2` (`projections.parquet` under `game_date=.../run=...`). Legacy `artifacts/sim_v2/projections` is ignored by `finalize_projections` unless `--allow-legacy-sim-projections-root` is set.

Validate pointer consistency for a date (fails non-zero on mismatch):
```bash
uv run python -m projections.cli.check_health check-artifact-pointers --date 2025-12-20
```

Pin the dashboard/API to a specific projections run (so a rescore doesn't get immediately replaced by live updates):
```bash
uv run python -m projections.cli.check_health pin-projections-run --date 2025-12-22 --run-id <PROJECTIONS_RUN_ID>
```

Auto-pin a run when invoking the live score script (useful for manual rescores):
```bash
LIVE_PIN_PROJECTIONS_RUN=1 /bin/bash scripts/run_live_score.sh
```

Clear the pin:
```bash
uv run python -m projections.cli.check_health pin-projections-run --date 2025-12-22
```

Debug missing minutes artifacts:
- Expected file: `$PROJECTIONS_DATA_ROOT/artifacts/minutes_v1/daily/<YYYY-MM-DD>/run=<MINUTES_RUN_ID>/minutes.parquet`
- Pointer: `$PROJECTIONS_DATA_ROOT/artifacts/minutes_v1/daily/<YYYY-MM-DD>/latest_run.json`
- If the parquet exists but the pointer is wrong/stale, re-run minutes scoring for the date (or update the pointer atomically).

Finalize projections with explicit upstream run IDs (recommended for backfills/debug):
```bash
uv run python -m projections.cli.finalize_projections \\
  --date 2025-12-20 \\
  --run-id <PROJECTIONS_RUN_ID> \\
  --minutes-run-id <MINUTES_RUN_ID> \\
  --sim-run-id <SIM_RUN_ID> \\
  --draft-group-id <DK_DRAFT_GROUP_ID> \\
  --data-root "$PROJECTIONS_DATA_ROOT"
```
