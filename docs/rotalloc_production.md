# RotAlloc production mode (minutes allocation)

## Enable / disable

- Config (default): `config/minutes_current_run.json`
  - `minutes_alloc_mode`: `"legacy"` or `"rotalloc_expk"`
  - `rotalloc_bundle_dir`: path to a directory containing `promote_config.json` and `models/`
- Kill switch (overrides config):
  - `PROJECTIONS_MINUTES_ALLOC_MODE=legacy` (force legacy)
  - `PROJECTIONS_MINUTES_ALLOC_MODE=rotalloc_expk` (force RotAlloc)

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

Notes:
- Sim source of truth is `worlds_fpts_v2` (`projections.parquet` under `game_date=.../run=...`). Legacy `artifacts/sim_v2/projections` is ignored by `finalize_projections` unless `--allow-legacy-sim-projections-root` is set.

Validate pointer consistency for a date (fails non-zero on mismatch):
```bash
uv run python -m projections.cli.check_health check-artifact-pointers --date 2025-12-20
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
