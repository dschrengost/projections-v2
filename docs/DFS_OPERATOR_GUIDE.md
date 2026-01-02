# DFS Operator Guide (Live Nightly Use)

This repo is optimized for **“play DFS tonight”**: fast reruns, late-news robustness, and artifacts that make failures obvious.

## Key Concepts

- `game_date`: slate date (YYYY-MM-DD)
- `run_id`: run-scoped identifier for *this* execution (timestamp-like)
- `run_as_of_ts`: what information was allowed to be used (UTC timestamp)
- **Source of truth for the optimizer**: unified projections parquet for a `run_id`

## One-Command Readiness Check

Run this anytime you’re about to build lineups (and again near lock):

```bash
uv run python -m projections.cli.dfs_readiness --date YYYY-MM-DD --strict
```

It validates:
- stale inputs near lock (odds / injuries / ESPN / Rotowire)
- missing artifacts (salaries / projections)
- run_id lineage mismatches
- invariants (no NaNs, minutes bounds, team minutes sum ~240, ownership sanity)

## Live-Day Run Recipe

Set your data root:

```bash
export PROJECTIONS_DATA_ROOT="/home/daniel/projections-data"
```

### 1) Scrape / refresh inputs

```bash
LIVE_START_DATE=YYYY-MM-DD scripts/run_live_scrape.sh
```

### 2) Build full projections (minutes → rates → sim → ownership → finalize)

```bash
LIVE_START_DATE=YYYY-MM-DD LIVE_SIM_PROFILE=today scripts/run_live_score.sh
```

### 3) Validate readiness

```bash
uv run python -m projections.cli.dfs_readiness --date YYYY-MM-DD --strict
```

### 4) Build lineups (expected-value projections)

The optimizer can now choose between **conditional** and **availability-weighted** means:

```bash
# Default (recommended when you can late swap and trust news): E[FPTS | plays]
export PROJECTIONS_OPTIMIZER_PROJ_MODE="cond"

# Optional (bake in availability risk): E[FPTS]
export PROJECTIONS_OPTIMIZER_PROJ_MODE="uncond"
```

If you’re using the optimizer API/service, this environment variable affects projection selection in the player pool builder.

## Locked Games: Freeze + Keep Full Slate

After games start, we want **two things**:
1) stop re-scoring already-tipped games (avoid post-lock “metrics drift”)
2) still keep a **full-slate** player pool for late swap tooling

This is handled by:
- `build_minutes_live --lock-buffer-minutes 0` (default): skips already-tipped games; use `-1` only for debugging/backfills
- `finalize_projections --merge-locked-games` (enabled in `scripts/run_live_score.sh`): pulls locked games from the **latest pre-tip projections run** and merges them into the current snapshot

For traceability:
- `artifacts/projections/YYYY-MM-DD/run=<run_id>/projections.parquet` includes `row_source_run_id` + `row_source_reason`
- `artifacts/projections/YYYY-MM-DD/run=<run_id>/summary.json` includes `locked_game_merge`

## Late News / Rerun Procedure

1) Re-run the scrape stage (or let `run_live_score.sh` do it):
```bash
LIVE_START_DATE=YYYY-MM-DD scripts/run_live_scrape.sh
```

2) Force a new full pipeline run:
```bash
LIVE_START_DATE=YYYY-MM-DD LIVE_FORCE_RUN=1 scripts/run_live_score.sh
```

3) Re-run readiness (near lock, treat failures as actionable):
```bash
uv run python -m projections.cli.dfs_readiness --date YYYY-MM-DD --strict
```

If the check says inputs are stale, the most common causes are:
- scraper failed (look for warnings printed by `projections.cli.live_pipeline`)
- upstream site lagged (you’ll see old `as_of_ts` / `ingested_ts`)

## What to Inspect When Something Looks Off

All paths below are rooted at `$PROJECTIONS_DATA_ROOT`.

- Minutes live features summary (now includes ESPN + Rotowire freshness):
  - `live/features_minutes_v1/YYYY-MM-DD/run=<run_id>/summary.json`
- Minutes scored output:
  - `artifacts/minutes_v1/daily/YYYY-MM-DD/run=<run_id>/minutes.parquet`
  - `artifacts/minutes_v1/daily/YYYY-MM-DD/run=<run_id>/summary.json`
- Rates scored output:
  - `gold/rates_v1_live/YYYY-MM-DD/run=<run_id>/rates.parquet`
  - `gold/rates_v1_live/YYYY-MM-DD/run=<run_id>/summary.json`
- Sim output + diagnostics:
  - `artifacts/sim_v2/worlds_fpts_v2/game_date=YYYY-MM-DD/run=<run_id>/projections.parquet`
  - `artifacts/sim_v2/worlds_fpts_v2/game_date=YYYY-MM-DD/run=<run_id>/metrics.json`
  - `artifacts/sim_v2/worlds_fpts_v2/game_date=YYYY-MM-DD/run=<run_id>/sim_diagnostics.json`
  - `artifacts/sim_v2/worlds_fpts_v2/game_date=YYYY-MM-DD/run=<run_id>/worlds_matrix.parquet` (per-player FPTS per world)
- Unified projections (optimizer input):
  - `artifacts/projections/YYYY-MM-DD/run=<run_id>/projections.parquet`
  - `artifacts/projections/YYYY-MM-DD/run=<run_id>/summary.json`
- Ownership predictions (per slate, with lock cache):
  - `silver/ownership_predictions/YYYY-MM-DD/run=<run_id>/<draft_group_id>.parquet`
  - `silver/ownership_predictions/YYYY-MM-DD/<draft_group_id>_locked.parquet`

### Spot-check Tail Worlds (Ceilings / Correlation)

The sim writes a full `worlds_matrix.parquet` (row = world index, col = `player_id`).

```bash
# Tail worlds for a player (uses latest_run.json when --run-id omitted)
uv run python scripts/sim_v2/inspect_worlds_matrix.py tails YYYY-MM-DD PLAYER_ID --top-k 25

# Inspect a specific world (slice to the player’s game)
uv run python scripts/sim_v2/inspect_worlds_matrix.py world YYYY-MM-DD WORLD_IDX --player-id PLAYER_ID --top-n 20
```

## Notes on Projection Semantics (Important)

- `dk_fpts_mean` is **conditional** on the player being active (`E[FPTS | plays]`).
- `dk_fpts_mean_uncond` is the **availability-weighted expected value** (`E[FPTS]`).
- For lineup building with reliable late news + swaps, prefer conditional (`cond`) so you don’t systematically fade Q players who are likely to be in.
- Use `uncond` when you explicitly want to pay for availability risk (or when inputs are stale and you want a conservative EV).
