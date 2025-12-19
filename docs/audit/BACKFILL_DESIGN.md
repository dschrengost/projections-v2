# Backfill Design — First-Class, Resumable, Deterministic (Same DAG as Daily)

Generated (UTC): 2025-12-19T01:51:26Z

This design makes historical backfills safe and reproducible by running the **same orchestrated DAG** used for daily runs, simply over a larger set of partitions. There is no separate backfill codepath.

## Goals

- Run daily and historical backfills via the **same DAG** and partition keys.
- Make replays deterministic via explicit **as-of selection** (no “latest” reads).
- Ensure reruns are safe via **immutable run directories** + **atomic publish**.
- Support resume after failure without manual cleanup.
- Produce per-partition manifests and a single backfill summary artifact.

## Core Concepts

### Partition keys

Use the minimal set per asset, but standardize names:

- `game_date` (ET): slate date
- `run_as_of_ts` (UTC): “what did we know as of time T?”
- `run_id` (string): unique execution id (stable across all steps for a given partition)
- `draft_group_id` (optional): DK slate id for ownership/finalize

### Deterministic snapshot selection (as-of)

For each input that is time-varying:

1) Identify the source’s history partitions (e.g., hourly injuries, odds by `run_ts`).
2) Select the **latest snapshot ≤ `run_as_of_ts`** for that `game_date` (or nearest applicable key).
3) Record the selected snapshot path(s) in the step manifest as explicit inputs.

If a source has no historical record (e.g., missing odds for a date), the manifest must mark it as a gap and downstream steps should either:
- run with “missing input policy” (explicitly configured), or
- fail fast with a clear error and a recommended remediation.

## Backfill Invariants (Rules)

1) **No mutable “latest” reads** during backfill (including “pick newest run directory by mtime”).
2) Every partition produces outputs in `.../run=<run_id>/...` (immutable) before any publish step.
3) Every partition emits a manifest (success/fail/skip) for every step.
4) Publish happens only after validations pass; publish updates are atomic.
5) Resume is driven by manifests, not by “what files happen to exist”.

## Partition Mapping (What keys apply where)

This is the target partition model (not necessarily current on disk today):

| Asset/Step | Partition keys | “As-of” selection required? | Notes |
|---|---|---:|---|
| `ingest.schedule` | `season`, `month` | No | Schedule is authoritative, mostly immutable |
| `ingest.injuries_raw` | `season`, `game_date`, `run_as_of_ts` | Yes | Backfill from stored hourly history when available |
| `ingest.odds_raw` | `season`, `game_date`, `run_as_of_ts` | Yes | Requires stored odds history; gaps must be explicit |
| `ingest.daily_lineups` | `season`, `game_date`, `run_as_of_ts` | Yes | Prefer stored bronze snapshots |
| `ingest.roster_snapshot` | `season`, `game_date`, `run_as_of_ts` | Yes | Depends on your historical retention (nightly snapshots) |
| `minutes.features_live` | `game_date`, `run_as_of_ts`, `run_id` | Yes | `run_id` must be stable and shared with scoring |
| `minutes.score` | `game_date`, `run_id` | No | Consumes feature partition selected above |
| `rates.features_live` | `game_date`, `run_id` | No | Deterministic given minutes features + season aggregates |
| `rates.score` | `game_date`, `run_id` | No | Deterministic given features + model run |
| `sim.run` | `game_date`, `run_id`, `profile`, `worlds` | No | Deterministic if seeded; record seed + profile in manifest |
| `ownership.score` | `game_date`, `draft_group_id`, `run_id`, `run_as_of_ts` | Yes | Must remove “skip if locked” semantics for backfills |
| `finalize.unified` | `game_date`, `draft_group_id`, `run_id` | No | Deterministic merge step (if inputs are deterministic) |
| `labels.boxscores` | `season`, `game_date` | No | Backfillable; external API/network dependent |
| `labels.minutes` | `season`, `game_date` | No | Deterministic given boxscores/labels |

## Concurrency Controls (Global + Per-Source)

Backfills should be aggressively parallel only where safe:

### Stage-level caps

- **Ingest stage** (network-bound): low parallelism; rate-limited
- **Feature build / scoring**: moderate parallelism; CPU-bound
- **Simulation**: strict caps; CPU + memory heavy
- **Publish/finalize**: low parallelism; avoid write contention

### Per-source rate limiting

Define concurrency limits per external source:

- NBA endpoints (schedule/boxscores/lineups/tracking): e.g., max 2–4 concurrent requests total
- Odds provider (oddstrader): e.g., max 1–2 concurrent; exponential backoff with jitter
- Any scraped HTML endpoints: serialize if they are brittle

In Prefect, implement with concurrency limits (tags/semaphores) rather than ad-hoc `sleep` loops.

## Resume/Checkpointing

Resume is driven by manifests:

- If a step manifest exists with `status=success` for `(asset, partition)`, skip by default.
- If `status=failed`, rerun only failed partitions unless `--force` is set.
- If a downstream step fails, upstream successful steps are not rerun unless explicitly requested.

This eliminates manual cleanup of partial directories and avoids “rerun makes state worse”.

## Provenance Outputs

### Per-partition manifests

Every step produces a manifest at:

`/home/daniel/projections-data/manifests/<asset>/game_date=<DATE>/run=<run_id>/manifest.json`

### Backfill summary artifact

Each backfill execution produces a single summary JSON/CSV:

`/home/daniel/projections-data/manifests/backfills/<backfill_id>/summary.json`

Contents:
- partitions attempted / succeeded / failed / skipped
- per-step durations and totals
- top error messages grouped by root cause signature
- input gaps encountered (e.g., missing odds history)

## Concrete CLI / Flow Interface Proposal

Expose a single interface that runs either daily or backfill depending on the date range:

```
pipeline run \
  --start YYYY-MM-DD \
  --end YYYY-MM-DD \
  [--season 2025] \
  [--as-of-mode {pretip,lock,custom}] \
  [--as-of-ts 2025-12-18T22:45:00Z] \
  [--max-parallel 4] \
  [--resume/--no-resume] \
  [--dry-run/--no-dry-run] \
  [--steps ingest,minutes,rates,sim,ownership,finalize] \
  [--worlds 25000] \
  [--profile sim_v3]
```

Behavior:
- Daily run is `--start=<today> --end=<today>`.
- Backfill is any multi-day range; same DAG, just mapped over partitions.
- `--as-of-mode` controls how to derive `run_as_of_ts`:
  - `pretip`: `first_tip_utc - lead_minutes`
  - `lock`: contest lock time if available
  - `custom`: use provided `--as-of-ts`
- `--resume` uses manifests to skip already-successful partitions.
- `--dry-run` plans/prints partitions + selected snapshots without executing.

## Known Gaps / Edge Cases (Must Be Explicit)

- Odds history may be incomplete for older dates; backfills must record gaps and either:
  - proceed with a configured “missing odds policy”, or
  - fail fast with a clear “cannot backfill without odds history” message.
- Current “roster age” assertions (`18h`/`720h`) are appropriate for live but should be configurable per run mode:
  - Daily: strict
  - Backfill: select roster snapshot by `run_as_of_ts` (or relax checks when history is known incomplete)
- Any step that currently uses “skip if locked” must be refactored into:
  - explicit `run_as_of_ts` selection + deterministic scoring, and
  - an explicit “do we publish?” policy (publish can still be skipped after lock, but scoring should be reproducible).

