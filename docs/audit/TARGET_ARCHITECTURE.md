# Target Architecture — Reliable, Reproducible, Observable (Daily + Backfill = Same DAG)

Generated (UTC): 2025-12-19T01:51:26Z  
Updated (UTC): 2025-12-19T04:35:21Z

This is the proposed “minimal but robust” target architecture that prioritizes:

- **Reliability**: reruns are safe; partial outputs don’t poison downstream steps
- **Repeatability**: historical rebuilds are deterministic (no hidden “latest” reads)
- **Observability**: failures are diagnosable in minutes (single pane of glass + run manifests)

It is designed to be adopted **incrementally**: Phase 1 wraps existing scripts as black-box tasks without rewriting the pipeline internals (see `docs/audit/MIGRATION_PLAN.md`).

## Non-Negotiables (as required)

1) One orchestrator is the source of truth for schedule + dependencies  
2) **Daily runs and backfills use the same DAG** via partitions (no separate backfill codepath)  
3) Each step emits a **run manifest** (inputs + outputs + checks + durations + status)  
4) Production “current” artifacts are published via **atomic promotion** (temp/run_id → promote)  
5) Centralized visibility for “what ran, what failed, why” in one UI

## Orchestrator Evaluation (Brief)

### Option A: Prefect Cloud (control plane + local workers)

**Pros:**
- Minimal operational overhead (no server/DB/UI for you to run).
- Fast to start; great if you accept SaaS as the “source of truth” UI.

**Cons (given your pivot):**
- Adds an external dependency (Cloud availability + connectivity).
- Run history/retention limits can force you to build a second “truth store” anyway (manifests).

### Option B: Prefect self-hosted server (DB/UI run by you) — Recommended (as requested)

**Why it fits your constraints:**
- Removes reliance on a third-party control plane; everything runs on your infrastructure.
- Lets you choose retention policies and keep operational history longer than typical free SaaS tiers.
- Your pipeline already depends on local services (systemd, MLflow sqlite, local data root); adding one more service is manageable if kept minimal.

**Operational requirements (be explicit):**
- Run Prefect Server (API + UI) and its DB (typically Postgres).
- Back up the Prefect DB (or accept that Prefect UI history is ephemeral and rely on manifests as the durable record).
- Keep server and workers supervised (systemd or containers).

**Cons:**
- You own the uptime and upgrades of the orchestrator control plane.
- Requires basic ops hygiene (DB backups, disk monitoring, service health checks).

### Option C: Dagster (asset-based) — Strong long-term, slower short-term

**Pros:**
- Asset model fits your medallion/data-asset world extremely well.
- Great observability/checks for assets, partitions, and materializations.

**Cons (for “fast wins”):**
- Higher upfront modeling cost; you’ll likely need more refactors earlier.
- Black-box script wrapping is possible, but the real payoff requires assetization.

### Option D: Airflow — Not recommended here

**Reason:**
- Heavier operational footprint and complexity.
- For “local heavy compute + many Python scripts”, Prefect/Dagster are typically a better fit and faster to iterate.

**Decision:** Use **Prefect self-hosted server + local workers** as the single source of truth for schedules and run status. Keep **local manifests** under `/home/daniel/projections-data/manifests/...` as the durable system of record for run provenance and long-term history.

## Target Data Model: Partitions and Run Identity

Today, many steps depend on wall-clock time and mutable “latest” inputs. The target design makes these explicit.

### Canonical partition keys

Not every asset needs all keys; pick the minimal set per asset.

- `game_date` (ET date): the slate date being produced/consumed
- `run_as_of_ts` (UTC timestamp): the “data cut” timestamp for as-of correctness
- `run_id` (string): a unique id for a concrete execution instance (e.g., `20251219T020143Z`)
- `draft_group_id` (string/int): DK slate id for ownership/finalize
- `season`/`month`: for season/month partitioned assets (schedule, monthly snapshots)

### Rules

1) Every *materialization* step writes to an **immutable run directory** keyed by its partition keys (at least `game_date` + `run_id`).  
2) No step reads from “latest” during backfill; the orchestrator passes `run_as_of_ts` and/or exact input partitions.  
3) If a “current/production” view is needed, it is a **pointer** (json or symlink) updated atomically after success.

## Run Manifests (Required for Every Step)

### Manifest storage

Store manifests under a single root for easy querying and backfill summaries:

`/home/daniel/projections-data/manifests/<asset_name>/game_date=<YYYY-MM-DD>/run=<run_id>/manifest.json`

(For non-`game_date` assets: use the relevant partition keys in the path.)

### Minimal manifest schema (v1)

Each step emits a JSON with:

- `manifest_version`: `"v1"`
- `asset`: canonical step/asset name (e.g., `minutes.features_live`)
- `partition`: `{game_date, run_as_of_ts, run_id, draft_group_id, season, month}`
- `status`: `success|failed|skipped`
- `timing`: `{started_at, finished_at, duration_s}`
- `inputs`: list of `{name, path, pointer_hash?, row_count?, file_count?, size_bytes?, mtime?}`
- `outputs`: list of `{name, path, row_count, null_counts?, min_max?, unique_checks?}`
- `checks`: explicit validations run (and pass/fail)
- `error`: `{type, message, traceback_snippet}` on failure
- `environment`: `{git_sha, code_version, hostname, python, uv_env, orchestrator_run_id}`

This is intentionally minimal. It’s enough to answer, in minutes:

- What input partitions were used?
- Where did outputs land?
- Were outputs complete and sane?
- How long did it take?
- What failed and where?

## Data Contracts / Expectations

You already depend on `pandera` and `pydantic`. The minimal approach:

- **Pydantic** for:
  - pointer files (e.g., `current_run.json`)
  - manifest schema
  - “run config” objects passed through tasks
- **Pandera** for:
  - boundary schemas between stages (bronze→silver, silver→features, features→scores)
  - small critical checks: required columns, dtypes, key uniqueness, min/max constraints

Avoid Great Expectations initially (heavier operational footprint; more config than value for early phases).

## Atomic Publish (“temp → promote”)

Wherever “current” outputs exist, publish must be atomic.

### Pattern

1) Write step outputs to a staging path:
   - `.../run=<run_id>/_staging/<uuid>/...`
2) Validate outputs (pandera checks + basic row counts).
3) Atomically promote:
   - Use `os.replace()` for pointer JSON files (atomic on POSIX).
   - Use atomic directory rename for final output folder when possible.
4) Update `current_run.json` (or symlink) only after promotion succeeds.

This prevents consumers from reading half-written parquet or a pointer to non-existent data.

## Centralized Logging + “Single Pane of Glass”

### Primary UI

- Prefect UI is the **single source of truth** for:
  - schedules
  - flow runs and task runs
  - retries and failures
  - per-partition backfill progress

### Log strategy (pragmatic)

Phase 1 (fast wins):
- Use Prefect task logs as the canonical log stream.
- Stop relying on scattered journald for “what happened”.
- Store stdout/stderr of wrapped scripts as Prefect logs + attach as Prefect artifacts for failed runs.

Phase 2 (optional):
- Add a lightweight local log aggregator (e.g., Loki + promtail) only if needed.

## Proposed DAG (Assets/Jobs)

This DAG reflects the *existing pipeline structure*, but makes partitions explicit and removes “time gating” as a correctness mechanism.

```mermaid
graph TD
  Sched[Scheduler (Prefect)] --> Scrape[ingest.scrape_all(game_date, run_as_of_ts)]
  Scrape --> MinutesFeat[minutes.features_live(game_date, run_as_of_ts, run_id)]
  MinutesFeat --> MinutesScore[minutes.score(game_date, run_id, model_run)]
  MinutesScore --> RatesFeat[rates.features_live(game_date, run_id)]
  RatesFeat --> RatesScore[rates.score(game_date, run_id, model_run)]
  RatesScore --> Sim[sim.run(game_date, run_id, profile, worlds)]
  RatesScore --> Salaries[dk.salaries(game_date)]:::optional
  Salaries --> Ownership[ownership.score(game_date, draft_group_id, run_id, model_run)]:::optional
  Sim --> Finalize[finalize.unified(game_date, draft_group_id, run_id)]
  Ownership --> Finalize
  Finalize --> Health[health.checks(game_date, run_id)]

  Box[boxscores(game_date)] --> Labels[minutes.labels(game_date)]
  Track[tracking.scrape(date)] --> TrackRoles[tracking.roles(season_range)]

  classDef optional fill:#f6f6f6,stroke:#999,stroke-dasharray: 3 3;
```

### Notes

- `run_id` is generated once per `game_date` run (daily or backfill) and passed through every step that writes run-scoped outputs.
- `run_as_of_ts` is required for anything that needs “as-of correctness” (scrapes, snapshot selection, “what did we know at time T”).
- Ownership is *not* “skip if locked”. Instead:
  - For daily runs, you choose `run_as_of_ts` (e.g., 15 minutes before first tip) and run it deterministically.
  - For backfills, you choose the historical `run_as_of_ts` partition.

## How Existing Scripts Fit (No Rewrite Needed Initially)

Phase 1 uses Prefect tasks that call the existing scripts:

- `scripts/run_live_scrape.sh` → `ingest.scrape_all`
- `scripts/run_live_score.sh` → split into multiple Prefect tasks by calling the underlying CLIs directly (preferred) or wrap the script as one task initially (fastest)
- `scripts/run_boxscores.sh` → `boxscores`
- `scripts/run_post_slate_minutes_labels.sh` → `minutes.labels`
- `scripts/run_tracking_daily.sh` → `tracking.scrape` + `tracking.roles`

Each wrapper task:
- passes explicit parameters (`game_date`, `run_id`, `run_as_of_ts`, `draft_group_id`)
- captures stdout/stderr into Prefect logs
- emits a manifest JSON on success/failure

## “Current / Production” Pointers

Standardize all “what’s current?” pointers as explicit, validated JSON under a single config root:

- Model pointers (already present):
  - `config/minutes_current_run.json`
  - `config/rates_current_run.json`
  - `config/usage_shares_current_run.json`
  - (Add missing pointers for any other production heads)
- Data pointers (new):
  - `/home/daniel/projections-data/gold/<asset>/game_date=<DATE>/current_run.json`
  - `/home/daniel/projections-data/live/<asset>/game_date=<DATE>/current_run.json`

Update pointers only via atomic replace after validation.
