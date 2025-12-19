# Migration Plan — Incremental, Rollbackable, “Fast Wins First”

Generated (UTC): 2025-12-19T01:51:26Z  
Updated (UTC): 2025-12-19T04:35:21Z

This plan avoids a big-bang rewrite. It moves scheduling, dependencies, retries, and observability into a single orchestrator (target: **self-hosted Prefect Server + local workers**) while initially treating existing scripts as black boxes.

Key references:
- Inventory: `docs/audit/pipeline_inventory.json`
- Root causes: `docs/audit/ROOT_CAUSES.md`
- Target design: `docs/audit/TARGET_ARCHITECTURE.md`
- Backfill strategy: `docs/audit/BACKFILL_DESIGN.md`

## Phase 1 — Fast Wins (1–3 days)

Goal: one scheduler + one UI + consistent logs, without changing pipeline internals.

### Tasks

1) **Stand up self-hosted Prefect + local worker**
   - Run Prefect Server (UI + API) and its DB (typically Postgres), supervised via systemd or containers.
   - Run one local Prefect worker on the machine (kept alive by a single systemd service) pointing at the local Prefect API URL.
   - Decide what is durable “truth”:
     - Prefect DB backed up (long UI history), and/or
     - manifests under `/home/daniel/projections-data/manifests/...` as the durable record (recommended regardless).

2) **Wrap existing systemd jobs as orchestrated tasks (black-box)**
   - Create flows that run the existing entrypoints as subprocess tasks:
     - `scripts/run_live_scrape.sh`
     - `scripts/run_live_score.sh`
     - `scripts/run_boxscores.sh`
     - `scripts/run_post_slate_minutes_labels.sh`
     - `scripts/run_tracking_daily.sh`
     - `python -m projections.cli.freeze_slates freeze-pending`
     - `python -m scripts.dk.run_daily_salaries ...`
     - `python -m scripts.analyze_accuracy ...`
   - Ensure each task captures stdout/stderr into orchestrator logs.

3) **Centralize schedules in the orchestrator**
   - Recreate systemd timer cadences as Prefect schedules.
   - Keep the current schedules initially (don’t change “when” yet).

   Practical sequencing recommendation:
   - Start by migrating only `live-score` and `live-scrape` schedules first, leaving the other systemd timers untouched until Phase 1 is stable.

4) **Minimal run manifests (no internal rewrites)**
   - For each task, emit a manifest JSON with:
     - `{task_name, start/end, exit_code, duration, key output paths found, error snippet if failed}`
   - Store under `/home/daniel/projections-data/manifests/...` as described in `TARGET_ARCHITECTURE.md`.

5) **Disable duplicate schedulers only after validation**
   - First run Prefect schedules in “observe-only” mode (run alongside systemd but with tasks disabled / dry-run where possible).
   - Then disable:
     - user systemd timers: `~/.config/systemd/user/live-score.*`, `~/.config/systemd/user/live-scrape.*`
     - system timers for jobs migrated to Prefect (starting with `live-score.timer` and `live-scrape.timer`)

6) **Replace broken failure alerting**
   - Stop relying on systemd `OnFailure=` (currently points to a missing unit).
   - Use Prefect notifications (email/Slack) for failed flow runs.

### Risks and mitigations

- **Risk:** Prefect schedule double-runs with systemd while migrating.  
  **Mitigation:** Start with manual triggers; only disable systemd after several successful Prefect-driven runs.

- **Risk:** Black-box scripts don’t expose step-level retries.  
  **Mitigation:** Retry at the task level in Phase 1; split into step-level tasks in Phase 2.

### Rollback

- Re-enable systemd timers and disable Prefect schedules (or pause Prefect deployments).
- Because Phase 1 does not change pipeline code paths, rollback is low-risk.

## Phase 2 — Make the DAG First-Class (1–2 weeks)

Goal: explicit partitions, idempotent writes, atomic publish, and step-level observability.

### Tasks

1) **Split `run_live_score.sh` into orchestrator-native steps**
   - Replace the monolith with discrete tasks that call underlying CLIs directly:
     - `projections.cli.live_pipeline` (ingest)
     - `projections.cli.build_minutes_live` (features)
     - `projections.cli.score_minutes_v1`
     - `projections.cli.build_rates_features_live`
     - `projections.cli.score_rates_live`
     - `scripts.sim_v2.run_sim_live`
     - `projections.cli.score_ownership_live`
     - `projections.cli.finalize_projections`
     - `projections.cli.check_health`
   - This enables targeted retries and makes failures diagnosable in minutes.

2) **Standardize run identity + partitions**
   - Generate one `run_id` per `game_date` run; pass it through every step.
   - Require `run_as_of_ts` for time-varying inputs and record selections in manifests.

3) **Atomic publish everywhere**
   - Introduce a consistent “temp → promote” convention for gold/live outputs.
   - Ensure pointer updates (`current_run.json`, `latest_run.json`) are atomic (`os.replace`).

4) **Data contracts at boundaries (minimal but strict)**
   - Add Pandera checks on critical boundaries:
     - features → scoring inputs
     - required season aggregate columns for rates features
     - tracking roles required columns
   - Fail fast with clear “contract broken” messages and record in manifests.

5) **Remove time-gated correctness logic**
   - Replace “skip if locked” with explicit `run_as_of_ts` partitions.
   - Daily publishes can still be gated (publish policy), but scoring should be deterministic.

6) **Make pipeline status append-only**
   - Replace/augment `bronze/pipeline_status` with manifest-driven append-only run events (or a simple DuckDB table).

### Risks and mitigations

- **Risk:** This phase touches code and storage contracts.  
  **Mitigation:** Keep a compatibility layer: continue writing existing paths while adding new run-scoped paths and pointers.

### Rollback

- Keep Phase 1 “black-box” flows runnable while iterating on Phase 2.
- If a refactor introduces instability, revert to calling the monolith scripts while preserving orchestrator scheduling/visibility.

## Phase 3 — Optional Infra Changes (Only if it reduces toil)

Goal: reduce operational complexity, not “cloud for cloud’s sake”.

Possible upgrades (only with clear ROI):

- **Storage**: move large artifacts (worlds, parquet) to object storage if disk management is a recurring pain.
- **Central logs**: add Loki/promtail if Prefect logs + manifests aren’t sufficient.
- **Compute isolation**: containerize workers if dependency drift becomes a problem.

Explicit non-goals:
- Migrating to cloud compute or managed warehouses unless it demonstrably reduces daily ops and total complexity.
