# Operations Runbook — Daily Runs, Reruns, Backfills, and Triage

Generated (UTC): 2025-12-19T01:51:26Z  
Updated (UTC): 2025-12-19T04:35:21Z

This runbook is written for the **target orchestrated pipeline** described in `docs/audit/TARGET_ARCHITECTURE.md` (target: **self-hosted Prefect Server + local worker**). It is designed to make failures diagnosable in minutes and eliminate daily manual cleanup.

## Key Concepts

- `game_date`: ET slate date being produced
- `run_as_of_ts`: UTC timestamp for “as-of correctness” (what data was allowed to be used)
- `run_id`: unique id for a concrete execution instance; stable across all steps for a partition
- **Run manifests**: per-step JSON artifacts recording inputs, outputs, checks, durations, and failure reason

Manifest root (target):

`/home/daniel/projections-data/manifests/<asset>/game_date=<DATE>/run=<run_id>/manifest.json`

## Daily Operation

### What happens automatically

- The orchestrator triggers the daily/live flow(s) on schedule.
- Each step logs to the orchestrator UI and writes a manifest on completion.
- “Production/current” pointers (e.g., `current_run.json`) are updated only after all validations pass.

### What to check each day (2-minute checklist)

1) Open orchestrator UI → verify today’s `live` flow run is `SUCCESS`.
2) If `FAILED`, open the failing task run → read the error + retry count.
3) Confirm manifests exist for the expected steps for `game_date=<today>`.
4) Confirm the “current” pointers were updated (if the publish step ran).

## Reruns / Safe Retry Rules

### Golden rule

Reruns must never require manual deletion of outputs. If a rerun requires cleanup, the step is not truly idempotent yet.

### Safe retry categories

**Category A — safe to retry in-place (same `run_id`)**
- Steps that write only under `.../run=<run_id>/...` and do not publish until the end.
- Examples: feature builds, model scoring, sim worlds generation (if run-scoped).

**Category B — retry by creating a new `run_id`**
- Steps that currently have mutable publishes or interact with “current” pointers directly.
- Rationale: keep the failed run intact for forensics; rerun creates a clean immutable output set.

**Category C — do not retry blindly**
- Steps failing due to contract violations (missing required columns, schema drift).
- Steps failing due to permissions/ownership problems.
- Fix underlying cause first, then retry.

### Recommended rerun procedure

1) Retry the failed task from the orchestrator UI (same flow run) if it’s Category A.
2) If still failing, trigger a new flow run for the same `game_date` (new `run_id`) and allow publish to update “current”.
3) Never manually edit or delete data partitions to “get unstuck”; instead rely on manifests + new run_id.

## Backfills (First-Class)

Backfills run the exact same DAG as daily runs over a larger set of `game_date` partitions. See `docs/audit/BACKFILL_DESIGN.md`.

### Interface (target)

```
pipeline run \
  --start YYYY-MM-DD \
  --end YYYY-MM-DD \
  --as-of-mode pretip \
  --max-parallel 4 \
  --resume
```

### Resume procedure

1) Re-run the same command with `--resume`.
2) The orchestrator skips partitions where manifests show `success`.
3) Only failed partitions are retried; no manual cleanup needed.

### What “done” looks like

- Backfill summary artifact exists:
  - `/home/daniel/projections-data/manifests/backfills/<backfill_id>/summary.json`
- For every partition, manifests exist for each configured step.

## Promoting Model Heads to “Current/Production”

Model promotion should be **atomic** and reversible.

Current selector files (as of audit):
- `config/minutes_current_run.json`
- `config/rates_current_run.json`
- `config/usage_shares_current_run.json`
- Ownership model run is hardcoded in `projections/cli/score_ownership_live.py` (target: move to a config pointer)

### Safe promotion procedure (target)

1) Train/evaluate the candidate run; produce a model artifact directory with a `meta.json`.
2) Run a smoke scoring job against a recent `game_date` (no publish) and confirm expected output shape.
3) Update the relevant `config/*_current_run.json` pointer via atomic write.
4) Trigger a daily flow run for today and confirm manifests indicate the new model run was used.
5) Rollback = restore the previous pointer file contents and rerun.

## Where to Find Things

- Orchestrator UI:
  - flow runs / task runs / retry history / logs
- Manifests:
  - `/home/daniel/projections-data/manifests/...`
- External cron logs (until migrated):
  - `/home/daniel/dkresults/logs/*.log`
- Legacy systemd logs (during migration):
  - `journalctl -u <unit>`

## Triage Flowchart (On-Call Style)

```
START
  |
  v
Orchestrator shows FAILURE?
  |-- no --> DONE
  |
  v
Identify failing task (ingest / minutes / rates / sim / ownership / finalize)
  |
  +--> Network/DNS/SSL error?
  |      - Check endpoint availability
  |      - Retry task (with backoff); if persistent, mark as upstream outage
  |
  +--> Missing input partition / FileNotFound?
  |      - Check manifests for upstream step success
  |      - Verify run_id propagation (same run_id across steps)
  |      - Rerun upstream step(s) for same partition
  |
  +--> Contract/schema violation (missing columns)?
  |      - Inspect upstream output schema (manifest output stats)
  |      - Treat as code/data contract regression; do not “blind retry”
  |
  +--> Permission denied?
  |      - Inspect ownership/mode on target directories
  |      - Fix permissions at the filesystem level; then retry
  |
  +--> Port/service flapping (APIs)?
         - Check for port conflicts / duplicate services
         - Fix service config; unrelated to ETL success but affects visibility
```

## Common Failure Signatures (from current system) and Fast Diagnosis

These are current observed patterns that the target system should surface clearly via manifests:

- `address already in use (0.0.0.0:8501)` → API port conflict (`live-api.service` vs `minutes-dashboard.service`)
- `Requested run=... missing under .../features_minutes_v1/...` → run_id mismatch (legacy runner)
- `Roster snapshot is Xh old` → backfill ran without correct as-of selection or without backfill mode
- `missing required columns [...]` → schema drift; upstream contract broken
- `Temporary failure in name resolution` / `SSL handshake timed out` → network fragility; retry with backoff/jitter
