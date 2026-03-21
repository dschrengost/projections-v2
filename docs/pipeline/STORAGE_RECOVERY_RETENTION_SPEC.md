# Storage Recovery And Retention Redesign Spec
**Suggested filename:** `docs/pipeline/STORAGE_RECOVERY_RETENTION_SPEC.md`

## 0) Repository Anchors Used
- [docs/00_REPO_MAP.md](/home/daniel/projects/projections-v2/docs/00_REPO_MAP.md)
- [docs/10_CONTROL_PLANE.md](/home/daniel/projects/projections-v2/docs/10_CONTROL_PLANE.md)
- [docs/20_DATA_CONTRACTS.md](/home/daniel/projects/projections-v2/docs/20_DATA_CONTRACTS.md)
- [projections/paths.py](/home/daniel/projects/projections-v2/projections/paths.py)
- [projections/pipeline/control_plane.py](/home/daniel/projects/projections-v2/projections/pipeline/control_plane.py)
- [projections/pipeline/writer_guard.py](/home/daniel/projects/projections-v2/projections/pipeline/writer_guard.py)
- [prefect_flows/live_nba_pipeline_v3.py](/home/daniel/projects/projections-v2/prefect_flows/live_nba_pipeline_v3.py)
- [scripts/sim_v2/generate_worlds_fpts_v2.py](/home/daniel/projects/projections-v2/scripts/sim_v2/generate_worlds_fpts_v2.py)
- [scripts/sim_v2/run_sim_live.py](/home/daniel/projects/projections-v2/scripts/sim_v2/run_sim_live.py)
- [projections/contest_sim/contest_sim_service.py](/home/daniel/projects/projections-v2/projections/contest_sim/contest_sim_service.py)
- [scripts/cleanup_runs.py](/home/daniel/projects/projections-v2/scripts/cleanup_runs.py)
- [prefect.yaml](/home/daniel/projects/projections-v2/prefect.yaml)

---

## 1) Executive Summary
The outage pressure is primarily a storage-retention failure, not a modeling failure.

Observed on active data:
- `artifacts` is dominant at ~`567G`.
- `artifacts/gtv2_worlds` is ~`303G`.
- `artifacts/sim_v2` is ~`256G` (almost entirely `worlds_fpts_v2`).
- Model/training assets are much smaller (`training/runs` ~`14G`).

The live system currently persists many high-frequency world/sim payloads per slate date and run, while consumers mostly need:
- latest promoted run,
- pinned runs,
- one canonical pre-tip snapshot per slate/start bucket,
- minimal debug recency.

This is an operational storage-policy and lifecycle-control redesign:
- add deterministic canonical selection,
- split metadata vs payload retention,
- tier heavy payloads to archive HDD,
- add fail-closed free-space guards,
- re-enable automation only after deterministic dry-run and first manual slate validation.

---

## 2) Current State (Stabilized But Fragile)
Current runtime condition:
- Headless boot is functional.
- Minutes dashboard service is currently `disabled/inactive` (checked).
- Prefect worker is `enabled` but currently `failed` (not actively processing).
- Active roots are symlinked:
  - `/home/daniel/projections-data -> /mnt/relief/linux-rescue/projections-data`
  - `/home/daniel/prod -> /mnt/relief/linux-rescue/prod`
- `/mnt/relief` is `ntfs3`, ~`89%` used.
- `/` root ext4 is ~`90%` used.
- Prefect schedules still exist in config ([prefect.yaml](/home/daniel/projects/projections-v2/prefect.yaml)), so reactivating worker without guards can restart write pressure quickly.

Risk constraints:
- Root SSD requires cautious write profile until the platform is stable, even though current SMART checks are clean.
- Current hot path is on NTFS emergency mount, acceptable short-term but not ideal for sustained high-write Linux workloads.
- Heavy artifact families currently have no strict canonical retention enforcement.
- Pointer contracts are mixed (`LATEST/current.json` + `latest_run.json`), and consumers like contest sim still rely on legacy pointer behavior in places.

Recent host-health findings (2026-03-17 checks):
- Root NVMe (`/dev/nvme1n1`, CT1000P1SSD8): SMART `PASSED`, 0 media/data-integrity errors, 0 NVMe error-log entries.
- Relief NVMe (`/dev/nvme0n1`, Samsung 990 EVO): SMART `PASSED`, 0 media/data-integrity errors, 0 NVMe error-log entries.
- Logs show repeated unclean shutdown/journal recovery events; no persistent NVMe I/O error storm was observed in recent kernel logs.
- Operational conclusion: treat instability as primarily operational (space pressure + unclean shutdown), not confirmed SSD media failure.

---

## 3) Storage Tier Design
### Host Device Role Decision (Recovery Baseline)
Use a simple role split first; defer RAID until the system is stable:

| Device | Role | Notes |
|---|---|---|
| `/dev/nvme1n1` (Ubuntu/root SSD) | OS + repo + services | Keep heavy data writes off root where possible. |
| `/dev/nvme0n1` (Samsung SSD) | Hot data (`PROJECTIONS_DATA_ROOT`) | Preferred long-term on native Linux FS; NTFS rescue mount is temporary. |
| IronWolf HDD (new) | Warm/Cold archive | Canonical/pinned payload archives + retention receipts. |

RAID policy for this recovery program:
- Do **not** introduce RAID during initial recovery and retention rollout.
- RAID is a later optimization decision after stable operations.
- If revisited later, evaluate SSD-only RAID1 for hot data (not SSD+HDD mixed RAID), and keep archive copies regardless.

### Tier Definition
| Tier | Physical target | Purpose | Write profile |
|---|---|---|---|
| Hot | `/home/daniel/projections-data` (currently Samsung via symlink) | Active live pipeline inputs/outputs, pointers, current day payloads | Frequent read/write |
| Warm (optional logical) | `/mnt/archive/projections-data-archive/warm` (on IronWolf) | Recent canonical payloads and pinned payloads, still directly readable | Moderate |
| Cold/Archive | `/mnt/archive/projections-data-archive/cold` (on IronWolf) | Older payload bundles, long-term storage | Low |

### Placement Policy
| Data family | Hot | Warm | Cold |
|---|---|---|---|
| Repo code + active configs | Yes | No | No |
| `artifacts/runs/*` manifests | Yes (indefinite) | Mirror optional | Optional |
| `live/*` current day | Yes | No | No |
| `artifacts/gtv2_worlds` payload | Current day + canonical recent + pinned | Canonical + pinned | Aged canonical/pinned |
| `artifacts/sim_v2/worlds_fpts_v2` payload | Current day debug + canonical recent | Canonical + pinned | Aged canonical/pinned |
| `training/runs` large payloads | Active experiments only | Recent promoted refs | Older runs |
| `training/datasets` + `training/snapshots_minutes_v1` | Yes (default keep hot) | Optional mirror | Optional |
| `bronze/dk_contests/nba_gpp_data` raw contest CSVs | Last 7 days hot | Older recent dates | Older dates |
| `analytics/contest_results/*.parquet` normalized tables | Yes (keep hot) | Optional mirror | Optional backup |
| Live build inputs (`bronze`, `silver`, `gold`) for in-season operation | Yes | No (default) | No (default) |
| `gold/prediction_logs_minutes` | Recent rolling window hot | Older recent runs | Older runs |
| `builds/optimizer` + `builds/contest_sim` | Recent and pinned builds hot | Older recent builds | Older builds |
| `reports/experiments` | Recent window hot | Optional | Older runs |
| `tmp/*` (scratch/triton smoke/temp) | Very short TTL hot only | No | No |
| Legacy duplicate trees (`br/*`, top-level dated contest dirs) | No new writes; quarantine on hot | Optional staging | Archive as bulk legacy set |
| `bronze`/raw history | Hot rolling window | Optional | Older partitions |
| Reports/manifests/retention receipts | Yes | Mirror optional | Optional backup |

Notes:
- Phase 1 can run with Hot + Cold only; Warm is logical separation on same HDD path.
- Archive should be native Linux FS (`ext4` or `xfs`) on IronWolf; do not keep long-term heavy churn on NTFS rescue mount.
- IronWolf is archive/cold storage in this design, not part of live RAID.

---

## 4) Retention Policy Design
## 4.1 Canonical Key
Use deterministic key:
`(model_family, game_date, slate_signature, start_time_bucket_utc)`

Definitions:
- `model_family`: `gtv2_worlds` or `sim_v2_worlds_fpts_v2`.
- `game_date`: partition date.
- `slate_signature`: stable hash of sorted `(game_id, tip_ts)` from run manifest data.
- `start_time_bucket_utc`: 30-minute floor of earliest `tip_ts` in that slate signature.

Primary metadata source:
- run manifest at `artifacts/runs/nba_live/game_date=.../run=.../manifest.json` (pointer payload includes `manifest_path`).
Fallback for legacy runs:
- `sim_manifest.json`, `projections.parquet` `tip_ts`, or day-level pointer metadata.

## 4.2 Canonical Selection Rule
For each canonical key:
1. Candidate runs are runs in that family/date with valid `as_of_ts`.
2. Select run with max `as_of_ts` such that `as_of_ts <= first_tip_ts - lead_time`.
3. Default `lead_time = 2 minutes`.
4. Tie-breaker is lexicographic max `run_id`.
5. If no pre-tip candidate exists, pick earliest post-tip run and mark `canonical_degraded=true`.

## 4.3 Family-Specific Retention
### `artifacts/gtv2_worlds`
- Keep on hot:
  - latest promoted run for active date,
  - one canonical per key for recent window,
  - one latest same-day debug run,
  - pinned runs.
- Archive:
  - canonical and pinned payloads indefinitely (until manual policy revision).
- Delete from hot:
  - noncanonical payloads after archive verification and guard delay.
- Keep metadata-only indefinitely:
  - manifest linkage, retention decision, payload inventory.

### `artifacts/sim_v2/worlds_fpts_v2`
- New-style `run=*` directories:
  - same policy as gtv2.
- Legacy day-level `world=*.parquet` (no run dirs):
  - treat as legacy bulk payload.
  - archive day partitions older than policy window.
  - keep small metadata receipt in hot.
  - no immediate rebuild requirement for legacy payloads.

### `training/runs`
- Keep on hot:
  - runs referenced by active production selectors/promotions,
  - very recent active experiment window.
- Archive:
  - older unreferenced runs.
- Metadata-only indefinitely:
  - run summary, metrics index, artifact pointer map.

### Other relevant families
- `artifacts/projections` currently small relative to worlds; keep existing cleanup but align with canonical retention metadata.
- `bronze/gold/live` not primary storage driver; defer aggressive changes until world retention is stable.

## 4.4 Protection Rules
- Current-day protection: no prune/delete for `game_date=today`.
- Pinned-run protection: never prune payload for pinned run IDs.
- Pointer protection: any run referenced by `LATEST/current.json`, `latest_run.json`, `pinned_run.json`, or blessed pointer is protected.
- Two-man rule for destructive mode in first rollout: planner output must be reviewed before execute.

## 4.5 Contest Results + Datasets + Live Input Policy Addendum
### Contest result CSVs (`bronze/dk_contests/nba_gpp_data`)
- Keep raw contest CSVs hot for 7 days by default.
- Archive raw contest CSV dates older than 7 days via a weekly automated retention job.
- Do not delete raw contest CSV payload until:
  - archive copy verification is complete, and
  - `analytics/contest_results` has been refreshed for that date window, and
  - archive-aware read fallback is implemented for APIs that query historical contests.

### Normalized contest tables (`analytics/contest_results`)
- Keep hot by default; footprint is small relative to worlds/sim.
- Retain hot copy for fast API access (`flashback_api`, contest analytics, ownership workflows).

### Training datasets
- Keep `training/datasets` and `training/snapshots_minutes_v1` on hot storage by default.
- Do not include these families in first-wave automated prune/archive.
- Only archive by explicit opt-in policy if hot pressure later requires it.

### Live feature-input data
- Treat the following as hot-required operational data:
  - `bronze/*` sources used by live ingestion (including injuries/props/contest feeds),
  - `silver/*` snapshots used by live builders,
  - `gold/*` feature/label stores referenced by live paths,
  - `live/features_minutes_v1` and `live/features_gtv2_v1`.
- Default policy: no automated archive/prune during active season for these families.

## 4.6 Additional Growth Families (Prevent Next Storage Incident)
### `gold/prediction_logs_minutes`
- Default window: keep last 30 days hot.
- Archive older run partitions monthly.
- Keep run-level metadata/index hot for traceability.

### `builds/optimizer` and `builds/contest_sim`
- Keep pinned and last 14 days hot.
- Weekly archive job moves older builds.
- Keep manifest/index JSON hot for UI lineage.

### `reports/experiments`
- Keep last 30 days hot by default.
- Archive older report trees unless explicitly pinned to an incident/model-promotion ticket.

### `tmp/*`
- Enforce TTL cleanup (default: remove directories older than 7 days).
- Exclusions: active lock files and currently referenced run directories.
- No archival requirement for temporary scratch output.

### Legacy duplicate trees (`br/*` and top-level contest date folders)
- Treat as legacy/orphan candidates until proven active.
- Inventory and hash once, then archive as bulk legacy package.
- Block new writes into legacy roots after cutover.

---

## 5) Metadata vs Payload Split
Design principle: always preserve lineage metadata; persist heavy payload only when policy says it is needed.

### Keep for every run (metadata, small)
- Existing run manifest (`artifacts/runs/.../manifest.json`).
- New retention decision record:
  - classification (`canonical|debug|pinned|noncanonical|protected`),
  - canonical key,
  - source pointers seen,
  - payload file list with byte sizes,
  - archive receipt reference when moved.
- New audit index row per run for deterministic replays.

### Keep only for selected runs (payload, heavy)
- `worlds.parquet` (gtv2) or `worlds_matrix.parquet`/legacy world files (sim_v2).
- optional large debug payloads only inside explicit debug window.
- `minutes_matrix.parquet` only when explicitly enabled and selected.

### Proposed retention metadata path
`$HOT_ROOT/artifacts/retention/v1/<family>/game_date=<date>/run=<run_id>/decision.json`
`$HOT_ROOT/artifacts/retention/v1/<family>/game_date=<date>/run=<run_id>/archive_receipt.json`

---

## 6) Tooling To Build
Implement as reusable library + CLI wrappers matching current style (`python -m projections.cli.<module>`).

## 6.1 Inventory / Audit Tool
- Module: `projections/storage_retention/inventory.py`
- CLI: `projections/cli/storage_inventory.py`
- Purpose: enumerate families, runs, payload sizes, pointer references, risk summary.
- Inputs: `--data-root`, `--families`, `--start-date`, `--end-date`.
- Outputs: JSON + CSV report under `artifacts/retention/reports/`.
- Dry-run: always non-destructive.
- Failure behavior: hard fail on unreadable roots unless `--skip-errors`.
- Safety: read-only, no pointer writes.

## 6.2 Canonical Selector
- Module: `projections/storage_retention/canonical.py`
- CLI: `projections/cli/storage_select_canonical.py`
- Purpose: deterministic classification for each run.
- Inputs: run inventory, manifests, policy config.
- Outputs: per-run decisions + per-key canonical map.
- Dry-run: default; `--write-decisions` persists decision JSON only.
- Failure behavior: if missing timestamps, classify as `unknown` and protect from delete.
- Safety: no payload move/delete.

## 6.3 Archive Mover
- Module: `projections/storage_retention/archive.py`
- CLI: `projections/cli/storage_archive.py`
- Purpose: copy/move eligible payload from hot to archive with verification.
- Inputs: decision files + archive root.
- Outputs: archive payload, `archive_receipt.json`.
- Dry-run: default prints planned bytes/files.
- Failure behavior: never delete source if verification fails.
- Safety: atomic receipt generation; idempotent re-run.

## 6.4 Prune Planner / Deleter
- Module: `projections/storage_retention/prune.py`
- CLI: `projections/cli/storage_prune.py`
- Purpose: produce delete plan and optionally execute for verified candidates.
- Inputs: decisions + receipts + protection lists.
- Outputs: prune plan report and delete ledger.
- Dry-run: default.
- Failure behavior: stop on first policy violation in execute mode.
- Safety: requires verified archive receipt and non-protected classification.

## 6.5 Free-Space Guard (Fail-Closed)
- Module: `projections/storage_retention/guard.py`
- CLI: `projections/cli/storage_guard.py`
- Purpose: block heavy payload writes when hot space is below threshold.
- Inputs: threshold config and mount paths.
- Outputs: health JSON + exit code for flow gating.
- Dry-run: N/A (read-only check).
- Failure behavior: non-zero exit to abort world persistence steps.
- Safety: no writes except optional report file.

## 6.6 Path Resolver / Storage Abstraction Helpers
- Module: `projections/storage_retention/paths.py`
- Purpose: centralize hot/archive path resolution and payload location checks.
- Inputs: env vars + policy config.
- Outputs: canonical resolved paths.
- Dry-run: N/A.
- Failure behavior: explicit errors on unresolved roots.
- Safety: pure resolution logic.

## 6.7 Weekly Retention Orchestrator (Automated, Low Frequency)
- Module: `projections/storage_retention/scheduler.py`
- CLI: `projections/cli/storage_retention_weekly.py`
- Purpose: run weekly archive/prune cycle for low-risk families (first targets: raw contest result CSVs older than 7 days, old builds, old experiment reports).
- Inputs: policy config + date window + max-bytes/max-files caps.
- Outputs: weekly run report, archive receipts, prune ledger.
- Dry-run: default; execute requires explicit `--execute`.
- Failure behavior: fail closed on any verification mismatch and stop remaining deletes.
- Safety: idempotent planning, capped batch size, no-delete-before-verified-archive.

## 6.8 Temp/Legacy Sweep Tool
- Module: `projections/storage_retention/sweep.py`
- CLI: `projections/cli/storage_sweep.py`
- Purpose: enforce TTL cleanup for `tmp/*` and quarantine/archive legacy duplicate roots (`br/*`, top-level dated contest dirs).
- Inputs: policy config + include/exclude family filters.
- Outputs: sweep report, deleted-temp ledger, legacy quarantine manifest.
- Dry-run: default.
- Failure behavior: skip-and-report for unreadable paths; fail closed on configured protected roots.
- Safety: TTL deletion limited to designated temp roots; legacy roots are moved/copied only with explicit execute flag.

---

## 7) Path / Config Architecture
Keep current compatibility with [projections/paths.py](/home/daniel/projects/projections-v2/projections/paths.py), then layer storage-specific roots.

### New env/config contract
- `PROJECTIONS_DATA_ROOT` (existing logical root, unchanged).
- `PROJECTIONS_HOT_ROOT` (default = `PROJECTIONS_DATA_ROOT`).
- `PROJECTIONS_ARCHIVE_ROOT` (required for archive operations).
- `PROJECTIONS_WORLDS_ROOT` (optional override for worlds family logical root).
- `PROJECTIONS_RUN_METADATA_ROOT` (default `HOT_ROOT/artifacts/runs`).
- `PROJECTIONS_TRAINING_ARCHIVE_ROOT` (default under archive root).
- `PROJECTIONS_CONTEST_ARCHIVE_ROOT` (optional override for contest CSV archive target).
- `PROJECTIONS_BUILDS_ARCHIVE_ROOT` (optional override for builds/report artifacts archive target).
- `PROJECTIONS_LEGACY_ARCHIVE_ROOT` (optional override for one-time legacy bulk archive).

### New policy config
Add `config/storage_retention.yaml` with:
- thresholds,
- windows,
- protection rules,
- family-specific file patterns,
- archive destination templates,
- do-not-touch families (`training/datasets`, `training/snapshots_minutes_v1`, live input roots),
- contest raw CSV hot window (`7d`) and weekly archive cadence,
- prediction logs/builds/reports windows (default `30d/14d/30d`),
- tmp TTL (`7d`) and legacy-root quarantine rules.

### Resolution pattern
All new retention tools resolve through one helper path layer.
Existing producers/consumers stay unchanged initially.
Phase 2 can add archive-aware fallback reads for non-hot historical runs only.

---

## 8) Staged Rollout Plan
## Stage A: Before HDD Arrives
1. Keep automation disabled.
2. Implement inventory + canonical selector + planner in dry-run mode only.
3. Produce baseline reports and canonical maps for last 60 days.
4. Review protected/pinned pointer integrity.

## Stage B: HDD Arrival Day
1. Provision IronWolf with native Linux filesystem.
2. Mount at stable path, add fstab entry with safe options.
3. Reboot validation: mount persists and writable.
4. Do not enable Prefect worker yet.
5. Keep current non-RAID layout for recovery rollout (no RAID changes in Stage B).

## Stage C: Archive Ready
1. Configure `PROJECTIONS_ARCHIVE_ROOT`.
2. Run inventory and canonical selection again.
3. Execute first archive move on one low-risk date.
4. Verify archive receipts and source payload integrity before any prune.
5. Implement and validate archive-aware historical contest reads before contest CSV prune.

## Stage D: First Prune Pass
1. Execute prune on same low-risk date only.
2. Confirm pointers still resolve to hot payload.
3. Confirm contest-sim and API smoke checks on that date.
4. Run first weekly contest CSV archive/prune in dry-run, then execute with capped batch.
5. Run first temp TTL sweep in dry-run, then execute with conservative cap.

## Stage E: Manual Bring-Up
1. Run one manual reduced-persistence pipeline execution.
2. Confirm storage delta is bounded.
3. Run first manual slate end-to-end.
4. Monitor space and pointer health.
5. Validate that heavy live writes stay on hot SSD path and not root filesystem.

## Stage F: Controlled Automation Re-enable
1. Enable dashboard service first.
2. Enable Prefect worker only after guard checks are active.
3. Keep frequent retention dry-runs for first week.
4. Switch retention execute mode only after stable outcomes.

## Stage G: Post-Stabilization Topology Review (Deferred)
1. After at least 2-4 stable weeks, review whether RAID is still needed.
2. Re-evaluate SSD health, unclean shutdown frequency, and operational risk.
3. If redundancy is still required, design a dedicated SSD-only RAID1 migration as a separate project.

---

## 9) Guardrails
- Hot free-space warning threshold: `<15%` or `<150GB`.
- Hot free-space hard stop for heavy payload persistence: `<10%` or `<100GB`.
- Root filesystem hard stop for nonessential writes: `<50GB`.
- No heavy payload persistence for non-selected runs once phase-2 gating is enabled.
- Current-day runs are protected from prune/delete.
- Pinned/blessed/latest referenced runs are always protected.
- No delete without verified archive receipt.
- Max delete cap per execution batch (bytes/files) to limit blast radius.
- Locking: retention execute mode must refuse to run if live writer lock is active.
- Rollback: pointer repoint + archive restore by receipt must be documented and tested.
- Do-not-touch guard: automated retention must skip `training/datasets`, `training/snapshots_minutes_v1`, and in-season live input families unless explicitly overridden.
- Contest CSV guard: do not prune raw contest CSVs until analytics index refresh is complete for the same date window.
- Temp TTL guard: cleanup is only allowed under configured temp roots (`tmp/*`), never under live/artifacts roots.
- Legacy-root guard: no deletion of legacy duplicate trees until bulk archive manifest is verified.

---

## 10) Validation / Acceptance Criteria
System is “back online” only when all pass:

1. Path resolution:
- hot and archive roots resolve correctly from config/env.
- retention tools generate deterministic absolute paths.

2. Mount resiliency:
- archive mount survives reboot.
- write/read/checksum sanity passes.

3. Canonical determinism:
- canonical selector returns identical output on repeated runs over same inputs.

4. Dry-run sanity:
- archive/prune plans match expected protected/canonical behavior.
- no protected run appears in delete plan.

5. Archive safety:
- first execute move writes receipts and verifies payload.
- no source delete occurs on verification failure.

6. Prune safety:
- prune executes only on receipt-verified non-protected runs.
- pointer targets remain available on hot tier.

7. Pipeline behavior:
- one manual reduced-persistence run succeeds.
- one manual slate succeeds with bounded artifact growth.

8. Operational safety:
- free-space guard blocks heavy persistence when threshold breached.
- logs/reports are clear enough for on-call triage.

9. Contest data continuity:
- historical contest endpoints continue to work for dates older than 7 days after archive.
- weekly contest retention job completes with no missing-index gaps.

10. Live input continuity:
- live feature build succeeds with no missing hot input datasets due to retention actions.

11. Auxiliary growth containment:
- weekly retention reports show bounded growth for `builds/*`, `reports/experiments`, and `gold/prediction_logs_minutes`.
- `tmp/*` does not exceed configured cap after TTL sweep.

12. Legacy data handling:
- legacy duplicate trees are either explicitly retained or archived with manifest; no silent orphan growth.

13. Topology objective met:
- IronWolf is serving archive/cold tiers.
- Hot pipeline writes are isolated to hot SSD path.
- No RAID-related migration changes were required during initial recovery.

---

## 11) Unknowns / Repo-Specific Questions
1. V3 run metadata split:
- v3 flow writes runtime artifacts under `artifacts/runs/nba_live_v3`, but manifest pointer paths still commonly reference `artifacts/runs/nba_live`. Confirm canonical source of truth before coding selector assumptions.

2. Slate identity completeness:
- `input_change_set.changed_games` may not always contain full slate game list. Confirm preferred slate-signature source for deterministic canonical key.

3. sim_v2 legacy day layout:
- old `world=*.parquet` day directories lack run-level manifest data. Confirm retention treatment (bulk archive-only vs deeper reconstruction).

4. Historical run_id format:
- some run IDs are non-timestamp strings. Selector/pruner must not assume strict `YYYYMMDDTHHMMSSZ` format.

5. Consumer expectations:
- verify whether any downstream job still requires historical noncanonical worlds payloads beyond canonical/pinned.

6. Pointer standardization:
- some consumers still look for `latest_run.json` only. Confirm whether to standardize all readers to `LATEST/current.json` + fallback during rollout.

7. Contest-read fallback scope:
- confirm which APIs/scripts must read archived contest CSVs and whether fallback should be added in `projections/api/contest_service.py` only or also in post-contest scripts.

8. Legacy duplicate roots:
- confirm whether `/home/daniel/projections-data/br` and top-level dated contest directories are still used by any active job or are safe to classify as legacy-only.

9. Prediction log dependency:
- confirm required retention horizon for `gold/prediction_logs_minutes` consumers before enforcing 30-day default.

10. Future RAID need:
- after stabilization, confirm whether operational risk still justifies SSD RAID1 migration versus continuing with single-hot-SSD + archive policy.

---

## 12) Concrete Implementation Order
1. Add `config/storage_retention.yaml` schema and defaults.
2. Add `projections/storage_retention/paths.py` and integrate env resolution.
3. Implement read-only inventory tool + CLI.
4. Implement deterministic canonical selector + CLI.
5. Implement retention decision persistence (`decision.json`) only.
6. Implement archive mover with verification and receipt writing.
7. Implement prune planner (dry-run only).
8. Implement prune execute mode with strict protections and lock checks.
9. Add free-space guard CLI and integrate into world-producing flow steps.
10. Add pointer integrity audit (ensure pointed runs still hot-available).
11. Add archive-aware fallback for historical contest reads (hot -> archive).
12. Add weekly contest retention CLI/deployment (`7d hot`, capped execute).
13. Add policy families/windows for `prediction_logs_minutes`, `builds/*`, `reports/experiments`, `tmp/*`.
14. Implement temp/legacy sweep CLI with strict root guards.
15. Run full dry-run across recent dates; review outputs.
16. Execute first single-date archive move + verify.
17. Execute first single-date prune + verify.
18. Run first weekly contest archive pass (dry-run then execute).
19. Run first weekly auxiliary-family archive pass (builds/reports/prediction logs) with capped batch.
20. Run first temp TTL sweep pass (dry-run then execute).
21. Run first manual pipeline/smoke tests.
22. Re-enable dashboard service.
23. Re-enable Prefect worker only after guard and retention jobs are verified stable.
24. After one stable week, enable stricter payload gating for noncanonical writes.
25. Schedule post-stabilization topology review (RAID defer/revisit decision gate).

---

This packet is spec-only and implementation-ready for post-HDD arrival execution.
