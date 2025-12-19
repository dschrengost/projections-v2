# Root Cause Analysis — Why Daily Runs Need Manual Intervention

Generated (UTC): 2025-12-19T01:51:26Z

This document ranks the **structural** causes of repeated manual intervention and low diagnosability. The goal is not “bug lists”, but the systemic patterns that make failures frequent, hard to reproduce, and slow to triage.

For detailed inventory and evidence sources, see `docs/audit/pipeline_inventory.json`.

## Top Root Causes (Ranked by Impact)

### 1) Multiple overlapping schedulers and duplicate triggers (systemd system + systemd user + cron)

**Impact:** High. Creates concurrency hazards, ambiguous “what ran?”, unexpected environment differences, and increases log noise that hides real failures.

**Evidence:**
- Duplicate enabled user timers in `~/.config/systemd/user/`:
  - `~/.config/systemd/user/live-score.timer`
  - `~/.config/systemd/user/live-scrape.timer`
- Duplicate enabled system timers in `/etc/systemd/system/`:
  - `/etc/systemd/system/live-score.timer`
  - `/etc/systemd/system/live-scrape.timer`
- Config drift between duplicates (example):
  - User `live-score.service` sets `LIVE_DISABLE_TIP_WINDOW=0`
  - System `live-score.service` sets `LIVE_DISABLE_TIP_WINDOW=1`
  - This changes whether the runner respects schedule gating and changes run behavior.
- External cron jobs in `crontab -l` invoke `/home/daniel/sim-v2/...` and write to `/home/daniel/dkresults/logs`, creating an external dependency chain not visible in the systemd DAG.

### 2) Unstable or inconsistent `run_id`/partition semantics across steps (and time-dependent behavior)

**Impact:** High. Leads to missing-path failures, non-reproducible runs, and “rerun makes it worse” behavior because different steps read different partitions.

**Evidence:**
- Legacy runner generates mismatched run IDs:
  - `scripts/run_live_pipeline.sh` calls `projections.cli.build_minutes_live` **without** `--run-id`, then generates a new `RUN_ID` and calls `projections.cli.score_minutes_v1 --run-id <new>`.
  - Resulting failure in journald:
    - `journalctl -u live-pipeline.service` shows `FileNotFoundError: Requested run=... missing under /home/daniel/projections-data/live/features_minutes_v1/2025-12-18`
- “Latest-by-mtime” is used as a hidden selector:
  - `scripts/run_live_sim.sh` uses `ls -1t ... | head -1` to pick the latest minutes run directory when copying into gold.
- Many jobs implicitly depend on wall-clock time:
  - `scripts/run_live_score.sh` generates `LIVE_RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)` and uses tip-time gating for ownership.

### 3) Mutable “latest” inputs and non-atomic publishes (partial outputs are readable)

**Impact:** High. Produces flakey downstream behavior: a consumer can read half-written parquet or a pointer that has been updated before the data is complete.

**Evidence:**
- `bronze/pipeline_status` is “latest-only” and overwrites per `(job_name, target_date)`:
  - `/home/daniel/projections-data/bronze/pipeline_status/*`
  - This makes it impossible to reconstruct a full run timeline and makes root-cause for regressions difficult.
- Multiple assets maintain both “history” and a mutable “latest view” (examples):
  - `/home/daniel/projections-data/bronze/injuries_raw/season=.../date=.../injuries.parquet` (overwritten)
  - `/home/daniel/projections-data/bronze/odds_raw/season=.../date=.../odds.parquet` (overwritten)
- `scripts/run_live_score.sh` publishes minutes to gold via `cp` to:
  - `/home/daniel/projections-data/gold/projections_minutes_v1/game_date=<DATE>/minutes.parquet`
  - This is not an atomic “temp → promote” publish; downstream readers can observe intermediate states.

### 4) Alerting is configured but ineffective; observability is fragmented and not run-centric

**Impact:** High. Failures are not surfaced quickly and are slow to diagnose because there is no single “run dashboard” and no structured per-step manifests.

**Evidence:**
- Several critical services specify `OnFailure=status-email-user@%n.service`:
  - `/etc/systemd/system/live-score.service`
  - `/etc/systemd/system/live-scrape.service`
  - `/etc/systemd/system/live-sim.service`
- Missing failure handler:
  - `/etc/systemd/system/status-email-user@.service` does not exist
  - journald shows “Failed to enqueue OnFailure job” when failures occur (captured in inventory).
- Logs live in multiple places:
  - systemd/journald for most units
  - `/home/daniel/dkresults/logs/*.log` for cron-driven external jobs
  - mutable `bronze/pipeline_status` for some step status

### 5) Time-gated logic embedded in runners breaks backfills and forces special-case reruns

**Impact:** High for backfills, medium for daily. Produces partial historical runs and requires manual “do it differently for backfill” intervention.

**Evidence:**
- Ownership step is skipped once the first game has tipped:
  - `scripts/run_live_score.sh` checks first tip vs now and skips `projections.cli.score_ownership_live` if locked.
  - This makes the naive date-loop backfill (`scripts/backfill_season.sh`) incomplete for historical slates.
- Roster age thresholds cause failures when running older dates unless `--backfill-mode` is enabled:
  - `/home/daniel/projections-data/bronze/pipeline_status/build_minutes_live_2025-10-22.json`:
    - “Roster snapshot is 1083.3h old… exceeds 720h limit.”
  - `/home/daniel/projections-data/bronze/pipeline_status/build_minutes_live_2025-12-01.json`:
    - “Roster snapshot is 93.9h old… exceeds 18h limit.”

### 6) Schema drift + missing data contracts/expectations at stage boundaries

**Impact:** Medium-high. Failures show up late as runtime exceptions rather than early, explainable contract violations.

**Evidence:**
- Missing feature columns:
  - `/home/daniel/projections-data/bronze/pipeline_status/build_rates_features_live_2025-10-22.json`:
    - “features_rates_v1 missing required columns: ['season_fg2_pct', 'season_fg3_pct', 'season_ft_pct']”
- Tracking roles crash from missing expected columns:
  - `journalctl -u tracking-daily.service` shows:
    - `AttributeError: 'numpy.float64' object has no attribute 'fillna'`
  - Likely from `projections/tracking/roles.py` calling `.fillna()` on a scalar due to absent input columns.

### 7) External network fragility without orchestrator-level retries/backoff/jitter/rate limiting

**Impact:** Medium-high. Causes spurious failures that require manual restarts and reduce confidence in run completeness.

**Evidence:**
- DNS failure in pipeline_status:
  - `/home/daniel/projections-data/bronze/pipeline_status/injuries_live_2025-12-15.json`:
    - `[Errno -3] Temporary failure in name resolution`
- Oddstrader SSL handshake timeouts:
  - captured in `journalctl -u live-scrape.service` (inventory highlights show `TimeoutError: SSL handshake timed out ... oddstrader.com`)
- Schedule endpoint returns unexpected empties:
  - `/home/daniel/projections-data/bronze/pipeline_status/schedule_live_2025-12-09_2025-12-15.json`:
    - “NBA schedule API returned zero games for requested window.”

### 8) Permissions/ownership inconsistencies across pipeline outputs

**Impact:** Medium. Causes hard failures that look like logic errors but are actually ops/ownership drift.

**Evidence:**
- Permission denied writing features parquet:
  - `/home/daniel/projections-data/bronze/pipeline_status/build_minutes_live_2025-12-04.json`:
    - `[Errno 13] Permission denied: '/home/daniel/projections-data/live/features_minutes_v1/2025-12-04/run=20251205T003000Z/features.parquet'`
- This is consistent with “mixed schedulers / mixed environments” creating directories with inconsistent owners or modes.

### 9) Always-on service port conflicts create constant restarts and noisy logs

**Impact:** Medium. Doesn’t always break projections generation directly, but makes the “status surface” unreliable and burns attention.

**Evidence:**
- `minutes-dashboard.service` and `live-api.service` both bind to `0.0.0.0:8501`:
  - `systemd/minutes-dashboard.service` and `systemd/live-api.service` (repo templates)
  - `/etc/systemd/system/minutes-dashboard.service` and `/etc/systemd/system/live-api.service` (installed)
- `journalctl -u live-api.service` repeatedly shows:
  - `ERROR: [Errno 98] ... address already in use`
  - restart counter in the thousands (e.g., `restart counter is at 1827`)

### 10) “Legacy/disabled” components remain installed/enabled and complicate the mental model

**Impact:** Medium. Increases cognitive load and the surface area for surprise failures.

**Evidence:**
- `live-pipeline.service` is **enabled** as a oneshot service (runs on boot), while its timers are disabled:
  - `/etc/systemd/system/live-pipeline.service` + `/etc/systemd/system/live-pipeline.service.d/override.conf`
- `live-pipeline-weekend.service` claims to skip odds via `LIVE_FLAGS=--skip-odds`, but a drop-in override clears `LIVE_FLAGS`, so the intended behavior is not effective:
  - `/etc/systemd/system/live-pipeline-weekend.service`
  - `/etc/systemd/system/live-pipeline-weekend.service.d/override.conf`

## Summary of What “Fixes the Class of Problem”

These root causes point to a consistent direction:

1) A single orchestrator as the source of truth (remove duplicates; systemd only keeps worker(s) alive)  
2) Partitioned, run-id-driven DAG where “daily” and “backfill” are the same workflow  
3) Atomic publish + immutable run directories + consistent pointers  
4) Per-step run manifests + centralized UI/logging + real alerting  
5) Minimal contracts/expectations at boundaries to detect drift early

Those concrete proposals are detailed in `docs/audit/TARGET_ARCHITECTURE.md` and `docs/audit/MIGRATION_PLAN.md`.

