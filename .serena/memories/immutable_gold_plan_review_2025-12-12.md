# Immutable gold plan review (2025-12-12)

## Key decisions
- Use `as_of_ts` as the *semantic* timestamp for time-travel selection (training/backtests + gold slate freezing). Use `ingested_ts` for audit/monitoring and as a fallback selector only when `as_of_ts` is unavailable/unreliable.
- Bronze partition keys are ET-domain keys (America/New_York): `date=` is a domain date, not UTC calendar day.
  - `injuries_raw`: `date=` is injury report date (ET); history stored as `hour=HH/` keyed by report_time.
  - `odds_raw`: `date=` is game_date (ET) derived from schedule per `game_id`; history stored as `run_ts=.../` (per-ingest run) to avoid intra-hour overwrite.
- Hourly downsampling is acceptable for injuries (upstream is effectively hourly); odds should remain per-run at least until gold slates are frozen, then compact/prune via manifests.

## Live pipeline trace + robustness findings
- systemd entrypoints:
  - `live-score.timer` → `scripts/run_live_score.sh` runs scrape (live_pipeline) + minutes + rates + sim + ownership + finalize.
  - Additional timers exist (`live-scrape`, `live-pipeline-*`, `live-rates`, `live-sim`) that can overlap and cause concurrent parquet writes.
- Issues spotted:
  - `scripts/run_live_pipeline.sh` run-id drift: does not pass `--run-id` to `build_minutes_live` but passes a different `--run-id` to `score_minutes_v1`.
  - `scripts/run_live_rates.sh` generates a fresh RUN_ID and passes `--strict`; often fails because minutes features for that run id do not exist.
  - `scripts/run_live_score.sh` RUN_AS_OF_TS derivation uses `LIVE_SCHEDULE_PATH` defaulting to a non-partitioned path; should reuse computed `SCHEDULE_PATH`.
  - `scripts/run_live_sim.sh` fallback looks under `${DATA_ROOT}/artifacts/...` even though minutes artifacts are repo-local by default.
- Recommended hardening options:
  - Option A (recommended): make `run_live_score.sh` the single orchestrator; disable redundant timers.
  - Option B: keep timers but add a shared `flock` lock + fix run-id propagation and schedule/minutes path plumbing.
  - Strongly recommended: atomic writes for parquet and pointer JSON (`tmp` + `os.replace`) to prevent partial reads.

## Plan doc updated
- Edited `docs/pipeline/IMMUTABLE_GOLD_IMPLEMENTATION_PLAN.md` to include explicit timestamp semantics, ET partitioning rules, odds partitioning by game_date, safer `read_bronze_day` (prefer history partitions to avoid dual-write double counting), and a dedicated Live Pipeline Trace & Hardening section.