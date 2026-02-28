# Control Plane

Pipeline orchestration, Prefect flows, and systemd services for `projections-v2`.

## Prefect Architecture

### Main Flow

The canonical scheduled pipeline is `nba_live_pipeline_v3_flow` in `prefect_flows/live_nba_pipeline_v3.py`. It orchestrates:

1. **Data scraping and run freeze** - core inputs + manifest stamp
2. **Feature building** - GTV2 live feature contract build
3. **Preflight parity gate** - schema/order/dtype/null + transform/integrity checks
4. **Model scoring + world generation** - GameTransformerV2 score/sample path
5. **Projection finalization** - Unified projections artifact
6. **Postflight gate + publish** - contract validation then atomic pointer promotion

### V3 Flow (Phase 4 Redesign)

`prefect_flows/live_nba_pipeline_v3.py` is the redesign-first flow for GameTransformerV2 cutover. It keeps a strict linear DAG and hard gates before publish:

1. Scrape/freeze run inputs
2. Build GTV2 live features
3. Preflight parity gate (schema/order/dtype/null policy, transform/integrity, freshness)
4. Score -> world generation -> finalize projections
5. Postflight gate (world contracts + projection schema/key sanity)
6. Atomic pointer promotion via `projections.pipeline.control_plane.promote_run_pointer`

Legacy flow remains available as manual rollback deployment: `nba-live-pipeline-legacy`.

### Deployment

Prefect deployments are defined in `prefect.yaml`:

```bash
# Deploy all flows
uv run prefect deploy --all

# Prefect CLI does not currently persist all deployment fields from `prefect.yaml`
# (notably `concurrency_options`). Re-apply overrides after deploy.
uv run python tools/prefect_apply_deployment_overrides.py

# View deployments
uv run prefect deployment ls
```

### Flow Schedules

| Flow | Schedule | Purpose |
|------|----------|---------|
| `nba-live-pipeline` | Every 15 min from 8-10 AM ET, every 5 min from 11 AM-10:30 PM ET | Canonical v3 prediction pipeline |
| `nba-live-pipeline-legacy` | Manual only (no schedule) | Rollback deployment (pre-v3 flow) |
| `boxscores-etl` | 3:30 AM daily | Scrape previous-day boxscores to bronze + legacy labels |
| `minutes-labels-refresh` | 3:40 AM daily | Materialize `gold/labels_minutes_v1` from boxscore raw partitions |
| `rates-training-base-refresh` | 4:05 AM daily | Refresh rates training base partitions |
| `gamerotation-scrape` | 4:15 AM daily | Scrape NBA Stats GameRotation feed |
| `rotation-priors-update` | 4:45 AM daily | Rebuild rotation priors after scrape |
| `nightly-eval` | 3 AM daily | Model evaluation |
| `minutes-retrain-pipeline` | Weekly (Tue 10 AM ET) | Minutes recency retrain + head-to-head eval vs prod |
| `rates-retrain-pipeline` | Weekly trigger (Tue 10 AM ET, biweekly gate in flow) | Rates recency retrain + calibration diagnostics + head-to-head guardrails + auto-promotion |
| `rates-calibration-monitor` | Weekly (Tue 9 AM ET) | Calibration diagnostics (decile curves / efficiency heads) for current production rates run |

## V3 Cutover And Rollback

Use the existing control-plane pointer model (`LATEST/current.json`) for v3 as well.

### GTV2 Bundle Promotion (Canonical)

Before non-placeholder v3 scoring can run, package and promote a GTV2 bundle with a
parity manifest:

```bash
uv run python scripts/rotation/promote_game_transformer_v2_bundle.py \
  --candidate-root /home/daniel/projections-data/training/runs/game_transformer_v2_phase3_c3_multiseed_20260224T142815Z \
  --seed 123
```

This writes:

1. Candidate freeze artifact:
   - `/home/daniel/projections-data/training/runs/<candidate>/promoted_phase3.json`
2. Promoted bundle:
   - `/home/daniel/projections-data/artifacts/game_transformer_v2/bundles/<bundle_name>/`
   - required files: `model.pt`, `config.json`, `parity_manifest.json`, `promotion_meta.json`
3. Canonical pointer:
   - `/home/daniel/projections-data/artifacts/game_transformer_v2/bundle_current` (symlink)

Optional props-source drift audit command:

```bash
uv run python scripts/rotation/check_action_props_source_drift.py \
  --start-date 2025-12-09 \
  --end-date 2026-02-11
```

### Cutover (promote a completed run)

1. Confirm run artifacts and gate reports exist:
   - `$PROJECTIONS_DATA_ROOT/artifacts/runs/nba_live_v3/game_date=<DATE>/run=<RUN_ID>/preflight_report.json`
   - `$PROJECTIONS_DATA_ROOT/artifacts/runs/nba_live_v3/game_date=<DATE>/run=<RUN_ID>/postflight_report.json`
2. Promote dataset pointers (handled by v3 flow when `promote_pointers=true`):
   - `live/features_gtv2_v1/<DATE>/LATEST/current.json`
   - `artifacts/gtv2_scores/game_date=<DATE>/LATEST/current.json`
   - `artifacts/gtv2_worlds/game_date=<DATE>/LATEST/current.json`
   - `artifacts/projections/<DATE>/LATEST/current.json`

### Rollback (re-point to previous known-good run)

Use `promote_run_pointer` with the previous `run_id` and corresponding run manifest path:

```bash
uv run python - <<'PY'
from pathlib import Path
from projections.pipeline.control_plane import promote_run_pointer

data_root = Path("/home/daniel/projections-data")
game_date = "2026-01-18"
run_id = "20260118T224500Z"  # known-good prior run
manifest_path = data_root / "artifacts" / "runs" / "nba_live" / f"game_date={game_date}" / f"run={run_id}" / "manifest.json"

targets = [
    data_root / "live" / "features_gtv2_v1" / game_date,
    data_root / "artifacts" / "gtv2_scores" / f"game_date={game_date}",
    data_root / "artifacts" / "gtv2_worlds" / f"game_date={game_date}",
    data_root / "artifacts" / "projections" / game_date,
]
for t in targets:
    promote_run_pointer(dataset_dir=t, run_id=run_id, manifest_path=manifest_path, extra={"entrypoint": "manual-rollback"})
print("rollback pointers promoted")
PY
```

## Failed-Gate Incident Capture (V3)

For any preflight/postflight hard failure, capture these artifacts in the incident ticket:

1. Run context:
   - `run_id`, `game_date`, `as_of_ts`
   - runtime stamp log line (`[runtime-stamp]`)
2. Gate payloads:
   - `preflight_report.json` and/or `postflight_report.json` from run directory
3. Parity metadata:
   - bundle parity manifest path (`parity_manifest.json`)
   - `integrity.parity_manifest_hash`
   - `integrity.config_hash`
   - `integrity.git_sha`
4. Feature/projection evidence:
   - run-scoped feature runtime manifest (`feature_runtime_manifest.json`)
   - failing output parquet path and row count

### PBP Vendor Daily Refresh

`rotation-priors-update` is the canonical daily integration point for vendor PBP refresh.

- Fetch task: `projections/cli/pbp_vendor_fetch_daily_zip.py`
- Ingest task: `projections/cli/pbp_vendor_ingest.py`
- Required env: `PBP_VENDOR_DAILY_URL`
- Default deployment behavior:
  - `run_pbp_ingest: true`
  - `pbp_fetch_daily_zip: true`
  - `pbp_allow_qa_failures: true` (writes QA artifacts and permits publish on known outliers)

## Systemd Services

Services live in `infra/systemd/` and are installed to `~/.config/systemd/user/`.

### Core Services

| Service | Purpose |
|---------|---------|
| `prefect-worker.service` | Prefect work pool worker |
| `live-api.service` | FastAPI backend |
| `minutes-dashboard.service` | React frontend |
| `mlflow.service` | MLflow tracking server |

### Management Commands

```bash
# Enable and start a service
systemctl --user enable --now prefect-worker.service

# Check status
systemctl --user status live-api.service

# View logs
journalctl --user -u prefect-worker.service -f

# Restart all services
systemctl --user restart prefect-worker live-api minutes-dashboard
```

## Pipeline Timers

Timers trigger pipeline runs on schedule:

| Timer | Triggers | Schedule |
|-------|----------|----------|
| `live-pipeline.timer` | `live-pipeline.service` | Game-time schedule |
| `nightly-eval.timer` | `nightly-eval.service` | 3 AM daily |

## Monitoring

- **Prefect UI**: http://localhost:4200
- **MLflow UI**: http://localhost:5000
- **API docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:3000

## RMH Shadow Mode (Minutes)

The live pipeline can optionally run the Rotation Minutes Hurdle model (RMH v1.1) in **shadow mode** alongside production minutes scoring.

- Enable: `RMH_SHADOW_ENABLED=1`
- Configure bundle: `RMH_ARTIFACT_DIR=/full/path/to/artifacts/rotation_minutes_hurdle_v1/<run_id>`
- Optional label: `RMH_MODEL_LABEL="RMH v1.1"`
- Output (dev namespace): `$PROJECTIONS_DATA_ROOT/artifacts/minutes_models/daily/model_id=rmh_v1_1/<game_date>/run=<run_id>/minutes.parquet`

Shadow failures are non-blocking and will log with `[rmh-shadow]`.

## See Also

- [00_REPO_MAP.md](./00_REPO_MAP.md) - Repository structure
- [30_DEV_PLAYBOOK.md](./30_DEV_PLAYBOOK.md) - Local development setup
