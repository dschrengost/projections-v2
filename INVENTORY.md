# Inventory of Existing Repo

Generated: 2026-04-09 UTC

Scope:
- Phase 1 only.
- Only file added in this phase: `INVENTORY.md`.
- Inventory is based on direct code/config/file inspection in this checkout plus the authoritative docs in `docs/00_REPO_MAP.md`, `docs/10_CONTROL_PLANE.md`, `docs/20_DATA_CONTRACTS.md`, `docs/30_DEV_PLAYBOOK.md`, and `README.md`.

Important doc drift note:
- The authoritative docs are not perfectly aligned. `docs/10_CONTROL_PLANE.md` says the canonical scheduled flow is `prefect_flows/live_nba_pipeline_v3.py`, while `docs/00_REPO_MAP.md` still lists `prefect_flows/live_pipeline.py` as the live pipeline flow. The codebase and configs are clearly centered on `live_nba_pipeline_v3.py` now.

## 1. Top-Level Structure

### Excluded Directories and Local State

These were intentionally not traversed in the structure tree below.

| Path | Size | Why excluded |
| --- | ---: | --- |
| `.git/` | 102M | Explicitly excluded by task |
| `.venv/` | 8.0G | Virtualenv |
| `.venv-v3/` | 7.8G | Virtualenv |
| `.venv.bak.20260123112937/` | 11M | Virtualenv backup |
| `.venv.broken.20260224T203926/` | 7.9G | Virtualenv backup |
| `.venv.py313.bak.20260324T121234Z/` | 7.9G | Virtualenv backup |
| `.venv311/` | 8.0G | Virtualenv |
| `.venv311.bak.20260309T104328/` | 24K | Virtualenv backup |
| `.venv_worker/` | 7.6M | Virtualenv |
| `data/` | 552M | Repo-local fallback data root; not traversed |
| `analytics/` | 16K | Local analytics data (`contests.duckdb`) |
| `artifacts/` | 20K | Tracked artifact stubs/registry; excluded as data/artifact dir |
| `runs/` | 4.4M | Local run outputs |
| `mlruns/` | 28K | MLflow run metadata |
| `lightning_logs/` | 60K | Training logs |
| `models/` | 2.9M | Checked-in model artifacts |
| `handoffs/` | 52M | Archived handoff bundles |
| `node_modules/` | not present at repo root | Explicitly excluded by task if present |

### Directory-Focused Tree to Depth 3

This is a directory-focused tree to depth 3 plus repo-root files. I excluded `.git`, `__pycache__`, `node_modules`, virtualenvs, and the local data/artifact directories listed above.

```text
├── .claude/
├── .serena/
│   ├── cache/
│   │   └── python/
│   └── memories/
├── config/
│   └── experiments/
├── docs/
│   ├── archive/
│   ├── audit/
│   ├── contest-sim/
│   ├── diagnostics/
│   ├── entry-manager/
│   ├── findings/
│   ├── joint_rotation_rates_v1/
│   ├── late-swap/
│   ├── memos/
│   ├── minutes/
│   ├── misc/
│   ├── notes/
│   ├── optimizer/
│   ├── ownership/
│   ├── pipeline/
│   ├── prefect/
│   ├── props/
│   ├── rotation_v1/
│   ├── simulator/
│   ├── STINTS/
│   └── systemd/
├── infra/
│   └── systemd/
├── notebooks/
├── obsolete/
│   ├── code/
│   │   ├── fpts_v1/
│   │   └── sim_v1/
│   ├── configs/
│   ├── docs/
│   │   └── projections/
│   ├── logs/
│   ├── scripts/
│   │   ├── fpts/
│   │   ├── minutes_nn/
│   │   ├── sim_old/
│   │   └── sim_v1/
│   ├── tests/
│   │   └── fpts/
│   └── workflows/
├── prefect_flows/
│   └── __pycache___rootbak_20260313T180434Z/
├── projections/
│   ├── alloc/
│   ├── api/
│   ├── archetypes/
│   │   └── __pycache___rootbak_20260313T180434Z/
│   ├── builders/
│   │   └── __pycache___rootbak_20260313T180434Z/
│   ├── cli/
│   │   └── __pycache___rootbak_20260313T180434Z/
│   ├── contest_sim/
│   ├── dk/
│   ├── etl/
│   │   ├── __pycache___rootbak_20260313T180434Z/
│   │   └── etl/
│   ├── eval/
│   ├── fd/
│   ├── features/
│   │   └── __pycache___rootbak_20260313T180434Z/
│   ├── fpts_v2/
│   ├── jobs/
│   ├── labels/
│   │   └── __pycache___rootbak_20260313T180434Z/
│   ├── late_swap/
│   ├── math/
│   ├── metrics/
│   ├── minutes/
│   ├── minutes_alloc/
│   ├── minutes_v1/
│   │   ├── __pycache___rootbak_20260313T180434Z/
│   │   └── calibration_utils/
│   ├── minutes_v3/
│   ├── mlflow_utils/
│   ├── ops/
│   │   └── __pycache___rootbak_20260313T180434Z/
│   ├── optimizer/
│   │   ├── backends/
│   │   └── objective/
│   ├── overrides/
│   ├── ownership_v1/
│   ├── ownership_v2/
│   ├── pbp/
│   ├── pipeline/
│   │   ├── __pycache___rootbak_20260313T180434Z/
│   │   └── training/
│   ├── pipelines/
│   ├── post_contest/
│   ├── projection_ops/
│   ├── rates_v1/
│   ├── registry/
│   ├── rotation/
│   ├── rotations/
│   ├── sim_v2/
│   ├── storage_retention/
│   ├── tracking/
│   ├── usage_shares_v1/
│   └── validation/
├── scrapers/
│   ├── __pycache___rootbak_20260313T180434Z/
│   ├── action_network/
│   ├── dk_contests/
│   └── linestar/
│       ├── .pw_linestar_profile/
│       ├── backfill/
│       ├── captures/
│       └── out/
├── scripts/
│   ├── contest_results/
│   ├── contests/
│   ├── deploy/
│   ├── diagnostics/
│   ├── dk/
│   ├── experiments/
│   ├── fd/
│   ├── lineups/
│   ├── minutes/
│   ├── minutes_debug/
│   ├── optimizer/
│   ├── ownership/
│   ├── rates/
│   ├── rotation/
│   ├── sim_v2/
│   ├── storage/
│   ├── tracking/
│   ├── triton/
│   │   └── model_repository/
│   └── usage_shares_v1/
├── sql/
├── systemd/
├── tests/
│   ├── alloc/
│   ├── api/
│   ├── cli/
│   ├── contest_sim/
│   ├── fd/
│   ├── features/
│   ├── jobs/
│   ├── lineups_eval/
│   ├── minutes_v1/
│   ├── optimizer/
│   ├── ownership_v1/
│   ├── ownership_v2/
│   ├── pbp/
│   ├── pipeline/
│   ├── post_contest/
│   ├── prefect/
│   ├── prefect_flows/
│   ├── projection_ops/
│   ├── rates/
│   ├── registry/
│   ├── rotation/
│   ├── rotations/
│   ├── sim_v2/
│   ├── storage_retention/
│   ├── test_scrapers/
│   └── test_tracking/
├── tools/
│   └── audit/
├── web/
│   └── minutes-dashboard/
│       ├── dist/
│       ├── public/
│       └── src/
├── .dashboard-8501.log
├── .deploy_info
├── .env
├── .gitignore
├── .prefectignore
├── AGENTS.md
├── CLAUDE.md
├── Findings.md
├── gtv2_anchor_preserve_trials_v1.json
├── gtv2_astreb_a90_d8_single_trial.json
├── gtv2_astreb_gapfix_trials_v1.json
├── gtv2_astreb_light_multiseed.json
├── gtv2_astreb_multiseed_candidates_v1.json
├── gtv2_astreb_stability_trials_v1.json
├── gtv2_autonomous_trials_focus_v2.json
├── gtv2_autonomous_trials_v1.json
├── gtv2_mixedmask_graft_trials_v1.json
├── gtv2_mixedmask_mode_trials_v1.json
├── gtv2_mixedmask_trials_v2.json
├── gtv2_promotion_alignment_rqs_ctrl_20260311.json
├── gtv2_robust_realism_trials_v1.json
├── gtv2_robust_realism_trials_v2.json
├── gtv2_rqs_ctrl_parity_trials_v1.json
├── gtv2_rqs_focus_trials_v1.json
├── gtv2_rqs_opportunity_trials_v1.json
├── hs_err_pid789048.log
├── mlflow.db
├── poetry.lock
├── prefect-worker.service
├── prefect.yaml
├── pyproject.toml
├── README.md
├── replay_pid789048.log
├── run_stack.sh
└── uv.lock
```

### Top-Level Directory Descriptions

- `.claude/`: Local agent/editor config; this checkout only contains `settings.local.json`, not product code.
- `.serena/`: Serena project metadata, cache, and a large set of prior session memory notes; local tooling state, not runtime code.
- `analytics/`: Local DuckDB analytics scratch area; currently just `contests.duckdb`.
- `artifacts/`: Tiny tracked artifact stub area with a registry manifest, not the real production artifact store.
- `config/`: Active JSON/YAML selector files, model/profile settings, and experiment configs used by CLI/Prefect runtime.
- `data/`: Repo-local fallback data root with medallion-style folders (`bronze`, `gold`, `labels`, `live`, `artifacts`).
- `docs/`: Canonical docs plus many design specs, audits, and historical notes.
- `handoffs/`: Archived external-agent handoff bundle(s), not application runtime code.
- `infra/`: Deployment assets; specifically contains the newer installable systemd unit templates.
- `lightning_logs/`: Local model-training log directories from prior experiments.
- `models/`: Checked-in small model asset(s), currently an `ownership_v1` bundle.
- `notebooks/`: Empty placeholder directory; only `.gitkeep` is present.
- `obsolete/`: Explicitly archived code, scripts, docs, tests, and logs from older stacks.
- `prefect_flows/`: Prefect flow entrypoints for live pipeline orchestration, retraining, ETL, and maintenance jobs.
- `projections/`: Main Python package for the API, ETL, feature building, model inference/training, simulation, optimizer, and ops layers.
- `scrapers/`: Standalone scraper modules plus vendor-specific scratch assets/captures under some subtrees.
- `scripts/`: Training, evaluation, diagnostic, and deployment helper scripts layered on top of `projections`.
- `sql/`: Contest schema and view definitions.
- `systemd/`: Older repo-level service templates; looks superseded by `infra/systemd/` for installation.
- `tests/`: Large pytest suite covering API, pipeline, rotation/GTv2, sim, optimizer, ownership, ETL, and storage-retention behavior.
- `tools/`: One-off inspection, audit, recovery, and local operator utilities.
- `web/`: React/Vite minutes dashboard frontend; built assets live in `web/minutes-dashboard/dist/`.

## 2. Component Map

### GTv2 Transformer

- Paths:
  - `projections/rotation/game_transformer_v2.py`
  - `projections/rotation/joint_minutes.py`
  - `projections/rotation/joint_active_set.py`
  - `projections/rotation/joint_game_flow.py`
  - `projections/rotation/assist_heads.py`
  - `projections/rotation/rebound_heads.py`
  - `projections/rotation/usage_share_head.py`
  - `projections/rotation/team_budget_heads.py`
  - `projections/rotation/efficiency_head.py`
  - `projections/rotation/possession_backbone.py`
  - `scripts/rotation/train_game_transformer_v2.py`
  - `projections/pipeline/gtv2_inference_runtime.py`
  - `projections/pipeline/gtv2_live_features.py`
  - `scripts/rotation/eval_game_transformer_v2.py`
  - `scripts/rotation/promote_game_transformer_v2_bundle.py`
  - `scripts/triton/model_repository/gtv2_scorer/1/model.py`
- Purpose:
  - This is the joint game-level transformer stack for live NBA rotation/minutes inference. It models active-set membership and minutes together, then optionally predicts flow, efficiency, usage share, assist share, rebound share, and team budget heads from the same backbone.
  - The code is structured around a bundle contract: a `config.json` with feature columns and normalization stats plus a `model.pt` checkpoint.
- Entry points:
  - Training: `python scripts/rotation/train_game_transformer_v2.py ...`
  - Local inference: functions in `projections/pipeline/gtv2_inference_runtime.py`
  - Live feature build: `projections/pipeline/gtv2_live_features.py`
  - Promotion: `python scripts/rotation/promote_game_transformer_v2_bundle.py ...`
  - Triton serving: `scripts/triton/model_repository/gtv2_scorer/1/model.py`
- Direct internal dependencies:
  - Core model imports the head modules listed above plus `projections.rotation.set_model.zfill_game_id_series`.
  - Live inference depends on the joint dataset contract in `scripts/rotation/build_joint_rotation_rates_dataset_v1.py`.
  - Live feature build reuses `projections.rotation.live_features_v1` and the older rotation-set live prior loaders.
- External dependencies:
  - `torch`
  - `numpy`
  - `pandas`
  - Triton Python backend runtime for the server deployment path
- State:
  - Actively used. This is the current canonical minutes/rotation modeling path for the v3 live pipeline.

### Older LightGBM Projection Stack

- Paths:
  - Minutes: `projections/models/minutes_lgbm.py`, `projections/minutes_v1/`, `projections/cli/build_minutes_live.py`, `projections/cli/score_minutes_v1.py`
  - Rates: `scripts/rates/train_rates_v1.py`, `projections/cli/score_rates_live.py`, `projections/rates_v1/loader.py`
  - Ownership v1: `projections/ownership_v1/`, `scripts/ownership/train_ownership_v1.py`, `projections/cli/score_ownership_live.py`
  - FPTS LGBM: `projections/models/fpts_lgbm.py`
- Purpose:
  - `minutes_v1` is the older player-level LightGBM minutes system with feature building, play-prob modeling, conformal adjustments, and multiple post-process layers.
  - `rates_v1` is still a tree-model stack for per-minute stat rates and remains part of the current overall production stack even after GTv2 cutover.
  - `ownership_v1` is the older tree-based ownership model and is still preserved as a fallback path.
  - `fpts_lgbm.py` appears to be an older fantasy-points-per-minute training script whose source dependencies have drifted.
- Entry points:
  - Minutes live build: `python -m projections.cli.build_minutes_live ...`
  - Minutes scoring: `python -m projections.cli.score_minutes_v1 ...`
  - Rates training: `python scripts/rates/train_rates_v1.py ...`
  - Ownership live scoring: `python -m projections.cli.score_ownership_live ...`
  - FPTS training: `python -m projections.models.fpts_lgbm ...`
- Direct internal dependencies:
  - Minutes stack imports `projections.minutes_v1.*`, `projections.minutes_alloc.*`, `projections.models.feature_contract`, `projections.registry.model_resolver`, and shared feature builders.
  - Rates stack depends on live minutes/rates features and rates loaders.
  - Ownership v1 depends on `projections.ownership_v1.loader`, `calibration`, and feature-prep helpers.
  - `projections/models/fpts_lgbm.py` imports `projections.fpts_v1.*`, but that package is missing from the live repo and only appears under `obsolete/`.
- External dependencies:
  - `lightgbm`
  - `scikit-learn`
  - `joblib`
  - `mlflow`
  - `numpy`
  - `pandas`
  - `typer`
- State:
  - Minutes LGBM: legacy but still present as a real fallback/older production path.
  - Rates LGBM: actively used.
  - Ownership v1: fallback but still wired.
  - FPTS LGBM: likely dead or heavily drifted.

### Generate-Worlds / World Generation

- Paths:
  - GTv2 worlds: `projections/rotation/sample_worlds_v2.py`, `scripts/rotation/generate_worlds_game_transformer_v2.py`
  - Older sim worlds: `scripts/sim_v2/generate_worlds_fpts_v2.py`, `scripts/sim_v2/run_sim_live.py`, `projections/sim_v2/*`
- Purpose:
  - `sample_worlds_v2.py` samples player/game worlds directly off the GTv2 model outputs, including minutes uncertainty and flow-derived stat surfaces.
  - `generate_worlds_fpts_v2.py` is the older simulation stack that combines minutes plus rates plus multiple noise layers and many fallback paths to produce fantasy-point worlds.
- Entry points:
  - GTv2 wrapper: `python scripts/rotation/generate_worlds_game_transformer_v2.py ...`
  - Older sim live run: `python scripts/sim_v2/run_sim_live.py ...`
- Direct internal dependencies:
  - GTv2 worlds import `projections.rotation.game_transformer_v2`, `gtv2_promotion_hybrid`, `joint_minutes`, `set_model`, `projections.projections_bundle`.
  - Older sim worlds import `projections.sim_v2.*`, `projections.ops.overrides`, `projections.fpts_v2.scoring`, `projections.minutes`.
- External dependencies:
  - `torch`, `numpy`, `pandas` for GTv2 worlds
  - `numpy`, `pandas`, `typer` for older sim
- State:
  - GTv2 worlds: active.
  - Older `sim_v2` worlds: still used by legacy/fallback paths and downstream tooling; structurally older and more brittle.

### Portfolio Optimizer

- Paths:
  - Core modern portfolio layer: `projections/contest_sim/portfolio_optimizer.py`
  - Contest-sim integration: `projections/contest_sim/contest_sim_service.py`, `projections/api/contest_sim_api.py`
  - Older lineup optimizer: `projections/optimizer/`, especially `projections/optimizer/nba_optimizer.py`
- Purpose:
  - The modern portfolio layer optimizes lineup selection after contest simulation, with min-uniques, exposure caps, covariance-aware EV retention, and diagnostics.
  - The older `optimizer/` package is still the lineup-construction engine and supports DFS roster constraints and backends such as PuLP and CP-SAT.
- Entry points:
  - API router: `projections/api/contest_sim_api.py`
  - Direct lineup optimizer script/class: `projections/optimizer/nba_optimizer.py`
- Direct internal dependencies:
  - Portfolio layer depends on contest-sim worlds loading, field library, payout config, and optimizer service player-pool builders.
  - Older optimizer is more self-contained but sits beside `backends/`, `objective/`, and the API services.
- External dependencies:
  - `numpy`
  - `pandas`
  - `pyarrow`
  - `pulp`
  - `ortools` elsewhere in optimizer package
  - `fastapi` for the API layer
- State:
  - Portfolio optimizer: actively used.
  - Old `nba_optimizer.py`: active but older/more monolithic.

### Ownership Models

- Paths:
  - `projections/ownership_v1/`
  - `projections/ownership_v2/`
  - `projections/cli/score_ownership_live.py`
  - `config/ownership_current_run.json`
- Purpose:
  - `ownership_v2` is a slate-level transformer bundle loader and inference path.
  - `ownership_v1` is the older tree-based scorer with calibration and target-sum normalization still preserved for fallback.
- Entry points:
  - Live scoring CLI: `python -m projections.cli.score_ownership_live ...`
  - Bundle load/infer functions: `projections/ownership_v2/inference.py`
- Direct internal dependencies:
  - `ownership_v2` imports `slate_transformer` and `projections.paths`.
  - `ownership_v1` imports loader/calibration/schema helpers and is selected inside `score_ownership_live.py`.
- External dependencies:
  - `torch` for v2
  - `numpy`, `pandas`
  - `lightgbm` indirectly via v1 bundle loading
- State:
  - `ownership_v2`: active.
  - `ownership_v1`: active fallback.

### Scrapers

- Paths:
  - `scrapers/`
  - `projections/etl/`
  - `projections/data/nba/tracking_client.py` and `scripts/tracking/*`
- Purpose:
  - Scrapers fetch schedule, injuries, odds, props, RealGM depth charts, lineups, DraftKings data, and tracking data; ETL modules normalize and persist these into bronze/silver/labels.
  - The scraper layer is mixed: some modules are clean single-purpose scrapers, while others include local captures or deprecated files.
- Entry points:
  - Many `python -m projections.cli.*` ETL CLIs
  - Direct scripts under `scripts/tracking`, `scripts/dk`, `scripts/fd`
  - Standalone modules such as `scrapers/realgm_depth_charts.py`
- Direct internal dependencies:
  - ETL modules use `projections.paths`, `projections.etl.storage`, `projections.minutes_v1.schemas`, shared resolvers/builders.
  - Scrapers feed data into CLI and Prefect flow layers.
- External dependencies:
  - `requests`, `httpx`
  - `playwright`
  - `beautifulsoup4`, `lxml`
  - `nba_api`
  - `tabula-py`, `jpype1`, Java for PDF-style workflows
- State:
  - Mixed but mostly active. `scrapers/basketball_reference(deprecated).py` is explicitly deprecated.

### Data Layer Interface

- Paths:
  - `projections/paths.py`
  - `projections/etl/storage.py`
  - `projections/model_selectors.py`
  - Many direct readers/writers across `projections/cli`, `projections/api`, `projections/etl`, `projections/pipeline`, `scripts/*`, `prefect_flows/*`
- Purpose:
  - There is a small root-resolution helper and a bronze-layout helper, but there is not a single enforced data-access layer. Most code still joins medallion paths directly.
- Entry points:
  - All major CLI and Prefect paths call into the data root.
- Direct internal dependencies:
  - `projections.paths` is the closest thing to a common entrypoint.
  - `projections.etl.storage` is a common bronze helper.
- External dependencies:
  - `pandas`, `pyarrow`, `json`, `pathlib`
- State:
  - Actively used but highly decentralized.

### Sim Layer

- Paths:
  - `projections/sim_v2/`
  - `scripts/sim_v2/`
  - `projections/rotation/sample_worlds_v2.py`
  - `projections/contest_sim/`
- Purpose:
  - There are now two simulation surfaces: GTv2-native worlds and the older `sim_v2` minutes/rates-driven simulator. Contest simulation then consumes worlds matrices and scores lineups/portfolios.
- Entry points:
  - `scripts/sim_v2/run_sim_live.py`
  - `scripts/rotation/generate_worlds_game_transformer_v2.py`
  - API contest-sim routes
- Direct internal dependencies:
  - `sim_v2` depends on minutes outputs, rates outputs, overrides, play-prob policy, and game-script/noise modules.
  - Contest sim depends on the resulting worlds matrices and payout/field library code.
- External dependencies:
  - `numpy`, `pandas`, `pyarrow`, `typer`
- State:
  - GTv2 worlds are active.
  - `sim_v2` is still significant and not fully removed.

### FastAPI / Triton Serving

- Paths:
  - `projections/api/minutes_api.py`
  - `projections/api/*.py`
  - `projections/pipeline/triton_inference_client.py`
  - `scripts/triton/model_repository/gtv2_scorer/1/model.py`
  - `scripts/triton/setup_gtv2_model_repo.py`
  - `infra/systemd/minutes-dashboard.service`
- Purpose:
  - The FastAPI app serves projections, contest-sim APIs, ops routes, and the built React dashboard.
  - Triton is used as an optional inference server for GTv2 scoring and world generation.
- Entry points:
  - FastAPI factory: `projections.api.minutes_api:create_app`
  - Triton Python backend model entry: `scripts/triton/model_repository/gtv2_scorer/1/model.py`
- Direct internal dependencies:
  - FastAPI layer imports optimizer, contest sim, live status, ops, props, control-plane, and projection bundle helpers.
  - Triton backend reuses the same local `gtv2_inference_runtime` and `sample_worlds_v2` code.
- External dependencies:
  - `fastapi`, `uvicorn`
  - `requests` for Triton HTTP client
  - `torch`, `numpy`, `pandas` inside Triton model runtime
- State:
  - Active.

### Tests

- Paths:
  - `tests/`
- Purpose:
  - Large pytest suite with 346 `test_*.py` files. Coverage is especially deep for GTv2 rotation code, live pipeline v3, API endpoints, contest sim, and storage guards.
- Entry points:
  - `uv run pytest -q`
- Direct internal dependencies:
  - Imports mirror almost every major runtime package.
- External dependencies:
  - `pytest`
  - Usual runtime packages (`pandas`, `numpy`, `torch`, `fastapi`, etc.) as needed by the test target
- State:
  - Active and substantial.

### Notebooks

- Paths:
  - `notebooks/`
- Purpose:
  - Placeholder only; there are no actual notebooks in this checkout.
- Entry points:
  - None
- Direct internal dependencies:
  - None
- External dependencies:
  - None
- State:
  - Inactive/empty.

### Configuration / Settings Management

- Paths:
  - `config/`
  - `projections/model_selectors.py`
  - `projections/pipeline/control_plane.py`
  - `projections/runtime_stamp.py`
  - `prefect.yaml`
- Purpose:
  - The repo uses JSON/YAML files for model selectors, runtime profiles, and promotion knobs. `model_selectors.py` overlays runtime selector files under the data root on top of repo-local defaults. `control_plane.py` manages run IDs and pointer promotion. `runtime_stamp.py` records git/config provenance at runtime.
- Entry points:
  - Prefect flows
  - API startup
  - CLI loaders that call selector helpers
- Direct internal dependencies:
  - `projections.paths`
  - Writer guard / manifest helpers
- External dependencies:
  - Mostly standard library, plus `pyyaml` where YAML is loaded
- State:
  - Active.

### Agent / Skill Scaffolding

- Paths:
  - `AGENTS.md`
  - `CLAUDE.md`
  - `.serena/`
  - `.claude/`
- Purpose:
  - These are local assistant/operator instruction and memory files, not product runtime code. `.serena/memories` is effectively a running engineering journal of prior AI-assisted changes.
- Entry points:
  - None in application runtime
- Direct internal dependencies:
  - None
- External dependencies:
  - None
- State:
  - Local tooling only.

## 3. Data Layer Interface

### Where the Data Directory Lives

- Production path, repeatedly referenced in docs/configs:
  - `/home/daniel/projections-data`
- Repo-local fallback path from `projections/paths.py`:
  - `/home/daniel/projects/projections-v2/data`
- Current shell state during this inventory:
  - `PROJECTIONS_DATA_ROOT` was not exported, so code using `projections.paths.get_data_root()` would fall back to repo-local `./data`.
- Important inconsistency:
  - Not all code uses `projections.paths`. Some modules still hardcode `/home/daniel/projections-data` or use ad hoc env lookups. `projections/pipeline/status.py` is one example of a direct hardcoded fallback to `/home/daniel/projections-data`.

### Observed / Documented Layout

From `README.md`, `docs/20_DATA_CONTRACTS.md`, `projections/etl/storage.py`, and the repo-local fallback `data/` tree:

```text
$PROJECTIONS_DATA_ROOT/
├── bronze/
│   └── <dataset>/season=YYYY/date=YYYY-MM-DD/
│       ├── <file>.parquet                    # legacy latest-view
│       ├── hour=HH/<file>.parquet            # append-only injuries-style history
│       └── run_ts=YYYYMMDDTHHMMSSZ/<file>.parquet
├── silver/
│   └── <dataset>/season=YYYY/month=MM/.../*.parquet
├── gold/
│   └── feature/training/prior tables and some prediction outputs, mostly parquet
├── live/
│   └── run-scoped live artifacts, e.g.
│       ├── features_minutes_v1/YYYY-MM-DD/run=<ts>/features.parquet
│       ├── features_gtv2_v1/YYYY-MM-DD/run=<ts>/...
│       └── features_rates_v1/YYYY-MM-DD/run=<ts>/...
├── labels/
│   └── immutable label parquet files, e.g. boxscore labels
├── artifacts/
│   ├── minutes_v1/daily/YYYY-MM-DD/run=<id>/minutes.parquet
│   ├── sim_v2/worlds_fpts_v2/game_date=YYYY-MM-DD/run=<id>/
│   ├── gtv2_worlds/game_date=YYYY-MM-DD/run=<id>/
│   ├── projections/YYYY-MM-DD/run=<id>/
│   └── game_transformer_v2/bundles/... or bundle_current
├── training/
│   ├── datasets/joint_rotation_rates_v1_*/...
│   └── runs/game_transformer_v2_*/...
└── control_plane/
    └── model_selectors/*.json
```

Primary on-disk formats:
- Parquet for nearly all medallion data, model features, labels, worlds, and projections
- JSON for selectors, manifests, latest pointers, runtime stamps, and run summaries
- CSV for small side outputs such as ID lists
- DuckDB in `analytics/`
- Occasional raw JSON inputs in bronze, especially DraftKings/action-network/props style feeds

### How Code Reads from It

Short answer:
- There is a small shared root/path helper, but there is no single data-access module.
- Data root access is spread across the repo.

Closest thing to shared infrastructure:
- `projections/paths.py`
  - `get_data_root()`
  - `data_path(...)`
- `projections/etl/storage.py`
  - bronze partition helpers and atomic writes
- `projections/model_selectors.py`
  - runtime-vs-repo selector resolution

Actual access pattern:
- Most code still constructs `bronze/`, `silver/`, `gold/`, `live/`, `labels/`, `artifacts/`, or `training/` paths directly.
- A simple grep for data-root/path patterns matched 339 Python modules in this checkout.

Grouped by package prefix, the biggest path-touching clusters are:

- `projections/cli` (58 files)
  - This is where most feature builders, scorers, audits, and backfills directly read/write the data lake.
- `scripts/diagnostics` (24 files)
  - Many diagnostic scripts open medallion and artifact paths directly.
- `scripts/rotation` (23 files)
  - Joint dataset building, GTv2 eval/promotion/training helpers, priors builders.
- `scripts/sim_v2` (16 files)
  - Older worlds generation and sim diagnostics.
- `projections/api` (14 files)
  - API routes read published artifact trees directly.
- `scripts/minutes` (13 files)
  - Older minutes debug/backfill utilities.
- `projections/etl` (11 files)
  - ETL writers/readers into bronze/silver/labels.
- `scripts/experiments` (11 files)
  - Old and new experiment runners against the data lake.
- `scripts/ownership` (10 files)
  - Ownership model builders/trainers.
- `scripts/rates` (9 files)
  - Rates training-base build and audits.
- `projections/pipeline` (8 files)
  - Run manifests, pointer promotion, validation, live orchestration.
- `projections/contest_sim` (7 files)
  - Worlds loaders and field-library access.

The most important individual modules touching the data root for a reset/port are:

- Root resolution:
  - `projections/paths.py`
  - `projections/model_selectors.py`
- Bronze write/read contract:
  - `projections/etl/storage.py`
- Live input build:
  - `projections/cli/build_minutes_live.py`
  - `projections/builders/features_builder.py`
  - `projections/minutes_v1/features.py`
  - `projections/pipeline/gtv2_live_features.py`
  - `projections/rotation/live_features_v1.py`
- Training dataset build:
  - `scripts/rotation/build_joint_rotation_rates_dataset_v1.py`
  - `scripts/rotation/build_rotation_train_dataset_v1.py`
  - `scripts/rates/build_training_base.py`
- Published artifact readers:
  - `projections/api/minutes_api.py`
  - `projections/contest_sim/contest_sim_service.py`
  - `projections/ownership_v2/inference.py`
  - `projections/cli/score_minutes_v1.py`
  - `projections/cli/score_ownership_live.py`
  - `projections/cli/score_rates_live.py`
- Live orchestration:
  - `prefect_flows/live_nba_pipeline_v3.py`
  - `prefect_flows/live_nba_pipeline.py`
  - `prefect_flows/live_pipeline.py`

### Schemas / Data Contracts

Documented contracts:
- `docs/20_DATA_CONTRACTS.md`
  - minutes feature contract
  - worlds contracts
  - rotation priors
  - depth-chart prior contract
  - play-prob policy fields
- `docs/joint_rotation_rates_v1/GAME_TRANSFORMER_SPEC.md`
  - GTv2 training/live feature contract and rollout details

Code-level contracts and schemas:
- `projections/minutes_v1/schemas.py`
  - includes `FEATURES_MINUTES_V1_SCHEMA`, `BOX_SCORE_LABELS_SCHEMA`, and validation/enforcement helpers
- `scripts/rotation/build_joint_rotation_rates_dataset_v1.py`
  - defines output files and a stable training contract (`features.parquet`, `labels_minutes.parquet`, `labels_rates.parquet`, `team_game_index.parquet`, `manifest.json`)
- `projections/etl/storage.py`
  - documents bronze layout contract, including the coexistence of legacy flat daily files and append-only history subpartitions

Assessment:
- Contracts exist, but enforcement is uneven.
- GTv2 is more contract-driven than the older minutes/rates/sim_v2 stack.
- The repo does not currently present a single versioned data-access boundary; the contract is spread across docs, schemas, and path conventions.

## 4. The Minutes Head Specifically

This section focuses on the current transformer minutes head, not the old `minutes_v1` LightGBM scorer, because the planned reset is explicitly aiming at a transformer for minutes/rotations.

### Exact File Paths

Model definition:
- `projections/rotation/game_transformer_v2.py`
- `projections/rotation/joint_minutes.py`

Training loop:
- `scripts/rotation/train_game_transformer_v2.py`

Inference:
- `projections/pipeline/gtv2_inference_runtime.py`
- `projections/pipeline/gtv2_live_features.py`
- Triton path: `scripts/triton/model_repository/gtv2_scorer/1/model.py`

Pre/post-processing and supporting contracts:
- `scripts/rotation/build_joint_rotation_rates_dataset_v1.py`
- `projections/rotation/live_features_v1.py`
- `projections/rotation/set_model.py`
- `projections/rotation/utils.py`
- `projections/rotation/alloc_mask.py`
- `projections/rotation/sample_worlds_v2.py`

Auxiliary heads that share the same backbone:
- AST: `projections/rotation/assist_heads.py`, `team_budget_heads.py`
- REB: `projections/rotation/rebound_heads.py`, `team_budget_heads.py`
- Usage: `projections/rotation/usage_share_head.py`
- Flow: `projections/rotation/joint_game_flow.py`
- Efficiency: `projections/rotation/efficiency_head.py`
- Possessions: `projections/rotation/possession_backbone.py`

### Input Features It Consumes and Where They Come From

Immediate scoring input:
- A feature-complete per-player DataFrame matching the bundle’s `config.json` fields:
  - `feature_columns`
  - `game_feature_columns`
  - `team_feature_columns`

Live source of those features:
- `projections/pipeline/gtv2_live_features.py`
  - Starts from live minutes features
  - Applies the same lineup/game-context transforms used in training
  - Joins rotation priors
  - Joins tracking roles if the bundle expects those columns

Underlying source layers for live features:
- Live minutes features:
  - `projections/cli/build_minutes_live.py`
  - `projections/builders/features_builder.py`
  - `projections/minutes_v1/features.py`
- Rotation priors:
  - `silver/rotation_priors_v1/...`
  - loader in `projections/rotation/live_features_v1.py`
- Tracking roles:
  - `gold/tracking_roles/season=*/game_date=*/tracking_roles.parquet`
- DNP history:
  - loaded through `_load_rotation_historical_features_for_dnp` in `projections/cli/score_minutes_rotation_set_v1.py`
- Game-context contract:
  - imported from `scripts/rotation/build_joint_rotation_rates_dataset_v1.py`

Training input source:
- `scripts/rotation/build_joint_rotation_rates_dataset_v1.py` produces:
  - `features.parquet`
  - `labels_minutes.parquet`
  - `labels_rates.parquet`
  - `team_game_index.parquet`
  - `manifest.json`

### Output Format

Local deterministic scorer output from `score_gtv2_features_df(...)`:
- `minutes_deterministic`
- `active_deterministic`
- `active_logit`
- `active_prob_proxy`
- key columns copied through (`game_id`, `team_id`, `player_id`, `game_date`, etc.)

Model-level output object also contains optional head outputs for:
- active-set count/membership
- minutes
- flow
- efficiency
- usage share
- team points budget
- team AST budget
- assist share
- team rebound budget
- rebound share

World-generation output:
- `worlds.parquet`
- `projections.parquet`
- summary JSON written alongside run outputs in the sampler path

### Model Artifact Locations

Default training run output:
- `scripts/rotation/train_game_transformer_v2.py` writes by default to:
  - `$PROJECTIONS_DATA_ROOT/training/runs/game_transformer_v2_<timestamp>/`

Typical training run contents:
- `model.pt`
- `checkpoint_stable.pt`
- `config.json`
- `history.json`
- `summary.json`
- `checkpoint_candidates/`

Promoted inference bundle:
- Current config points to:
  - `/home/daniel/projections-data/artifacts/game_transformer_v2/bundle_current`
- Promotion tooling writes bundle directories under:
  - `/home/daniel/projections-data/artifacts/game_transformer_v2/bundles/...`

Current selector/config:
- `config/gtv2_inference_current.json`
  - `bundle_dir`: `/home/daniel/projections-data/artifacts/game_transformer_v2/bundle_current`
  - also points at optional sparse hybrid and minutes-uncertainty auxiliary artifacts under `/home/daniel/projections-data/training/runs/...`

### Minimum File Set Needed to Run Inference Standalone in a New Repo

If the new repo only needs deterministic inference on an already-prepared feature DataFrame, the smallest practical code copy is:

- `projections/pipeline/gtv2_inference_runtime.py`
- `projections/rotation/game_transformer_v2.py`
- `projections/rotation/joint_active_set.py`
- `projections/rotation/joint_minutes.py`
- `projections/rotation/joint_game_flow.py`
- `projections/rotation/assist_heads.py`
- `projections/rotation/rebound_heads.py`
- `projections/rotation/usage_share_head.py`
- `projections/rotation/team_budget_heads.py`
- `projections/rotation/efficiency_head.py`
- `projections/rotation/possession_backbone.py`
- `projections/rotation/set_model.py`
- `projections/rotation/utils.py`
- `projections/rotation/alloc_mask.py`
- The promoted bundle files:
  - `config.json`
  - `model.pt`

That minimum set assumes:
- you are not rebuilding live features in the new repo
- you already have a DataFrame with the exact bundle feature contract

If the new repo also wants live feature parity, the minimal set expands immediately to include:
- `projections/pipeline/gtv2_live_features.py`
- `projections/rotation/live_features_v1.py`
- `scripts/rotation/build_joint_rotation_rates_dataset_v1.py`
- `projections/cli/build_minutes_live.py`
- `projections/builders/features_builder.py`
- `projections/minutes_v1/features.py`
- plus the data contracts and upstream medallion readers they depend on

### Hidden Coupling

- The bundle `config.json` is not optional metadata; it carries the real model contract:
  - feature columns
  - per-feature mean/std
  - game/team feature column lists
  - model architecture knobs
- `gtv2_live_features.py` imports training-contract helpers from a script module:
  - `scripts/rotation/build_joint_rotation_rates_dataset_v1.py`
  - That is a real coupling between runtime inference and the training-script namespace.
- Live feature build depends on older rotation-set code:
  - `projections.rotation.live_features_v1`
  - `projections.cli.score_minutes_rotation_set_v1._load_rotation_historical_features_for_dnp`
- The current selector file (`config/gtv2_inference_current.json`) also carries optional hybrid/uncertainty auxiliary paths, so “the minutes head” is not just `model.pt` plus columns anymore.
- `set_model.py` is imported even when only `zfill_game_id_series` is needed, so its transitive imports matter.
- v3 flow can route through local inference or Triton, so live production semantics also depend on `prefect_flows/live_nba_pipeline_v3.py`.

## 5. Dependencies

### Manifest Files Present

- Root manifest:
  - `pyproject.toml`
- Additional manifests:
  - `scripts/triton/requirements-gtv2-runtime.txt`
  - `scrapers/dk_contests/pyproject.toml`
- Not present:
  - root `requirements.txt`
  - root `environment.yml`

### Full Manifest Contents

#### `pyproject.toml`

```toml
[project]
name = "projections-v2"
version = "0.1.0"
description = "Maintainable NBA minutes prediction pipeline scaffolding."
readme = "README.md"
requires-python = ">=3.11"
authors = [
  { name = "Daniel Schrengost" }
]
dependencies = [
  # Native numeric stack is a common source of hard crashes (SIGSEGV/abort) under load.
  # Keep upper bounds to avoid silently rolling PROD onto bleeding-edge wheels.
  "numpy>=1.26,<2.0",
  "pandas>=2.2,<2.3",
  "scipy>=1.11,<1.16",
  "scikit-learn>=1.5,<1.7",
  "xgboost>=2.0",
  "torch>=2.2",
  "pyyaml>=6.0",
  "pydantic>=2.7",
  "typer>=0.12",
  "rich>=13.7",
  "prefect==3.6.7",
  "lightgbm>=4.6.0",
  "pandera>=0.26.1",
  "cvxpy>=1.7.3",
  "ortools>=9.11",
  "pyarrow>=10.0.1,<21.0",
  "tabula>=1.0.5",
  "httpx>=0.28.1",
  "tabula-py>=2.10.0",
  "pypdf2>=3.0.1",
  "fastapi>=0.115.5",
  "uvicorn>=0.38.0",
  "pulp>=2.8",
  "requests>=2.32.5",
  "duckdb>=1.4.2",
  "mlflow>=3.7.0",
  "python-multipart>=0.0.21",
  "unidecode (>=1.4.0,<2.0.0)",
  "beautifulsoup4 (>=4.14.3,<5.0.0)",
  "lxml (>=6.0.2,<7.0.0)",
  "repomix>=0.3.4",
  "nba_api>=1.4",
  "playwright>=1.57.0",
  "jpype1>=1.6.0",
  "tqdm>=4.67.3",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.2",
  "ruff>=0.5"
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["projections"]

[tool.uv]
dev-dependencies = [
  "pytest>=8.2",
  "ruff>=0.5"
]

[tool.pytest.ini_options]
addopts = "-ra"
pythonpath = ["."]
testpaths = ["tests"]
```

#### `scripts/triton/requirements-gtv2-runtime.txt`

```text
numpy>=1.26,<2.0
pandas>=2.2,<2.3
pyarrow>=10.0.1,<22.0
torch>=2.2
PyYAML>=6.0
```

#### `scrapers/dk_contests/pyproject.toml`

```toml
[project]
name = "dk-contests"
version = "0.1.0"
description = "DraftKings NBA GPP contest scraper"
requires-python = ">=3.10"
dependencies = [
    "requests>=2.28",
    "python-dotenv>=1.0",
    "playwright>=1.40",
]

[project.scripts]
scrape-gpp = "nba_gpp_scraper:main"
download-results = "download_contest_results:main"
```

### Flagged Subset: Dependencies Actually Imported by the GTv2 Minutes-Head Minimum File Set

Using import-grep across the minimum standalone scoring file set listed in Section 4, the external packages directly imported are:

- `numpy`
- `pandas`
- `torch`

Notably absent from the minimum deterministic scorer:
- `lightgbm`
- `scikit-learn`
- `pyarrow`
- `prefect`
- `fastapi`

Interpretation:
- A standalone GTv2 minutes scorer can be much slimmer than the current repo.
- The heavy dependency footprint mostly comes from the older stacks, orchestration, API layer, and ETL/scraper surface.

## 6. Pipeline Failure Points

Search scope used:
- `scripts/rotation/generate_worlds_game_transformer_v2.py`
- `projections/rotation/sample_worlds_v2.py`
- `scripts/sim_v2/generate_worlds_fpts_v2.py`
- `prefect_flows/live_nba_pipeline_v3.py`
- recent logs present in repo root and `obsolete/logs/`

### GTv2 Generate-Worlds Module

- `scripts/rotation/generate_worlds_game_transformer_v2.py`
  - This is only a thin wrapper around `projections.rotation.sample_worlds_v2.main`.
  - No `TODO`, `FIXME`, or `HACK` comments found.

- `projections/rotation/sample_worlds_v2.py`
  - I found no `TODO`/`FIXME`/`HACK` comments in the file.
  - The code does contain non-fatal warning paths and fallback semantics:
    - `fallback_sigma` for minutes uncertainty
    - allocation fallbacks when an active mask is empty
    - warnings for non-strict world-contract violations
  - This is much cleaner than the old `sim_v2` world generator, but it is still not a tiny or fully isolated module.

Assessment:
- The GTv2 worlds sampler itself does not look abandoned.
- The fragility around GTv2 worlds is mostly in orchestration and data I/O, not obvious “paper over it” comments inside the wrapper.

### Older `sim_v2` World Generation

The brittle area is here, not in the GTv2 wrapper.

Key findings in `scripts/sim_v2/generate_worlds_fpts_v2.py`:
- Many broad `except Exception` blocks.
- Numerous warning-and-continue branches.
- Multiple explicit fallback/degrade behaviors:
  - failed status overrides load
  - missing FG% predictions
  - missing rates noise params
  - missing minutes noise params
  - failure building minutes sigma
  - failed override artifact writes
  - failed `pre_sim_reconcile`
  - degrading from `model_space_v1` to legacy backend if play-prob is missing
  - failed writes for `metrics.json`, `sim_manifest.json`, `sim_diagnostics.json`, `worlds_matrix.parquet`, `minutes_matrix.parquet`

Interpretation:
- `generate_worlds_fpts_v2.py` is a classic “fragile boundary accumulator”: lots of conditional degradation paths, lots of IO warnings, lots of broad exception handling.
- This file should be treated as high-risk for selective porting. It contains a lot of behavior, not just sampling.

### Prefect v3 Flow Retry Logic

Key findings in `prefect_flows/live_nba_pipeline_v3.py`:
- Subprocess crash retry loop for certain native-crash exit codes:
  - `-11`, `-7`, `-6`, `134`, `135`, `139`
- Retry logic specifically around corrupt parquet validation:
  - checks for strings like `corrupt snappy compressed data` and `corrupt data page`
- Retry/fallback logic for parquet writes with alternate compression
- Stale-fallback logic when current materialized run data is unreadable
- Fallback to previous minutes-feature runs if `build_minutes_live` crashes with a segfault-like exit code
- Prefect task retries on:
  - `scrape-core-inputs`
  - `score-ownership`

Interpretation:
- The GTv2 production path is trying hard to survive native/data corruption failures.
- The primary fragile boundaries are parquet IO, subprocess/native crashes, and stale artifact recovery.

### Recent Logs in Repo

- `hs_err_pid789048.log`
  - Java fatal error in OpenJDK 17
  - Stack shows `org.apache.pdfbox.io.RandomAccessBufferedFileInputStream`
  - This points at a PDFBox/Tabula/Java crash path, not directly at GTv2 worlds
- `replay_pid789048.log`
  - Consistent with the same PDFBox/Java crash context

Interpretation:
- There is real native-runtime fragility in this repo, but the concrete log present here implicates Java/PDF tooling, not the GTv2 sampler.
- That matters because the dependency surface for a reset should aggressively avoid dragging Java/PDF paths forward unless they are essential.

### TODO / FIXME / HACK Search Result

Result for the world-generation files searched:
- No `TODO`, `FIXME`, or `HACK` markers found in:
  - `scripts/rotation/generate_worlds_game_transformer_v2.py`
  - `projections/rotation/sample_worlds_v2.py`
  - `scripts/sim_v2/generate_worlds_fpts_v2.py`
  - `prefect_flows/live_nba_pipeline_v3.py`

That does not mean they are clean. It means the brittleness is expressed as retries, fallback branches, and broad exception handling rather than explicit markers.

## 7. What Looks Dead or Superseded

Conservative flags only.

| Path | Why it looks dead / superseded |
| --- | --- |
| `obsolete/` | Explicitly marked archived; contains old code, docs, tests, and logs. |
| `notebooks/` | Only `.gitkeep` exists; no notebooks are present. |
| `prefect_flows/hello.py` | Tiny demo flow, no references found anywhere else in repo. |
| `projections/etl/etl/` | Duplicate package namespace under `projections/etl/etl/`; no imports of `projections.etl.etl` were found, and there are same-named modules in `projections/etl/`. |
| `systemd/minutes-dashboard.service` | Looks superseded by `infra/systemd/minutes-dashboard.service`; the install script points at `infra/systemd/...`, and docs call `systemd/` files templates. |
| `scrapers/basketball_reference(deprecated).py` | Explicitly labeled deprecated in filename. |
| `prefect_flows/live_pipeline.py` | Older wrapper-style flow; docs outside the authoritative set explicitly call it legacy/deprecated, and authoritative control-plane docs now point at v3. |
| `scripts/run_live_pipeline.sh` | Script is disabled in production by its own guard message and older audit docs describe it as legacy/broken. |
| `projections/models/fpts_lgbm.py` | Imports `projections.fpts_v1.*`, but there is no live `projections/fpts_v1/` package in this checkout; only `obsolete/` contains that code. No active callers found beyond the module itself and docs discussing drift. |

## Reset-Oriented Takeaways

- The cleanest reusable modeling asset is the GTv2 minutes/rotation bundle contract, not the old `minutes_v1` scorer.
- The cleanest reusable infrastructure asset is `projections/paths.py` plus a very small subset of data-contract helpers, but not the broader repo-wide pathing pattern. Data access is too decentralized to copy wholesale.
- The biggest porting risk is hidden coupling in live feature construction:
  - old minutes builder
  - rotation priors
  - training-contract transforms imported from script modules
  - published selector files
- The largest obvious “do not carry this forward unchanged” candidate is the old `sim_v2` worlds generator.

Stop here for Phase 1 review.
