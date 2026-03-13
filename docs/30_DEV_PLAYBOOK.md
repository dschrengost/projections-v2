# Developer Playbook

Setup, workflows, and common tasks for developing `projections-v2`.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Java (for tabula-py PDF parsing)
- Node.js 18+ (for frontend)
- Playwright Chromium (for RealGM depth chart scraper)

## Initial Setup

```bash
# Clone and enter repo
cd ~/projects/projections-v2

# Install Python dependencies
uv sync

# Install Playwright browser runtime used by RealGM scraper
uv run playwright install chromium

# Verify installation
uv run python -c "import projections; print('OK')"

# Run tests
uv run pytest -q
```

## Common Commands

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_scrapers/test_injuries.py

# Run with coverage
uv run pytest --cov=projections --cov-report=html
```

### Linting

```bash
# Check code style
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### GPU Training

Training is CUDA-first where supported, but Triton remains inference-only.

```bash
# Verify CUDA visibility
nvidia-smi
uv run python -c "import torch; print('cuda_available=', torch.cuda.is_available())"

# PyTorch trainers (auto selects cuda -> mps -> cpu)
uv run python scripts/rotation/train_game_transformer_v2.py --device auto
uv run python scripts/usage_shares_v1/train_nn.py --device auto --num-workers 4

# LightGBM trainers (auto probes cuda and falls back to cpu)
uv run python scripts/rates/train_rates_v1.py --lgbm-device auto
uv run python scripts/ownership/train_ownership_v1.py --lgbm-device auto
```

Reference: `docs/pipeline/GPU_TRAINING_SPEC.md`

### CLI Tools (Development Only)

These commands are for local development and debugging. **Production execution
must go through Prefect** (see below).

```bash
# Score minutes model (dev only)
uv run python -m projections.cli.score_minutes_v1 --date 2025-01-01

# Build features (dev only)
uv run python -m projections.cli.build_minutes_live --date 2025-01-01

# Finalize projections (dev only)
uv run python -m projections.cli.finalize_projections --date 2025-01-01
```

### RealGM Depth Chart Prior Ops

```bash
# Manually scrape + persist RealGM depth charts into bronze
uv run python -m projections.cli.scrape_realgm_depth_charts scrape --date 2026-01-18

# Rebuild/update RealGM->canonical player crosswalk from a minutes run
uv run python -m projections.cli.build_realgm_player_crosswalk run --date 2026-01-18
```

Operator notes:
- Manual crosswalk overrides live at `bronze/realgm/player_id_crosswalk_overrides.csv`.
- Required override columns: `realgm_player_id`, `player_id`.
- Live pipeline logs include `[dc-prior]`, `[dc-cap]`, `[dc-disagree]`, `[dc-crosswalk]`, and `[dc-alert]`.
- DNP-history inference guardrail logs under `[dnp-guardrail]` and is configured in `config/depth_chart_prior.json`.
- `PROJECTIONS_DC_CROSSWALK_WARN_MIN_MATCH_RATE` controls crosswalk warning threshold (default `0.30`).

### Play-Prob Policy Tuning

```bash
# Grid-search guarded-v2 play_prob_policy knobs against realized played labels
uv run python -m scripts.diagnostics.grid_search_play_prob_policy \
  --start 2026-01-15 \
  --end 2026-02-09 \
  --holdout-days 5
```

Notes:
- Uses `artifacts/minutes_v1/daily/<date>/run=*/effective_minutes.parquet` + `labels/season=*/boxscore_labels.parquet`.
- Writes ranking outputs under `artifacts/tuning/play_prob_policy/<run_id>/`.

### Rotation Eval (rot_eval)

```bash
# Baseline: prior_topn candidate pool, no gate
uv run python -m projections.cli.rot_eval \
  --rot-bundle $ROT_BUNDLE \
  --run-id prior_topn_nogate \
  --minutes-prior-parquet $PRIORS \
  --candidate-pool prior_topn \
  --candidate-top-n 11 \
  --no-gate

# Candidate pool: predictor_threshold, no gate
uv run python -m projections.cli.rot_eval \
  --rot-bundle $ROT_BUNDLE \
  --run-id predictor_pool_nogate \
  --minutes-prior-parquet $PRIORS \
  --rotation-predictor-bundle $PRED_BUNDLE \
  --gate-feature-source cached_all \
  --candidate-pool predictor_threshold \
  --pool-max-size 11 \
  --t-ge15 0.35 \
  --t-ge5 0.35 \
  --always-include-top-n 8 \
  --no-gate \
  --baseline-out-dir $BASELINE_OUT_DIR

# Candidate pool: predictor_threshold, with gate
uv run python -m projections.cli.rot_eval \
  --rot-bundle $ROT_BUNDLE \
  --run-id predictor_pool_gate \
  --minutes-prior-parquet $PRIORS \
  --rotation-predictor-bundle $PRED_BUNDLE \
  --gate-feature-source cached_all \
  --candidate-pool predictor_threshold \
  --pool-max-size 11 \
  --t-ge15 0.35 \
  --t-ge5 0.35 \
  --always-include-top-n 8 \
  --gate \
  --baseline-out-dir $BASELINE_OUT_DIR
```

### Production Pipeline (Prefect)

**Prefect is the single source of truth** for production orchestration.

```bash
# Trigger the canonical live pipeline
uv run prefect deployment run nba-live-pipeline-v3/nba-live-pipeline

# Trigger for a specific date
uv run prefect deployment run nba-live-pipeline-v3/nba-live-pipeline \
    --param game_date=2025-01-01

# Check deployment status
uv run prefect deployment ls

# View recent flow runs
uv run prefect flow-run ls --limit 10
```

Legacy shell scripts (`scripts/run_live_*.sh`) are gated and will refuse to run
unless `PROJECTIONS_ALLOW_LEGACY_SHELL_RUNNERS=1` is set. This gate exists to
prevent accidental direct execution in production.

### Ownership Live Selector And Rollback

Ownership source/model selection is controlled by:
- `config/ownership_current_run.json` (repo default)
- `$PROJECTIONS_DATA_ROOT/control_plane/model_selectors/ownership_current_run.json` (runtime override)

Example selector (internal transformer with v1 fallback):

```bash
cat > "$PROJECTIONS_DATA_ROOT/control_plane/model_selectors/ownership_current_run.json" <<'JSON'
{
  "source": "internal",
  "model_family": "ownership_v2",
  "model_run": "ownership_xfmr_v1_12ep_big",
  "gtv2_features_path": null,
  "fallback_source": "internal",
  "fallback_model_family": "ownership_v1",
  "fallback_model_run": "dk_only_v6_logit_chalk5_cleanbase_seed1337",
  "fallback_gtv2_features_path": null
}
JSON
```

Quick rollback to LineStar:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from projections.paths import data_path
dst = data_path() / "control_plane" / "model_selectors" / "ownership_current_run.json"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps({"source":"linestar","model_family":"ownership_v1","model_run":None}, indent=2), encoding="utf-8")
print(dst)
PY
```

Canary replay eval (run-scoped preds, namespaced lock-cache aware):

```bash
uv run python scripts/ownership/evaluate_ownership_production_path.py \
  --start-date 2026-02-01 \
  --end-date 2026-03-10 \
  --pred-snapshot locked \
  --pred-run-id 20260313T210000Z \
  --model-family ownership_v2 \
  --out-json reports/ownership_eval_v2.json \
  --out-md reports/ownership_eval_v2.md
```

### Running Services Locally

```bash
# Start API server
uv run uvicorn projections.api.minutes_api:app --reload --port 8000

# Start frontend (in web/ directory)
cd web && npm run dev
```

## Directory Conventions

| Directory | Purpose |
|-----------|---------|
| `scratch/` | Temporary work (gitignored) |
| `tools/` | One-off debugging scripts |
| `obsolete/` | Deprecated code for reference |
| `docs/archive/` | Historical planning documents |

## Commit Guidelines

- Use imperative mood: "Add feature" not "Added feature"
- Prefix with scope when helpful: `scraper: add Rotowire support`
- Keep commits focused and atomic
- Include test updates with code changes

## PR Checklist

- [ ] Tests pass (`uv run pytest`)
- [ ] Linting passes (`uv run ruff check .`)
- [ ] No new warnings introduced
- [ ] Documentation updated if needed
- [ ] Breaking changes called out

## Troubleshooting

### Import Errors

If you get import errors, ensure you're using `uv run`:
```bash
uv run python -c "import projections"
```

### Java/Tabula Issues

Some scrapers need Java. Install OpenJDK:
```bash
sudo apt install openjdk-11-jre-headless
```

### Prefect Connection

Ensure Prefect API is running:
```bash
prefect server status
# or
systemctl --user status prefect-worker
```

### Snapshot Coverage Recovery

If monthly silver snapshots are missing historical games (for example after an unexpected overwrite),
rebuild them from bronze history:

```bash
uv run python scripts/rebuild_silver_snapshots_from_bronze.py \
  --start-date 2025-12-01 \
  --end-date 2026-01-31 \
  --season 2025
```

If you intentionally need to allow a smaller rebuilt snapshot (rare), pass:
`--allow-snapshot-regression`

Then rebuild gold features for affected months:

```bash
uv run python -m projections.pipelines.build_features_minutes_v1 \
  --start-date 2025-12-01 --end-date 2025-12-31 --season 2025 --month 12
```

Safety guardrails now in place:
- `projections.etl.odds` and `projections.etl.injuries` refuse non-regressive snapshot overwrites by default.
- Use `--allow-snapshot-regression` only for intentional recovery operations.
- `build_features_minutes_v1` now upserts into existing month outputs by default (`--merge-with-existing`), so daily jobs do not clobber full month partitions.
- `projections.etl.odds` / `projections.etl.injuries` now include schedule-aware coverage checks:
  no-game days are treated as expected empty windows, while game days with zero overlap fail fast unless `--no-strict-schedule-coverage` is set.

## See Also

- [00_REPO_MAP.md](./00_REPO_MAP.md) - Repository structure
- [10_CONTROL_PLANE.md](./10_CONTROL_PLANE.md) - Service management
- [AGENTS.md](../AGENTS.md) - AI agent guidelines
