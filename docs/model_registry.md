# Model Registry

The model registry provides lightweight version tracking and promotion for ML models in this project.

## Quick Start

```bash
# List all registered models
uv run python -m projections.registry.cli list

# List versions of a specific model
uv run python -m projections.registry.cli list minutes_v1_lgbm

# Show details for production version
uv run python -m projections.registry.cli show minutes_v1_lgbm

# Promote a version to production
uv run python -m projections.registry.cli promote minutes_v1_lgbm <run_id> --stage prod

# Get production model path (for scripts)
MODEL_PATH=$(uv run python -m projections.registry.cli production minutes_v1_lgbm)
```

## Registered Models

| Model | Trainer | Description |
|-------|---------|-------------|
| `minutes_v1_lgbm` | `projections/models/minutes_lgbm.py` | LightGBM quantile regression for minutes |
| `fpts_v1_lgbm` | `projections/models/fpts_lgbm.py` | LightGBM for fantasy points per minute |
| `rates_v1_lgbm` | `scripts/rates/train_rates_v1.py` | Multi-target LightGBM for per-minute rates |

## How It Works

### Auto-Registration

After training any model, it is automatically registered:

```
[registry] Registered minutes_v1_lgbm v20241204T220000Z (stage=dev)
```

The registry stores:
- **version**: Run ID (timestamp-based)
- **artifact_path**: Path to model artifacts
- **metrics**: Validation MAE, coverage, etc.
- **training_start/end**: Date range used for training
- **stage**: `dev` → `staging` → `prod`

### Promotion Workflow

```
┌─────┐    promote     ┌─────────┐    promote     ┌──────┐
│ dev │ ──────────────►│ staging │ ──────────────►│ prod │
└─────┘   --stage      └─────────┘   --stage      └──────┘
         staging                     prod
```

```bash
# Promote to staging for testing
uv run python -m projections.registry.cli promote minutes_v1_lgbm <run_id> --stage staging

# After validation, promote to production
uv run python -m projections.registry.cli promote minutes_v1_lgbm <run_id> --stage prod
```

### Using Production Models in Scripts

```bash
#!/bin/bash
MODEL_PATH=$(uv run python -m projections.registry.cli production minutes_v1_lgbm)
echo "Loading model from: $MODEL_PATH"
```

### Programmatic Access

```python
from projections.registry import load_manifest, get_production_model

# Get production model info
manifest = load_manifest()
prod = get_production_model(manifest, "minutes_v1_lgbm")
if prod:
    print(f"Production: {prod.artifact_path}")
    print(f"Metrics: {prod.metrics}")
```

## Manifest Location

The registry manifest is stored at:
```
artifacts/registry/manifest.json
```

This file is JSON and can be committed to version control for auditability.

## CLI Reference

| Command | Description |
|---------|-------------|
| `list [model]` | List all models, or versions of a specific model |
| `show <model> [version]` | Show details (defaults to production) |
| `promote <model> <version> --stage <stage>` | Promote to dev/staging/prod |
| `production <model>` | Print production artifact path (for scripting) |

## Adding Registry to New Trainers

To add registry support to a new trainer:

```python
from projections.registry.manifest import load_manifest, save_manifest, register_model

# After training completes:
manifest = load_manifest()
register_model(
    manifest,
    model_name="my_new_model",
    version=run_id,
    run_id=run_id,
    artifact_path=str(run_dir),
    training_start="2024-01-01",
    training_end="2024-06-30",
    metrics={"val_mae": my_mae, ...},
    description="Description of this training run",
)
save_manifest(manifest)
```
