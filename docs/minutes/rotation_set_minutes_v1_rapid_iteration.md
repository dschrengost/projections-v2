# Rotation-Set Minutes V1: Rapid Iteration Guide

This is the “do not make me search the repo” checklist for retraining and promoting the production **rotation-set minutes** model (`rotation_set_minutes_v1`).

The production scorer runs in **primary** mode (transformer is canonical minutes; baseline is only a guardrail/fallback).

## Source Of Truth (Production)

- Live config: `config/rotation_set_minutes_live.json`
- Current prod model artifact directory: `artifacts/rotation_set_minutes/...` (see `model_dir` in the config)
- Current prod training dataset: read `dataset_dir` from `<model_dir>/manifest.json`

Every trained model artifact contains:

- `model.pt` (weights)
- `feature_columns.json` (what live inference must provide)
- `config.json` (architecture + training knobs)
- `manifest.json` (dataset path + full train/val metrics history)

## Prereqs (Data Coverage)

Rotation priors must exist (used by both dataset build and live inference):

- `$PROJECTIONS_DATA_ROOT/silver/rotation_priors_v1/team_game_priors/`
- `$PROJECTIONS_DATA_ROOT/silver/rotation_priors_v1/player_game_priors/`

If those roots are missing, build them:

```bash
uv run python scripts/rotation/build_rotation_dataset_v1.py
uv run python scripts/rotation/build_rotation_priors_v1.py
```

Alternatively, run the Prefect flow that does the same (plus optional PBP ingest): `prefect_flows/rotation_priors_update.py`.

## Fast Path (No New Features)

If you are only changing:

- training hyperparams / loss weights
- model architecture switches already supported by the trainer

…you can **reuse the existing dataset** from the current prod model and retrain immediately.

## Full Path (New Features)

If you add/remove features:

1. Update the training dataset builder: `scripts/rotation/build_rotation_train_dataset_v1.py`
2. Ensure live inference will have those same columns:
   - rotation-set live features are assembled in `projections/rotation/live_features_v1.py`
   - the scorer will fail (by design) if any required feature in `feature_columns.json` is missing
3. Rebuild dataset → retrain → smoke test → promote

## Commands

### 1) Find Current Prod Model + Dataset

```bash
cat config/rotation_set_minutes_live.json
MODEL_DIR=$(jq -r .model_dir config/rotation_set_minutes_live.json)
cat "$MODEL_DIR/manifest.json" | jq '{created_at, git_sha, dataset_dir}'

# Base minutes dataset used to build the rotation training dataset (important for parity).
ROT_DATASET_DIR=$(jq -r .dataset_dir "$MODEL_DIR/manifest.json")
MINUTES_DATASET_DIR=$(dirname "$(jq -r .source.minutes_features "$ROT_DATASET_DIR/manifest.json")")
echo "$MINUTES_DATASET_DIR"
```

### 2) (Optional) Rebuild The Training Dataset

Use this when adding new features or changing the dataset window.

Common pattern: last 365 days anchored to a slate date (example `2026-02-20`).

```bash
OUT_DIR="$PROJECTIONS_DATA_ROOT/training/datasets/rotation_train_v1_lookback365_$(date -u +%Y%m%dT%H%M%SZ)"

uv run python scripts/rotation/build_rotation_train_dataset_v1.py \
  --minutes-dataset-dir "$MINUTES_DATASET_DIR" \
  --out-dir "$OUT_DIR" \
  --lookback-days 365 \
  --anchor-date 2026-02-20
```

Notes:

- Action props features are enabled by default (`--action-props`). Disable via `--no-action-props`.
- Prop-derived implied-minutes features are attached by default (`--prop-implied-minutes`). Disable via `--no-prop-implied-minutes`.
- The model only uses a feature if it’s present in `<model_dir>/feature_columns.json` (training decides this).

### 3) Train A New Model (Prod-Equivalent Defaults)

Start by copying the current prod model architecture:

```bash
DATASET_DIR=$(jq -r .dataset_dir "$MODEL_DIR/manifest.json")

uv run python scripts/rotation/train_rotation_set_model_v1.py \
  --dataset-dir "$DATASET_DIR" \
  --model settransformer \
  --use-gate-head \
  --use-prior-head --prior-weight-col minutes_from_stints_prior_20 --prior-weight-floor 1.0 \
  --use-player-embeddings --player-embedding-dim 16 \
  --use-player-team-embeddings --player-team-hash-buckets 16384 --player-team-embedding-dim 8 \
  --alloc-activation entmax --entmax-alpha 1.5 --share-temperature 1.0 \
  --rot-loss-type bce \
  --epochs 6 --batch-size 32 --lr 1e-3 --device cpu
```

For rapid ablations:

- Fix the validation window so metrics are comparable:
  - `--val-start-date YYYY-MM-DD --val-end-date YYYY-MM-DD`
- Cap the training set:
  - `--max-team-games 400`
- Feature ablation for implied minutes missingness:
  - `--drop-implied-minutes-missingness`

### 4) Inspect Metrics (Without Guessing)

Training writes a full epoch history to `<new_model_dir>/manifest.json`.

This is the minimum set I look at:

- `val_mae_team_w` (primary selection metric in most runs)
- `val_k_hat_label_mae` (rotation size accuracy)
- `val_rotation_size_hat_mean` vs `val_rotation_size_label_mean`
- `diagnostics.val.dust_rate` (how often p50 collapses to ~0 for many players)
- `diagnostics.val.top8_share_mean` (concentration)

Quick readout:

```bash
NEW_DIR="artifacts/rotation_set_minutes/<your_new_run_dir>"
cat "$NEW_DIR/manifest.json" | jq '{created_at, dataset_dir, diagnostics, counts, history: (.history[-1])}'
```

### 5) Smoke Test On A Real Live Run (No Deploy Required)

Pick a slate date + run id from gold minutes outputs:

```bash
cat "$PROJECTIONS_DATA_ROOT/gold/projections_minutes_v1/game_date=2026-02-20/latest_run.json"
```

Then score using your candidate model without touching prod config:

```bash
RUN_ID="<run_id_from_latest_run_json>"
NEW_DIR="artifacts/rotation_set_minutes/<your_new_run_dir>"

uv run python -m projections.cli.score_minutes_rotation_set_v1 \
  --date 2026-02-20 \
  --run-id "$RUN_ID" \
  --scoring-mode primary \
  --rotation-config config/rotation_set_minutes_live.json \
  --model-dir "$NEW_DIR" \
  --artifact-root "$PROJECTIONS_DATA_ROOT/debug/minutes_v1/daily"
```

Inspect:

- `$PROJECTIONS_DATA_ROOT/debug/minutes_v1/daily/2026-02-20/run=$RUN_ID/minutes.parquet`
- Look for obvious pathologies: confirmed starters at 0, rotation size exploding, etc.

### 6) Promote / Rollback

Promotion is just:

1. Update `config/rotation_set_minutes_live.json` `model_dir` → new artifact dir
2. PR/merge/deploy

Rollback is a one-line config revert.

## Common “Why Did It Miss?” Debug Checklist

When a player gets projected ~0 despite recent minutes, check these columns in the rotation-set live features:

- `player_prior_n_games_5/10/20`
- `minutes_from_stints_prior_5/10/20` and `*_missing`
- `consecutive_active_dnp`, `active_but_dnp_rate_last10`
- `gate_prob` (if low, they’ll get squeezed to ~0 after 240 scaling)

If `player_prior_n_games_* == 0` unexpectedly, you likely have a **rotation_priors_v1 coverage issue** for that player (common for players who recently logged minutes but have sparse/zero PBP stint rows).
