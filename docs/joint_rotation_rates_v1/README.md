# Joint Rotation+Minutes+Rates V1

This folder is the working spec for a unified model that predicts:

- rotation membership,
- team-constrained minutes,
- per-minute rates,
- and optional covariance structure for simulation.

## Why

Current production flow predicts minutes and rates in separate models, then combines later in sim. A joint model should better capture:

- non-linear interactions between role/minutes and per-minute production,
- context-dependent behavior in injury/vacancy regimes,
- and cross-player dependencies currently approximated by post-hoc noise/factors.

## Scope

In scope:

- unified architecture design,
- dataset + label contract,
- training objective and staged optimization,
- rollout plan that preserves current downstream schemas.

Out of scope (initially):

- replacing optimizer/sim consumers,
- changing public output schema for live minutes/rates files,
- in-turn production promotion.

## Document Map

- `docs/joint_rotation_rates_v1/01_ARCHITECTURE.md`
- `docs/joint_rotation_rates_v1/02_DATA_AND_LABELS.md`
- `docs/joint_rotation_rates_v1/03_TRAINING_AND_ROLLOUT.md`

## Dataset Builder (Implemented)

Script:

- `scripts/rotation/build_joint_rotation_rates_dataset_v1.py`

Example:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/rotation/build_joint_rotation_rates_dataset_v1.py \
  --rotation-dataset-dir /home/daniel/projections-data/training/datasets/rotation_train_v1_rolechange_20260219 \
  --lookback-days 365 \
  --anchor-date 2026-02-20 \
  --out-dir /home/daniel/projections-data/training/datasets/joint_rotation_rates_v1_20260220
```

## Trainer (Implemented)

Script:

- `scripts/rotation/train_joint_rotation_rates_model_v1.py`

Example:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/rotation/train_joint_rotation_rates_model_v1.py \
  --dataset-dir /home/daniel/projections-data/training/datasets/joint_rotation_rates_v1_20260221T163500Z \
  --out-dir /home/daniel/projections-data/artifacts/joint_rotation_rates_v1/runs \
  --run-tag joint_rotation_rates_v1_full \
  --epochs 6 \
  --batch-size 32 \
  --num-workers 8 \
  --use-prior-head --prior-weight-floor 1.0 \
  --use-team-embeddings --team-embedding-dim 8 \
  --use-player-embeddings --player-embedding-dim 16 \
  --use-player-team-embeddings --player-team-hash-buckets 16384 --player-team-embedding-dim 8 \
  --rotation-minutes-threshold 6.0 \
  --gate-bce-weight 1.0 \
  --minutes-out-weight 0.25 \
  --k-target 9.5 --k-target-source fixed --k-reg-weight 0.05 \
  --anti-smear-weight 0.05 --anti-smear-floor 4.0
```

## Baselines To Beat

- Rotation/minutes transformer flow documented in:
  - `docs/minutes/rotation_set_minutes_v1_rapid_iteration.md`
- Rates GBM flow:
  - `scripts/rates/train_rates_v1.py`
- Sim integration point that currently combines minutes and rates:
  - `scripts/sim_v2/generate_worlds_fpts_v2.py`
