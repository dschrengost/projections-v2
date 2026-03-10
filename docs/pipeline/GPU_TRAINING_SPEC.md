# GPU Training Spec (CUDA-First)

Status: v0.1 (March 10, 2026)
Owner: projections-v2

## 1. Objective

Move training off CPU-default paths so model iteration is GPU-first where technically supported, while retaining safe CPU fallback behavior.

## 2. Key Clarification

Triton is **inference serving infrastructure**, not a training backend.

Training in this repo now uses:
- Native PyTorch on CUDA for neural trainers.
- LightGBM `device_type` backends (`cuda`/`gpu`) for tree trainers when supported by the local LightGBM build.

## 3. Scope

### 3.1 PyTorch / Lightning trainers (CUDA-first)

Default behavior is now `auto` device selection:
1. CUDA if available
2. MPS if available
3. CPU fallback

Updated scripts:
- `scripts/rotation/train_joint_rotation_rates_model_v1.py`
- `scripts/rotation/train_rotation_set_model_v1.py`
- `scripts/rotation/train_game_transformer_v2.py`
- `scripts/rotation/train_fttransformer_tabular_baseline_v1.py`
- `scripts/usage_shares_v1/train_nn.py`
- `scripts/usage_shares_v1/train_nn_residual.py`

Runtime improvements applied in these scripts:
- `--device auto` or accelerator auto defaults.
- CUDA TF32 enabled (`matmul` + `cudnn`) for faster training on Ampere+ GPUs.
- GPU-friendly DataLoader settings (`pin_memory`, optional `spawn` workers, `persistent_workers`, `prefetch_factor`).

### 3.2 LightGBM trainers (GPU-capable with safe fallback)

Added a shared backend resolver and param injector:
- `projections/lgbm_device.py`

Behavior for `--lgbm-device auto`:
1. Probe `device_type=cuda`
2. Fall back to `device_type=cpu`

`device_type=gpu` (OpenCL) remains available via explicit `--lgbm-device gpu`.

Updated scripts:
- `scripts/rates/train_rates_v1.py`
- `scripts/ownership/train_ownership_v1.py`
- `scripts/usage_shares_v1/train_lgbm.py`
- `scripts/usage_shares_v1/train_lgbm_residual.py`
- `scripts/train_minute_share_mixture_v0.py`
- `scripts/experiments/train_rotation_minutes_lgbm.py`

## 4. CLI Contract Changes

### 4.1 New/updated GPU flags

PyTorch:
- `--device auto` default on updated scripts.
- `--num-workers` exposed where missing.

LightGBM:
- `--lgbm-device {auto,cpu,cuda,gpu}`
- Optional `--lgbm-num-threads` where relevant.

### 4.2 Logging

All updated trainers now print requested vs resolved backend so operators can verify whether a run actually used GPU.

## 5. Runbook

## 5.1 Pre-flight

```bash
nvidia-smi
uv run python -c "import torch; print('cuda_available=', torch.cuda.is_available())"
```

## 5.2 Typical commands

```bash
# Game transformer family (PyTorch)
uv run python scripts/rotation/train_game_transformer_v2.py --device auto
uv run python scripts/rotation/train_joint_rotation_rates_model_v1.py --device auto
uv run python scripts/rotation/train_rotation_set_model_v1.py --device auto

# Usage shares NN
uv run python scripts/usage_shares_v1/train_nn.py --device auto --num-workers 4
uv run python scripts/usage_shares_v1/train_nn_residual.py --device auto --num-workers 4

# LightGBM families
uv run python scripts/rates/train_rates_v1.py --lgbm-device auto
uv run python scripts/ownership/train_ownership_v1.py --lgbm-device auto
uv run python scripts/usage_shares_v1/train_lgbm.py --lgbm-device auto
uv run python scripts/usage_shares_v1/train_lgbm_residual.py --lgbm-device auto
```

## 6. Validation Completed

- Ruff lint on all touched training files: pass.
- CLI `--help` smoke checks: pass for all updated scripts in current env.
- Exception in current env: `train_fttransformer_tabular_baseline_v1.py` requires optional dependency `pytorch-tabular`.

## 7. Open Optimization Backlog

1. Mixed precision training (`autocast`/`GradScaler`) for rotation PyTorch loops.
2. Optional `torch.compile` path for stable long-running training jobs.
3. Data pipeline throughput improvements (cached tensors / parquet pre-materialization for hot loops).
4. Multi-GPU strategy (DDP) only if dataset sizes justify complexity.

## 8. Non-Goals

- No Triton-based training path.
- No change to CPU-only statistical scripts where GPU acceleration is not meaningful.
