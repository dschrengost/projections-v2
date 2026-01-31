# Minute Share Feature Contract Fix Summary

## Problem
The minute_share model was trained with 23 features, but **16 were missing from live inference**, leading to silent failures and invalid projections. Additionally, the initial fix revealed a "flatness" issue where starters received too few minutes (~20) because the model was distributing shares too evenly across deep bench players.

## Solution

### 1. Created Feature Contract Module
**File:** `projections/models/feature_contract.py`
- Defines `ALLOWED_FEATURES` (58 features) and `FORBIDDEN_FEATURES` (16 leaky features).
- Implements `assert_no_leakage()` and parity validation logic.

### 2. Updated Training Pipeline
**File:** `projections/cli/train_minute_share.py`
- Enforces strict feature parity during training.
- Saves `feature_contract.json` artifact for live validation.

### 3. Updated Logic & Inference Algorithm
**File:** `projections/minutes_v1/minute_share.py`
- **Zero-out Logic:** Added `is_out` parameter to `predict_minutes` to strictly zero out players marked OUT before normalization.
- **Sharpening:** Added `sharpen_exponent` parameter (power transform) to fix distribution flatness.

### 4. Updated Live Scoring CLI
**File:** `projections/cli/score_minutes_v1.py`
- Added support for `MinuteShareArtifacts` bundles.
- Implemented `_score_rows_share` handler which:
    - Validates feature contract (hard fail on mismatch).
    - Zeros out shares for OUT players (combining `status`, `lineup_role`, `is_out`).
    - Applies **distribution sharpening** (exponent=2.0) to ensure starters get realistic workloads (~30+ min).

## Results (12-23 Slate Dry Run)

| Metric | Legacy/Broken | Initial Fix (Flat) | Final Fix (Sharpened) |
|--------|---------------|-------------------|----------------------|
| Starters Mean | ~29.5 | 20.2 | **30.6** |
| Bench Mean | ~16.3 | 11.1 | **15.4** |
| Zeroed Players | ~43% | 4% | **39.2%** |
| Max Minutes | 37.7 | 26.4 | **41.1** |
| Team Sums | 1.01 | 1.00 | **1.00** |

## Final Feature List (58 features)
All 58 features are confirmed available in live inference.
Forbidden features (box-score stats, salary, biometrics) are strictly excluded.

## Reproduction
To run inference with the new model:
```bash
uv run python -m projections.cli.score_minutes_v1 \
    --date 2025-12-23 \
    --mode live \
    --bundle-dir artifacts/minute_share/contract_v2_fold_10 \
    --features-path /path/to/live/features.parquet \
    --minutes-output both \
    --artifact-root artifacts/minute_share/inference_test
```
