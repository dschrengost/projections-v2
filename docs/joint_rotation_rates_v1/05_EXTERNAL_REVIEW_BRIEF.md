# Joint Rotation+Minutes+Rates (v1) External Review Brief

## 1) Objective

Train one team-set transformer that jointly predicts:
- minutes allocation (team-constrained to 240),
- per-minute stat rates (9 targets),
- efficiency rates (3 targets).

The intent is to replace separate minutes/rates modeling with a single architecture that can learn cross-task structure.

## 2) Code + Artifacts

- Model definition: `projections/rotation/joint_set_model_v1.py`
- Dataset builder: `scripts/rotation/build_joint_rotation_rates_dataset_v1.py`
- Trainer: `scripts/rotation/train_joint_rotation_rates_model_v1.py`
- Current eval script (joint vs current): `scripts/rotation/eval_joint_vs_current.py`
- Sweep runner: `scripts/rotation/sweep_joint_rotation_rates_v1.py`
- Dataset used: `/home/daniel/projections-data/training/datasets/joint_rotation_rates_v1_20260221T163500Z`
- Example trained feature list artifact: `/home/daniel/projections-data/artifacts/joint_rotation_rates_v1/runs/joint_sweep_005_t005_lr-6p00em04_weight_decay-1p00em05_dropout-5p00em02_20260222T150026Z/feature_columns.json`

## 3) Architecture Summary

- Backbone: `SetTransformerMinutesModel` (per-team player set; permutation-invariant).
- Minutes head:
- Gate + share formulation with simplex allocation (`entmax`) and hard team total constraint of 240 minutes.
- Optional prior weighting via `minutes_from_stints_prior_20`.
- Optional embeddings:
- team embeddings, player embeddings, hashed player-team embeddings.
- Joint heads on shared embeddings:
- 9 rate outputs: `fga2_per_min`, `fga3_per_min`, `fta_per_min`, `ast_per_min`, `tov_per_min`, `oreb_per_min`, `dreb_per_min`, `stl_per_min`, `blk_per_min`.
- 3 efficiency outputs: `fg2_pct_label`, `fg3_pct_label`, `ft_pct_label` (bounded by sigmoid-affine clipping ranges).

## 4) Training Setup

- Multi-objective loss (weighted sum):
- minutes MAE (`lambda_minutes`)
- rates MAE (`lambda_rates`)
- efficiency MAE (`lambda_eff`)
- rotation BCE (`lambda_rot`)
- gate BCE (`gate_bce_weight`)
- minutes-out penalty (`minutes_out_weight`)
- K regularizer (`k_reg_weight`, target rotation size `k_target`)
- anti-smear penalty (`anti_smear_weight`, `anti_smear_floor`)
- Optimizer: `AdamW`.
- Validation split: last 14 game dates.
- In current dataset/run:
- feature columns used by model: 310
- train team-games: 2101
- val team-games: 214
- date range: train `2025-02-21`..`2026-01-28`, val `2026-01-29`..`2026-02-11`

## 5) Feature / Label Contract

- Dataset builder starts from rotation training features/labels and joins rates labels from `gold/rates_training_base`.
- Outputs:
- `features.parquet`
- `labels_minutes.parquet` (`minutes_label`)
- `labels_rates.parquet` (rates + efficiency + eligibility flags)
- `team_game_index.parquet`
- `manifest.json`
- Dataset snapshot (`joint_rotation_rates_v1_20260221T163500Z`) stats:
- rows: 40,803
- team-games: 2,315
- feature columns in raw dataset file: 371
- rates labels present on ~31.37% of rows
- rates loss-eligible rows: 12,799
- Key training filters:
- excludes DNP-blind rolling minute features from model input,
- excludes same-game leakage-prone rotation-summary features,
- uses `rates_loss_eligible` masking for rate/eff losses.

## 6) Experiments Completed

### A) Full training run (earlier baseline)

- Run: `joint_rotation_rates_v1_full_20260221T212552Z`
- Best validation snapshot:
- `val_loss=3.5989`
- `val_minutes_mae=2.9392`
- `val_rates_mae=0.06145`
- `val_eff_mae=0.18315`
- `val_anti_smear=0.45268`

### B) Automated hyperparameter sweep

- Sweep root: `/home/daniel/projections-data/artifacts/joint_rotation_rates_v1/sweeps/joint_hp_20260222T145828Z`
- 12 trials (`quick` preset), 9 successful, 3 failed.
- Best trial:
- `t005_lr-6p00em04_weight_decay-1p00em05_dropout-5p00em02`
- Run id: `joint_sweep_005_t005_lr-6p00em04_weight_decay-1p00em05_dropout-5p00em02_20260222T150026Z`
- Best metrics:
- `val_loss=3.6059`
- `val_minutes_mae=2.8408`
- `val_rates_mae=0.06031`
- `val_eff_mae=0.18234`
- `val_anti_smear=0.35990`
- Failed trials crashed with runtime signals (`-11`, `-6`) and one `double free or corruption` error in torch runtime.

### C) Raw-vs-raw comparison against current production models (fresh)

- Report: `/tmp/joint_eval_vs_current_20260222T151040Z.json`
- Eval window: `2026-01-29`..`2026-02-11` (214 team-games, 3,763 rows)
- Minutes MAE:
- joint `2.8067`
- current `2.6897`
- Rates MAE (9 targets, eligible rows):
- joint `0.06039`
- current `0.06716`
- Efficiency MAE:
- joint `0.18482`
- current unavailable (`NaN`) in this eval path (no comparable current eff columns in joined frame).

## 7) Known Gaps / Risks

- Some hyperparameter points trigger low-level runtime crashes (torch CPU path) instead of clean training failures.
- Current raw-vs-raw eval has incomplete efficiency parity against production rates outputs.
- Sim-level calibration parity is still an open question; prior sim comparisons showed large distribution shifts when noise assumptions differ.

## 8) Immediate Questions For External Reviewer

- Are current multi-task loss weights well-balanced, or should dynamic weighting (e.g., uncertainty weighting/GradNorm) be used?
- Is shared trunk + linear heads sufficient, or should rates/eff heads use deeper conditional decoding on gate/minutes features?
- Should the minutes gate objective be calibrated separately (temperature/platt/isotonic) before feeding downstream sim?
- How should we stabilize training to avoid runtime crashes at higher LR / some configs (precision, batching, torch settings)?
- What is the best evaluation protocol for true production parity (raw metrics + sim outcome metrics + optimizer-level KPIs)?
