# 03 Training And Rollout

## Training Strategy (Staged)

Joint training should be staged to avoid identifiability drift between minutes and rates.

## Stage A: Minutes/Rotation Warm Start

Goal:

- recover current rotation-set minutes behavior with the shared encoder + minutes heads only.

Train:

- `L_minutes`,
- `L_rotation`,
- `L_k_regularizer`,
- `L_gate_bce`, `L_minutes_out`, and `L_anti_smear` (now wired in the joint trainer).

Stop gate:

- parity with current `rotation_set_minutes_v1` validation metrics.

## Stage B: Rates Head Bring-Up

Goal:

- train rates/efficiency heads on top of stable role/minutes representation.

Train:

- freeze or partially freeze encoder + minutes heads,
- optimize `L_rates + L_efficiency`.

Use rates mask/weights:

- minimum-minutes and/or minute-weighted labels.

## Stage C: Joint Fine-Tuning

Goal:

- let shared representation adapt across all tasks while preserving minutes quality.

Train:

- unfreeze full network,
- optimize combined objective:
  - `L_total = w_m * L_minutes + w_rot * L_rotation + w_k * L_k + w_r * L_rates + w_e * L_eff + w_c * L_counts`.

`L_counts`:

- consistency between predicted minutes and rates vs realized counts.
- start with low weight to avoid overpowering base heads.

Suggested initial weights:

- `w_m=1.0`,
- `w_rot=0.4`,
- `w_k=0.05`,
- `w_r=0.6`,
- `w_e=0.2`,
- `w_c=0.1`.

These are starting points, not promotion defaults.

## Evaluation

## Task Metrics

Minutes/rotation:

- existing `val_mae_team_w`,
- `val_k_hat_label_mae`,
- rotation-size diagnostics,
- dust/concentration diagnostics.

Rates:

- per-target MAE/RMSE,
- calibration by minutes buckets,
- injury/vacancy slice performance.

## Joint Metrics

1. Count reconstruction:
- MAE on realized game stats from `minutes_hat * rates_hat`.

2. World-level sim diagnostics:
- compare simulated stat and FPTS distribution moments,
- compare teammate/opponent covariance calibration,
- monitor sparse/injury slate behavior.

3. Decision quality:
- downstream lineup quality deltas in historical contest replay (if available).

## Rollout Plan

1. Offline backtest only (no live writes).
2. Shadow mode in live pipeline:
- score joint model alongside current models,
- write to debug namespace only.
3. Compare for fixed date window with pre-agreed promotion gates.
4. Switch scorer to joint outputs while preserving same output schema paths.
5. Keep one-line rollback via config selector.

## Output Compatibility Requirement

The joint scorer must write outputs compatible with:

- current minutes consumers,
- current rates consumers,
- existing sim join logic (no immediate downstream rewrite required).

## Implementation Backlog (Initial)

1. Add a new model module:
- `projections/rotation/joint_set_model_v1.py`

2. Add a joint dataset builder (implemented):
- `scripts/rotation/build_joint_rotation_rates_dataset_v1.py`

3. Add a joint trainer (implemented):
- `scripts/rotation/train_joint_rotation_rates_model_v1.py`

4. Add a joint live scorer (shadow-first):
- `projections/cli/score_joint_rotation_rates_live.py`

5. Add artifact loader/config selector:
- `config/joint_rotation_rates_current_run.json` (proposed),
- plus runtime stamp inclusion.

6. Add validation scripts:
- world-level diagnostics adapter for joint outputs,
- task-level regression checks.

## Immediate Next Sprint (Concrete)

1. Build a minimal joint model (implemented):
- shared encoder + existing gate/share minutes heads + simple multi-output rates head.
2. Train on a capped window (`max-team-games`) and produce first parity report against current baselines (implemented).
