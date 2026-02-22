# 04 Live Scorer Plan

## Objective

Build a production-grade live scorer for the joint rotation/minutes/rates model that:

- scores from live pre-tip features,
- writes sim-consumable artifacts (`effective_minutes.parquet`, `effective_rates.parquet`),
- preserves anti-leak semantics,
- and prevents "surprise starter -> zero minutes" misses.

## Problem Statement

Recent slates exposed a critical failure mode:

1. player is correctly identified as starting,
2. priors are weak (low historical minutes / low prior play-prob),
3. joint scorer outputs ~0 minutes anyway,
4. downstream sim/optimizer inherit that miss.

This is unacceptable for live operation and must be guarded by inference-time rules.

## Proposed Entry Point

Add a new CLI:

- `projections.cli.score_joint_rotation_rates_live`

Responsibilities:

1. load live features for a date/run-as-of timestamp,
2. run joint model inference (minutes + rates + efficiency),
3. apply post-inference live guardrails,
4. write canonical live outputs + manifest metadata.

## Data Contracts

The scorer must emit files with schemas compatible with current downstream consumers:

- Minutes output:
  - include `minutes_final`, `minutes_p50` (or equivalent), plus key identity columns
  - write `effective_minutes.parquet`
- Rates output:
  - include rate targets (`*_per_min`) and efficiency columns (`fg2_pct`, `fg3_pct`, `ft_pct`)
  - write `effective_rates.parquet`
- Metadata:
  - run id, model run id, config hash, feature source path, timestamp

## Training -> Inference Parity (Hard Gate)

The joint live scorer must enforce train/infer parity before publishing outputs.

### Parity Requirements

1. **Feature schema parity**
   - all required training features present at inference,
   - no missing key identity columns (`game_id`, `team_id`, `player_id`, `game_date`),
   - expected numeric/categorical coercions applied consistently.
2. **Semantic parity**
   - same prior columns and definitions as training (for example `minutes_from_stints_prior_20`, `prior_play_prob`),
   - same embedding index construction rules as training,
   - same normalization inputs (`feature_mean`, `feature_std`) loaded from model artifact.
3. **Imputation parity**
   - deterministic fill policy for absent/NaN features,
   - no silent fallback that changes feature meaning.
4. **Output contract parity**
   - emitted minutes/rates files remain compatible with sim and downstream readers.

### Parity Artifacts

Each run should write a parity artifact (for example `parity_report.json`) including:

- expected feature count vs present feature count,
- missing required columns,
- unexpected columns (informational),
- dtype/coercion warnings,
- null-rate summary for key prior/starter fields,
- pass/fail flag and failure reason.

If parity fails, scorer should fail closed (do not publish live outputs).

## Starter Whiff Guardrails (Required)

### 1. Starter Rescue Floor

If a player is a confirmed starter (or high-confidence projected starter), enforce a minimum minutes floor unless hard-out.

Suggested initial defaults (tunable):

- confirmed starter floor: `14.0`
- projected high-confidence starter floor: `12.0`

Hard exclusions:

- `is_out=1` / explicit inactive status / manual hard inactive override.

### 2. Gate/Eligibility Override

At inference:

- force starter eligibility in allocation mask (unless hard-out),
- optionally boost gate logits for starters before allocation.

### 3. Team-240 Preserving Redistribution

If rescue floor increases one or more starters:

- reduce minutes from lowest-priority non-starters,
- preserve team total = 240,
- avoid violating hard floors/constraints for locked starters.

### 4. Diagnostics + Alerting

Emit a machine-readable report for all rescue events:

- player id, team id, starter signal source,
- pre-rescue minutes, post-rescue minutes,
- donor players adjusted,
- reason code.

## Deployment Approach

This stack is experimental. We will run it live directly, monitor outcomes, and revert immediately if quality is unacceptable.

### Immediate Build

- Implement `projections.cli.score_joint_rotation_rates_live`.
- Score minutes/rates directly from live features.
- Emit `effective_minutes.parquet` and `effective_rates.parquet` in downstream-compatible locations.
- Apply starter rescue guardrails at inference time.
- Run parity audit before publish; fail closed on parity errors.

### Live Plug-In

- Point live sim/scoring path to joint outputs for real slates.
- Keep current model outputs available as fallback during the same window.
- Log model/version/runtime stamp for each run for clean attribution.

### Fast Revert

- Revert by switching the live path back to current minutes/rates producers.
- No retraining required for rollback; this is a config/entrypoint switch.

## Acceptance Criteria

1. No confirmed starter gets 0 minutes unless hard-out/inactive.
2. Team totals remain 240 after guardrails.
3. Train/infer parity report passes on live runs.
4. No schema regressions for sim consumption.
5. No NaN/invalid outputs in scored artifacts.
6. Comparable or better downstream sim quality vs current baseline.

## Risks

- Over-aggressive floors can inflate weak starters and hurt global calibration.
- Redistribution rules can create unintended donor instability if not bounded.
- Starter flags can be noisy near lock; source confidence must be encoded.

## Open Decisions

1. Exact floor values by starter signal type and injury regime.
2. Priority order for minute donors (bench depth rank vs model confidence vs priors).
3. Whether rescue applies only to confirmed starters or also projected starters.
4. Whether to couple rescue with `play_prob` adjustments for downstream consistency.

## Suggested Next Steps

1. Implement the joint live scorer CLI with artifact writing.
2. Add starter rescue + redistribution + diagnostics.
3. Wire it into live sim for direct testing.
4. Keep one-command rollback ready in config/runbook.
