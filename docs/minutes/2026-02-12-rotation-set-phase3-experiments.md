# Rotation-Set Phase 3 Experiment Matrix (Embeddings + K Supervision)

## Goal

Run controlled ablations for active-rotation membership and bench-tail allocation:

1. Baseline gate model (`k_target_source=fixed`, no player embeddings).
2. Add player identity embeddings.
3. Add label-derived K supervision (`k_target_source=label`).
4. Add blended K supervision (`k_target_source=blend`).

Evaluate all runs with close-to-tip live snapshots and catastrophic miss slices.

## Canonical Train Command

```bash
uv run python scripts/rotation/train_rotation_set_model_v1.py \
  --dataset-dir /home/daniel/projections-data/training/datasets/rotation_train_v1_20260103 \
  --model settransformer \
  --use-gate-head \
  --alloc-activation entmax \
  --rot-loss-type focal \
  --epochs 12 \
  --batch-size 32 \
  --lr 1e-3 \
  --device cpu
```

## Ablation Overrides

- Baseline:
```bash
--k-target-source fixed --k-target 9.5
```

- + player embeddings:
```bash
--use-player-embeddings --player-embedding-dim 16 \
--use-player-team-embeddings --player-team-hash-buckets 16384 --player-team-embedding-dim 8
```

- + label-K supervision:
```bash
--k-target-source label
```

- + blended K supervision:
```bash
--k-target-source blend --k-target 9.5 --k-target-blend 0.5
```

## Primary Metrics To Compare

- `val_k_hat_label_mae`
- `val_rotation_size_label_mean` vs `val_rotation_size_hat_mean`
- close-to-tip `ghost_dnp.rate`
- close-to-tip `missed_run.rate`
- `tail_minutes_mae_team240`

## Run Metadata (executed)

- Train dataset: `/home/daniel/projections-data/training/datasets/rotation_train_v1_20260103`
- Shared train args: settransformer + gate head, `epochs=12`, `batch_size=32`, `lr=1e-3`, `device=cpu`.
- Additional train override used for all ablations: `--max-team-minutes-gap 220` (default `2.0` yielded an empty train split on this dataset snapshot).
- Live eval window: `2026-01-02` last-before-tip snapshots.
- Snapshot age diagnostics (all four runs): `minutes_to_tip_p50=4.6`, `minutes_to_tip_p90=59.6`.

## Results

### Offline train/val snapshot

| Variant | best `mae_team_w` | `val_k_hat_label_mae` |
|---|---:|---:|
| baseline_fixed | 7.9940 | 0.257 |
| emb_fixed | 8.5076 | 0.297 |
| emb_labelk | 8.2011 | 0.411 |
| emb_blendk | 8.4779 | 0.321 |

### Live close-to-tip (candidate outputs)

| Variant | `player_mae` | `top9_player_mae_team240` | `tail_minutes_mae_team240` | `ghost_rate_active_only` | `missed_rate_active_only` |
|---|---:|---:|---:|---:|---:|
| baseline_fixed | 7.9844 | 9.0112 | 46.4199 | 0.1412 | 0.2118 |
| emb_fixed | 7.9896 | 9.0220 | 46.4147 | 0.1412 | 0.2118 |
| emb_labelk | 7.9954 | 9.0307 | 46.4382 | 0.1412 | 0.2118 |
| emb_blendk | 8.0137 | 9.0532 | 46.5590 | 0.1412 | 0.2118 |

### Context baselines (same live window)

| Model | `player_mae` | `top9_player_mae_team240` | `tail_minutes_mae_team240` | `ghost_rate_active_only` | `missed_rate_active_only` |
|---|---:|---:|---:|---:|---:|
| `current` (from live logs) | 5.3308 | 8.9809 | 21.4012 | 0.0087 | 0.0000 |
| `baseline_top8` heuristic | 7.6635 | 12.8446 | 11.5514 | 0.0087 | 0.1824 |
| phase3 candidate (best) | 7.9844 | 9.0112 | 46.4199 | 0.1412 | 0.2118 |

### Strict close-to-tip slice (`--max-minutes-to-tip 30`)

Rows kept: `246` player rows / `14` team-games.

| Variant | `player_mae` | `top9_player_mae_team240` | `tail_minutes_mae_team240` | `ghost_rate_active_only` | `missed_rate_active_only` |
|---|---:|---:|---:|---:|---:|
| baseline_fixed | 7.9360 | 9.0490 | 44.6211 | 0.1322 | 0.2101 |
| emb_fixed | 7.9318 | 9.0401 | 44.6285 | 0.1322 | 0.2101 |
| emb_labelk | 7.9368 | 9.0486 | 44.6403 | 0.1322 | 0.2101 |
| emb_blendk | 7.9556 | 9.0715 | 44.7637 | 0.1322 | 0.2101 |

Context on same strict slice:

- `current`: `player_mae=5.0757`, `ghost_rate_active_only=0.0115`, `missed_rate_active_only=0.0000`
- `baseline_top8`: `player_mae=7.3990`, `ghost_rate_active_only=0.0115`, `missed_rate_active_only=0.1765`

## Readout

- Player embeddings + label/blend K targets did not improve the production pain metrics in this run.
- Catastrophic misses are effectively unchanged across all four ablations.
- Active-only ghost rate remains very high (14.1%), which confirms the active-rotation membership problem is still unresolved.
- The model appears to enforce team-240 strongly but over-allocates tail minutes (large positive tail bias), which is consistent with bench smear.

## Decision

- Do not promote these embedding/K-target variants as-is.
- Keep this branch as scaffolding and move to a role-aware membership experiment:
  - add coach/role stability features and substitution-role priors,
  - train explicit active-only membership targets (not just minute threshold proxies),
  - evaluate with the active-only catastrophic panel as a hard gate.

## Inference Mask Ablation (No Retrain, baseline_fixed model)

Purpose: isolate how much behavior is coming from hard inference gating vs learned outputs.

### A) `alloc_mask_mode=strict` (current behavior)

- Output shape on sample run (`20260102T235959Z`): exactly `9` players with `>0` minutes per team-game.
- Positive minutes are near-uniform (`~26-27` for each of the 9 players), indicating weak learned ranking inside mask.
- Guardrails: no pathology fallback, sparsify active.

### B) `alloc_mask_mode=not_out` (all non-OUT eligible)

- Raw model allocations become too flat (`~11-16` players with `>=8` minutes per team-game).
- Guardrails pathology fallback trips on **all 20 team-games** (`min_max_minutes,min_top5_sum`), so outputs degrade to baseline fallback behavior.
- Live metrics (all-games, active-only catastrophic):
  - `player_mae=7.6091`
  - `missed_rate_active_only=0.0000`
  - `ghost_rate_active_only=0.1610`
  - `tail_minutes_mae_team240=62.0085`

### Interpretation

- This confirms the core failure mode: model membership signal is not strong enough on its own.
- `strict` mask over-constrains and causes over-zero misses; `not_out` removes constraint and causes over-smear plus pathological flatness.
- The production pain is not solved by embeddings/K-target tweaks alone; we need stronger membership supervision and richer role/context features.

## Phase 3.1 (next run) - concrete spec

1. Train target split
   - Keep minutes head, but add explicit active-only membership target (`y_rot_active = 1[minutes>=X and active]`).
   - Keep K supervision, but evaluate against active-only rotation size labels.

2. Feature additions (highest leverage first)
   - `last_n_games_active_rot_rank` (within-team minute/share rank).
   - `first_sub_pattern` / `bench_entry_order_prior` (role continuity proxy).
   - `lineup_continuity_5` (overlap with prior-game top rotation).
   - `coach_tenure_proxy` or stable team-style latent.

3. Inference contract
   - Replace hard strict pre-mask as default with learned membership post-processing:
     - compute `gate_prob` on all non-OUT rows,
     - derive `K_hat` and apply cardinality-aware top-K on `gate_prob` (or sampled K for worlds),
     - allocate shares only inside selected set.

4. Acceptance gate (must pass before rollout)
   - Active-only close-to-tip:
     - `ghost_rate_active_only` lower than current strict mode.
     - `missed_rate_active_only` lower than current strict mode.
   - Tail realism:
     - lower `tail_minutes_mae_team240`.
   - No full-slate pathology fallback bursts.
