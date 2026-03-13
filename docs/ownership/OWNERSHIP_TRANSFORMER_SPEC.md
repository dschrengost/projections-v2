# Ownership Transformer v2: Slate-Level Ownership Model

## Spec Status: DRAFT v0.2 (2026-03-13)

Related production specs:

- [Game Transformer v2 Spec](../joint_rotation_rates_v1/GAME_TRANSFORMER_SPEC.md)
- [Live Pipeline Production Spec](../pipeline/LIVE_PIPELINE_PRODUCTION_SPEC.md)
- [Ownership System Overview](./README.md)

---

## 1. Motivation

### 1.1 Current gap

`ownership_v1` is a row-wise `LightGBM` model trained against slate ownership labels.
It is useful, but it has a persistent failure mode:

- predictions are too flat
- dust is overpredicted
- true mega-chalk is underpredicted

This is not just a calibration issue. Ownership is a slate allocation problem with a hard
mass constraint and strong contextual interactions across players.

### 1.2 What this spec changes

This revision makes four decisions:

1. Model ownership at the full-slate level, not as independent player rows.
2. Predict slate logits jointly and normalize in-model to the exact roster-slot mass.
3. Use in-house DK contest labels as the canonical supervision source.
4. Allow optional enrichment from `GTV2` latent player states when available.

### 1.3 Non-negotiable requirements

- Preserve the downstream live ownership output contract: one prediction row per draftable player.
- Enforce exact slate sum constraints by construction.
- Support all classic GPP slates, including classic sub-slates such as early/late/night.
- Avoid any runtime dependency on `LineStar`.
- Degrade safely when `GTV2` enrichments are unavailable at inference time.

---

## 2. Problem Framing

### 2.1 Target object

For a DraftKings classic slate with player pool `P`, the target is:

```
o in R^P
```

where:

- `o_i >= 0`
- `sum_i o_i = R * 100`
- `R = 8` for DK NBA classic

This is a constrained slate share distribution, not an unconstrained per-player regression.

### 2.2 Why row-wise models flatten

Row-wise models are biased toward conditional averages because they do not directly represent:

- competition among similarly priced players
- salary-tier crowding
- positional scarcity within a slate
- team and game concentration effects
- the hard mass transfer from one player gaining ownership to another losing it

Post-hoc calibration can reshape outputs, but it does not fix the structural limitation that
the model never learned the slate equilibrium jointly.

---

## 3. Design Principles

1. Treat ownership as a slate distribution:
   the model should see the entire player pool at once.

2. Enforce constraints in the model:
   exact slate sums should come from the forward pass, not a large downstream patch.

3. Keep live integration practical:
   the model must support a non-`GTV2` path and an enriched path under the same scorer contract.

4. Prefer in-house labels and features:
   `LineStar` is not a production dependency.

5. Optimize for DFS usefulness, not just mean error:
   top-chalk ordering matters more than tiny improvements on the long tail.

---

## 4. Architecture Overview

### 4.1 Training unit

A training example is a full DK draft group slate, not a player row.

Each slate contains:

- up to `P_max` playable draftable players
- per-player tabular features
- optional per-player `GTV2` latent features
- a mask indicating which slots are real players
- ownership labels expanded to the full salary universe

### 4.2 Sequence layout

The default v1 sequence is player-only:

```
[player_1] [player_2] ... [player_P]
```

with learned embeddings from:

- continuous projection and context features
- categorical position/team/opponent signals
- optional `GTV2` latent state channels

No special game token is required in v1. Slate context is learned through full self-attention.

### 4.3 Backbone

The model is a standard encoder-style transformer over slate player tokens:

- input projection from tabular feature vector to `d_model`
- `N` transformer encoder blocks
- layer norm + residual connections
- output scalar logit per player

Current strong experimental configuration:

- `d_model = 128`
- `num_layers = 3`
- `num_heads = 4`
- `hidden_dim = 256`

### 4.4 Output head

For masked player logits `z_i`, predict ownership share via masked softmax:

```
p_i = exp(z_i) / sum_j exp(z_j)
o_i = p_i * (R * 100)
```

Properties:

- `o_i >= 0`
- exact slate sum by construction
- every prediction is relative to the rest of the slate

### 4.5 Optional `GTV2` enrichment

When available, append `GTV2` latent signals to each player token.

Current enriched feature set includes:

- `gtv2_minutes_deterministic`
- `gtv2_active_logit`
- `gtv2_active_prob_proxy`
- `gtv2_state_*` latent state dimensions

These features are intended to expose richer rotation and role context than the ownership model
can infer from tabular projections alone.

### 4.6 Missing-feature fallback

`GTV2` coverage is not guaranteed. The ownership model must support missing enrichments by:

- merging on `(date, player_name)` after name normalization
- zero-filling missing `gtv2_*` columns
- optionally adding a binary `has_gtv2_features` indicator in future revisions

This fallback behavior is part of the live contract and must match training-time behavior.

---

## 5. Loss Design

### 5.1 Base objective

The default training loss is a slate-level ownership-share fit objective:

- share-space loss on normalized ownership distribution
- auxiliary weighted absolute error in percent space

This keeps optimization aligned with the exact constrained output.

### 5.2 Chalk emphasis

Ownership quality is judged disproportionately by high-owned plays.
The training loop therefore supports additional emphasis on top-owned players via:

- top-k weighted error terms
- underprediction penalties for high-owned players

Early experiments show that very aggressive tail loss overcorrects, so tail emphasis should remain
bounded and be evaluated against full-slate metrics, not top-tail bias alone.

### 5.3 Calibration stance

Calibration is not the primary design lever for `ownership_v2`.

Observed behavior from the in-house `LGBM` baseline:

- flat predictions were mostly structural
- simple power and softmax calibration did not materially improve top-k recall

Calibration may still be used later for mild reshaping, but architecture and feature quality are
the primary drivers of improvement.

---

## 6. Data and Labels

### 6.1 Canonical label source

Use DK contest results aggregated by slate:

- input: `bronze/dk_contests/nba_gpp_data/<date>/results/contest_*_results.csv`
- builder: `scrapers/dk_contests/build_ownership_data.py`
- output: `bronze/dk_contests/ownership_by_slate/all_ownership.parquet`

### 6.2 Contest filtering policy

Keep:

- `Classic` GPP contests
- classic sub-slates such as early, late, and night when they are true classic tournaments

Exclude:

- showdown
- single-game
- tiers
- cash-like formats such as double-ups or head-to-heads if they appear in source data

The ownership label builder is responsible for enforcing this policy before slate aggregation.

### 6.3 Slate aggregation policy

Contest results are clustered into slates by player-pool overlap.
Within a slate cluster:

- missing players in a given contest are treated as `0%` for that contest
- ownership is aggregated across qualifying contests

This avoids inflated per-slate sums from partial contest coverage.

### 6.4 Full salary-universe expansion

Training rows must be expanded to the full DK salary slate, not just players observed in contest CSVs.

For every matched slate:

- join to the full DK salary file
- keep every draftable salary row
- zero-fill ownership labels for players absent from contest results

This is required for a correct slate sum target and for realistic long-tail behavior.

### 6.5 Base feature sources

Current in-house base features come from pre-lock artifacts already used elsewhere in the repo:

- DK salaries
- projection features
- minutes features
- slate-relative ranks and z-scores
- structural context such as salary tiers and positions
- optional historical DK ownership priors

The ownership training base is built by:

- `scripts/ownership/build_ownership_inhouse_base.py`

### 6.6 `GTV2` enrichment source

The current offline enrichment builder is:

- `scripts/ownership/build_gtv2_ownership_embeddings.py`

Inputs:

- ownership base parquet
- historical `GTV2` feature parquet
- `GTV2` bundle dir

Output:

- parquet of per-player latent features merged into the ownership base

This path currently uses historical `GTV2` rotation training features for backfill experiments.

### 6.7 Train/validation split

Use date-based splits only. Ownership labels are highly date-correlated and same-day leakage is not acceptable.

Current benchmark slice for the strong transformer runs:

- training dates before `2026-01-25`
- validation dates `2026-01-25` through `2026-02-12`

### 6.8 Current experimental quality

Best pre-`GTV2` transformer run:

- run: `ownership_xfmr_v1_12ep_big`
- MAE: `3.071`
- pooled Spearman: `0.863`
- Top-5 hit: `0.400`
- Top-10 hit: `0.516`
- Top-20 hit: `0.598`

Best current enriched run:

- run: `ownership_xfmr_v1_12ep_big_gtv2`
- MAE: `2.851`
- pooled Spearman: `0.873`
- Top-5 hit: `0.480`
- Top-10 hit: `0.524`
- Top-20 hit: `0.642`

Reference in-house `LGBM` baseline:

- run: `inhouse_v2_v6_logit_chalk5_clean`
- MAE: `3.312`
- pooled Spearman: `0.822`
- Top-5 hit: `0.320`
- Top-10 hit: `0.460`
- Top-20 hit: `0.576`

Interpretation:

- the transformer architecture is clearly better than the cleaned `LGBM` baseline
- `GTV2` latent enrichment materially improves top-chalk behavior
- some top-tail underprediction remains, but the direction is correct

---

## 7. Artifact Contract

### 7.1 Training artifacts

`ownership_v2` training artifacts should live under:

```
artifacts/ownership_v2/runs/<run_id>/
```

Minimum expected contents:

- `model.pt`
- `config.json`
- `feature_columns.json`
- `meta.json`
- `val_predictions.csv`
- training summary metrics

### 7.2 Metadata requirements

`meta.json` should record:

- training base path
- date split
- model hyperparameters
- whether `GTV2` features were used
- exact feature column list
- key validation metrics

This is required for reproducibility and promotion decisions.

---

## 8. Live Integration Design

### 8.1 Current state

Live production ownership scoring currently runs through:

- `projections/cli/score_ownership_live.py`

As of `2026-03-13`, this path supports both:

- `ownership_v1` (`LightGBM`)
- `ownership_v2` (slate transformer)

via explicit `--model-family` routing.

### 8.2 Required `ownership_v2` live scorer

`ownership_v2` live inference now runs through the same CLI entrypoint with explicit model-family routing:

```bash
uv run python -m projections.cli.score_ownership_live \
  --date 2026-03-13 \
  --run-id <run_id> \
  --model-family ownership_v2 \
  --model-run <ownership_v2_run_id>
```

Implemented behavior:

1. loads `model.pt` and `config.json`
2. rebuilds the exact transformer feature frame for the live slate
3. applies the same feature ordering used in training
4. handles optional `GTV2` enrichment
5. emits predictions under the current silver output contract

Implementation location:

- CLI routing + live features/scoring: `projections/cli/score_ownership_live.py`
- Transformer artifact loader + scorer: `projections/ownership_v2/inference.py`

### 8.3 Live inputs

Minimum live inputs for the non-`GTV2` transformer:

- DK salaries for the draft group
- live projection features
- live minutes features
- slate-relative engineered features used in training

Optional live enrichment inputs:

- `GTV2` live features or hidden states derived from the active production `GTV2` path

### 8.4 Output contract

The scorer preserves downstream compatibility by writing the same core columns:

- `player_id`
- `player_name`
- `salary`
- `pos`
- `team`
- `proj_fpts`
- `pred_own_pct`
- `pred_own_pct_raw`
- `game_date`
- `run_id`
- `model_run`
- `model_family`

If the transformer uses masked softmax ownership directly, `pred_own_pct_raw` should still be
defined as the pre-normalization score used by downstream diagnostics.

### 8.5 Lock persistence

The lock-cache implementation is now namespaced by scorer family and run:

- `silver/ownership_predictions/<date>/<draft_group_id>_locked__<model_family>__<model_run>.parquet`

Compatibility note:

- legacy `*_locked.parquet` is still read/written for `ownership_v1` compatibility only.
- `ownership_v2` never reuses `ownership_v1` lock caches.

### 8.6 `GTV2` live integration choices

There are two viable rollout options:

1. ship the non-`GTV2` transformer first
2. ship a richer path that computes `GTV2` enrichments live

Recommended order:

1. productionize the plain transformer scorer first
2. add optional `GTV2` enrichment behind a feature flag
3. promote the enriched path only after train/infer parity is proven

### 8.7 Failure handling

Current behavior:

- `GTV2` features are missing
  - scorer zero-fills required `gtv2_*` columns and continues.
- feature columns are mismatched
  - scorer fails loudly before writing output.
- model artifacts are missing/invalid
  - scorer fails loudly before writing output.

This preserves safe degradation while protecting train/infer feature-contract integrity.

---

## 9. Promotion Criteria

`ownership_v2` should not replace `ownership_v1` in production until it satisfies all of:

1. beats the current `ownership_v1` baseline on the fixed validation slice
2. preserves exact slate sum constraints in live scoring
3. shows no train/infer feature drift on a live replay sample
4. passes a downstream optimizer sanity check on several recent slates
5. has an operational rollback path to `ownership_v1`

### 9.1 Current recommendation

The best current candidate is:

- `ownership_xfmr_v1_12ep_big_gtv2`

But the recommended first live deployment candidate is:

- non-`GTV2` transformer path

Reason:

- simpler live feature contract
- lower train/infer drift risk
- still materially better than `ownership_v1` `LGBM`

---

## 10. Implementation Plan

### Phase 1: Spec and training stabilization

- keep this document current as architecture and contracts change
- maintain the fixed benchmark slice
- continue comparing against `ownership_v1`

### Phase 2: Live scorer

Status: **implemented** (2026-03-13)

- `ownership_v2` live scoring entry point implemented in `score_ownership_live.py`
- artifact loader + feature-contract validation implemented in `ownership_v2/inference.py`
- outputs written to existing silver ownership path (same downstream contract)
- lock cache namespaced by model-family/model-run

### Phase 3: Optional `GTV2` live enrichment

Status: **implemented with safe fallback**

- scorer supports optional `--gtv2-features-path`
- scorer attempts to auto-load from live GTV2 score artifacts when available
- missing-feature fallback + zero-fill implemented
- coverage diagnostics are now emitted in run-scoped health summary JSON

### Phase 4: Production canary

Status: **completed** (2026-03-13)

- canary replay window executed for `2026-03-10` through `2026-03-12`
  - `ownership_v1` canary run: `20260313T170000Zcanaryv1`
  - `ownership_v2` canary run: `20260313T170500Zcanaryv2`
- production-path replay metrics favored `ownership_v2` on this window:
  - MAE: `3.3229` (`v2`) vs `3.3319` (`v1`)
  - RMSE: `7.4280` (`v2`) vs `8.0151` (`v1`)
  - pooled Spearman: `0.7548` (`v2`) vs `0.7347` (`v1`)
  - recall@10 / recall@20: `0.1667 / 0.3167` (`v2`) vs `0.1333 / 0.3000` (`v1`)
- concentration sanity check on main slates passed:
  - no sum-contract violations
  - no extreme single-player concentration (`max pred_own_pct` <= `66.7%` on sampled main slates)
- production selector/runtime-stamp verification completed and live pointer switched to internal transformer run `20260313T164435Z`

---

## 11. Open Questions

1. Should the live scorer support both plain and enriched transformer bundles, or should those be separate run families?
2. Do we want explicit slate tokens or team/game pooling tokens in the next architecture revision?
3. Should high-owned-player ranking be optimized with an explicit pairwise ranking loss?
4. What is the final live source of `GTV2` latent features: saved hidden states, recomputed embeddings, or a smaller distilled feature subset?

Question 5 is now resolved:

- lock-cache snapshots are model-family/model-run scoped and no longer mix scorers.

---

## 12. Pre-Live Checklist (Updated 2026-03-13)

Completed:

1. **Canonical flow wiring** ✅
   - both `prefect_flows/live_nba_pipeline.py` and `prefect_flows/live_nba_pipeline_v3.py` now route ownership through selector-driven source/model logic with fallback.

2. **Model run selector contract** ✅
   - `config/ownership_current_run.json` added.
   - runtime selector path support added under `$PROJECTIONS_DATA_ROOT/control_plane/model_selectors/ownership_current_run.json`.
   - manifest/runtime-stamp now track `ownership_current_run`.

3. **Operational guardrails** ✅
   - run-scoped `ownership_health_summary.json` emitted by `score_ownership_live`.
   - includes per-slate sum checks, lock-cache status, and GTV2 row-coverage diagnostics.

4. **Runbook/docs updates** ✅
   - control-plane/dev playbook updated with selector + rollback workflow and replay-eval command.

5. **Replay tooling for parity/canary** ✅
   - `scripts/ownership/evaluate_ownership_production_path.py` now supports:
     - run-scoped predictions (`--pred-run-id`)
     - namespaced lock-cache files
     - model-family filtering (`--model-family`)
6. **Execute canary replay and review results** ✅
   - replay window run for `2026-03-10..2026-03-12` with isolated run IDs:
     - `ownership_v1`: `20260313T170000Zcanaryv1`
     - `ownership_v2`: `20260313T170500Zcanaryv2`
   - `ownership_v2` improved production-path RMSE, rank correlation, and recall@k on this window.
   - concentration sanity check on sampled main slates showed no pathological behavior.

7. **Production deploy + selector verification** ✅
   - PROD deploy completed and runtime selector confirmed at:
     - `/home/daniel/projections-data/control_plane/model_selectors/ownership_current_run.json`
   - Prefect flow run `3c07ffa7-94f0-4fcb-9532-f7bbda475237` runtime stamp reported:
     - `ownership_source=internal`
     - `ownership_model_family=ownership_v2`
     - `ownership_model_run=ownership_xfmr_v1_12ep_big`
   - live ownership pointer promoted to run `20260313T164435Z` (`prefect-v3`) with `ownership_v2` output files.
