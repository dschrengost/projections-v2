# NBA minutes stack deep dive: rotation realism + calibration guardrails

## 1) Current system end-to-end (what’s actually used)

### Production entrypoints
- Live feature build: `projections/cli/build_minutes_live.py` (called by `scripts/run_live_score.sh`)
  - Writes: `${PROJECTIONS_DATA_ROOT}/live/features_minutes_v1/<YYYY-MM-DD>/run=<RUN_ID>/features.parquet`
- Minutes scoring: `projections/cli/score_minutes_v1.py` (called by `scripts/run_live_score.sh`)
  - Loads model bundle via `config/minutes_current_run.json` → `bundle_dir` (minutes_v1 LGBM quantiles)
  - Optional allocator via `minutes_alloc_mode` (config) or `PROJECTIONS_MINUTES_ALLOC_MODE` (env)
  - Writes: `${PROJECTIONS_DATA_ROOT}/artifacts/minutes_v1/daily/<YYYY-MM-DD>/run=<RUN_ID>/minutes.parquet`
  - Copies to sim-consumed location (via `scripts/run_live_score.sh`):
    - `${PROJECTIONS_DATA_ROOT}/gold/projections_minutes_v1/game_date=<YYYY-MM-DD>/minutes.parquet`
    - `${PROJECTIONS_DATA_ROOT}/gold/projections_minutes_v1/game_date=<YYYY-MM-DD>/latest_run.json`
- Sim worlds generation: `scripts/sim_v2/generate_worlds_fpts_v2.py` (invoked via `scripts/run_live_sim.sh`)
  - Loads sim profile from `config/sim_v2_profiles.json` (live script defaults to `SIM_PROFILE=sim_v3`)
  - Consumes `minutes_p10/p50/p90` + `play_prob` from minutes parquet
  - 240 enforcement + rotation pruning: `projections/sim_v2/minutes_noise.py`

### Active minutes models
- `minutes_v1` (primary): LightGBM quantile bundle (see `projections/models/minutes_lgbm.py` for trainer)
- RotAlloc (optional allocation layer): `projections/minutes_alloc/rotalloc_production.py`
  - Loads models from `artifacts/experiments/lgbm_rotalloc_final_v1/models/` (rot8 classifier + minutes regressor)
  - Versioned production knobs: `config/rotalloc_production.json`

## 2) How `p10/p50/p90` (+ `play_prob`) are produced and consumed

### Scoring (`projections/cli/score_minutes_v1.py`)
- Base model:
  - Predicts quantiles (`0.1/0.5/0.9`) → calibrates tails (if bundle has a calibrator) → conformal adjustments.
  - Produces `minutes_p10`, `minutes_p50`, `minutes_p90` (and `_cond` aliases).
  - Predicts `play_prob` (if play-prob head enabled in bundle).
- Hidden post-processing in scorer:
  - OUT handling: forces `play_prob=0` and zeroes minutes for `status==OUT` / `lineup_role==out` (and optionally ESPN OUTs).
  - Optional “ghost minutes” caps: `_apply_rotation_minutes_caps()` clamps minutes for low-rotation players when `rotation_prob` exists.
  - Allocator-mode overrides:
    - For RotAlloc-family allocator modes, scorer forces `reconcile_team_minutes=none` and disables upside adjustment (tails are derived from legacy deltas; sim adds variance).

### Allocation / 240 enforcement
- `rotalloc_expk` (current production allocator): `score_rotalloc_minutes()` returns per-player
  - `eligible_flag` (eligibility mask),
  - `p_rot` (rotation prob),
  - `mu_cond` (conditional minutes mean),
  - plus an allocation `minutes_mean` that sums to 240 per team-game (waterfill + caps).
- `rotalloc_fringe_alpha` (new, flagged): replaces the deterministic allocation with a “shape layer”:
  - Core/fringe blend of RotAlloc proxy weights with a historical-minutes proxy (default: `roll_mean_5`).
  - Still respects RotAlloc eligibility and enforces sum-to-240 per team-game.
  - Tails (`minutes_p10/p90`) are re-centered using legacy deltas around the new `p50`.

### Sim consumption (`scripts/sim_v2/generate_worlds_fpts_v2.py`)
- Samples availability per-world using `play_prob` (Bernoulli, if enabled in the sim profile).
- Samples minutes per-world using `minutes_p10/p50/p90` (or a structured minutes noise path, depending on profile flags).
- Applies team-level enforcement and rotation pruning via `projections/sim_v2/minutes_noise.py`:
  - 240-minute reconciliation,
  - optional rotation caps (`max_rotation_size`) and protected core (`protected_rotation_size`),
  - optional behavior to preserve the input rotation (when upstream allocator already handled eligibility/240).

## 3) Diagnosing the “bench spread too wide / 6th–8th men crushed” failure mode

### What’s happening
The main failure mode is *allocator-induced flattening* when the RotAlloc proxy weights are diffuse:
- RotAlloc allocates across a 9–11 player eligible set with similar weights → minutes become too uniform across ranks ~6–11.
- That inflates rank 9–11 minutes and “crushes” the heavy bench (6th–8th men).

Second-order contributor (base model):
- Raw minutes can “leak” into deep bench buckets; reconciliation reduces this but also redistributes errors across buckets.

### Quick quantification (allocator comparison)
Source: `runs/abtest_minutes_alloc_multi/20251225T184342/aggregate_summary_clean.json` (10 slates, 7 with labels).

| Allocator | Mean MAE ↓ | 6th-man MAE ↓ | top8 share ↑ | bench_crush ↓ |
|---|---:|---:|---:|---:|
| B = `rotalloc_expk` | 4.66 | 8.72 | 0.892 | 9.61 |
| E = `rotalloc_fringe_alpha` (ship) | 4.57 | 6.32 | 0.899 | 9.35 |

Interpretation:
- The core/fringe blend materially improves “6th man minutes realism” (large MAE drop) while keeping overall MAE roughly flat-to-better.
- top8 share increases (less minutes sprayed into the 9th–12th).

### Quantiles + reconciliation diagnostics (raw vs L2 reconcile)
Source: `runs/minutes_eval_raw_vs_reconciled_2025-11-25_2025-12-04.json` (5 dates).

- Raw p50 MAE: 5.11 → Reconciled p50 MAE: 4.07
- Deep bench minutes “leak” (sum of predicted minutes on actual <4min players): 710 → 167
- Quantile coverage (target ~0.90 each):
  - Raw: p10=0.819, p90=0.945
  - Reconciled: p10=0.936, p90=0.945

Notes:
- This is a helpful regression test for calibration/drift, but production RotAlloc modes currently force `reconcile_team_minutes=none` (allocator already enforces 240).

## 4) Shippable recommendations

### Ship today (flagged, minimal)
Enable `rotalloc_fringe_alpha` minutes allocation behind an explicit flag.
- Mode switch (no production change unless set):
  - `PROJECTIONS_MINUTES_ALLOC_MODE=rotalloc_fringe_alpha`
- Knobs (env overrides; safe defaults live in `config/rotalloc_production.json`):
  - `ROTALLOC_BLEND_K_CORE` (default 8)
  - `ROTALLOC_BLEND_ALPHA_CORE` (default 0.8)
  - `ROTALLOC_BLEND_ALPHA_FRINGE` (default 0.3)
  - `ROTALLOC_BLEND_SHARE_GAMMA` (default 1.0)
  - `ROTALLOC_BLEND_SHARE_COL` (default auto-select from `roll_mean_5`, `min_last5`, `min_last3`, `roll_mean_10`)

Why this is pragmatic:
- Doesn’t change schemas.
- Doesn’t require training a new model.
- Preserves RotAlloc eligibility + 240 constraint.
- Targets the observed failure mode directly (diffuse weights → over-flat bench).

### Next step (more robust, still small)
Swap the “share proxy” from historical minutes to a learned share signal (minute-share model), still gated by RotAlloc eligibility:
- Option 1: use `share_with_rotalloc_elig` (allocator C) where share predictions are scaled to 240 within eligibility.
- Option 2: keep `rotalloc_fringe_alpha`, but set `ROTALLOC_BLEND_SHARE_COL` to a produced share prediction column (once we attach it in scoring).

This should handle role-change games (new starters/returning players) better than purely historical proxies.

## 5) Repro commands

### Tests
- `uv run pytest -q`

### Allocator A/B/C/D/E/F comparison (multi-slate)
- `uv run python scripts/abtest_minutes_allocators.py multi --start-date 2025-11-25 --end-date 2025-12-04 --no-share-model`

### Raw vs reconciled evaluation (quantiles + leakage + realism)
- `uv run python scripts/minutes/eval_raw_vs_reconciled.py --start-date 2025-11-25 --end-date 2025-12-04 --season 2025 --output-json runs/minutes_eval_raw_vs_reconciled_2025-11-25_2025-12-04.json`

### One-shot minutes run on a slate (flagged allocator)
- Feature build + score (minimal):
  - `uv run python -m projections.cli.build_minutes_live --date 2025-12-26 --run-id test_20251226T000000Z`
  - `PROJECTIONS_MINUTES_ALLOC_MODE=rotalloc_fringe_alpha uv run python -m projections.cli.score_minutes_v1 --date 2025-12-26 --mode live --run-id test_20251226T000000Z`
- Full live pipeline (includes rates + copies to gold): `PROJECTIONS_MINUTES_ALLOC_MODE=rotalloc_fringe_alpha ./scripts/run_live_score.sh 2025-12-26`
