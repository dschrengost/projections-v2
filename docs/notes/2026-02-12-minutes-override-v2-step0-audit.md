# Minutes Override V2 Step-0 Audit (Worlds/Optimizer/Contest-Sim)

Date: 2026-02-12

This note documents the **current** manual minutes override pipeline and the exact conflict points for introducing v2 semantics without breaking the legacy path.

## 1) Where overrides are stored and how Gameview writes them

- Gameview/Ops writes overrides via `POST /api/ops/overrides` (`projections/api/ops_api.py`).
- Persistence location:
  - `data_root/artifacts/ops/overrides_v1/game_date=YYYY-MM-DD/overrides.json`
  - implemented in `projections/ops/overrides.py` (`overrides_path`, `upsert_overrides`, `load_overrides_map`).
- Payload shape is `{"version", "game_date", "updated_at", "overrides": [{"game_id","player_id","fields",...}]}`.

## 2) Where overrides are read and applied in the pipeline

- Minutes scoring writes baseline `minutes.parquet`.
- Effective layer (`projections/pipeline/effective_inputs.py`) calls `apply_overrides_to_minutes_df(...)` from `projections/ops/overrides.py` and writes `effective_minutes.parquet`.
- Sim worlds (`scripts/sim_v2/generate_worlds_fpts_v2.py`) prefers `effective_minutes.parquet` when loading minutes input.

## 3) Reconcile/allocation order relative to overrides

- `apply_overrides_to_minutes_df(...)` mutates minutes/status/play_prob and then reconciles to team=240 (hard-target path currently in `projections/ops/overrides.py`).
- World generation then re-samples availability/minutes and re-projects each team-world to 240 via `allocate_team_minutes_matrix(...)` (`projections/sim_v2/minutes_allocator.py`).
- So there are currently two points where allocation/reconciliation can move minutes:
  - effective layer reconcile
  - world-level team projection

## 4) Existing bench-zero / membership / play_prob gates that can zero players

In `scripts/sim_v2/generate_worlds_fpts_v2.py`:

- Availability draw from `play_prob_eff` (Bernoulli active mask).
- Team feasibility gate can resample/promote activity.
- `eligible_flag` filtering can remove players from active set.
- `bench_zero_mixture` can drop low-minute active players to zero.
- Hard-force inactive logic currently zeros players based on out-like status / play_prob / ops markers.

These gates execute **before** final team-240 projection and can zero players unless explicitly protected.

## 5) Exact conflict point (double-apply risk)

- `effective_minutes.parquet` already contains legacy override effects from `apply_overrides_to_minutes_df(...)`.
- If a new v2 compiler directly consumes those already-adjusted minutes and also re-applies override payloads, deltas/targets can be applied twice.

### Required v2 guardrail

When `minutes_override_mode=v2`:

- Legacy world path must be bypassed.
- Override payload must be applied **exactly once** in world generation.
- Baseline minutes used by v2 compiler must come from pre-override/model-origin columns where available (or equivalent baseline reconstruction), not from already-adjusted override outputs.

