# Play-Prob Policy Layer (Rotation Locks)

Goal: stabilize minutes feasibility by making availability semantics sane for rotation locks, without retraining models or changing downstream projection schemas.

## Why this exists

The sim pipeline samples `active_mask ~ Bernoulli(play_prob)` and then enforces team/world feasibility via resampling + deterministic promotion. When `play_prob` is too low for true rotation locks (often because the upstream model prices DNP risk conservatively), the feasibility gate ends up promoting players and creates a regime where `sim_p_active >> play_prob` for some players.

This policy layer makes rotation locks who are not on the injury report behave like ~certain actives, reducing resampling/promotions while keeping DNP pricing for fringe/injured players.

## Where it runs

- Policy implementation: `projections/sim_v2/play_prob_policy.py`
- Wired in sim generator: `scripts/sim_v2/generate_worlds_fpts_v2.py`
- Diagnostics: `scripts/diagnostics/world_sparsity_stats.py`

## Output semantics

The policy is used only for *availability sampling*:

- `play_prob_raw`: original `play_prob` (clipped to `[0, 1]`)
- `play_prob_eff`: effective probability used for availability draws / feasibility gate resampling

Downstream schemas are unchanged: sim outputs still include the original `play_prob` column, and do not persist the new policy columns.

## Rotation lock heuristic (v1)

`rotation_lock=True` if any:

- starter flag is true (`starter_flag` / `is_*_starter` / `is_starter` if present), OR
- conditional minutes p50 >= `rotation_lock_min_cond_p50` (default 18), OR
- within top-K by conditional minutes p50 on team (default K=8)

Conditional minutes p50 is resolved from the first available column in:
`cond_minutes_p50`, `minutes_p50_cond`, `baseline_minutes_p50`, `minutes_p50`, `minutes_mean`.

## Policy rules (v1)

In priority order:

1) OUT/INACTIVE/SUSPENDED => `play_prob_eff = 0.0`
2) rotation_lock AND "not listed" (healthy) => `play_prob_eff = max(play_prob_raw, rotation_lock_floor)`
3) PROBABLE => `play_prob_eff = max(play_prob_raw, probable_floor)`
4) Else => `play_prob_eff = play_prob_raw`

Notes:
- `rotation_lock_floor` defaults to `0.995` (intentionally not `1.0`).
- "Not listed" is represented in the minutes input as `status_bucket="healthy"` (typically raw status values `Ava` / `AVAIL`).
- Freshness gating is supported in config but is off by default (best-effort; requires timestamp columns).

## Config

Configured per profile in `config/sim_v2_profiles.json` under `play_prob_policy`.

Defaults (when the block is absent) are conservative: policy is disabled.

The production `sim_v3` profile enables the policy and sets `rotation_lock_min_cond_p50=8.0` to reduce feasibility resampling on typical slates.

Fields:

- `enabled` (bool, default false)
- `rotation_lock_floor` (float, default 0.995)
- `rotation_lock_min_cond_p50` (float, default 18.0)
- `rotation_lock_topk` (int, default 8)
- `probable_floor` (float, default 0.90)
- `require_fresh_injury_snapshot` (bool, default false)
- `freshness_minutes` (float, default 90.0)

## Reproducing diagnostics

Use the world sparsity diagnostic (defaults to the `sim_v3` profile):

```bash
uv run python -m scripts.diagnostics.world_sparsity_stats --date 2026-01-29 --n-worlds 1000 --profile sim_v3
```

For per-player audits:

```bash
uv run python -m scripts.diagnostics.world_sparsity_stats --date 2026-01-29 --n-worlds 1000 --profile sim_v3 --player-out /tmp/play_prob_policy_players.csv
```
