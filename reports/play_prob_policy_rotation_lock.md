# Play-Prob Policy for Rotation Locks (Report)

Date: 2026-01-29  
Worlds: N=1000  
Profile: `sim_v3`

## Goal

Reduce reliance on minutes feasibility resampling/promotions by applying a play-prob “policy layer” so rotation-lock players who are not on the injury report default to ~100% availability (0.995), while fringe/injured players keep their raw DNP risk.

## Repro

Policy disabled (baseline):

```bash
uv run python -m scripts.sim_v2.generate_worlds_fpts_v2 \
  --start-date 2026-01-29 --end-date 2026-01-29 \
  --n-worlds 1000 --profile sim_v3 \
  --profiles-path /tmp/sim_v2_profiles_sim_v3_nopolicy.json \
  --output-root /tmp/sim_v2_policy_run_before --run-id policy_before
```

Policy enabled (this branch defaults for `sim_v3`):

```bash
uv run python -m scripts.sim_v2.generate_worlds_fpts_v2 \
  --start-date 2026-01-29 --end-date 2026-01-29 \
  --n-worlds 1000 --profile sim_v3 \
  --output-root /tmp/sim_v2_policy_run_after --run-id policy_after
```

Optional: per-player audit + top offenders:

```bash
uv run python -m scripts.diagnostics.world_sparsity_stats \
  --date 2026-01-29 --n-worlds 1000 --profile sim_v3 \
  --hard-cap 41 --player-out /tmp/play_prob_policy_players.csv
```

## Results (minutes_physics)

From the `[sim-physics]` line emitted by `scripts/sim_v2/generate_worlds_fpts_v2.py`:

### Baseline (policy disabled)

- `frac_infeasible_pre_resample`: 0.8951
- `frac_promoted`: 0.4046
- `promoted_players_total`: 18207
- `cap` hit rate: ~0.0000 (no hard-cap bind in this run)

### Policy enabled

- `frac_infeasible_pre_resample`: 0.0446
- `frac_promoted`: 0.0000
- `promoted_players_total`: 0
- `cap` hit rate: ~0.0000

## Results (availability semantics)

Using `scripts.diagnostics.world_sparsity_stats` per-player audit (policy enabled):

- Policy reason counts: `rotation_lock_floor=150`, `probable_floor=2`, `out_like=44`, `raw=51`
- Rotation locks (excluding OUT / QUESTIONABLE):
  - `sim_p_active` mean=0.9926, p10=0.9930, p50=0.9970, p90=1.0000
  - `play_prob_raw` mean=0.6498
  - `play_prob_eff` mean=0.9898 (floored to 0.995 when applicable)
- Fringe (excluding OUT / QUESTIONABLE):
  - `sim_p_active` mean=0.3666
  - `play_prob_raw` mean=0.3626
  - Mean absolute deviation `|sim_p_active - play_prob_eff|` ≈ 0.0056 (p90 ≈ 0.0119)

## Notes / Limitations

- The `sim_v3` profile sets `rotation_lock_min_cond_p50=8.0` (in addition to top-K=8) to reduce feasibility resampling on typical slates.
- Injury “freshness” gating is implemented but disabled by default; it requires timestamp columns (e.g. `injury_as_of_ts`) in the sim input frame to be meaningful.
- Status semantics use `status_bucket` (derived from raw `status` like `OUT/Q/PROB/Ava`) to decide “not listed” vs injury-listed players.

