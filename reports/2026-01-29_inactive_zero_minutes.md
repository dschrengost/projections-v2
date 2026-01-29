# Inactive => Minutes == 0 invariant

## Goal
Confirm and enforce the invariant that inactive players receive exactly 0 minutes in sim_v2 minutes allocation.

## Evidence
- `scripts/sim_v2/generate_worlds_fpts_v2.py`: active_mask is sampled from play_prob and applied before allocation and after bench-zero; allocation uses active_mask to zero inactive players.
- `projections/sim_v2/minutes_allocator.py`: allocator zeroes inactive players and keeps them at 0 after allocation.
- `projections/sim_v2/minutes_stabilization.py`: reconciliation functions keep inactive minutes at 0.

## Reproduce
```bash
uv run python scripts/diagnostics/check_inactive_zero_minutes.py --date 2026-01-28 --profile baseline --n-worlds 200
```

This runs sim_v2 with dev assertions enabled and fails if any inactive player receives minutes.

## Latest run
- 2026-01-28, profile=baseline, n_worlds=200 → OK
