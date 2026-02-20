# Props Minutes Anchor (Draft)

## Overview
We want to use player props as a market-derived signal to **anchor minutes predictions** without
overriding the core minutes model or play-prob logic. The anchor should be inference-only,
anti-leak safe, and compatible with sim_v2.

This draft proposes a small, auditable layer that:
- Computes **implied minutes** from props.
- Blends them into minutes medians with **confidence-weighted** rules.
- Leaves `play_prob` unchanged (props are weak availability signals).
- Exposes diagnostics and is easy to disable or A/B in sim.

## Goals
- Use props to correct obvious minutes mispricing (starters w/ stale projections).
- Preserve existing minutes model and overrides behavior.
- Maintain stable team-minute totals.
- Keep sim_v2 behavior predictable and debuggable.

## Non-Goals
- Not a new minutes model head.
- Not a play-prob replacement.
- Not a training-time dependency in v1.

## Data Inputs
- **Props source**: Action Network JSON in `bronze/` already parsed by
  `projections/features/action_props.py`.
- **Per-minute rates**: rates_v1 outputs (or fallback to recent per-min rolling stats).
- **Minutes model**: minutes_v1 outputs (baseline p50, tails, play_prob).

## Prop Normalization
Use the existing Action props features:
- `an_<stat>_line`, `an_<stat>_p_over`, `an_<stat>_line_std`, `an_<stat>_books`
- `an_implied_activity_line`, `an_props_market_count`, `an_props_books_total`

For each prop market:
1. Compute a **no-vig median/mean proxy** from `line` and `p_over`.
2. Reject markets with insufficient books or high dispersion.

## Implied Minutes (Per Player)
Map props to per-minute rates:

```text
implied_minutes_stat = implied_stat / max(rate_per_min, min_rate)
```

Supported markets (initial pass):
- `pts`, `reb`, `ast`, `pra`, `pr`, `pa`, `ra`

Implied minutes aggregation:
- Use weighted median or trimmed mean across markets.
- Weight by `books`, inverse dispersion, and rates stability.

Guardrails:
- Clamp to `[min_minutes, max_minutes]` (e.g. [2, 44]).
- Skip if implied minutes differs wildly from base (e.g. > 18 minutes delta).
- Skip if rates are missing or too small (`rate_per_min < min_rate`).

## Anchor Blend
We want a **soft anchor**, not a hard override:

```text
minutes_p50_anchor = w * implied_minutes + (1 - w) * minutes_p50_cond
```

Where `w` is derived from market quality:
- `w = f(books, dispersion, market_count, props_asof_age, rates_conf)`
- Typical range: 0.0 to 0.4

Tails update:
- Shift p10/p90 by the same delta as p50
- Or scale tails by the ratio `minutes_p50_anchor / minutes_p50_cond`
- Clamp tails to valid bounds and re-order if needed

Team totals:
- Optional post-blend reconcile using existing L2 reconcile helpers
- Default: only reconcile if total minutes drift > threshold

## Where It Runs
Preferred: **effective inputs layer**, after ops overrides **and after depth-chart prior**:

1. minutes_v1 scoring produces baseline parquet
2. `apply_overrides_to_minutes_df` (keeps `*_model` columns)
3. depth-chart prior (existing)
4. **props anchor apply** (does not touch `*_model`)
5. write `effective_minutes.parquet`

Notes:
- If `status == OUT` or `play_prob <= 0`, skip anchor (Vegas minutes should not resurrect OUTs).
- If `ops_override_applied` or `minutes_lock_eff`, skip anchor.
- `*_model` columns remain the pre-anchor baseline.

## Sim_v2 Interaction
Sim uses the minutes columns in this order:
- `minutes_final` → `minutes_p50_cond` → `minutes_p50`

Anchoring should **update minutes medians** that sim consumes while **keeping
`play_prob` unchanged**. This means:
- Availability sampling remains governed by the existing play-prob policy.
- Anchored minutes affect `minutes_mean` and therefore FPTS output.

Optional sim controls:
- Add a profile knob (sim_v2 config) to choose minutes source:
  `minutes_source = baseline | anchored`.
- Keep `minutes_p50_model` and `minutes_p50_anchor` for A/B.

## Data Contract (Effective Minutes Additions)
Add columns to `effective_minutes.parquet`:
- `minutes_p50_anchor` (float)
- `minutes_anchor_weight` (float)
- `minutes_anchor_source` (string)
- `minutes_anchor_applied` (bool)
- `minutes_anchor_reason` (string)
- `minutes_anchor_as_of_ts` (timestamp)
- `minutes_anchor_market_count` (int)
- `minutes_anchor_books_total` (float)
- `minutes_anchor_dispersion` (float)

Summary additions in `effective_inputs_summary.json`:
- `props_anchor`: counts, coverage, top deltas, reasons skipped

## Config
New config file: `config/props_anchor_minutes.json`

Suggested fields:
- `enabled`: bool
- `min_books`: int
- `min_markets`: int
- `max_line_std`: float
- `min_rate_per_min`: float
- `max_minutes_delta`: float
- `weight_max`: float
- `reconcile_team_minutes`: bool
- `reconcile_delta_threshold`: float
- `asof_max_age_minutes`: float
- `skip_if_out`: bool
- `skip_if_play_prob_le`: float

## Rollout Plan
1. Implement anchor + diagnostics, using **internal rates** (primary) and a toggle for alternative rates.
2. Retrain minutes + run backtests on historical slates:
   - Compare minutes errors (MAE/MSE) and sim outputs (FPTS error, tail calibration).
   - A/B internal rates vs alternative per‑min source for implied minutes.
3. Roll out with conservative weights, then increase if backtests remain stable.

## Open Questions
- Should the anchor use rates_v1 outputs (circular but consistent) or rolling
  per-min rates from boxscores (more independent)?
- Should we allow props to nudge `play_prob` for players with strong markets? (currently no)
- Do we want prop-specific distributions to infer tails instead of shifting?
