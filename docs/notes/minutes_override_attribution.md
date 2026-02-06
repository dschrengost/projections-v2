# Minutes Override Attribution (m0 → m1 → m2 → m3)

This note documents (with code evidence) how **ops overrides** propagate through:

- `minutes.parquet` (baseline scorer output) → **m0**
- apply ops overrides with **reconcile OFF** → **m1**
- apply ops overrides with **reconcile ON** (effective layer) → **m2**
- sim worlds minutes allocation (post-mask project to 240; mean over worlds) → **m3**

and why operators can observe “wrong players get the vacated minutes”.

---

## The override system(s) in play

This note is about **ops overrides** (GameView “Manual Overrides”), persisted under:

- `artifacts/ops/overrides_v1/game_date=YYYY-MM-DD/overrides.json`: `projections/ops/overrides.py:90-97`.

Ops overrides include:
- `minutes_target` (absolute minutes) + `minutes_lock` (hard lock): enforced in effective layer + sim allocator (Stage 1A)
- `minutes_delta` (additive adjustment): `projections/api/ops_api.py:90-110`
- `ops_depth_role` (including `"out"`), plus `status`, `play_prob`, and minutes quantiles: `projections/ops/overrides.py:42-55`.

`minutes_delta` remains supported for backward compatibility, but is treated as **sugar** (delta → target+lock) when `minutes_target` is absent.

There is also a separate “Optimizer My Proj” override system (`projections/api/user_overrides.py`), but it does **not** drive sim minutes allocation; it affects optimizer pool values only.

---

## End-to-end flow diagram (live pipeline wiring)

The canonical live flow runs:

- `projections.cli.score_minutes_v1` → `minutes.parquet`: `prefect_flows/live_nba_pipeline.py:1131-1145` and `prefect_flows/live_nba_pipeline.py:490-509`
- `effective_inputs.write_effective_minutes_layer` → `effective_minutes.parquet`: `prefect_flows/live_nba_pipeline.py:1147-1154` and `projections/pipeline/effective_inputs.py:130-154`
- `scripts.sim_v2.run_sim_live` → sim worlds + projections: `prefect_flows/live_nba_pipeline.py:877-907` and `scripts/sim_v2/run_sim_live.py:68-93`

```mermaid
flowchart LR
  A[minutes.parquet\nbaseline scorer] -->|m0| B[ops override apply\nreconcile OFF]
  B -->|m1| C[ops override apply\nreconcile ON\n(effective layer)]
  C -->|m2| D[sim worlds\navailability mask\nteam-240 allocation]
  D -->|m3 mean| E[sim projections.parquet\n(+ optional minutes_matrix.parquet)]
```

---

## m0: baseline minutes (minutes.parquet)

Baseline minutes are produced by the scorer and written to:

- `data_root/artifacts/minutes_v1/daily/<date>/run=<run_id>/minutes.parquet`: `prefect_flows/live_nba_pipeline.py:490-509`.

Key minutes columns used downstream:
- `minutes_p50_cond` and `minutes_p50` (center), plus tails: `projections/cli/score_minutes_v1.py:1696`+.

---

## m1: ops overrides applied, reconcile OFF

Ops override application happens in:
- `projections/ops/overrides.apply_overrides_to_minutes_df`: `projections/ops/overrides.py:697-941`.

### minutes_delta

`minutes_delta` is applied as an **additive shift** to minutes quantiles and clipped to `[0, 48]`:
- `projections/ops/overrides.py:858-876`.

Delta’d players are treated as **locked** for the team reconciliation step:
- `projections/ops/overrides.py:877-881`.

### OUT handling

If ops sets `ops_depth_role="out"`, it is normalized into `status="out"`:
- `projections/ops/overrides.py:887-895`.

If `status="out"`, minutes columns are forced to `0` and `play_prob` to `0`:
- `projections/ops/overrides.py:896-909`.

With reconcile OFF, team totals may deviate from 240 (by design).

---

## m2: ops overrides applied, reconcile ON (effective layer)

The live pipeline writes an “effective inputs” layer:

- `effective_inputs.write_effective_minutes_layer` calls `apply_overrides_to_minutes_df(..., force_reconcile=True)`: `projections/pipeline/effective_inputs.py:42-60`.

Reconciliation to team=240 is invoked inside override application:
- `projections/ops/overrides.py:914-929`.

The underlying reconciliation algorithm is deterministic and tier-based:
- `projections/minutes/reconcile.py:175-375`.

Important tier semantics (affect *who absorbs* minutes):
- OUT includes `status=="out"` or `play_prob<=0`: `projections/minutes/reconcile.py:85-90`.
- Rotation vs cameo uses a minutes threshold: `projections/minutes/reconcile.py:91-98`.
- **Unknown defaults to rotation tier**, which tends to distribute minutes broadly: `projections/minutes/reconcile.py:101-110`.

Distribution order:
- Removing minutes: cameo first, then rotation: `projections/minutes/reconcile.py:328-337`.
- Adding minutes: rotation first, then cameo: `projections/minutes/reconcile.py:318-327`.

Lock relaxation (surprise mode):
- If locked minutes alone exceed 240, the reconcile step allows locked rows to be reduced anyway: `projections/minutes/reconcile.py:288-303`.

This is the most common reason operators observe “wrong players got the minutes”: reconciliation *must* offset a delta to preserve team totals, and the offset is computed by these tier rules unless explicitly routed/locked.

---

## m3: sim mean minutes after availability/masking + team-240 allocation

Sim loads minutes projections with a preference for `effective_minutes.parquet` when present:
- `scripts/sim_v2/generate_worlds_fpts_v2.py:1059-1197`.

Randomness + redistribution sources in sim:

1) Availability sampling from play probabilities:
- `active_mask = u < play_prob_eff`: `scripts/sim_v2/generate_worlds_fpts_v2.py:2551-2561`.

2) Minutes sampling (one of multiple backends), then masked by availability:
- masking defense-in-depth: `scripts/sim_v2/generate_worlds_fpts_v2.py:2792-2796`.

3) Optional “bench-zero mixture” drops fringe players to 0 and redistributes minutes:
- `scripts/sim_v2/generate_worlds_fpts_v2.py:2810-2833`.

4) Post-mask per-(team, world) reproject to **exactly 240** using a weighted, bounded allocator:
- call site: `scripts/sim_v2/generate_worlds_fpts_v2.py:2852-2876`
- allocator objective + waterfilling details: `projections/sim_v2/minutes_allocator.py:1-17`
- priority → weight transform: `projections/sim_v2/minutes_allocator.py:37-109`

### Where to get m3

By default, sim writes per-player mean summaries to:
- `artifacts/sim_v2/worlds_fpts_v2/game_date=.../run=.../projections.parquet` with `minutes_sim_mean[_uncond]`: `scripts/sim_v2/generate_worlds_fpts_v2.py:3625-3648`.

Optionally (behind an env var), it also writes the full minutes worlds matrix:
- `PROJECTIONS_SIM_WRITE_MINUTES_MATRIX=1` writes `minutes_matrix.parquet`: `scripts/sim_v2/generate_worlds_fpts_v2.py:3925-3936`.

When `minutes_matrix.parquet` is present, `m3` can be computed as `mean(minutes_matrix, axis=0)` and should closely match `minutes_sim_mean_uncond`.

---

## Why “wrong players get minutes” can happen (summary)

Concrete mechanisms (each can be present simultaneously):

1) **Effective-layer team reconcile** redistributes residual minutes by tier heuristics (cameo/rotation ordering), not by an operator-specified route.  
   Evidence: `projections/minutes/reconcile.py:318-337`.

2) **Locks can be relaxed when infeasible** (locked minutes exceed 240), causing even delta’d players to be reduced.  
   Evidence: `projections/minutes/reconcile.py:288-303`.

3) **Sim worlds reallocate minutes after masking** (availability draws + bench-zero), using a priority-weighted QP, so “vacated minutes” can flow to different players than the effective-layer reconcile did.  
   Evidence: masking + allocator call: `scripts/sim_v2/generate_worlds_fpts_v2.py:2551-2566` and `scripts/sim_v2/generate_worlds_fpts_v2.py:2852-2876`; allocator weights: `projections/sim_v2/minutes_allocator.py:37-109`.

---

## Stage-1 predictability options (proposal; not implemented here)

### Option A: Hard constraints (minutes_lock / minutes_target)

Idea: make overrides behave like constraints inside both:
- effective-layer reconciliation, and
- sim allocator.

Proposed new fields (ops overrides):
- `minutes_target: float | None` (absolute minutes)
- `minutes_lock: bool` (treat as fixed)
- `minutes_min: float | None`, `minutes_max: float | None` (bounds)

Implementation hooks:
- effective layer: `projections/ops/overrides.py:914-929` (construct `locked_mask`, pass caps/locks to reconcile)
- sim allocator: `projections/sim_v2/minutes_allocator.allocate_team_minutes_matrix` (extend to support fixed minutes / per-player bounds)

### Option B: Routing weights (absorb_weight / route_targets)

Idea: explicitly route residual minutes rather than relying on tier heuristics.

Proposed new fields:
- Player-level: `absorb_weight: float | None`
- Team-level: `route_targets: {player_id: weight}` + `route_mode`

Implementation hooks:
- `projections/ops/overrides.apply_overrides_to_minutes_df` between delta application and reconcile call: `projections/ops/overrides.py:858-929`
- optionally feed the same weights into sim allocation priority: `projections/sim_v2/minutes_allocator.py:105-109`
