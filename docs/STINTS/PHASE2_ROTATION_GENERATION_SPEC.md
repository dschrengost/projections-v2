# Phase 2 Spec: Rotation-Aware Minutes Generation (Base + Pluggable Generators)

Status: **Spec** (post-Phase 1 merge)  
Owner: Daniel  
Repo: `projections-v2`  
Data bundle: `projections-data/artifacts/pbp_v1/LATEST_PUBLISHED`

---

## Purpose

Replace “minutes regression as truth” with **rotation-aware generation** driven by the Phase 1 stint truth layer, while keeping
**downstream contracts stable**.

Phase 2 delivers:

- A canonical **rotation event stream** dataset derived from `stints` / `player_stints`
- A pluggable `RotationGenerator` interface
- An MVP `TemplateRotationGenerator` (stint-library sampler + regime selection)
- Optional `MembershipModel` (active / rotation membership probabilities) that can replace or blend with existing minutes priors
- Minutes **worlds/samples** (and derived moments) that can feed existing sim/optimizer components without breaking them

This base must support **natural upgrades** to:
- hazard-based substitution models
- learned sequence models (transformer)
- learned minutes/stint generators

---

## Non-Goals (Phase 2)

- Possession-level simulation
- Modeling fouls explicitly (only coarse proxies allowed)
- Replacing rates/FPTS heads
- Changing optimizer interface/CSV outputs
- Shipping a transformer in Phase 2.0

---

## Downstream Contract

### Must remain compatible with existing consumers
The rotation-aware minutes pipeline must emit the same *shape* of minutes outputs currently used downstream, e.g.:

- `minutes_mean`, `minutes_p10/p50/p90` (or equivalent)
- `play_prob` / `sim_p_active` (if your sim uses availability)
- optional: `starter_prob` / `starter_flag`

**No optimizer changes** required as long as these fields are provided in the projections table.

---

## High-Level Architecture

```
PBP Stints Bundle (Phase 1)
  ├─ pbp_events.parquet
  ├─ stints.parquet
  └─ player_stints.parquet
         ↓
Rotation Dataset Builder (Phase 2)
  ├─ rotation_events.parquet   (canonical event stream per team-game)
  ├─ rotation_labels.parquet   (membership labels, next-sub targets, regime labels)
  └─ rotation_templates.parquet (optional cached templates/features)
         ↓
RotationGenerator (pluggable)
  ├─ TemplateRotationGenerator (Phase 2.0 MVP)
  ├─ HazardRotationGenerator   (Phase 2.1+)
  └─ TransformerRotationGenerator (Phase 2.2+)
         ↓
Minutes Worlds + Summary
  ├─ minutes_worlds.npy/parquet (optional)
  └─ minutes_summary.parquet    (mean/p10/p50/p90 + play_prob)
```

---

## Core Concepts

### 1) Rotation Event Stream
A team-game is represented as a time-ordered sequence of **lineup states** with durations:

- `lineup_5` (player_ids)
- `t_start`, `t_end` in game-seconds (or period+clock encoding)
- `duration_sec = t_end - t_start`
- optional context at boundary (score diff bucket, etc.)

This is the canonical representation used by both the sampler and learned models.

### 2) Regime
A coarse latent that controls rotation depth / substitution intensity:

- `tight` (≈8-man)
- `normal` (≈9–10)
- `deep` (≈11–12)

MVP can label regimes rule-based from historical “played>=5” counts and/or minutes concentration.

### 3) Membership
Separate from exact minutes:
- `played_ge_1` (availability / appeared)
- `played_ge_5` (in-rotation)
- `starter`

These are derived from Phase 1 stints and are targets for membership modeling.

---

## Data Products (Silver/Gold)

All tables are **internal canonical**, not vendor-shaped.

### A) `rotation_events.parquet`
Grain: **team_id + game_id + segment_idx** (one row per continuous 5-man state for a team)

Required columns:
- `season_id`
- `game_id`
- `team_id`
- `opponent_team_id`
- `is_home` (bool)
- `segment_idx` (int, monotonic per team-game)
- `period` (int)
- `start_clock_sec` (int)  # remaining in period
- `end_clock_sec` (int)
- `duration_sec` (int)     # >= 0, can be 0
- `lineup_p1`..`lineup_p5` (internal player_id)
- `score_diff` (home_score - away_score at segment start, optional)
- `score_diff_bucket` (optional)
- `game_seconds_elapsed` (optional convenience)
- `raw_ref` (optional: stint_id range for traceability)

Acceptance:
- `duration_sec` sums to regulation+OT seconds for team-game
- exactly 5 players per segment
- deterministic ordering: period asc, clock desc, segment_idx asc

### B) `rotation_labels.parquet`
Grain: **team_id + game_id + player_id**

Columns:
- `played_ge_1` (bool)
- `played_ge_5` (bool)
- `minutes_actual` (float)
- `starter_actual` (bool)
- `regime_label` (categorical: tight/normal/deep) (team-game level duplicated OK)
- optional: role rank features from stints (minutes rank, start rate, etc.)

Acceptance:
- derived purely from stints/player_stints
- stable across runs for same pbp bundle

### C) `rotation_templates.parquet` (optional cache)
Grain: **template_id** (team-game template, or clustered template)

Contains:
- `template_id`
- `team_id`
- `season_id`
- `regime_label`
- compressed representation of the event stream (e.g., durations vector + lineup slot roles)
- template metadata (pace proxy, score diff distribution, etc.)

MVP can skip this and build templates on the fly; cache is a performance optimization.

---

## Interfaces

### `RotationGenerator`
Location: `projections/rotations/generator.py` (new module)

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

@dataclass
class TeamContext:
    season_id: str
    game_id: str
    team_id: int
    opponent_team_id: int
    is_home: bool
    vegas_spread: Optional[float] = None
    vegas_total: Optional[float] = None
    # candidate players available today
    candidate_player_ids: Optional[List[int]] = None
    starter_candidates: Optional[List[int]] = None
    # optional priors
    minutes_prior: Optional[Dict[int, float]] = None  # player_id -> mean minutes
    play_prob_prior: Optional[Dict[int, float]] = None
    # knobs
    n_worlds: int = 5000
    rng_seed: int = 0

@dataclass
class RotationWorlds:
    # minutes samples: player_id -> array[n_worlds]
    minutes_by_player: Dict[int, "np.ndarray"]
    # optional: starter indicator per world
    starter_by_player: Optional[Dict[int, "np.ndarray"]] = None
    # diagnostics
    diagnostics: Optional[Dict] = None

class RotationGenerator(Protocol):
    def generate(self, ctx: TeamContext) -> RotationWorlds: ...
```

**Contract:**
- Must output minutes samples consistent with 240+OT team minutes per world
- Must respect candidate set constraints (no minutes for players not in candidate set unless explicitly allowed)
- Must be deterministic given `rng_seed` and fixed inputs

### `MembershipModel` (optional)
Location: `projections/rotations/membership.py`

Outputs:
- `p_played_ge_1`, `p_played_ge_5`, `p_starter`

MVP can implement a rules-based baseline; later can be learned (LightGBM/logistic).

---

## Phase 2.0 MVP: TemplateRotationGenerator

### Objective
Generate realistic minutes distributions by sampling from historical rotation templates while respecting today’s roster/membership.

### Steps

1) **Template selection**
   - Choose regime label (tight/normal/deep)
     - MVP: rule-based using spread (optional) and number of likely rotation players
   - Select a historical team-game template from the same team (or similar team) and same regime
     - Backoff hierarchy:
       1. same team, same season
       2. same team, recent seasons
       3. league-wide templates matching regime + “rotation depth”
   - Optional: nearest-neighbor on context features (spread bucket, total bucket)

2) **Role mapping**
   Map template players → today’s candidate players using a role ordering:
   - starter group mapped to top starter candidates
   - bench roles mapped by minutes_prior rank or membership scores
   - enforce uniqueness and 5-man lineup validity

   If mapping fails (not enough candidates), backoff:
   - widen candidate set (allow “possible” players)
   - choose another template
   - ultimately fall back to minutes prior (rare)

3) **World generation**
   For each world:
   - sample a template (or sample small perturbations of durations)
   - apply mapping to produce lineup segments
   - compute minutes per player = sum durations across segments / 60

   Optional small noise:
   - jitter segment durations with conservation of total seconds
   - jitter regime selection per world

4) **Outputs**
   - per-player minutes array
   - implied `play_prob` = P(minutes >= 1)
   - implied `rotation_prob` = P(minutes >= 5)
   - optional `starter_prob` = P(starter)

### Diagnostics to emit
- mapping_success_rate
- template_source breakdown (same team vs fallback)
- per-world team minutes check
- minutes distribution summaries (p0/p5/p50/p95)

---

## Natural Upgrade Path

### Phase 2.1 HazardRotationGenerator (learned substitutions)
Uses the same `rotation_events` dataset to learn:
- hazard for next substitution time given state
- transition model for who subs out/in (set-valued action)

Implementation plugs into `RotationGenerator.generate()` and emits the same `RotationWorlds`.

### Phase 2.2 TransformerRotationGenerator
Uses the same event stream but models the rotation sequence as tokens (lineup states / sub actions / durations).
Constraints enforced during decoding:
- exactly 5 on floor
- substitution size constraints
- candidate set gating
- minutes conservation

Because we keep the same interface and datasets, this is an **implementation swap**, not a rewrite.

---

## Evaluation Plan (must exist before “shipping” into live)

### Offline backtest targets
Using historical games (where you know truth from stints):

1) **Rotation membership accuracy**
- Recall@k for played_ge_5 (rotation players) within candidate set
- False positives for played_ge_1=0

2) **Minutes distribution quality**
- calibration of P(minutes>=5) and P(minutes>=1)
- tail realism: frequency of 0–5 minute outcomes for fringe players
- per-team minutes concentration stats vs truth

3) **Failure modes**
- late scratch simulations (remove a starter, see if generator adapts)
- blowout-ish contexts (spread buckets)

### DFS-oriented metrics (later)
- lineup ROI uplift is noisy; start with minutes integrity proxies.

---

## CLI / Plumbing

### New CLIs (Phase 2)
- `projections/cli/build_rotation_dataset.py`
  - input: pbp bundle path (LATEST_PUBLISHED default)
  - output: `rotation_events.parquet`, `rotation_labels.parquet` into a new artifacts namespace
- `projections/cli/generate_minutes_from_rotations.py`
  - input: slate date / teams + context
  - output: minutes samples + summary parquet

### Artifacts layout
```
/home/daniel/projections-data/artifacts/rot_v1/<run_id>/
  rotation_events.parquet
  rotation_labels.parquet
  templates.parquet (optional)
  manifest.json
```

---

## Milestones

### M2.0 — Rotation dataset builder
- Build `rotation_events` and `rotation_labels` from pbp bundle
- QA: sums, ordering, 5-man invariants
- Publish `rot_v1` bundle

### M2.1 — TemplateRotationGenerator MVP
- Generate minutes worlds for a team-game using templates
- Determinism + conservation checks
- Emit summary minutes moments + play_prob

### M2.2 — Integrate as optional minutes source
- Add config flag: `minutes_source = regression|rotation_sampler|blend`
- Blend mode: weighted mixture of minutes prior + rotation sampler (optional)

### M2.3 — Evaluation harness
- Backtest on a holdout month of 2024–25
- Report membership/tail metrics

---

## Risks & Guardrails

### Risks
- Candidate set wrong on chaotic injury nights
- Role mapping can produce unrealistic lineups if priors are poor
- Template coverage gaps for novel rosters

### Guardrails
- Hard constraints: 5 on floor, no duplicate players, team minutes conserved
- Mapping diagnostics; fall back rather than emit nonsense
- Blend mode as safety valve early on

---

## Acceptance Criteria (Phase 2.0)

- Can build `rot_v1` dataset deterministically from `pbp_v1` bundle
- Can generate minutes worlds for any team-game with >99% mapping success on historical data
- Produces materially more realistic tails:
  - higher frequency of 0–5 minute outcomes for fringe players vs regression minutes
- Downstream consumers can use outputs without code changes (same minutes fields)

---

## Appendix: Definitions

- **Played**: minutes >= 1.0  
- **In rotation**: minutes >= 5.0  
- **Stint**: continuous on-court 5-man state interval; can be zero-duration; ordered deterministically by
  period asc, clock desc, tie-break by play_id.
