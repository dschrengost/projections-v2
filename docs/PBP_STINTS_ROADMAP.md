# Play-by-Play → Stints Project Roadmap

This document defines the **scope, phases, deliverables, and acceptance criteria** for introducing a
play-by-play–derived **stint / rotation truth layer** into the projections stack.

The goal is to improve **minutes integrity and rotation realism** without breaking downstream consumers.

---

## Project Goal

Establish a **trusted internal truth layer** for:
- Play-by-play events
- On-court stints (10-man lineup + time interval)
- Player participation by stint

Minutes, usage, and correlation will become **derived properties**, not primary predictions.

---

## Non-Goals (Phase 1)

- No new simulator architecture
- No optimizer changes
- No possession-level modeling
- No projection accuracy promises

Phase 1 is about **correctness, reproducibility, and trust**.

---

## Guiding Principles

1. **Canonicalize early** – downstream code never depends on vendor schemas.
2. **Stints are the primitive** – minutes are derived, not predicted.
3. **QA gates are mandatory** – failures block publishing.
4. **Historical ≠ Live** – separate ingestion modes, same canonical outputs.
5. **Downstream contracts stay stable** until explicitly revised.
6. **Identity is owned internally** – vendor names are never treated as canonical IDs.

---

## Data Sources

### Initial Scope
- Paid vendor play-by-play dataset
- **One historical season only** (recommended: most recent full season)

### Future Scope (Out of Phase 1)
- Live in-season feed
- NBA.com JSON as optional fallback / validation source

---

## Identity Resolution (REQUIRED)

The vendor dataset **does not provide stable player IDs**.  
Phase 1 therefore includes a mandatory **identity resolution layer**.

### Canonical Rule
- All internal tables use a stable, internal `player_id`
- Vendor-provided names are treated as *attributes*, never identifiers

### Canonical Dimension Tables

#### `players_dim`
```
player_id           (internal int or UUID)
canonical_name
first_name
last_name
suffix              (Jr, Sr, II, III, etc.)
normalized_name     (lowercased, accent-folded)
vendor_name
season_first_seen
season_last_seen
active_flag
notes
```

#### `teams_dim`
```
team_id
team_tricode
team_name
season_first_seen
season_last_seen
```

### Name Normalization Rules
- lowercase
- strip punctuation
- unicode accent folding (e.g. Dončić → Doncic)
- normalize suffixes (jr, sr, ii, iii)
- trim whitespace

### Mapping Strategy
1. Normalize vendor name
2. Attempt match to existing `players_dim`
3. If matched, reuse `player_id`
4. If unmatched:
   - log to `unmapped_players.parquet`
   - fail build OR explicitly allow audited auto-creation

### Acceptance Criteria
- No silent creation of player identities
- All canonical tables reference `player_id`
- Unmapped players are explicitly logged and reviewable

---

## Canonical Data Model

### 1. `pbp_events`
One row per recorded action.

Required fields:
- `game_id`
- `season`
- `event_idx` (monotonic per game)
- `period`
- `clock_sec` (seconds remaining in period)
- `team_id`, `team_tricode`
- `event_type` (normalized)
- `primary_player_id`
- Optional: `assist_id`, `block_id`, `steal_id`, `foul_drawn_id`
- `points`
- `shot_result`, `shot_distance`
- `x`, `y`
- `description`
- `raw` (optional JSON namespace for traceability)

---

### 2. `stints`
One row per **continuous on-court state**.

Fields:
- `game_id`
- `season`
- `stint_id` (monotonic per game)
- `period`
- `start_clock_sec`
- `end_clock_sec`
- `duration_sec`
- `home_lineup` (5 player IDs)
- `away_lineup` (5 player IDs)
- `start_event_idx`
- `end_event_idx`
- `end_reason` (substitution, end_period, etc.)

---

### 3. `player_stints`
One row per player per stint (10 rows per stint).

Fields:
- `game_id`
- `stint_id`
- `team_id`
- `player_id`
- `seconds_played`
- Optional aggregated stats:
  - `fga`, `fgm`, `3pa`, `3pm`
  - `fta`, `ftm`
  - `ast`, `tov`
  - `reb`, `blk`, `stl`
  - `pf`, `pts`

---

## Phase 1 — Data Foundation

### Objective
Produce **trusted stint tables** with full QA coverage.

---

### Phase 1.1 — Ingestion + Identity Resolution

**Deliverables**
- Vendor dataset loader
- Deterministic player and team identity mapping
- Normalized `pbp_events` parquet

**Acceptance Criteria**
- All players mapped to stable `player_id`
- Unmapped names logged and audited
- Raw vendor rows preserved or hash-logged

---

### Phase 1.2 — Stint Construction

**Logic**
- Events sorted deterministically
- On-court 10 identified per event
- Consecutive identical on-court states collapsed into stints

**Deliverables**
- `stints.parquet`
- `player_stints.parquet`

**Acceptance Criteria**
- Exactly 5 players per team per stint
- Stints have non-negative duration
- Stint boundaries reproducible run-to-run

---

### Phase 1.3 — QA Gates

#### Per-Game Checks
- Team seconds ≈ `5 * game_seconds` (± tolerance)
- No overlapping stints per player
- No phantom players
- Clock monotonicity within periods

#### Season-Level Checks
- % games passing QA
- Distribution of stint durations
- Top failure reasons logged

**Deliverables**
- `qa_report.json`
- `qa_failures.parquet`
- Summary markdown report

**Acceptance Criteria**
- ≥ 99% games pass hard QA
- All failures are explainable and categorized

---

### Phase 1.4 — Publishing

**Published Bundle**
- `pbp_events.parquet`
- `stints.parquet`
- `player_stints.parquet`
- `qa_report.json`
- `manifest.json`

**Manifest Includes**
- input file hashes
- git SHA
- schema version
- season ID
- run timestamp

**Acceptance Criteria**
- Bundle is immutable and self-describing
- Re-running produces identical outputs

---

## Phase 2 — Rotation-Derived Minutes (Preview)

### Objective
Replace direct minutes regression with **rotation-aware minutes generation**.

Minutes distributions are derived by sampling **plausible rotation paths**
from historical stint behavior.

Downstream output contracts remain unchanged.

---

## Downstream Impact Summary

| Phase | Optimizer | Sim | Dashboards |
|----|----|----|----|
| Phase 1 | No change | No change | Additive only |
| Phase 2 | No change | Minor internal | Optional metadata |
| Phase 3+ | Possible | Possible | Possible |

---

## Definition of Success

You can confidently say:

> “This stint table is a better representation of who played than our current minutes model.”

At that point, the project has already paid for itself.
