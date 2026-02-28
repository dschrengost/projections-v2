# Live Pipeline Production Spec

This document is the living production-readiness spec for the live inference
pipeline in `projections-v2`.

It is intended to play the same role for pipeline/system design that
`docs/joint_rotation_rates_v1/GAME_TRANSFORMER_SPEC.md` plays for model and
training design: one canonical document with explicit goals, boundaries,
contracts, rollout phases, and open questions.

This spec should be updated as architecture decisions are made or superseded.

## 1. Goals

1. Make live projections trustworthy near lock.
2. Guarantee reproducibility of every published run.
3. Reduce end-to-end latency so late news is reflected before it matters.
4. Support GPU-backed transformer inference without making the control plane
   fragile.
5. Define a production-standard MLOps path for promotion, rollback, and audit.
6. Improve storage resilience so a single disk failure is not a business-ending
   event.

## 2. Non-goals

1. This spec does not change model architecture directly.
2. This spec does not require rewriting the entire medallion layout into a new
   database system.
3. This spec does not assume a full cloud migration.
4. This spec does not require removing run-scoped artifacts or immutable
   histories.

## 3. Current Observed Failure Modes

### 3.1 Stale snapshot selection near lock

Recent example:

- On the 2026-02-27 slate, the last successful pre-lock features build still
  had James Harden as `Q` with `is_out=0`.
- The same build had Dennis Schroder correctly upgraded to confirmed starter.
- A later Rotowire lineup snapshot marked Harden `out`, but no later successful
  live rebuild consumed it.

Interpretation:

- We likely launched an expensive run from a stale-but-valid injury snapshot.
- Pipeline latency was long enough that input freshness degraded during the run.
- Recovery relied too heavily on the next scheduled run succeeding.

### 3.2 Full-slate reruns are too expensive

We currently pay the cost of reprocessing the whole slate too often. That:

- increases latency
- increases failure surface area
- makes late-news handling worse
- wastes compute when only one game changed

### 3.3 Source disagreement handling is under-specified

We do not yet have a clear production contract for disagreements between:

- official injury reports
- Rotowire lineups
- ESPN injuries
- props/odds implied activity signals

### 3.4 Operational resilience is weak

- live data is concentrated on a single SSD
- there is limited evidence of tested restore procedures
- the system is too dependent on one host staying healthy during lock windows

### 3.5 MLOps is insufficiently formalized

We have model artifacts and configs, but the promotion path needs harder
contracts around:

- candidate vs production state
- live shadow/canary policy
- rollback triggers
- latency budgets
- dataset/config provenance

## 4. Core Principles

### 4.1 Freeze first, score second

The system should freeze a run's inputs before expensive inference begins.
Scoring should consume a fixed input manifest, not "whatever is latest by the
time the scorer reaches a step."

### 4.2 Publish only from fresh-enough inputs

A technically successful run that used stale inputs is not a production-quality
run. Freshness must be a publish gate, not just metadata.

### 4.3 Prefer immutable inputs and run-scoped artifacts

The system must preserve the ability to answer:

- What did we know at a given `as_of_ts`?
- Which exact inputs fed a published run?
- Can we replay that run later?

This is more important than reducing the number of files on disk.

### 4.4 Compute should be incremental

Late-news updates should trigger the minimum work necessary. Per-game and
per-slate digests should determine what gets rebuilt.

### 4.5 Every production path needs a fallback

If the preferred path is too slow or unavailable, the system should degrade to
a clearly defined conservative mode rather than publish stale or partial data.

## 5. Canonical Production Architecture

### 5.1 Control-plane stages

The production live path should be organized into these logical stages:

1. `scrape_inputs`
   - scrape or ingest injuries, lineups, odds, salaries, schedule, roster
2. `freeze_run_inputs`
   - produce immutable run manifest with source timestamps and hashes
3. `compute_change_set`
   - identify which games changed materially relative to last published run
4. `build_features`
   - build only affected game features
5. `score_models`
   - run primary model path or fallback path
6. `finalize_outputs`
   - generate unified published artifacts
7. `postflight_validate`
   - freshness, schema, key, and sanity checks
8. `publish_atomic`
   - update pointers only after all gates pass

### 5.2 Required run identity

Every run must have:

- `run_id`
- `as_of_ts`
- `game_date`
- source timestamps by dataset
- config hash
- model bundle pointer/hash
- git sha / runtime stamp

### 5.3 Publish contract

A run is publishable only if:

1. required sources exist
2. freshness gates pass
3. model/path-specific parity checks pass
4. output schemas and keys pass
5. atomic pointer promotion succeeds

## 6. Input Freshness Spec

### 6.1 Freshness is a hard gate

For live games, freshness thresholds must exist for:

- official injuries
- lineup source
- odds
- props, if used in core scoring
- roster/schedule snapshots

### 6.2 Lock-window policy

Near scheduled NBA report windows, the pipeline should not immediately launch
full inference from the newest currently visible snapshot if that snapshot is
older than the expected report boundary.

Instead, implement:

- a bounded wait window around scheduled report times
- re-check loops for injuries and lineups
- explicit fail/warn behavior if expected updates do not arrive

### 6.3 Freshness metadata

Published artifacts should expose, per game:

- injury snapshot ts used
- lineup snapshot ts used
- odds snapshot ts used
- props snapshot ts used
- freshness age in minutes at publish time
- whether fallback logic was used

### 6.4 Stale-input guard

If a newer authoritative input exists after the currently published run for a
live game, the system should either:

- trigger an automatic rerun for that game, or
- raise an alert that the published result is stale

## 7. Delta-Driven Execution Spec

### 7.1 Material-change detection

We should define a per-game digest and a material-change contract.

Material examples:

- player ruled out / upgraded active
- projected starter -> confirmed starter
- confirmed starter -> out
- meaningful line move
- meaningful prop move
- salary/slate eligibility change

Non-material examples:

- benign ingest timestamp change with identical content
- non-slate player updates
- minor book count noise with no consensus line change

### 7.2 Per-game rebuilds

The default live reaction to late news should be:

- detect affected game(s)
- rebuild only those features
- score only those game(s)
- republish unified outputs atomically

### 7.3 Full-slate rebuild policy

Full-slate rebuilds remain valid for:

- early-morning baseline builds
- large slate-wide upstream refreshes
- schema/config/model changes
- explicit operator requests

## 8. Source Precedence and Disagreement Policy

### 8.1 Proposed precedence

For core availability:

1. official NBA injury report
2. confirmed lineup `out` signals from trusted lineup source
3. ESPN injuries as supporting fallback
4. model priors and props only as secondary context, never as authoritative
   availability

### 8.2 Disagreement handling

When sources disagree for the same player in a live game:

- record the disagreement in run diagnostics
- surface it in operator-visible tooling
- apply a conservative policy for high-impact players
- allow source-specific overrides only through explicit documented rules

### 8.3 High-impact player policy

We should define a class of high-impact players based on projected minutes,
usage, or downstream lineup sensitivity. Disagreements involving these players
should have stronger alerting and stricter publish gates.

## 9. Data Architecture Decision

### 9.1 Recommendation

Do not move the live system to a pure "transform in place" model as the source
of truth.

For this project, immutable raw inputs plus run-scoped artifacts are more
important than minimizing copies, because live DFS operations require exact
`as_of_ts` reasoning and replayability.

### 9.2 DuckDB role

DuckDB is still a strong fit as a compute/query layer on top of Parquet.

Recommended uses:

- backfills
- feature inspection
- incident investigation
- training-set assembly
- per-game change detection
- operational summary tables

Not recommended as the canonical live-state pattern:

- destructive in-place updates of the only source-of-truth live tables

### 9.3 Storage pattern

Keep the current layered concept, but make the contracts stricter:

- raw inputs append-only
- frozen run manifests
- run-scoped feature artifacts
- run-scoped scored artifacts
- atomic published pointers
- small operational summary tables for fast inspection

## 10. Model Runtime Architecture

### 10.1 Primary and fallback scoring paths

For live production, define two scoring modes:

1. primary path
   - transformer-based path
   - GPU-backed when available
2. fallback path
   - faster conservative scorer
   - CPU-safe
   - lower quality but bounded latency

### 10.2 Latency budgets

Set explicit latency budgets per stage:

- scrape/ingest
- freeze
- feature build
- score
- finalize
- publish

The end-to-end budget for a late-news single-game update should be materially
smaller than the current full-slate path.

### 10.3 GPU integration requirements

When the NVIDIA GPU is added:

- benchmark CPU vs GPU by stage
- avoid cold-start cost near lock
- keep exact runtime/env manifests for CUDA and model serving
- make GPU failure non-fatal by preserving the fallback scorer

### 10.4 Warm process / serving model

We should strongly consider a warm scoring process for the transformer model so
late-news updates do not pay repeated model load overhead.

## 11. MLOps Spec

### 11.1 Model states

Every model bundle should be in one of these states:

- `experimental`
- `candidate`
- `shadow`
- `canary`
- `production`
- `rolled_back`

### 11.2 Promotion contract

A promotion record should include:

- training window
- dataset version / digest
- feature contract version
- config hash
- model artifact hash
- offline metrics
- live shadow/canary notes if applicable
- latency benchmark
- known limitations

### 11.3 Rollback contract

Rollback must be explicit and fast:

- revert model pointer
- revert config pointer if necessary
- preserve failed run artifacts
- capture incident bundle

### 11.4 Monitoring domains

Track separately:

1. model quality
   - calibration
   - realized error
   - late-news sensitivity
2. system quality
   - freshness
   - latency
   - successful publish rate
   - stale publish incidents

### 11.5 Retraining path

Automated retraining is desirable, but no retrained model should auto-promote
without explicit production checks. Retrain automation and promotion automation
should remain separate concerns.

## 12. Observability and Incident Response

### 12.1 Required telemetry

Every live run should emit:

- per-stage timings
- per-game input freshness
- source disagreement counts
- change-set size and affected games
- model path used
- publish outcome

### 12.2 Operator surfaces

We should expose a lightweight operational view showing:

- latest published run
- newest available inputs by game
- stale-game detection
- current model path
- warnings/blockers

### 12.3 Incident bundle

For any failed or suspicious run, capture:

- run manifest
- source timestamps
- config/model hashes
- failing gate payloads
- output paths
- relevant source disagreement rows

## 13. Storage and Disaster Recovery Spec

### 13.1 Current risk

`/home/daniel/projections-data` living on a single SSD is below production
standard for this project.

### 13.2 Minimum required protections

1. second copy of critical data on separate physical media
2. automated scheduled backups
3. restore verification drills
4. separate backup coverage for:
   - promoted model bundles
   - configs and pointers
   - live artifacts
   - raw scrape inputs

### 13.3 Longer-term target

- local redundancy for hot live data
- off-machine backup for disaster recovery
- separation of hot live storage from archive/training storage

## 14. Production Readiness Gates

The pipeline should not be considered production-standard until all of the
following are true:

1. freshness gates exist and are enforced
2. stale published runs are detectable automatically
3. per-game rebuilds exist
4. primary/fallback scoring modes exist
5. latency budgets are measured and monitored
6. model promotion/rollback policy is documented and used
7. backup/restore is operational and tested

## 15. Implementation Plan

### 15.1 Phase 1: Control-plane hardening

1. Add lock-window freshness gates for injuries and lineups.
2. Add bounded wait policy around official report windows.
3. Stamp all runs with per-game source freshness metadata.
4. Add stale-publish detection against newest available inputs.

### 15.2 Phase 2: Incremental execution

1. Define per-game digest format.
2. Implement material-change detection.
3. Add per-game feature rebuild and scoring path.
4. Keep full-slate rebuild as fallback/operator mode.

### 15.3 Phase 3: Runtime architecture

1. Benchmark transformer latency on CPU.
2. Introduce GPU-backed primary scoring path.
3. Add warm-process inference path.
4. Add conservative CPU fallback scorer.

### 15.4 Phase 4: MLOps formalization

1. Introduce explicit model states and promotion records.
2. Formalize rollback triggers and incident capture.
3. Add model/system KPI monitoring split.
4. Separate retrain automation from promote automation.

### 15.5 Phase 5: Resilience and storage

1. Add redundant storage for critical artifacts.
2. Add scheduled backups and restore drills.
3. Separate hot live data from archive/training data as practical.

## 16. Task List

### A. Freshness and publish safety

- [ ] Add injury/lineup freshness gates to live publish path
- [ ] Add bounded wait loops around scheduled NBA report windows
- [ ] Add per-game source freshness fields to published metadata
- [ ] Add stale-publish detection and alerting

### B. Incremental pipeline execution

- [ ] Define per-game input digest contract
- [ ] Implement material-change detection
- [ ] Add per-game feature rebuild
- [ ] Add per-game scorer/finalizer path

### C. Runtime and inference

- [ ] Measure current end-to-end latency by stage
- [ ] Benchmark transformer CPU latency
- [ ] Integrate GPU-backed primary inference path
- [ ] Add warm-process inference
- [ ] Implement conservative fallback scorer

### D. Source quality and disagreements

- [ ] Define source precedence contract for availability
- [ ] Add disagreement diagnostics between official injuries, Rotowire, and ESPN
- [ ] Add stronger policy for high-impact-player disagreements

### E. MLOps

- [ ] Add model state taxonomy and promotion records
- [ ] Add explicit rollback playbook
- [ ] Add latency benchmarks to promotion requirements
- [ ] Define model KPIs vs system KPIs

### F. Storage and recovery

- [ ] Add second copy / backup target for `projections-data`
- [ ] Back up configs, pointers, and model bundles separately
- [ ] Document restore procedure
- [ ] Run restore drill and record result

## 17. Open Questions

1. Which source should be allowed to hard-zero a player pre-lock if the
   official injury feed is lagging but trusted lineup data marks the player
   out?
2. What is the acceptable late-news end-to-end SLA for a single-game rebuild?
3. Should the fallback scorer publish automatically if the transformer misses
   latency budget, or require operator approval?
4. Should we keep the current bronze/silver/gold layout exactly as-is, or add a
   DuckDB-powered operational layer for summaries and change detection?
5. How much of the live path should move to a long-running service versus
   remaining Prefect task-based?

## 18. Immediate Recommendations

If only a few items are tackled first, prioritize these:

1. freshness gates around injury and lineup report windows
2. stale-publish detection
3. per-game rebuilds
4. latency instrumentation
5. backup and restore for `projections-data`

