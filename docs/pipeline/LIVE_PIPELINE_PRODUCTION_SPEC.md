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

### 4.5 Every production path needs a safe failure mode

If the preferred path is too slow or unavailable, the system should degrade to
a clearly defined safe mode rather than publish stale or partial data.

For this project, "safe mode" does not mean publishing from an alternate
minutes model. It means bounded waits, game-scoped reruns, operator-visible
blocking states, and explicit non-publish behavior when the transformer path
cannot complete in time.

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
   - run transformer scoring path
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
- whether manual override logic was used

### 6.4 Stale-input guard

If a newer authoritative input exists after the currently published run for a
live game, the system should either:

- trigger an automatic rerun for that game, or
- raise an alert that the published result is stale

## 7. Delta-Driven Execution Spec

### 7.1 Material-change detection

Baseline production policy:

1. Compute a per-game digest from frozen `source_freshness` using source-local
   `content_digest` values when available.
2. Ignore pure timestamp churn when `content_digest` and `source_used` are
   unchanged.
3. Treat changes in `injuries`, `lineups`, and `roster` as material for pre-tip
   games.
4. Treat `odds` changes as material only within 180 minutes of tip.
5. Do not auto-trigger from `props` yet; persist the delta for diagnostics only
   until a cleaner contract is defined.
6. Ignore changes for games that are already at or past tip.
7. Fall back to a full-slate rerun when the model bundle, selector config, or
   slate composition changes.

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
2. confirmed lineup `out` signals from Rotowire
3. operator manual `OUT` overrides, when explicitly entered and audited
4. ESPN injuries only as an early-day planning/context source, not as a live
   authoritative source
5. model priors and props only as secondary context, never as authoritative
   availability

Policy note:

- Rotowire is trusted to zero a player pre-lock when it explicitly marks the
  player `out`, even if the official NBA report feed is lagging.
- ESPN should not be used to zero or activate players in the live pipeline.
  ESPN remains useful for early-day expectation-setting, longer-term injuries,
  and operator context.

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

### 10.1 Primary scoring path

For live production, define one canonical scoring mode:

1. transformer path
   - transformer-based path
   - CPU-backed today
   - GPU-backed once hardware is installed and validated

As of February 27, 2026, GPU inference is not yet available in production.
The near-term plan is to continue using CPU inference while preparing the
runtime, packaging, and observability needed for GPU cutover.

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

Initial target:

- single-game late-news rebuild target: under 2 minutes, subject to what the
  transformer path can reliably sustain in production

### 10.3 GPU integration requirements

When the NVIDIA GPU is added:

- benchmark CPU vs GPU by stage
- avoid cold-start cost near lock
- keep exact runtime/env manifests for CUDA and model serving
- fail closed if the GPU path is unavailable and CPU inference cannot satisfy
  the live SLA

### 10.4 Warm process / serving model

We should strongly consider a warm scoring process for the transformer model so
late-news updates do not pay repeated model load overhead.

### 10.5 No alternate-model fallback

We should not automatically publish from a second minutes model if the
transformer path misses SLA.

Production behavior should instead be:

1. prefer game-scoped reruns so the transformer only recomputes what changed
2. keep the transformer process warm
3. block publish or hold prior projections if the required transformer update
   cannot complete safely
4. expose the blocked state to the operator with clear freshness diagnostics

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
- whether lineup building is currently using the latest published run because a
  newer run is blocked or still in progress

### 12.2.1 Blocked/in-progress behavior

If a new run is blocked or still executing:

1. the latest published run remains available for lineup building
2. the operator must be notified that the next run is blocked, waiting, or in
   progress
3. the UI should clearly show the blocker reason at the game or slate level
4. the system should not present the blocked run as published data

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
4. transformer runtime behavior and safe failure modes exist
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
2. Implement material-change detection with an explicit baseline policy.
3. Add per-game feature rebuild and scoring path.
4. Merge partial reruns back into unified full-slate artifacts before publish.
5. Keep full-slate rebuild as operator mode.

### 15.3 Phase 3: Runtime architecture

1. Benchmark transformer latency on CPU.
2. Introduce GPU-backed primary scoring path.
3. Add warm-process inference path.
4. Add fail-closed handling and operator-visible blocked states instead of an
   alternate-model fallback.

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

- [x] Add injury/lineup freshness gates to live publish path
- [x] Add bounded wait loops around scheduled NBA report windows
- [x] Add per-game source freshness fields to published metadata
- [x] Add stale-publish detection and alerting

Status note:
- implemented in canonical v3; lock-window freshness is now diagnostic/advisory
  rather than publish-blocking, and continue real-slate validation for
  report-window waits

### B. Incremental pipeline execution

- [x] Define per-game input digest contract
- [x] Implement material-change detection
- [x] Add per-game feature rebuild
- [x] Add per-game scorer/finalizer path
- [x] Merge game-scoped reruns back into unified publish artifacts

### C. Runtime and inference

- [ ] Measure current end-to-end latency by stage
- [ ] Benchmark transformer CPU latency
- [ ] Integrate GPU-backed primary inference path
- [ ] Add warm-process inference
- [ ] Add fail-closed runtime handling when transformer SLA is missed

### D. Source quality and disagreements

- [x] Define source precedence contract for availability
- [ ] Add disagreement diagnostics between official injuries, Rotowire, and ESPN
- [ ] Add stronger policy for high-impact-player disagreements

Status note:
- baseline precedence is now defined in the spec and reflected in the live v3
  path, but disagreement diagnostics and escalation policy are still open

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

## 17. Decisions And Remaining Questions

### 17.1 Resolved decisions

1. Rotowire is allowed to hard-zero a player pre-lock when it explicitly marks
   the player `out`.
2. ESPN is not a live authoritative source for player availability. It is
   useful for early-day planning, long-term injuries, and operator context.
3. The target SLA for a single-game late-news rebuild is under 2 minutes, with
   final enforcement to be set after transformer benchmarking.
4. We will keep the current bronze/silver/gold layout. DuckDB remains a
   possible future addition for operational summaries, debugging, and change
   detection, but not as the core mutable live-state store.
5. We will not add an alternate-model fallback for live minutes scoring. If the
   transformer path cannot complete safely, the system should fail closed and
   surface the blockage.
6. When a run is blocked or still in progress, operators continue to build from
   the latest published run and are explicitly notified about the blocked or
   in-progress state.
7. Operators should be able to trigger single-game pipeline / inference runs.
8. Rotowire takes precedence for explicit `OUT` signals in the live pipeline.

### 17.2 Long-running service options

The live path does not need to move entirely out of Prefect. The practical
options are:

1. Keep the current Prefect-first architecture and optimize within it.
   - Best when orchestration, auditability, and simple operator control are the
     priority.
   - Add bounded waits, better freshness gates, game-scoped reruns, and a warm
     transformer subprocess.
2. Hybrid model: Prefect for orchestration, long-running local inference
   service for transformer scoring only.
   - Best if model load/warmup dominates late-news latency.
   - Prefect still owns scrape, freeze, build, finalize, and publish.
   - A local service owns loaded model weights, batching, and health checks.
3. Larger service-oriented cutover.
   - Move more of the live path into always-on services and use Prefect mainly
     for supervision.
   - Highest complexity and least justified near-term.

Recommendation:

- Use option 2 once the GPU arrives if transformer warmup/model load is a
  significant part of the critical path.
- Until then, keep Prefect as the control plane and focus on game-scoped work,
  freshness gates, and eliminating unnecessary backfills during lock windows.

### 17.3 Remaining open questions

1. Which stages should be allowed to skip or defer non-critical enrichments
   during the final lock window to preserve the late-news SLA?
2. Do we need a hard UX decision now on override expiry/clearing behavior
   beyond the current requirement for auditability and visibility?

## 18. Immediate Recommendations

If only a few items are tackled first, prioritize these:

1. freshness gates around injury and lineup report windows
2. stale-publish detection
3. per-game rebuilds
4. latency instrumentation
5. backup and restore for `projections-data`

## 19. Manual Override Spec

### 19.1 Recommendation

Do not support arbitrary operator minute boosts or direct edits to model output
in the live projection pipeline.

Those overrides tend to create incoherent downstream values, are easy to
overwrite accidentally, and make run replay harder to reason about.

Instead:

1. allow only audited manual `IN` / `OUT` status overrides in the live pipeline
2. keep any exposure, boost, or preference logic in the optimizer / contest sim
   layer, not in the canonical minutes projection layer

### 19.2 Supported override types

Allowed in live pipeline:

- manual `OUT`
- manual `IN` / clear-out, when explicitly entered by operator

Not allowed in live pipeline:

- direct minute boosts
- direct usage boosts
- direct fantasy-point overrides
- arbitrary feature edits

### 19.3 Why `IN` / `OUT` only

An `OUT` signal is qualitatively different from a minute boost. It changes the
availability state of the slate and should propagate through minutes,
downstream scoring, and optimizer eligibility in a coherent way.

A manual boost is usually an opinion layered on top of the model. That belongs
in downstream decision support, not in the source-of-truth live model output.

### 19.4 Override data model

Add an append-only manual overrides table, for example:

`$PROJECTIONS_DATA_ROOT/live/manual_overrides/game_date=<DATE>/manual_overrides.parquet`

This is the logical target contract. The short-term implementation may reuse
the existing ops/GameView override path documented in
`docs/pipeline/MANUAL_OVERRIDE_CONTRACT.md`.

Required fields:

- `override_id`
- `created_ts`
- `expires_ts` or `game_lock_ts`
- `game_date`
- `game_id`
- `player_id`
- `player_name`
- `team_id`
- `team_tricode`
- `override_type` with enum `force_out`, `force_in`
- `reason_code`
- `reason_text`
- `source_label`
- `entered_by`
- `active`
- `cleared_ts`
- `cleared_by`

### 19.5 Override application order

For live availability, apply in this order:

1. official NBA injury report
2. Rotowire `out` signal
3. active manual override layer

The manual layer should be explicit and operator-visible. It is not a hidden
mutation of raw source data.

Recommended rule:

- `force_out` can zero a player immediately, even before official/Rotowire
  confirmation
- `force_in` should be used sparingly and should clear only a prior manual
  `force_out` unless the operator explicitly confirms an override against source

## 20. Validation Notes

### 20.1 2026-02-28 non-publishing dry run

On February 28, 2026 at approximately 00:03 ET / 2026-02-28T05:03:37Z, we ran
the canonical v3 flow as a non-publishing dry run against the February 27, 2026
slate after all five games had already tipped.

Command shape:

- `game_date=2026-02-27`
- `placeholder_mode=false`
- `replay_mode=true`
- `promote_pointers=false`
- `sim_worlds=512`

Observed result:

- the flow completed end-to-end successfully
- run manifest freeze succeeded with populated `source_freshness`,
  `freshness_gates`, and `bounded_wait` fields
- preflight passed and carried the frozen freshness metadata forward
- scoring, worlds generation, finalization, and postflight all passed
- no pointers were promoted, as intended for the dry run

Important interpretation:

- because all February 27, 2026 games had already started, this run recorded
  `live_game_count=0`
- accordingly, `lock_window.checked_games=0` and the report-window wait logic
  was inactive
- this dry run validated the plumbing, manifest contract, and non-publishing
  control-plane behavior, but it did not validate the pre-lock freshness gate
  path on truly live games

Remaining validation needed:

- run the same canonical v3 flow on the next pre-lock slate and confirm:
  - lock-window freshness checks activate for games with `minutes_to_tip > 0`
  - bounded wait behavior triggers correctly around the 1 PM / 2:30 PM / 5 PM
    ET report windows when inputs are lagging
  - stale-publish blocking behaves correctly when newer injuries or lineups
    arrive after freeze and before publish

### 20.2 2026-02-28 Phase 2 baseline policy and execution status

As of February 28, 2026, the canonical v3 live flow implements the baseline
incremental execution policy described in Section 7:

- per-game digests are computed from frozen source freshness metadata
- timestamp-only refreshes do not trigger reruns when source content is
  unchanged
- `injuries`, `lineups`, and `roster` changes are material for pre-tip games
- `odds` changes are material only within 180 minutes of tip
- `props` changes are tracked in diagnostics but do not yet auto-trigger
  reruns
- live props now resolve from Rotowire only; Action Network is no longer part
  of the live critical path
- selector, bundle, and slate-composition changes force a full-slate rebuild

Execution behavior now supports:

- `skip` when no material pre-tip change exists
- `game_scoped` reruns for only the affected games
- merge-back of partial feature, score, worlds, and finalized projection
  artifacts into unified full-slate outputs before publish

### 20.3 Agent handoff (next implementation pass)

This section is the authoritative handoff for the next live-pipeline
implementation pass after the Phase 2 baseline landed on February 28, 2026.

What is already done:

- Phase 1 control-plane hardening is in the canonical v3 flow
- Phase 2 baseline digests, rerun policy, game-scoped execution, and merge-back
  are implemented
- LineStar ownership scoring runs on every canonical v3 execution and publishes
  run-scoped ownership artifacts, but it does not participate in delta
  detection or rerun policy
- focused tests for control-plane and incremental execution behavior are
  passing

What still needs to be done next:

1. Pre-lock live validation on a real slate.
   - run the canonical v3 flow before first tip with `promote_pointers=false`
   - confirm `rerun_plan` resolves sensibly as `skip`, `game_scoped`, or
     `full_slate`
   - inspect `input_change_set.json`, `preflight_report.json`,
     `postflight_report.json`, and, when applicable,
     `unified_artifacts_report.json`
   - do not treat the February 28, 2026 post-tip dry run as sufficient
     validation for lock-window behavior
2. Manual availability override integration in v3.
   - merge active `force_out` / `force_in` overrides into the live build path
     before final availability flags are computed
   - treat override-driven changes as material game deltas
   - stamp run artifacts with override diagnostics and
     `manual_override_used=true` when applicable
3. Operator-visible diagnostics.
   - surface `source_freshness`, `freshness_gates`, `input_change_set`, and
     `rerun_plan` in the operator-facing API / dashboard so blocked or skipped
     states are understandable without opening manifest JSON files
4. Materiality policy refinement after real-slate observation.
   - review whether the 180-minute odds threshold is too permissive or too
     conservative
   - decide whether `props` should become an auto-trigger source or remain
     diagnostic-only
   - keep timestamp-only churn suppression unless a concrete replay shows it is
     masking a real upstream change

Guardrails for the next pass:

- preserve full-slate publish artifacts even when only one game is rerun
- fail closed rather than publish partial or stale outputs
- keep the canonical implementation in
  `prefect_flows/live_nba_pipeline_v3.py` unless there is a documented reason
  to move shared logic into `projections/pipeline/`
- update this spec and targeted tests in the same change set as any policy
  change

### 19.6 Runtime behavior

When an active `force_out` exists:

- set availability status to `OUT`
- set `is_out=1`
- set play probability to `0`
- zero minutes distributions downstream
- mark published artifacts with `manual_override_used=true`
- include override metadata in run manifest and diagnostics

When an active `force_in` exists:

- clear a prior manual `force_out`
- rerun normal source precedence
- require explicit audit trail in diagnostics

### 19.7 UX / operator flow

Recommended operator flow:

1. operator enters `force_out` for player X with reason/source
2. pipeline detects manual override as a material change
3. affected game is rebuilt immediately
4. published output clearly shows "manual override active"
5. once official/Rotowire confirmation arrives, operator can clear the manual
   override or let it expire automatically at game lock

### 19.8 Overwrite protection

Manual overrides must not be silently lost on the next run.

Implementation rules:

1. store overrides outside run-scoped feature artifacts
2. merge active overrides into every live rebuild before scoring
3. preserve override records after run completion for audit
4. expire or clear overrides explicitly; never drop them implicitly because a
   new run started

### 19.9 Optimizer and contest sim policy

Keep "boost" style operator judgment in the optimizer / contest sim surface.

Examples:

- exposure caps
- exposure boosts
- preferred plays
- lineup group rules
- portfolio constraints

Recommended UX control:

- provide a single reset action for manual strategy inputs that are not manual
  `OUT` overrides
- do not let that reset silently clear active manual `OUT` states

These are user strategy controls, not canonical model-state edits.

### 19.10 Implementation plan

1. Add `manual_overrides` storage contract and schema.
2. Add CLI/API commands to create, clear, and list overrides.
3. Merge active overrides into `build_minutes_live` before availability flags
   are finalized.
4. Mark override-driven rebuilds as material game changes.
5. Surface active overrides in dashboard/operator tooling.
6. Add tests for:
   - `force_out` zeroing a player
   - override persistence across reruns
   - override expiry at lock
   - official-source confirmation after manual override

### 19.11 Implementation mapping to current code

Current repo surfaces already related to overrides:

- API surface: `projections/api/ops_api.py`
- persistence and application helpers: `projections/ops/overrides.py`
- effective-inputs materialization: `projections/pipeline/effective_inputs.py`
- legacy canonical live flow with effective inputs stage:
  `prefect_flows/live_nba_pipeline.py`
- v3 live flow that still needs a manual-availability hook:
  `prefect_flows/live_nba_pipeline_v3.py`
- live minutes feature build entry point:
  `projections/cli/build_minutes_live.py`
- downstream projection readers:
  `projections/cli/finalize_projections.py` and
  `projections/api/optimizer_service.py`

Recommended implementation path:

1. Reuse the existing GameView / ops override storage and API surfaces first,
   but restrict the live projection effect to manual `IN` / `OUT` semantics.
2. Add the manual-availability merge in `build_minutes_live` so the override
   participates in v3 before minutes features are finalized.
3. Preserve the existing effective-inputs path for legacy flow compatibility
   while narrowing or disabling minute/fpts delta behavior for production live
   runs.
4. Ensure finalize/API layers expose override metadata but do not re-apply the
   override a second time.

### 19.12 Shipping sequence

Recommended order of work:

1. Narrow the live policy first:
   - reject or ignore non-availability override fields in the live production
     path
   - keep boost-style controls only in optimizer / contest sim surfaces
2. Add manual availability support to v3:
   - load active ops overrides during `build_minutes_live`
   - translate them into `force_out` / `force_in` semantics before final
     availability flags are computed
3. Expose diagnostics:
   - write active override rows into run manifest / summary
   - surface `manual_override_used` in projections metadata and operator UI
4. Clean up the API and storage contract:
   - once the v3 path is stable, consider a dedicated
     `manual_availability_overrides` endpoint/storage layer if the broader
     `overrides_v1` payload remains too permissive
