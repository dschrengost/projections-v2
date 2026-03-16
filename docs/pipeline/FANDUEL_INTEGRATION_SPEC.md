# FanDuel Integration Spec: Multi-Site DFS Support

## Spec Status: LIVING v0.5 (2026-03-15)

### Implementation Log (2026-03-15)

- Added `projections/fd/` package with:
  - `api.py` (FanDuel fixture-list API client + clientId resolution)
  - `slates.py` (date/type slate discovery)
  - `normalize.py` (players payload → gold salaries normalization)
- Added automated ingestion script:
  - `scripts/fd/run_daily_salaries.py`
  - Bronze writes under `bronze/fd/fixture_lists/game_date=<date>/draft_group_id=<id>/`
  - Gold writes under `gold/dk_salaries/site=fd/...` (v1 storage decision retained)
- Wired optimizer service:
  - FD live slate discovery via `list_fixture_lists_for_date(...)`
  - FD bronze→gold fallback in `load_salaries_for_date(...)` when gold partition is missing
- Wired orchestration:
  - `prefect_flows/live_nba_pipeline.py` optional `fd_salaries_task` (env-gated, non-blocking)
  - `projections/pipeline/live_orchestrator.py` optional `step_fd_salaries` (env-gated, non-blocking)
- Updated CSV FD importer to preserve string IDs (hyphenated FanDuel player IDs).
- Added FD-focused tests under `tests/fd/` and extended salary fallback tests.
- Validated `scripts.fd.run_daily_salaries` against live FanDuel payloads on `2026-03-15`:
  - wrote bronze payloads and gold FD salary partitions for 14 draft groups
  - fixed payload reference resolution (`_members` / `_ref`) in FD normalizer
- Added contest-sim backend site parity pass:
  - site-aware request/response models across run/save/portfolio/field-library endpoints
  - FD lineup slot normalization + validation for run and portfolio outputs
  - site-aware ownership loading and site-partitioned field-library lookup/build paths
- Updated field-library quickbuild defaults for FD (`59000/60000`, lineup size 9) and preserved DK defaults.
- Added entry-manager build-apply guardrails so optimizer/contest-sim builds must match entry `site`.
- Added FD contest-sim API regression tests:
  - `tests/api/test_contest_sim_fd_output.py` (FD lineup normalization + invalid-lineup rejection).

Related production specs:

- [Live Pipeline Production Spec](./LIVE_PIPELINE_PRODUCTION_SPEC.md)
- [Inference Server Spec](./INFERENCE_SERVER_SPEC.md)
- [Game Transformer v2 Spec](../joint_rotation_rates_v1/GAME_TRANSFORMER_SPEC.md)
- [Ownership Transformer v2 Spec](../ownership/OWNERSHIP_TRANSFORMER_SPEC.md)

---

## 1. Motivation

### 1.1 Current gap

The app now has working FD ingestion plus optimizer-site support, but true end-to-end FD operations are still incomplete:

- contest-sim backend is now site-aware, but contest-sim UI is still DK-oriented
- entry-manager and late-swap backends are now mostly site-aware, but operator UI remains DK-oriented
- several production/ops checks and run metadata are still DK-centric
- some downstream model/output contracts still rely on DK-named compatibility columns

This creates a partial-parity state where FD optimizer can run, but adjacent operational workflows are not fully site-complete.

### 1.2 What this spec changes

This spec defines the integration as a multi-phase, contract-first rollout:

1. Make FanDuel slate and salary ingestion first-class.
2. Make optimizer pool/build/export paths truly site-aware.
3. Add FanDuel scoring outputs while preserving DK contracts.
4. Extend dashboard and operational tooling for site selection.
5. Add contest-sim site parity.
6. Add entry-manager + late-swap site parity.
7. Keep DK production stable throughout rollout.

### 1.3 Non-negotiable requirements

- Do not regress current DK live pipeline behavior.
- FanDuel lineup builds must satisfy FD roster and salary rules by construction.
- Site selection must be explicit at API boundaries (`dk` or `fd`), never inferred silently.
- Backward compatibility must be preserved for existing DK columns and saved artifacts.
- Rollout must be observable with per-site health/readiness checks.

---

## 2. Scope

### 2.1 In scope (v1)

- FanDuel slate discovery and salary ingestion.
- Site-aware optimizer pool/build/export APIs.
- FanDuel scoring (`fd_fpts`) from existing box-score world outputs.
- Dashboard site selector and slot rendering for optimizer surfaces.
- Site-aware readiness checks and runtime stamps.
- Contest sim site parity (`dk` + `fd`) for lineup scoring/portfolio flows.
- Late swap / entry manager site parity for FanDuel upload workflows.

### 2.2 Out of scope (initial rollout)

- FanDuel contest result ingestion and post-contest replay parity.
- FanDuel ownership model retraining (`ownership_v2`) beyond compatibility fallback.
- Rebuilding every historical table to dual-site immediately.

---

## 3. Current State Audit (2026-03-15)

### 3.1 Areas already partly site-aware

- `config/optimizer.yaml` has FD defaults (`59000/60000`, team limit 4).
- `projections/optimizer/optimizer_types.py` validates `site in {"dk","fd"}` with site-sized rosters.
- `projections/optimizer/cpsat_solver.py` enforces FD positional counts (`2 PG, 2 SG, 2 SF, 2 PF, 1 C`).
- `projections/optimizer/nba_optimizer.py` defines FD slot list and lineup size 9.
- `projections/api/optimizer_api.py` request models already include `site`.

### 3.2 DK-coupled blockers

Data ingestion and slates:

- FD ingestion path exists (`projections/fd/*`, `scripts/fd/run_daily_salaries.py`) but is operationally gated.
- Prefect/orchestrator FD salaries are env-gated and non-blocking (`PROJECTIONS_ENABLE_FD_SALARIES=1`).
- FanDuel API access can be blocked by anti-bot controls depending on runtime network/session.

Salary and pool contracts:

- `projections/api/optimizer_service.py:load_salaries_for_date()` is site-aware, but game metadata enrichment remains DK-centric where DK draftables are available.
- Salary helper naming and schema are DK-branded in `projections/dk/salaries_schema.py` (`dk_player_id`, `dk_salaries` path).
- Multiple consumers read `gold/dk_salaries/site=dk/...` directly.

Scoring and projection aliases:

- `projections/fpts_v2/scoring.py` exports both `compute_dk_fpts` and `compute_fd_fpts`.
- `projections/rotation/sample_worlds_v2.py` emits DK FPTS world and summary columns only.
- `projections/projections_bundle.py` compatibility aliases are DK-specific (`dk_fpts_*`, `sim_dk_fpts_*`).
- `projections/cli/finalize_projections.py` and ownership scorers expect DK columns by name.

Export and entry tooling:

- `projections/api/optimizer_api.py` has FD export path and FD slot-ordered lineup export.
- `projections/api/entry_manager_api.py` supports `site=dk|fd` upload/export/session flows with per-site persistence.
- Entry apply-build now rejects source-build/entry site mismatches (optimizer + contest-sim sources).
- Contest-sim backend endpoints are site-aware with FD lineup normalization and site-aware field-library cache selection.

Frontend:

- Optimizer page supports `site=fd`.
- Contest sim and entry manager UIs are still DK-oriented.
- Late swap UX remains DK assumptions (slot map, lock semantics, upload/export shape).

---

## 4. Design Decisions (v0.1)

1. Keep `fpts_sim_uncond_mean` as the canonical optimizer decision metric for all sites.
2. Add site-specific aliases (`dk_fpts_*`, `fd_fpts_*`) as compatibility projections, not primary internal keys.
3. Keep current salary partition root for v1 (`gold/dk_salaries/site=<site>/...`) to avoid a high-risk storage migration now.
4. Introduce a site adapter abstraction so DK and FD ingestion/export rules do not leak into generic optimizer service code.
5. Entry-manager/late-swap backend should run dual-site (`dk` + `fd`) behind explicit site parameters.

---

## 5. Target Architecture

### 5.1 Site adapter layer

Add a thin per-site adapter interface, implemented for `dk` and `fd`:

- list slates for date
- fetch and normalize salary payloads
- expose slot definitions and upload/export formatter
- expose site-specific scoring formula function

Candidate module layout:

- `projections/sites/base.py` (protocols/types)
- `projections/sites/dk_adapter.py`
- `projections/sites/fd_adapter.py`

### 5.2 Optimizer service boundary

`projections/api/optimizer_service.py` becomes site-orchestrating, not site-implementing:

- `get_slates_for_date(game_date, site, slate_type)`
- `load_salaries_for_date(game_date, draft_group_id, site)`
- export dispatch by site (`dk_csv`, `fd_csv`)

### 5.3 Scoring and projection contract

Generate both:

- canonical: `fpts_sim_cond_*`, `fpts_sim_uncond_*`
- site aliases:
  - DK: existing `dk_fpts_*` + `sim_dk_fpts_*`
  - FD: new `fd_fpts_*` + `sim_fd_fpts_*`

This allows legacy DK consumers and enables FD UX without changing core optimizer metric selection.

---

## 6. Data Contracts

### 6.1 Salaries parquet (v1 contract extension)

Continue writing to:

`$PROJECTIONS_DATA_ROOT/gold/dk_salaries/site=<site>/game_date=<date>/draft_group_id=<id>/salaries.parquet`

Required columns (site-neutral + compatibility):

- `site` (`dk` or `fd`)
- `game_date`
- `draft_group_id`
- `site_player_id` (new canonical identifier)
- `display_name`
- `positions` (list[str], site-native positions normalized to NBA roster semantics)
- `salary`
- `team_abbrev`
- `status`
- `is_swappable`
- `is_disabled`
- `raw_competition_ids`
- `raw_data`
- `dk_player_id` (optional, DK compatibility only)
- `fd_player_id` (optional, FD compatibility only)

### 6.2 Projection aliases

Canonical required for downstream optimization:

- `fpts_sim_uncond_mean`
- `fpts_sim_uncond_std`
- `fpts_sim_uncond_p90`

Site aliases:

- DK: existing aliases unchanged.
- FD: add alias family with identical quantile semantics, FD scoring formula.

### 6.3 Export contracts

- DK export remains current DraftKings draftable CSV format.
- FD export adds FanDuel-compatible upload format (exact header/order finalized in Open Questions).

---

## 7. Workstreams and Phases

## 7.1 Phase 0: Contract and interface prep

Deliverables:

- Add this living spec and lock v0.1 decisions.
- Add adapter interfaces and site-aware signatures in service layer.
- Add test scaffolding for site-parametrized API/service behavior.

Exit criteria:

- No behavior change for DK.
- CI tests pass with `dk` baseline plus placeholder `fd` adapter tests.

## 7.2 Phase 1: FanDuel slate/salary ingestion

Deliverables:

- Implement FanDuel ingestion adapter and normalizer.
- Add Prefect/CLI task(s) for FD salaries by date.
- Site-aware slate discovery endpoint (`/api/optimizer/slates?...&site=fd`).

Exit criteria:

- FD slate list and salary parquet appear for target dates.
- Readiness checks show FD salary availability for at least one live date.

## 7.3 Phase 2: Optimizer pool/build parity

Deliverables:

- Build pool for FD via shared `build_player_pool(..., site="fd")`.
- Ensure FD constraints, salary bounds, and lineup size are enforced in build path.
- Add FD export endpoint path and tests.

Exit criteria:

- `/api/optimizer/pool`, `/build`, `/build/{id}/lineups`, `/export` all pass with `site=fd`.
- Generated lineups always have 9 players and valid FD position counts.

## 7.4 Phase 3: FD scoring integration

Deliverables:

- Add `compute_fd_fpts` to scoring module.
- Compute/store `fd_fpts` worlds + summary aliases from existing sampled stat outputs.
- Thread FD columns through bundle/finalize layers without DK regressions.

Exit criteria:

- FD projection columns available in finalized outputs.
- DK column parity validated against pre-change baseline (within tolerance).

## 7.5 Phase 4: Dashboard and operator UX

Deliverables:

- Add site selector in optimizer UI and wire API calls by selected site.
- Add FD slot rendering/lineup cards/export actions.
- Add clear site labels in saved builds, job history, and status panels.

Exit criteria:

- Operator can run full FD flow from UI without manual API calls.
- DK UI behavior unchanged by default.

## 7.6 Phase 5: Contest Sim site parity

Deliverables:

- Add explicit `site` parameter across contest-sim API request models and service entrypoints.
- Make worlds/projection column resolution site-aware (`dk_fpts*` vs `fd_fpts*` aliases).
- Partition and cache generated field libraries by site.
- Update Contest Sim UI to pass selected site and render site-correct lineup metadata.

Exit criteria:

- `/api/contest-sim/*` endpoints run for `site=fd` and `site=dk`.
- Portfolio outputs and ranking metrics are available for FD runs with no DK regressions.

## 7.7 Phase 6: Entry Manager + Late Swap parity

Deliverables:

- Generalize entry-manager session state, parsers, and exports for `site=fd`.
- Add FD slot templates and ID mapping utilities for upload-compatible lineups.
- Make late-swap lock detection, candidate generation, and commit/export site-aware.
- Add FD operator UX in entry-manager/late-swap surfaces.

Exit criteria:

- Operator can upload, preview, swap, and export FanDuel entries end to end.
- Late-swap generated candidates always satisfy FD roster constraints.
- DK late-swap behavior unchanged.

## 7.8 Phase 7: Production hardening

Deliverables:

- Site-aware readiness checks and alarms.
- Runtime stamp includes site and adapter metadata.
- Rollback switches for FD-specific code paths.

Exit criteria:

- Successful canary slates across multiple dates.
- No DK incident caused by FD rollout.

---

## 8. Validation Plan

- Unit tests:
  - scoring formulas (`compute_dk_fpts`, `compute_fd_fpts`)
  - site adapters (slate parse, salary normalization)
  - pool builder joins and inactive filtering by site
- Integration tests:
  - optimizer API endpoints with `site=dk` and `site=fd`
  - export format validation for each site
- Contract checks:
  - salary schema required columns present
  - lineup cardinality and position feasibility checks
  - projection alias parity checks (canonical vs site aliases)

---

## 9. Risks and Mitigations

- Risk: Hidden DK-only assumptions in downstream tools.
  - Mitigation: add explicit site parameters and fail-fast for unsupported site paths.

- Risk: Alias sprawl and consumer confusion.
  - Mitigation: canonical columns remain primary; aliases documented as compatibility layer.

- Risk: UI logic tied to DK slot semantics.
  - Mitigation: centralize slot definitions and assignment per site in shared TS utilities.

- Risk: FD rollout impacting DK.
  - Mitigation: gate FD flows behind site flags and keep DK defaults unchanged.

---

## 10. Open Questions

### 10.1 Resolved decisions (2026-03-15)

1. Contest sim should accept FD lineups in v1 with an explicit site toggle.
2. Salary storage should be renamed long-term (do not keep `gold/dk_salaries/site=fd` forever).
3. `/api/optimizer/slates` should continue defaulting to `site=dk`, with easy UX toggle to `fd`.
4. FD entry-manager parity is required before enabling FD pipeline ingestion in production.

### 10.2 Remaining unknowns

1. FD ownership fallback policy is still undecided (null/optional vs heuristic prior).
2. Canonical late-swap lock-time source for FD is still undecided when fixture metadata conflicts with entry CSV timestamps.

### 10.3 Immediate discovery tasks

1. Decide ownership fallback policy after a first pass of FD contest-sim/entry-manager implementation.
2. Decide final FD lock-time source precedence (`fixture start` vs `entry cell hints`) and codify in late-swap docs/tests.
3. Complete contest-sim + entry-manager/late-swap UI site toggles and run operator QA pass.

---

## 11. Workstream Status

| Workstream | Status | Notes |
|---|---|---|
| Spec + contracts | in progress | v0.5 includes explicit contest-sim/entry-manager/late-swap backend status + handoff |
| FD ingestion | partial | Live FD scraper/normalizer + bronze/gold writes implemented; rollout env-gated |
| Optimizer API parity | partial | Site-aware slates/pool/build/export implemented; broader downstream integrations pending |
| FD scoring columns | partial | `compute_fd_fpts` + alias fallback in optimizer path; full downstream propagation pending |
| Dashboard multi-site UX | partial | Optimizer page supports site toggle; contest-sim and entry-manager still DK-oriented |
| Contest Sim parity | partial | Backend APIs now run `site=fd` with FD lineup normalization; frontend site toggle/rendering still pending |
| Entry Manager parity | partial | Backend supports FD upload/export/session paths with live-template parser; UI still DK-oriented |
| Late Swap parity | partial | Session preview/commit/export are site-aware for FD; lock-time policy finalization + UI work pending |
| Production hardening | pending | Readiness/runtime stamps/alerts remain mostly DK-centric |

---

## 12. Status Updates

### Status Update (2026-03-15, integration spec kickoff)

- Established this as a living spec with explicit phased rollout.
- Audited current codebase for FD-ready vs DK-coupled components.
- Locked v0.1 design decisions:
  - canonical `fpts_sim_uncond_mean` remains the optimizer objective metric,
  - site alias columns are compatibility surfaces,
  - no immediate storage-path migration in v1.

### Status Update (2026-03-15, implementation pass)

- Added site-aware optimizer API/service paths for FanDuel:
  - slates endpoint accepts `site=fd`,
  - pool/build/export flows now route by site,
  - FD CSV export formatter added with `PG,PG,SG,SG,SF,SF,PF,PF,C`.
- Added scoring support:
  - `compute_fd_fpts` now available in `projections.fpts_v2.scoring`.
- Added FD salary ingestion utility:
  - `projections/cli/import_fanduel_salaries.py` imports FanDuel CSV into gold salary partitions.
- Added UI support on Optimizer page:
  - DraftKings/FanDuel selector,
  - site-aware salary cap, lineup size, pool loading, build, and export behavior,
  - FD lineup card slot assignment.
- Added/updated tests:
  - FD scoring test coverage,
  - FD slate discovery path test,
  - FD export/slot assignment unit tests.

### Status Update (2026-03-15, FD ingestion validation + next scope)

- Validated FD daily ingestion with live payloads for `2026-03-15` and wrote 14 FD salary partitions.
- Confirmed remaining major integration gaps are downstream workflows, not ingestion:
  - Contest Sim API/service/frontend site handling,
  - Entry Manager site handling,
  - Late Swap site handling and export semantics.
- Elevated late swap from implicit follow-up to explicit required phase in this spec.

### Status Update (2026-03-15, owner decisions captured)

- Captured product decisions:
  - contest sim must support FD toggle in v1,
  - optimizer slates default remains DK with explicit FD toggle,
  - storage should migrate away from `gold/dk_salaries/site=fd`,
  - entry-manager parity is required before FD production ingestion enablement.
- Left explicit unknowns in place for:
  - FD ownership fallback policy,
  - FD late-swap lock-time source.

### Status Update (2026-03-15, FD entry-manager + late-swap backend pass)

- Integrated live FanDuel entry-template CSV contract (`entry_id,contest_id,contest_name,entry_fee,PG,PG,SG,SG,SF,SF,PF,PF,C,...`) into entry upload parser.
- Entry manager is now site-aware at API boundary (`site=dk|fd`) across:
  - upload/list/get/delete/apply-build/validate/export paths,
  - per-site entry state persistence under `data/entries/<date>/<site>/`.
- FanDuel entry export now preserves duplicate position columns and emits FD-compatible slot values.
- Late-swap sessions are now site-aware (`create/list/get/preview/pin/policy/commit/export`) with FD session storage and commit/export path support.
- Added regression coverage:
  - FD upload-template parser + persistence test,
  - FD export duplicate-slot contract test,
  - FD late-swap session lifecycle integration test.

### Status Update (2026-03-15, contest-sim FD lineup compliance pass)

- Confirmed FanDuel lineup/scoring rule coverage in optimizer + quickbuild paths:
  - FD roster shape (`PG,PG,SG,SG,SF,SF,PF,PF,C`) enforced by solver/slot assignment paths,
  - FD salary defaults (`59000/60000`) applied in quickbuild field-library generation.
- Contest-sim backend now emits FD-compliant lineup outputs when `site=fd`:
  - lineups are normalized to FD slot order,
  - invalid FD lineup constructions fail fast with `400`.
- Contest-sim persistence and retrieval are now site-aware:
  - saved run/portfolio build payloads carry `site`,
  - saved-build listing supports site filtering,
  - field-library list/build are site-aware and FD cache/version isolation is enforced.
- Entry-manager integration safety updated:
  - `apply-build` now validates source build site matches entry site before applying lineups.
- Added API regression coverage for FD contest sim:
  - `tests/api/test_contest_sim_fd_output.py` validates normalized FD run outputs and invalid-lineup rejection.

---

## 13. Agent Handoff (2026-03-15)

### Current state

- FD ingestion + optimizer paths are functional.
- Contest-sim backend is now site-aware and outputs FD-compliant lineups.
- Entry-manager + late-swap backends are site-aware for FD session/upload/export flows.
- Remaining gaps are primarily UI/operator workflow polish and production hardening.

### Next priority items

1. Add explicit site toggle + FD slot rendering across contest-sim UI surfaces.
2. Add explicit site toggle + FD slot rendering across entry-manager and late-swap UI surfaces.
3. Finalize FD ownership fallback policy and codify behavior in contest-sim docs/tests.
4. Finalize FD late-swap lock-time precedence policy and codify docs/tests.
5. Add one end-to-end FD operator smoke test: upload FD entry CSV -> apply contest-sim or optimizer build -> late-swap preview/commit -> export.

### Notes for next agent

- Live FanDuel template CSV used for parser contract testing exists at:
  - `docs/pipeline/FanDuel-NBA-2026-03-15-127613-entries-upload-template.csv`
- FanDuel slate/salary API access does not currently use account auth, but runtime anti-bot/network behavior can still cause fetch failures; keep ingestion env-gated in production rollout.
