# Contest Flashback: Post-Contest Simulation Spec

## Spec Status: DRAFT v0.1 (2026-03-05)

---

## 1. Motivation

### 1.1 Current gap

Our current contest sim answers a forward-looking question:

```
Given user lineups and a modeled field, what are the ROI/rate outcomes under our worlds?
```

That is useful for slate-time decisions, but it does not answer the postmortem question users
actually care about after lock:

```
Given the actual contest field that entered, how good was my lineup ex ante?
```

SaberSim-style "contest flashback" is essentially:

1. reconstruct the real opponent field from contest results CSVs,
2. rescore that field under pre-lock simulated worlds,
3. report the user's simulated ROI/rates against the real field composition.

This is different from our current generated-field calibration work. Generated-field sim tries to
approximate what the field might look like. Flashback uses the field that actually showed up.

### 1.2 What this spec changes

This spec makes three hard decisions:

1. Treat post-contest replay as an extension of contest sim, not a separate simulator.
2. Use the actual contest lineup field as the primary field library when available.
3. Separate three modes clearly:
   - exact replay: full contest field available,
   - anchored emulation: partial contest field available,
   - synthetic fallback: no usable field CSV, use current generated-field library.

This draft now also treats match quality and replay attribution as first-class outputs. Replay is
not usable unless the user can see:

- entered lineup replay ROI and rates,
- field-vs-modeled drift,
- optimizer regret,
- player-name resolution quality and unresolved examples.

### 1.3 Non-negotiable requirements

- Use only information available as of contest lock when generating user/field outcomes.
- Reuse the current world-scoring and payout engine where possible.
- Preserve exact field composition when full results CSVs are available.
- Support duplicate lineups and tie-splitting correctly.
- Produce user-facing metrics centered on ROI and rates, not only realized finish.
- Make provenance explicit: which world bundle, which contest CSV, which payout table, which mode.

---

## 2. Design Principles

1. Exact field beats calibrated field:
   if we have the actual entries, do not replace them with a synthetic approximation.

2. Ex post field, ex ante outcomes:
   the field can come from post-contest CSVs, but player outcomes must come from the pre-lock world
   bundle or as-of-lock snapshot.

3. Weighted library remains the core abstraction:
   an actual contest field is just a weighted field library with much stronger grounding.

4. Be explicit about what is simulated vs observed:
   observed lineups and payout structure; simulated player outcomes and lineup profits.

5. Keep the output interpretable:
   "you finished 4,112th, but your lineup had 18.4% cash rate and 1.07 sim ROI against the actual
   field" is the product, not just a rank histogram.

6. Resolution quality must be observable:
   if DK names do not map cleanly to internal `player_id`s, the product must say so directly.

---

## 3. Product Modes

## 3.1 Exact replay mode

Use when we have a full contest results CSV with all entries or all unique lineups plus duplicate
counts.

Inputs:

- actual contest lineups,
- actual contest payout table or recoverable payout tiers,
- lock-time world bundle,
- optional user entries if the user wants portfolio-level results.

Behavior:

- dedupe identical field lineups into weighted unique entries,
- rescore every unique lineup in every simulated world,
- apply exact payout/tie logic using the actual field weights,
- report per-lineup and portfolio metrics.

This is the primary target mode.

## 3.2 Anchored emulation mode

Use when the contest CSV is partial or sampled, for example:

- only top finishers are available,
- only a subset of entries is exported,
- payout table is incomplete,
- lineup strings are present for only some rows.

Behavior:

- lock in the observed subset exactly,
- infer missing field mass with a constrained generator,
- force the completed field to match observed ownership/duplication/salary targets,
- mark results as estimated, not exact.

This is the "emulate Contest Flashback when the scrape is incomplete" mode.

## 3.3 Synthetic fallback mode

Use when no usable contest field is available.

Behavior:

- delegate to the current `generated_field` workflow,
- use historical calibration from bronze contest CSVs,
- present this as field estimation, not flashback replay.

---

## 4. Problem Formulation

For one contest with `N` entries, let:

- `L_j` be lineup `j`,
- `w_j` be the multiplicity of unique lineup `j` after dedupe,
- `Y^(m)` be simulated player fantasy points in world `m`,
- `S_j^(m)` be lineup score for lineup `j` in world `m`,
- `P(rank)` be the payout function after tie-splitting and rake,
- `f_j^(m)` be profit for lineup `j` in world `m`.

Then in replay mode:

1. observed field composition is fixed by the contest CSV,
2. simulated lineup outcomes come from our lock-time world model,
3. user sim ROI is:

```
sim_roi_j = E_m[f_j^(m)] / entry_fee
```

and user rate metrics are:

```
Pr(win), Pr(top_1pct), Pr(top_5pct), Pr(cash)
```

The critical point is that lineup composition is observed, while lineup performance remains random
under our model.

---

## 5. How We Emulate This

## 5.1 Exact replay is mostly not "emulation"

If a nightly results CSV contains the full contest field, the right implementation is simple:

1. parse all entries,
2. canonicalize each lineup to internal player IDs and slot order,
3. collapse duplicates into a weighted unique-lineup library,
4. score that library against our pre-lock worlds,
5. run the normal payout engine.

That is already enough to produce sim ROI against the real field.

So the main technical problem is not "how do we simulate opponents?".
It is:

1. how do we normalize contest CSVs into a trustworthy field library,
2. how do we align those lineups to the correct lock-time slate/world bundle,
3. how do we complete the field when the scrape is partial.

## 5.2 Exact replay algorithm

For one contest:

1. Resolve contest metadata:
   - `site`, `sport`, `contest_id`, `draft_group_id`, `game_date`, `lock_ts`, `entry_fee`.
2. Load the authoritative pre-lock world bundle for that slate.
3. Load and normalize all contest entries from nightly scrape.
4. Convert each lineup string into:
   - ordered slots,
   - internal player IDs,
   - canonical unordered lineup key for duplicate detection,
   - optional salary / ownership / entry-name metadata.
5. Dedupe identical lineups:
   - one unique lineup row,
   - `field_weight = duplicate_count`.
6. Score each unique lineup over all worlds:
   - matrix or sparse-dot over player world scores.
7. Rank all unique lineups per world with weights.
8. Apply payout tiers with tie-splitting using weights.
9. Extract per-user metrics for any entries belonging to the user.

This fits the current weighted field engine directly.

## 5.3 Anchored emulation algorithm

When the full field is not available, emulate the missing mass in layers:

1. Keep observed lineups exact.
2. Estimate the missing entrant count:
   - from contest metadata,
   - from payout table depth,
   - or from the highest observed rank / count.
3. Build observed target statistics:
   - player ownership from observed entries,
   - duplicate distribution,
   - salary and unused-salary histograms,
   - total ownership histogram,
   - stack/correlation patterns if stable.
4. Generate a candidate completion pool from current contest-sim field builders.
5. Reweight the candidate pool so the combined field:
   - preserves all observed lineups,
   - matches target distributions as closely as possible,
   - fills the remaining entrant mass only.
6. Compress the completed field back into a weighted library.

Conceptually:

```
completed_field = observed_exact + reweighted_synthetic_remainder
```

This is the closest defensible approximation when our scrape is incomplete.

## 5.4 Name resolution requirements

Contest flashback depends on mapping raw DraftKings lineup strings to internal `player_id`s. That
layer must be observable, not silent.

Resolution order:

1. exact normalized name match,
2. slate-constrained DK draftable-name match,
3. unique first-initial plus last-name signature match,
4. conservative fuzzy match inside the slate player pool,
5. explicit override from alias control-plane file.

Control-plane alias overrides:

```text
$PROJECTIONS_DATA_ROOT/control_plane/contest_results/player_alias_overrides.json
```

The UI and API should surface:

- slot resolution rate,
- unresolved slot count,
- ambiguous match count,
- fuzzy match count,
- preview rows for unresolved, ambiguous, and fuzzy examples.

If match quality is weak, player-level calibration should be treated as low-confidence for that run.

## 5.5 Why this is credible

The field composition problem is much easier post contest than pre contest because:

- we know exactly which slate and payout structure ran,

## 5.6 User-facing replay surface

The Flashback page should present a replay in this order:

1. entered lineups with `sim_roi`, `sim_cash_rate`, `sim_top1pct_rate`, and `sim_win_rate`,
2. replay takeaways for the user's entered set,
3. match-quality diagnostics,
4. regret summary,
5. field summary,
6. preview tables for player, lineup, field, and regret datasets.

The page is not just a parquet preview tool. It must answer:

- were my lineups good ex ante?
- was this mostly variance, projection error, field error, or selection error?
- can I trust the player-name resolution for this replay?

## 5.7 Export provenance requirements

To make replay regret meaningful, export artifacts must persist the contest-sim lineage that produced
the final uploaded set.

`EntryFileState` should retain:

- `source_build_source`
- `source_build_id`
- `source_build_kind`
- `source_build_name`
- `source_portfolio_build_id`
- `source_run_build_id`
- `source_selection_mode`

Export manifests should copy that provenance so flashback can prefer the full saved contest-sim run
candidate universe over `eval_lineups.csv` from the final exported subset.

Historical exports without provenance can be backfilled heuristically when:

- the export CSV can be reconstructed to internal player IDs,
- a saved portfolio build on the same slate has the same exported lineup multiset,
- or the exported lineup set is a clean subset of a nearby saved portfolio build.

Backfill tool:

```text
projections.cli.backfill_export_lineage
```

For the recent recovery window, the system successfully backfilled recent manifests and linked them
to:

- `source_portfolio_build_id`
- `source_run_build_id`
- `source_selection_mode`
- `lineage_backfill`

This is good enough for recent-history replay regret, but new exports should not rely on heuristic
matching.

## 5.8 Known repair gap: observed field IDs vs world namespace

Recent investigation showed a separate replay-quality issue in the observed opponent field:
some contest-result player names resolve to IDs that are not present in the pre-lock worlds bundle.

This affects only a small subset of opponent-field slots, but it can still contaminate replay
quality and regret if left untreated.

Observed examples on `2026-03-05` included:

- `EJ Harkless`
- `Keshad Johnson`
- `Tolu Smith`
- `Jonathan Isaac`
- `Kyrie Irving`

The saved contest-sim source runs themselves were clean. The issue is in replay field-resolution for
certain fringe players from contest results CSVs.

Required follow-up:

- constrain replay resolution to the canonical projection/worlds namespace,
- do not silently accept fallback IDs that are absent from the worlds matrix,
- surface opponent-field missing-player counts in replay diagnostics.

## 5.9 Current high-priority fixes

The highest-priority product and correctness fixes are:

1. Opponent-field canonical ID repair
   - Current blocker for trustworthy replay on some recent slates.
   - Symptom: observed field contains player IDs absent from the worlds matrix.
   - Effect: malformed opponent lineups, broken field summaries, inflated/deflated replay ROI.
   - Required fix: resolve contest-result names only to canonical internal IDs present in the
     projection/worlds namespace for that slate.

2. Replay trust diagnostics
   - Flashback must explicitly show when a replay is not trustworthy.
   - Required summary fields:
     - `opponent_missing_player_id_count`
     - `opponent_missing_player_examples`
     - `candidate_universe_source`
     - `candidate_universe_lineup_count`
     - `replay_trust_status`
   - If field summaries are physically impossible, the run should be flagged, not interpreted.

3. Candidate-universe provenance
   - Fixed for new exports and partially backfilled for recent history.
   - Replay must prefer:
     - `source_run_build_id`
     - then explicit override
     - then exported subset fallback
   - UI should show which source was used so selection-regret interpretation is grounded.

4. Historical reruns for recent slates
   - After opponent-field repair, recent high-value slates should be rerun.
   - Priority window:
     - `2026-02-28` through `2026-03-05`
   - Priority contests:
     - flagship / high-entry contests actually played by the user

5. Field-summary sanity checks
   - Current examples showed impossible values like near-zero salary totals and zero ownership sums.
   - Required guardrail:
     - fail or flag replay when field feature aggregates are outside plausible ranges.

## 5.10 Implementation update (2026-03-06)

Implemented in code:

- replay resolution is now constrained to the canonical worlds namespace for the slate:
  - slots resolving to player IDs outside worlds are marked unresolved and surfaced in diagnostics
- replay analytics summary now emits:
  - `replay_trust_status`
  - `replay_trust_issues`
  - `opponent_missing_player_id_count`
  - `opponent_missing_player_examples`
  - `candidate_universe_source`
  - `candidate_universe_lineup_count`
  - candidate missing-player diagnostics
- candidate lineup universes are filtered against worlds before regret scoring
- field sanity checks now feed replay trust classification (`trusted` / `warning` / `broken`)
- regret outputs now include human-readable lineup strings in addition to lineup keys/player IDs:
  - `best_entered_lineup_players`
  - `actual_best_entered_lineup_players`
  - `best_candidate_lineup_players`
  - `best_finalset_lineup_players`
- replay summary now includes actionable decision diagnostics:
  - `decision_guidance`
  - `attribution_summary` (`primary`, `confidence`, `reasons`)

Still pending:

- anchored-emulation completion for partial contest fields
- historical reruns for priority slates after trust-hardening rollout
- stricter as-of-lock world-bundle provenance checks in replay summary

## 5.11 Concrete known bad example

The `2026-03-03` `NBA $25K mini-MAX [150 Entry Max]` replay is currently not trustworthy.

Observed failures:

- `candidate_unique_count = 6` while the user entered `40` lineups
- field summary values were physically impossible:
  - near-zero salary totals
  - near-zero team counts
  - zero ownership sums
- opponent field contained player IDs absent from the worlds matrix

Interpretation:

- entered-lineup scoring looked plausible,
- opponent-field reconstruction did not,
- selection regret and slate-level replay takeaways for that run should not be trusted until the
  opponent-field repair is complete.
- we often have the actual lineups,
- even partial CSVs give strong constraints on duplication and ownership shape,
- we no longer need to guess what the field wanted to do; we only need to simulate outcomes for the
  field that entered.

That means replay mode should be materially more reliable than our pre-lock generated-field mode.

---

## 6. Data Contracts

## 6.1 Bronze inputs

Expected upstream raw inputs:

- `bronze/dk_contests/nba_gpp_data/<YYYY-MM-DD>/results/contest_<id>_results.csv`
- optional contest summary CSVs already used in historical calibration
- optional user entry export / saved-build export if we want explicit portfolio matching

Required raw fields or derivable equivalents:

- `contest_id`
- `contest_name`
- `entry_id` or stable row identifier
- `rank`
- `lineup_string`
- `fantasy_points` or realized score
- `prize`
- `entry_fee`
- `%Drafted` or drafted ownership fields when present

## 6.2 Silver normalized contracts

Add a normalized replay layer:

- `silver/post_contest/contest_entries/site=dk/sport=nba/date=<date>/contest_id=<id>/entries.parquet`
- `silver/post_contest/contest_payouts/site=dk/sport=nba/date=<date>/contest_id=<id>/payouts.parquet`
- `silver/post_contest/contest_meta/site=dk/sport=nba/date=<date>/contest_id=<id>/meta.json`

`entries.parquet` should contain one row per observed entry with:

- contest identifiers,
- rank / prize / realized score,
- parsed slot-level player IDs,
- canonical lineup key,
- duplicate group key,
- observed ownership aggregates when available.

Serving rule:

- normalized replay tables are the preferred serving layer for API/UI reads,
- raw bronze results remain the freshness fallback when normalized tables are stale or missing,
- replay execution may read raw bronze inputs directly, but downstream analytics and dashboards
  should materialize normalized contracts.

## 6.3 Replay-ready field library contract

For the simulator, represent the contest as:

- `unique_lineups.parquet`
- one row per unique lineup,
- `field_weight`,
- optional observed realized finish summaries,
- provenance fields:
  - source CSV path,
  - parse version,
  - normalization version,
  - slate snapshot / world bundle ID.

This should look as close as possible to the current field-library abstraction.

## 6.4 World-bundle contract

Flashback must use an as-of-lock snapshot, not today's latest model state.

Required identifiers:

- `game_date`
- `draft_group_id`
- `run_id`
- `as_of_ts`
- model bundle IDs relevant to minutes / rates / world generation

If an exact lock-time world bundle is unavailable, we may allow a nearest-pre-lock fallback, but the
run must be labeled as approximate.

---

## 7. Architecture Overview

## 7.1 High-level factorization

For one post-contest replay:

```
bronze results csv
-> normalize / crosswalk
-> exact or anchored field library
-> load lock-time worlds
-> score unique lineups by world
-> payout engine with field weights
-> user replay metrics and diagnostics
```

## 7.2 Major components

1. Contest results normalizer
   - parse nightly CSVs,
   - map players to internal IDs,
   - canonicalize lineup keys,
   - write normalized silver tables.

2. Replay field builder
   - exact mode: dedupe to weights,
   - anchored mode: preserve observed + fill missing mass,
   - synthetic mode: delegate to current field-library manager.

3. Lock-time world resolver
   - choose the correct slate/run snapshot,
   - enforce `as_of_ts <= lock_ts`,
   - surface provenance in output.

4. Replay scorer
   - reuse current contest sim scoring and payout logic,
   - accept actual-field libraries with weights.

5. Replay metrics layer
   - sim ROI and rate metrics,
   - realized-vs-sim comparisons,
   - variance/luck diagnostics,
   - portfolio aggregation.

6. Flashback serving/index layer
   - fast path from normalized `analytics/contest_results/*` tables,
   - fallback path from raw `bronze/dk_contests/nba_gpp_data/<date>/results/*.csv`,
   - same response contract regardless of source,
   - explicit freshness-over-speed tradeoff in favor of raw fallback.

## 7.3 Reuse of existing modules

The current contest-sim system already has most of the hard runtime pieces:

- field library schema and manager,
- payout engine,
- lineups scored across worlds,
- ROI/rate summaries,
- API/UI surfaces for contest sim.

New work is primarily:

- ingestion + normalization of actual results CSVs,
- exact-field library construction,
- lock-time world provenance,
- anchored completion logic for partial scrapes,
- product/UI semantics for replay vs generated-field.

Operational note:

- contest discovery should not hard-fail when normalized contest tables lag behind the nightly raw
  scrape,
- the API should fall back to raw results discovery for the requested date,
- this makes flashback usable for the latest scraped contests while preserving normalized parquet as
  the stable serving/index layer.

---

## 8. Scoring and Metrics

## 8.1 Primary outputs

Per lineup:

- simulated ROI
- win / top 1% / top 5% / cash rates
- mean finish percentile
- payout distribution quantiles
- duplicate count in actual field
- realized finish / realized payout

Per portfolio:

- aggregate sim ROI
- profit distribution
- probability of portfolio-level profit
- realized vs simulated aggregate outcome

## 8.2 Postmortem diagnostics

The best product value is not only "what was my sim ROI?" but also "why did my actual result differ?".

Add diagnostics such as:

- realized percentile vs sim percentile distribution,
- realized payout minus simulated mean payout,
- ownership leverage vs actual field,
- dupe burden vs expected,
- lineup-level downside/upside attribution from worlds.

## 8.3 Important semantic rule

Flashback metrics should be interpreted as:

```
How this lineup would perform against the actual contest field under our ex ante belief distribution.
```

They are not:

- a claim about true skill,
- a claim that the model knew the future,
- a replacement for realized payout.

---

## 9. Correctness Rules

## 9.1 Player identity and slate alignment

The replay run must fail fast if:

- a lineup contains unmapped players,
- player IDs map to a different slate than the contest,
- roster-slot legality cannot be reconstructed,
- `as_of_ts > lock_ts`.

## 9.2 Duplicate handling

Duplicates must be represented through field weights, not row replication.

Consequences:

- runtime stays tractable,
- tie-splitting remains exact,
- actual-field dupes are handled naturally by the payout engine.

## 9.3 Payout fidelity

Priority order for payout tiers:

1. exact payout table parsed from contest results,
2. exact contest metadata from scraped summary page,
3. contest archetype fallback from config.

Mode 3 is acceptable only for anchored/synthetic runs, not preferred for exact replay.

---

## 10. Implementation Roadmap

### Phase 1: Exact replay foundation

- Normalize nightly contest result CSVs into silver replay tables.
- Build `actual_field` library writer from normalized entries.
- Add lock-time world resolver with fail-fast provenance checks.
- Reuse contest-sim payout/scoring engine to run replay on actual fields.
- Expose a basic API endpoint returning per-entry replay metrics.

### Phase 2: Product wiring

- Add replay mode to dashboard/API.
- Allow filtering by date / contest / user handle / entry name.
- Show realized result next to simulated ROI/rates.
- Add provenance panel:
  - contest CSV source,
  - mode,
  - run ID,
  - as-of timestamp,
  - payout source.

### Phase 3: Anchored emulation

- Implement partial-field detection and coverage diagnostics.
- Add constrained completion of missing entrant mass.
- Label outputs as `anchored_emulation`.
- Add confidence/coverage metadata:
  - observed entrant fraction,
  - payout completeness,
  - player-ID mapping success rate.

### Phase 4: Historical analytics and calibration

- Aggregate replay outputs across slates.
- Measure how replay ROI tracks realized outcomes.
- Compare exact replay vs synthetic generated-field estimates.
- Use this to improve pre-lock field generation and dupe modeling.

### Phase 5: Acquisition hardening and automation

- Split the control plane into:
  - `acquire_dk_results`
  - `normalize_dk_results`
  - `run_flashback_for_played_contests`
  - `aggregate_replay_calibration`
- Keep DK authentication isolated to acquisition only.
- Use browser-state handoff as the primary DraftKings auth mechanism.
- Allow downstream replay/calibration jobs to run from already-landed raw files even if acquisition
  fails on a given night.
- Add explicit alerting when browser state has expired or acquisition coverage is incomplete.

---

## 11. Concrete Module Plan

## 11.1 New files

- `projections/post_contest/contest_results_normalizer.py`
- `projections/post_contest/replay_field_builder.py`
- `projections/post_contest/replay_world_resolver.py`
- `projections/post_contest/replay_service.py`
- `projections/post_contest/replay_models.py`
- `docs/contest-sim/POST_CONTEST_FLASHBACK_SPEC.md`

## 11.2 Existing files to reuse

- `projections/contest_sim/field_library.py`
- `projections/contest_sim/field_library_manager.py`
- `projections/contest_sim/contest_sim_service.py`
- `projections/api/contest_service.py`
- contest sim API and dashboard surfaces where replay mode should be exposed

---

## 12. Open Questions

### Resolved in this draft

- Should this be a separate simulator?

No. It should be a new field-source mode inside contest sim.

- What is the core abstraction?

An actual contest field is a weighted field library with stronger provenance.

- How do we emulate SaberSim-style flashback?

Exact replay when full CSVs exist; anchored completion only when the scrape is partial.

### Open

- Do our nightly result scrapes contain the full field for the target contests, or only a sampled/top-N
  slice in some cases?
- What is the best canonical join key for user portfolios: handle, entry ID, saved build ID, or uploaded
  CSV?
- Do we want replay metrics stored under `gold/` for longitudinal user analytics, or computed on demand
  only?
- Should replay support contest-specific payout curves exactly, or standardize some very large contests
  for speed?

### Immediate operational next steps

1. Implement browser-state handoff for DK contest acquisition.
2. Store the authenticated browser state in a control-plane path outside the repo.
3. Update nightly acquisition to reuse that state file instead of attempting fresh headless login.
4. Refresh normalized `contest_inventory` and `user_entries` from landed raw files after each
   acquisition run.
5. Trigger flashback replay and calibration jobs from normalized outputs, not from live DK access.

### Browser-state handoff recommendation

Preferred near-term auth approach:

1. Log into DraftKings in a real browser session.
2. Export a Playwright-compatible `storage_state.json` or equivalent cookie/session file.
3. Sync that state file to the server.
4. Point nightly DK acquisition at that state file.
5. Refresh the state manually when DraftKings expires the session.

Why this is the recommended path:

- materially more reliable than full headless DK login automation,
- constrains auth fragility to one reusable state file,
- keeps replay, normalization, and calibration independent from live login state once files land.

---

## 13. Summary

The shortest correct framing is:

1. contest flashback is not a new simulator,
2. it is contest sim with the actual contest field substituted in as the field library,
3. the only true emulation problem is filling missing field mass when the scrape is incomplete.

That makes the implementation path practical:

- normalize nightly result CSVs,
- convert them into replay-ready weighted field libraries,
- resolve the correct lock-time worlds,
- reuse the current scoring and payout engine,
- report sim ROI/rates plus realized-vs-sim diagnostics.

This should give us a credible postmortem product with much stronger grounding than pre-lock
generated-field sim, while still fitting cleanly into the existing contest-sim architecture.

---

## 14. API and UI Surface

Flashback should remain a separate product surface from pre-lock contest sim, even though both use
the same core payout/world-scoring engine.

### 14.1 API

Initial API shape:

- `GET /api/flashback/contests`
  - list likely contests for a given `date` and `user_pattern`
- `POST /api/flashback/run`
  - run exact replay analytics for one contest and return summary + preview rows
- `POST /api/flashback/calibration/run`
  - build aggregate replay-calibration artifacts and return summary + preview rows

### 14.2 Dashboard

The dashboard should expose a separate top-level `Flashback` page, not a hidden mode inside the
existing `Contest Sim` page.

The first UI pass should support:

1. date + user-pattern contest discovery
2. exact replay execution for one contest
3. preview tables for:
   - lineup calibration
   - player calibration
   - field calibration
   - regret summary
4. trigger + preview the aggregate calibration artifacts

Rationale:

- `Contest Sim` is pre-lock and decision-time
- `Flashback` is post-lock and diagnostic/calibration-time
- inputs, semantics, and operator workflows are different enough that they should not share the same
  page state

---

## Appendix A: Replay Analytics (Merged 2026-03-06)

Source migrated from `docs/contest-sim/POST_CONTEST_REPLAY_ANALYTICS_SPEC.md`.

### Contest Flashback Analytics: Calibration and Regret Spec

## Spec Status: DRAFT v0.1 (2026-03-05)

---

## 1. Motivation

The replay module answers:

```
How did my entered lineups project against the actual contest field under our pre-lock worlds?
```

That is necessary but not sufficient.

The bigger product and modeling opportunity is:

```
What did replay reveal about the quality of our upstream player model,
world generator, field model, and lineup-selection process?
```

This spec defines the analytics layer that turns one or many replay runs into calibration and
regret datasets.

---

## 2. Goals

1. Produce replay-derived calibration artifacts that are useful for:
   - minutes calibration,
   - player FPTS/world calibration,
   - field-model calibration,
   - optimizer and final-set regret analysis.
2. Keep outputs in columnar datasets suitable for nightly batch jobs and downstream dashboards.
3. Distinguish clearly between:
   - player-model error,
   - field-model error,
   - selection/optimizer regret,
   - realized variance.

---

## 3. Non-goals

- Direct online training in this phase.
- Anchored emulation for partial contest scrapes in this phase.

The implementation now includes a first-pass dashboard surface, but this spec remains the source of
truth for which replay analytics outputs that UI should prioritize.

---

## 4. Core Principle

Replay analytics should operate on the following chain:

```
pre-lock player/world beliefs
-> actual entered field
-> entered/candidate lineup replay outcomes
-> realized contest outcomes
```

This lets us decompose slate outcomes into:

- distribution miss,
- field miss,
- selection miss,
- luck.

---

## 5. Artifact Set

Each replay analytics run writes:

1. `player_calibration.parquet`
2. `lineup_calibration.parquet`
3. `field_calibration.parquet`
4. `regret_summary.parquet`
5. `summary.json`

Optional inputs may also add:

- candidate-pool evaluation sourced from contest export manifests / `eval_lineups.csv`

---

## 6. Player Calibration Contract

One row per player on the slate union used by:

- actual contest field,
- modeled/generated field library,
- or the scored world matrix.

Required columns:

- identifiers:
  - `game_date`
  - `contest_id`
  - `draft_group_id`
  - `player_id`
  - `player_name`
  - `team`
  - `positions`
- pre-lock model state:
  - `proj_fpts`
  - `proj_ownership_pct`
  - `salary`
- actual field state:
  - `actual_contest_own_pct`
  - `actual_opponent_own_pct`
  - `modeled_field_own_pct`
  - `actual_player_fpts`
  - `actual_minutes`
- simulated world distribution:
  - `sim_mean_fpts`
  - `sim_p10_fpts`
  - `sim_p50_fpts`
  - `sim_p90_fpts`
  - `actual_fpts_sim_percentile`
  - `sim_mean_minutes`
  - `sim_p10_minutes`
  - `sim_p50_minutes`
  - `sim_p90_minutes`
  - `actual_minutes_sim_percentile`
- calibration deltas:
  - `actual_vs_modeled_own_diff_pct`
  - `actual_vs_proj_own_diff_pct`

Primary use:

- minutes model calibration,
- player distribution calibration,
- ownership/model field calibration.

---

## 7. Lineup Calibration Contract

One row per unique lineup across:

- entered user lineups,
- optional candidate/eval pool lineups.

Required columns:

- identifiers:
  - `game_date`
  - `contest_id`
  - `draft_group_id`
  - `lineup_key`
  - `lineup_source` (`entered` or `candidate`)
  - `is_entered`
- lineup membership:
  - `player_ids_json`
- realized outcome:
  - `realized_points`
  - `realized_rank`
  - `realized_prize`
  - `realized_score_sim_percentile`
- replay outcome:
  - `sim_mean`
  - `sim_std`
  - `sim_p90`
  - `sim_p95`
  - `sim_roi`
  - `sim_cash_rate`
  - `sim_top1pct_rate`
  - `sim_win_rate`
- field interaction:
  - `actual_total_dupe_count`
  - `opponent_dupe_count`
- lineup features:
  - `salary_total`
  - `salary_left`
  - `projected_own_sum`
  - `num_teams`
  - `max_from_team`
  - `num_games`
  - `max_from_game`

Primary use:

- lineup-level calibration,
- identifying over/underpriced lineup archetypes,
- diagnosing selection and optimizer misses.

---

## 8. Field Calibration Contract

One row per contest replay run.

Required columns:

- identifiers:
  - `game_date`
  - `contest_id`
  - `draft_group_id`
  - `contest_name`
- actual opponent field summaries:
  - `actual_field_size`
  - `actual_unique_lineups`
  - `actual_dupe_rate`
  - `actual_salary_total_mean`
  - `actual_salary_left_mean`
  - `actual_projected_own_sum_mean`
  - `actual_num_teams_mean`
  - `actual_max_from_team_mean`
- modeled/generated field summaries:
  - `modeled_field_version`
  - `modeled_field_size_weighted`
  - `modeled_unique_lineups`
  - `modeled_dupe_rate`
  - `modeled_salary_total_mean`
  - `modeled_salary_left_mean`
  - `modeled_projected_own_sum_mean`
  - `modeled_num_teams_mean`
  - `modeled_max_from_team_mean`
- distance metrics:
  - `player_ownership_mae_pct`
  - `player_ownership_rmse_pct`
  - `top20_player_ownership_mae_pct`
  - `salary_left_hist_l1`
  - `projected_own_sum_hist_l1`
  - `dupe_hist_l1`

Primary use:

- calibrating field library weights,
- calibrating ownership and duplication assumptions,
- contest-bucket field sharpness analysis.

---

## 9. Regret Summary Contract

One row per contest replay run.

Required columns:

- identifiers:
  - `game_date`
  - `contest_id`
  - `draft_group_id`
- entered set:
  - `entered_unique_count`
  - `best_entered_lineup_key`
  - `best_entered_sim_roi`
  - `best_entered_sim_cash_rate`
  - `best_entered_realized_rank`
  - `best_entered_realized_prize`
  - `actual_best_entered_lineup_key`
  - `actual_best_entered_rank`
  - `actual_best_entered_prize`
- candidate pool:
  - `candidate_pool_available`
  - `candidate_manifest_path`

## 9.1 Replay summary payload contract

Each flashback run should also emit a compact `summary.json` payload for the UI with:

- `counts`
- `artifacts`
- `resolution`
- `user_replay_summary`
- `decision_guidance`
- `attribution_summary`
- `field_summary`
- `regret_summary`

`resolution` should include scalar diagnostics and preview examples:

- `resolved_entry_count`
- `unresolved_entry_count`
- `resolved_slot_count`
- `unresolved_slot_count`
- `slot_resolution_rate`
- `ambiguous_name_count`
- `fuzzy_match_count`
- `unresolved_examples`
- `ambiguous_examples`
- `fuzzy_examples`

`user_replay_summary` should include compact entered-set metrics:

- `entered_lineup_count`
- `best_sim_roi`
- `avg_sim_roi`
- `best_sim_cash_rate`
- `best_realized_rank`
- `best_realized_prize`
- `avg_realized_rank`
- `avg_realized_score_sim_percentile`

`attribution_summary` should include compact diagnosis metadata:

- `primary` (e.g. `selection`, `field_model`, `projection_or_generation`, `variance`, `mixed`)
- `confidence` (`low`/`medium`/`high`)
- `reasons` (short machine-readable tags)

The dashboard should prioritize these summaries over raw parquet previews.

## 9.2 Candidate-regret provenance

When an export manifest contains `source_run_build_id`, replay analytics should use the saved
contest-sim run build as the candidate universe for regret analysis. This is preferred over
`eval_lineups.csv`, which only reflects the exported subset.

Priority order:

1. `source_run_build_id` from export manifest
2. explicit `candidate_manifest_path` override
3. `eval_lineups.csv` fallback

This changes regret semantics from export-subset regret toward true candidate-pool regret when
lineage is available.

## 9.3 Historical lineage backfill

Recent historical exports can be repaired by matching export CSV lineups back to saved portfolio
builds on the same slate.

Backfill output should stamp:

- `source_build_id`
- `source_portfolio_build_id`
- `source_run_build_id`
- `source_selection_mode`
- `lineage_backfill`

`lineage_backfill` should record:

- `matched_by`
- `matched_at_utc`
- `match_ratio`
- `exact_lineup_multiset_match`
- `created_delta_seconds`
- `mapped_export_lineup_count`
- `unmapped_export_rows`

This is acceptable for recent recovery windows, but explicit export provenance remains the
authoritative path.

## 9.4 Known replay-quality caveat

Even with correct export lineage, replay can still be degraded if the observed contest field
contains player IDs that are absent from the worlds matrix.

Implications:

- entered-lineup regret can be sourced from the correct full candidate universe,
- but opponent-field scoring may still be slightly distorted until replay field-resolution is
  constrained to canonical world/player IDs.

Recommended additional diagnostics in future summary payloads:

- `opponent_missing_player_id_count`
- `opponent_missing_player_examples`
- `candidate_missing_player_id_count`

## 9.5 Current high-priority replay analytics fixes

1. Candidate-source visibility
   - Every replay run should record whether regret came from:
     - full saved contest-sim run build,
     - saved portfolio/export lineage,
     - exported subset fallback.
   - Current UI copy should not imply true candidate-pool regret when only the exported subset was
     available.

2. Replay trust status
   - Analytics summary should classify runs into:
     - `trusted`
     - `warning`
     - `broken`
   - Example criteria:
     - missing opponent IDs in worlds,
     - impossible field feature aggregates,
     - tiny candidate universe relative to entered set,
     - unresolved lineup slots.

3. Field-feature sanity validation
   - Reject or flag runs where:
     - `actual_salary_total_mean` is implausibly low,
     - `actual_num_teams_mean` is implausibly low,
     - `actual_projected_own_sum_mean` is zero or implausible,
     - histogram distances are degenerate because the underlying features are malformed.

4. Historical export-lineage backfill coverage
   - Recent manifests were backfilled successfully when a clean saved-portfolio match existed.
   - Remaining historical exports without lineage should be treated as lower-confidence regret runs.

## 9.6 Implementation update (2026-03-06)

Implemented in code:

- replay trust classification is now emitted per run:
  - `trusted`
  - `warning`
  - `broken`
- trust checks include:
  - unresolved/outside-world slot resolution diagnostics
  - opponent missing-player IDs in worlds
  - candidate missing-player IDs in worlds
  - field-feature sanity checks
  - candidate-universe size sanity relative to entered set
- candidate-source visibility is now explicit in summary and regret outputs:
  - `candidate_universe_source`
  - `candidate_universe_lineup_count`
- candidate lineups are filtered to worlds namespace before simulation/regret scoring
- regret summary now carries human-readable lineup strings (not only IDs/keys)
- replay summary now emits `decision_guidance` and `attribution_summary` for immediate operator interpretation

Remaining high-value follow-ups:

- run historical replay reruns for `2026-02-28` through `2026-03-05` using the new trust checks
- add scorecard-level attribution rollups (projection vs field vs selection vs variance)
- strengthen lock-time provenance checks in summary payloads (`as_of_ts`, world run lineage)

## 9.7 Concrete known bad run

The replay for:

- `game_date=2026-03-03`
- `contest_id=188511762`
- `contest_name=NBA $25K mini-MAX [150 Entry Max]`

is a known bad run for interpretation.

Specific issues observed:

- `candidate_unique_count = 6`
- `entered_lineup_count = 40`
- impossible field summary aggregates
- opponent field IDs absent from worlds

This run should be used as a regression test for the replay-field repair work.

Current phase semantic note:

- `finalset == entered set`
- if a richer saved-build/final-set source appears later, the same table can distinguish them

Primary use:

- optimizer regret,
- export/final-set quality review,
- identifying where candidate quality exceeded chosen entries.

---

## 10. Data Sources

Replay analytics may use:

- raw contest results CSVs from `bronze/dk_contests/nba_gpp_data/...`
- normalized contest inventory and user-entry tables from `analytics/contest_results/...`
- replay-prepared exact field outputs
- player world matrices from contest sim world loaders
- boxscore minute labels from `labels/season=*/boxscore_labels.parquet`
- DK contest export manifests and `eval_lineups.csv` from `contests/dk/...`
- generated field libraries from contest sim cache/build manager

Serving/freshness rule:

- normalized analytics tables are the preferred index for dashboards and APIs,
- raw contest results are the freshness fallback when normalized tables lag the scrape,
- replay analytics outputs should continue to materialize normalized parquet artifacts so downstream
  calibration jobs never depend on ad hoc CSV scans.

---

## 11. Nightly Job Shape

For each played contest:

1. run exact replay,
2. write replay outputs,
3. build replay analytics artifacts,
4. append/merge into longitudinal analytics tables,
5. emit summary rows for dashboards and model audits.

Failure policy:

- exact replay must fail if the scraped field is incomplete,
- analytics should still write partial artifacts only if replay itself completed,
- candidate regret is optional and should degrade gracefully when no export manifest is found.

Control-plane rule:

- replay analytics and calibration must never depend on live DraftKings authentication,
- they operate only on landed raw bronze files and normalized derivatives,
- the only auth-dependent upstream stage is contest acquisition.

Recommended next step:

- adopt browser-state handoff for DK acquisition so nightly replay analytics runs from landed files
  rather than attempting fresh headless login on the same host.

---

## 12. Downstream Uses

## 12.1 Minutes and player-world calibration

Use player rows to check:

- whether actual minutes land in the intended percentile bands,
- whether actual FPTS land inside simulated score bands,
- which player archetypes systematically miss.

## 12.2 Field-model calibration

Use field rows to tune:

- ownership-informed weights,
- dupe modeling,
- salary-left and chalk-shape assumptions,
- contest-type-specific field priors.

## 12.3 Optimizer calibration

Use lineup and regret rows to measure:

- whether high-sim candidates existed pre-lock,
- whether entered sets missed them,
- whether the objective/constraints are suppressing profitable constructions.

---

## 13. Summary

Replay analytics is the layer that turns contest flashback from a user-facing novelty into a model
improvement system.

The correct output is not one score.

It is a structured set of player, lineup, field, and regret artifacts that let us answer:

- were our player distributions wrong,
- was our field model wrong,
- did the optimizer miss better lineups,
- or did we simply run bad?

---

## 14. Calibration Jobs Built on Replay Analytics

Replay analytics artifacts are not the final training set for core models.
They are the calibration layer on top of those models.

This phase adds four batch calibration jobs.

### 14.1 Player/world calibration job

Inputs:

- replay `player_calibration.parquet`

Outputs:

- `gold/replay_calibration/player_fpts_calibration.parquet`
- `gold/replay_calibration/player_minutes_calibration.parquet`

Method:

- bucket rows by projected FPTS and projected ownership,
- compute actual-minus-sim mean bias,
- compute `below_p10_rate`, `above_p90_rate`, and outside-band rate,
- emit:
  - `recommended_mean_shift`
  - `recommended_variance_scale`

Use:

- post-hoc calibration of player means and world dispersion
- identifying buckets with under-wide or over-wide tails

### 14.2 Ownership recalibration job

Inputs:

- replay `player_calibration.parquet`

Outputs:

- `gold/replay_calibration/ownership_recalibration.parquet`

Method:

- bucket by projected ownership,
- compare projected ownership to actual contest ownership and actual opponent ownership,
- emit:
  - `recommended_delta`
  - `recommended_multiplier`
  - `monotone_target_own`

Use:

- ownership-model recalibration
- contest-sim field weight recalibration

### 14.3 Field-model calibration job

Inputs:

- replay `field_calibration.parquet`

Outputs:

- `gold/replay_calibration/field_model_calibration.parquet`

Method:

- bucket contests by field size,
- aggregate ownership error, dupe-rate error, salary-left error, and histogram distances

Use:

- field-library generator tuning by contest bucket
- identifying where generated fields are too chalky, too unique, or salary-shape wrong

### 14.4 Optimizer regret calibration job

Inputs:

- replay `regret_summary.parquet`
- replay `lineup_calibration.parquet`

Outputs:

- `gold/replay_calibration/optimizer_regret_by_contest.parquet`
- `gold/replay_calibration/optimizer_regret_by_bucket.parquet`
- `gold/replay_calibration/optimizer_regret_examples.parquet`

Method:

- preserve contest-level regret rows,
- bucket regret by field size,
- join best entered vs best candidate lineup examples

Use:

- compare optimizer/final-set versions
- identify systematic selection misses
- separate model miss from selection miss

---

## Appendix B: Agent Handoff (Merged 2026-03-06)

Source migrated from `docs/contest-sim/FLASHBACK_AGENT_HANDOFF_2026-03-06.md`.

### Flashback Agent Handoff

## Date

`2026-03-06`

## Current state

Flashback is live and materially improved, but not fully trustworthy for every recent slate.

Working pieces:

- Flashback UI/API is live in the dashboard.
- Entered-lineup replay metrics are surfaced in the UI.
- Match-quality diagnostics are surfaced in the UI.
- Export provenance is now persisted for new exports.
- Recent export manifests were heuristically backfilled to saved portfolio and source run builds.
- Replay regret now prefers full saved contest-sim run lineage when available.

## What was fixed

1. Entered replay lineups scoring as all-zero
   - Root cause: replay produced player IDs like `1628368.0` instead of canonical `1628368`.
   - Fix: canonicalized replay player IDs.

2. Flashback UI not surfacing useful replay outputs
   - Added entered-lineup replay table, takeaways, match quality, regret summary, field summary,
     and unresolved/ambiguous/fuzzy match previews.

3. Export lineage missing
   - `EntryFileState` now records contest-sim source provenance.
   - Export manifests now persist that provenance.

4. Historical export lineage for the recent window
   - Added `projections.cli.backfill_export_lineage`.
   - Successfully matched and stamped recent export manifests.

## Recent lineage backfill result

Window backfilled:

- `2026-02-28` through `2026-03-05`

Matched manifests:

- `9`

Important exact matches on `2026-03-05`:

- `export_20260305T215413Z_2e7503` -> portfolio `08346651-e051-4be3-bb50-0dab52d4378b` -> run `932062ed-ddcc-4b82-8c63-ff6bbec277d2`
- `export_20260305T220439Z_a96a50` -> portfolio `4f4c849d-3751-4057-bca8-fd63c1874ebb` -> run `64c8aa38-e7ce-4d90-932c-a574d8930817`
- `export_20260305T232412Z_1d3e8c` -> portfolio `3acf4940-faf4-459f-99bb-d46736ee7dd9` -> run `1d6a9417-5c46-4083-b6c5-a8a7977896d4`
- `export_20260305T234815Z_f72751` -> portfolio `a7daf9eb-6741-4541-92e6-48fd3c2af371` -> run `37043623-faec-4a67-9e65-7046213261df`

## Current blocker

The main remaining blocker is replay opponent-field resolution.

### Problem

Some contest-result player names are resolving to IDs that are absent from the worlds matrix.

This corrupts:

- opponent-field scoring
- field summary aggregates
- regret interpretation

### Important nuance

The saved contest-sim source runs that were checked for `2026-03-05` were clean against the worlds
matrix.

The problem is not the saved contest-sim run lineage.
It is the replay reconstruction of the observed opponent field.

## Concrete known bad run

Bad example:

- `game_date=2026-03-03`
- `contest_id=188511762`
- `contest_name=NBA $25K mini-MAX [150 Entry Max]`

Observed issues:

- `candidate_unique_count = 6`
- user entered `40` lineups
- field summary values were physically impossible
- opponent field had IDs absent from worlds

Conclusion:

- do not trust replay takeaways or selection regret for that run
- use it as the regression test for replay-field repair

## Names/IDs observed in bad opponent fields

Examples found during investigation:

- `EJ Harkless`
- `Keshad Johnson`
- `Tolu Smith`
- `Wendell Moore Jr.`
- `Emanuel Miller`
- `Tyler Smith`
- `Rocco Zikarsky`
- `Kyrie Irving`
- `Jonathan Isaac`

These names were appearing with IDs absent from worlds in replay field construction.

## Next recommended work

1. Constrain replay field resolution to canonical world/player IDs
   - Do not accept fallback IDs absent from the worlds namespace.
   - Add diagnostics for opponent-field missing IDs.

2. Add replay trust flags
   - `trusted`
   - `warning`
   - `broken`

3. Add field-summary sanity checks
   - Fail or flag runs with impossible salary/team/ownership aggregates.

4. Rerun recent flashback slates after the field-resolution fix
   - Priority window:
     - `2026-02-28` through `2026-03-05`

5. Only then interpret slate-level attribution
   - variance
   - candidate-generation miss
   - field-model miss
   - selection miss

## Files most relevant for next agent

- [replay_service.py](/home/daniel/projects/projections-v2/projections/post_contest/replay_service.py)
- [replay_analytics_service.py](/home/daniel/projects/projections-v2/projections/post_contest/replay_analytics_service.py)
- [flashback_api.py](/home/daniel/projects/projections-v2/projections/api/flashback_api.py)
- [entry_manager_api.py](/home/daniel/projects/projections-v2/projections/api/entry_manager_api.py)
- [FlashbackPage.tsx](/home/daniel/projects/projections-v2/web/minutes-dashboard/src/pages/FlashbackPage.tsx)
- [backfill_export_lineage.py](/home/daniel/projects/projections-v2/projections/cli/backfill_export_lineage.py)
- [POST_CONTEST_FLASHBACK_SPEC.md](/home/daniel/projects/projections-v2/docs/contest-sim/POST_CONTEST_FLASHBACK_SPEC.md)
- [POST_CONTEST_REPLAY_ANALYTICS_SPEC.md](/home/daniel/projects/projections-v2/docs/contest-sim/POST_CONTEST_REPLAY_ANALYTICS_SPEC.md)
