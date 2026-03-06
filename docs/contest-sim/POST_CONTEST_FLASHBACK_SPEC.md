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

## 5.4 Why this is credible

The field composition problem is much easier post contest than pre contest because:

- we know exactly which slate and payout structure ran,
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
