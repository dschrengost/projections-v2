# Contest Flashback Analytics: Calibration and Regret Spec

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
