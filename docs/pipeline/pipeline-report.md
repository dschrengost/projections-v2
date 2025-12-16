• Here’s a concise report on the current gaps and why the FPTS outputs are capped around ~35 even with Jokic on the slate.

  What’s broken / gaps

  - Wrong feature source for live rates: score_rates_live is fed live/features_minutes_v1 (74 cols) instead of a rates
    feature set. The model run rates_v1_stage3_clean_20251202_144334 expects the 41-stage3 feature columns (minutes_pred_,
    season_ per-minute stats, tracking, vacancy, pace/ratings, odds). Almost all of those are missing.
  - Silent zero-filling: In score_rates_live, any missing model features are created and filled with 0 before predict_rates.
    With missing season/track/vacancy/odds columns, every per-minute rate collapses toward 0. Example: Jokic rates row
    (run=20251204T013000Z) has pred_fga2_per_min ~0.126, pred_fga3_per_min ~0.069, pred_fta_per_min ~0.08, ast ~0.06, dreb
    ~0.12 — orders of magnitude too low for a 2+ fpm player, because inputs are zeroed.
  - Live feature coverage mismatch: There is no live minutes_for_rates build for 2025-12-03 (gold/minutes_for_rates stops at
    earlier dates). We’re using minutes_v1 features which lack all the season context columns.
  - Past missing join key: game_date was absent in rates_v1_live; fixed now, but not the primary issue for low FPTS.
  - Output symptom: After fixing join types, worlds/projections show 230 players (minutes rows), but fpts_mean max ≈35
    because per-minute rates are near zero for everyone.

  Recommendations to harden (no patches applied here)

  1. Build a proper live rates feature pipeline:
      - Mirror the training/rates_training_base schema (stage3_context feature set). Use the same transformations:
        minutes_pred_p50/spread/play_prob from minutes_for_rates, season_* per-minute aggregates, spread/total/team_itt/
        opp_itt, tracking features, vacancy features, pace/offensive/defensive ratings, odds flags.
      - Write to a live path (e.g., live/features_rates_v1/<date>/run=<id>/features.parquet) and point score_rates_live to
        it.
      - Ensure feature columns exactly match feature_cols.json for the model run.
  2. Make missing feature handling strict:
      - In score_rates_live, instead of zero-filling, raise if any required feature is missing. If certain columns
        legitimately can be NA, add an explicit imputation strategy consistent with training (e.g., fill with medians used in
        training, not zeros).
      - Add a preflight check comparing live features cols vs model feature_cols.
  3. Add sanity checks on rates outputs:
      - Per-day guardrails: reject/alert if per-minute rates median/max are implausibly low (e.g., fga2_per_min median <0.2)
        or if top predicted fpts_mean < some threshold (e.g., 45–50 on a full slate).
      - Alert if number of rows < expected (e.g., minutes rows) or if the join indicator counts show large right_only sets.
  4. Ensure data freshness for context features:
      - Keep season aggregates, tracking, vacancy, and odds inputs up to date and available to the live feature builder. Add
        checks that all players have non-null season stats; otherwise fall back to historical medians, not zeros.