• Data Availability Matrix (Option A: team totals + shares allocation)

  | Needed field | Avail | Best source(s) on disk | Where it’s joined/used in code | Notes |
  |---|---:|---|---|---|
  | Player-game FGA/FTA/TOV labels | ✅ | /home/daniel/projections-data/gold/rates_training_base/season=*/game_date=*/rates_training_base.parquet | (Derive) |
  rates_training_base has minutes_actual, fga2_per_min, fga3_per_min, fta_per_min, tov_per_min → derive totals: FGA=(fga2+fga3)*min, FTA=fta*min, TOV=tov*min. |
  | Team totals for FGA/FTA/TOV | ✅ | same as above | (Derive) | sum_team(x) via groupby (game_id, team_id); use to compute shares. |
  | ORB / DRB | ✅ | rates_training_base (per-min) + /home/daniel/projections-data/gold/fpts_training_base/.../fpts_training_base.parquet (game totals) | sim uses
  oreb_per_min/dreb_per_min | For shares MVP you don’t need these, but they exist. |
  | Pace / possessions | ⚠️ | rates_training_base has team_pace_szn, opp_pace_szn | — | Season pace exists; game possessions/pace not present as a direct label. |
  | Pregame minutes preds (p10/p50/p90) + play prob | ✅ | /home/daniel/projections-data/gold/projections_minutes_v1/game_date=*/minutes.parquet and also in
  rates_training_base (minutes_pred_*) | minutes inputs to sim | Sim loads minutes from minutes_v1 (see scripts/sim_v2/generate_worlds_fpts_v2.py:780). |
  | Starter / role flags | ✅ | minutes parquet + /home/daniel/projections-data/silver/roster_nightly/.../roster.parquet + rates_training_base.is_starter | minutes
  reconciliation + features | Roster has as_of_ts, starter_flag, lineup_role. |
  | Salary (DK) | ✅/⚠️ | /home/daniel/projections-data/gold/dk_salaries/site=dk/game_date=*/draft_group_id=*/salaries.parquet | merged by name in projections/
  cli/finalize_projections.py:176 | Available, but current “unified” join uses normalized player_name (not player_id) → potential mismatches/leakage-risk for
  historical training snapshots. |
  | Positions | ✅ | minutes.parquet.pos_bucket; rates_training_base.position_primary + position flags | used across stack | DK eligibility positions are in
  salaries parquet (list), NBA position is elsewhere. |
  | Injuries snapshot + timestamped “as-of” | ✅ | /home/daniel/projections-data/silver/injuries_snapshot/.../injuries_snapshot.parquet | snapshot selection
  helpers in projections/minutes_v1/snapshots.py:132 | You can select latest as_of_ts <= tip_ts to avoid leakage. |
  | Vegas total/spread + timestamped “as-of” | ✅ | /home/daniel/projections-data/silver/odds_snapshot/.../odds_snapshot.parquet | merged into minutes via
  projections/features/game_env.py:61 | minutes.parquet already carries spread_home, total, odds_as_of_ts. |
  | “Unified projections” artifact schema (final) | ✅ | /home/daniel/projections-data/artifacts/projections/YYYY-MM-DD/run=*/projections.parquet | built in
  projections/cli/finalize_projections.py:220 | Contains pts_mean/reb_mean/ast_mean/... plus salary, pred_own_pct, etc. |
  | sim_v2 outputs schema | ✅ | /home/daniel/projections-data/artifacts/sim_v2/projections/game_date=*/projections.parquet | written by scripts/sim_v2/
  generate_worlds_fpts_v2.py:1234 | Includes dk_fpts_* quantiles + pts_mean/reb_mean/... stat means. |

  Best existing data source for FGA/FTA/TOV share labels

  - Use /home/daniel/projections-data/gold/rates_training_base/.../rates_training_base.parquet (not fpts_training_base).
  - Rationale: fpts_training_base has tov but does not have FGA/FTA attempts; rates_training_base has the per-minute attempt rates + minutes_actual, which
    deterministically reconstructs attempts (validated via scripts/diagnostics/usage_share_prep.py:78).

  Gaps & minimal new datasets

  - Gaps:
      - Salary features are available, but the current “unified projections” merge is name-based (projections/cli/finalize_projections.py:220), and DK salaries
        parquet has no explicit as_of_ts → I would treat salary as “nice-to-have” for the first no-leak share model.
      - “Teammate-out aggregates” exist partially in rates_training_base (team_minutes_vacated, team_usage_vacated), but other “out flags” are currently null in
        schema (so if you want richer injury-context features, you’ll need to compute them from silver/injuries_snapshot + silver/roster_nightly).
  - Minimal new dataset to add (optional, but recommended for reproducibility): gold/usage_shares_training_base/season=*/game_date=*/
    usage_shares_training_base.parquet built from rates_training_base with derived columns fga/fga_share/fta_share/tov_share plus selected pregame-safe features.
    You can also skip materializing and train directly from rates_training_base initially.

  Where to implement in sim_v2 (reverse map) + profile knobs

  - Integration point: scripts/sim_v2/generate_worlds_fpts_v2.py:1077 where stat_totals are generated per target (currently independent per-player noise).
      - This is the minimal place to replace/override {fga2,fga3,fta,tov} generation with:
          1. sample (or compute) team total opportunities per world per team,
          2. sample per-player share vectors (softmax within team),
          3. allocate conserved totals to players (respecting active_mask + min_minutes_active_cutoff).
      - Vegas points anchoring is already applied after stat generation at scripts/sim_v2/generate_worlds_fpts_v2.py:1114.
  - Profile config loading is centralized in projections/sim_v2/config.py:69 and config/sim_v2_profiles.json:1.
  - Proposed new profile knobs (defaults OFF):
      - usage_shares.enabled: false
      - usage_shares.targets: ["fga", "fta", "tov"] (I’d do all 3 immediately; labels/features are already present and clean)
      - usage_shares.share_temperature: 1.0
      - usage_shares.share_noise_std: 0.15 (Gaussian noise on logits before softmax)
      - usage_shares.min_minutes_active_cutoff: 2.0
      - usage_shares.fallback: "rate_weighted" (fallback weights from pred_*_per_min * minutes_worlds when model/features missing)
      - (Later) team_context.enabled / team_opportunities.source to choose how to sample team totals (sum-of-player means vs explicit team-level head).

  Share model training design (within constraints)

  - Labels (from rates_training_base):
      - fga = (fga2_per_min+fga3_per_min)*minutes_actual, fta = fta_per_min*minutes_actual, tov = tov_per_min*minutes_actual
      - share_x = x_i / sum_team(x) for each (game_id, team_id).
  - Model form (practical with LGBM):
      - Train 3 independent regressors to predict log(count_x + alpha) (e.g. alpha=0.5), then at inference compute weight_i = exp(pred_i) and normalize via softmax
        within team-game to get shares.
  - MVP feature set (all pregame-safe and already present in rates_training_base):
      - minutes_pred_p50, minutes_pred_play_prob, is_starter
      - position flags / position_primary
      - season priors: season_fga_per_min, season_fta_per_min, season_tov_per_min, season_ast_per_min
      - team context: team_itt, opp_itt, spread_close, total_close
      - vacancy/team context: vac_*_szn, team_minutes_vacated, team_usage_vacated
      - role tracking: track_role_cluster, track_*_per_min_szn
  - Proposed builder script + output schema:
      - Script: scripts/usage_shares_v1/build_training_base.py (new)
      - Output columns: {season, game_date, game_id, team_id, player_id, minutes_pred_p50, is_starter, …features…, fga, fta, tov, fga_share, fta_share, tov_share}
  - Metrics:
      - Share MAE (per target), per-team KL divergence between actual vs predicted share vectors, and top-1 share accuracy (argmax within team).

  Correlations (minimal)

  - “For free” once you conserve team totals:
      - Negative within-team correlations in FGA/FTA/TOV (and thus points) because opportunities are allocated from a fixed team pool.
      - Existing enforce_team_240 already induces within-team minutes coupling; shares layer adds coupling for usage/opportunities on top.
  - Out of scope (but next-step ideas): a single latent “pace factor” per game-world and “team offense factor” per team-world to drive both opportunity totals and
    efficiency; keep out of this PR unless trivial.

  Draft config snippet (additive; defaults OFF)

  - Add under each profile in config/sim_v2_profiles.json:1:

  "usage_shares": {
    "enabled": false,
    "targets": ["fga", "fta", "tov"],
    "share_temperature": 1.0,
    "share_noise_std": 0.15,
    "min_minutes_active_cutoff": 2.0,
    "fallback": "rate_weighted"
  }

  Optional diagnostics (added)

  - New script: scripts/diagnostics/usage_share_prep.py:1
  - Runs: uv run python -m scripts.diagnostics.usage_share_prep --start-date 2025-11-03 --end-date 2025-11-03 --max-days 1