## Minutes V1 Model Roadmap (Rebuilt Dataset)

Last updated: 2025-11-21  
Scope: LightGBM quantile minutes model (`minutes_v1`) using rebuilt gold labels/features under `/home/daniel/projections-data`.

---

### 1. Lock the Data Contract

- **Gold labels**  
  - Source: `/home/daniel/projections-data/gold/labels_minutes_v1/season=YYYY/game_date=YYYY-MM-DD/labels.parquet`  
  - Built via:  
    - `uv run python -m projections.cli.build_minutes_labels --start-date <start> --end-date <end> --data-root /home/daniel/projections-data`
  - Invariants:
    - `minutes` numeric, no NaNs for completed games.
    - `starter_flag` / `starter_flag_label` present and derived from boxscore.

- **Gold features**  
  - Source: `/home/daniel/projections-data/gold/features_minutes_v1/season=YYYY/month=MM/features.parquet`  
  - Built via:  
    - `uv run python -m projections.pipelines.build_features_minutes_v1 --start-date <start> --end-date <end> --data-root /home/daniel/projections-data --season YYYY --month MM`
  - Invariants:
    - Schema `FEATURES_MINUTES_V1_SCHEMA` enforced.
    - `feature_columns` for all minutes models are the 43-column set written alongside bundles.
    - Live features (`/live/features_minutes_v1/...`) use the same `MinutesFeatureBuilder` and feature set.

- **Live contract**  
  - Live scoring (`score_minutes_v1`) and the minutes API must:
    - Use a bundle whose `feature_columns` exactly match the gold feature schema.
    - See status/return fields (`status`, `games_since_return`, `days_since_return`, `ramp_flag`, `restriction_flag`, `injury_snapshot_missing`) with the same semantics as training.

Action items:
- [x] Rebuild labels 2022–2025 with `build_minutes_labels`.
- [x] Rebuild features 2022–2025 with `build_features_minutes_v1`.
- [x] Confirm live features match bundle `feature_columns`.

---

### 2. Baseline Analysis on Rebuilt Data

Targets:
- Baseline bundle: `artifacts/minutes_v1/v1_full_calibration`.
- New bundles: `v1_full_history_rebuilt_20251121`, `v1_rebuilt_prodconfig_20251121`.

Steps:
1. **Static metrics comparison**
   - Read `metrics.json` for each bundle and track:
     - `val_mae_p50`, `val_mae_0_10`, `val_mae_10_20`, `val_mae_20_30`, `val_mae_30_plus`.
     - `val_p10_cond_playable`, `val_p90_cond_playable`, `val_winkler_cond_playable`.
   - Source: `artifacts/minutes_v1/<run_id>/metrics.json`.

2. **Sequential backtest comparison (live-like window)**  
   - Command pattern:
     - `uv run python -m projections.cli.sequential_backtest --start 2025-10-01 --end 2025-11-18 --run-id <run_id> --data-root /home/daniel/projections-data --features /home/daniel/projections-data/gold/features_minutes_v1 --artifact-root artifacts/minutes_v1 --reports-root reports/minutes_v1[_suffix] --target-col minutes --window-days 21 --min-n 400 --history-months 1`
   - Inspect:
     - `.../rolling_backtest_summary.json` for P10/P90 overall/daily coverage and Winkler.
     - `p10_coverage_daily.csv`, `p90_coverage_daily.csv` for temporal behavior.

3. **Bucket-level error and coverage**
   - Use `metrics.json` + backtest outputs to compare:
     - Starters vs bench (`starter` buckets).
     - Minutes buckets (`0–10`, `10–20`, `20–30`, `30+`).
     - Status buckets (AVAILABLE/OUT/Q/PROB/UNKNOWN) and `injury_snapshot_missing`.

Deliverable:
- A short written comparison (can live alongside this doc) summarizing where `v1_rebuilt_*` is better/worse than `v1_full_calibration`.
- Latest run summary: `docs/minutes_v1_model_roadmap_results.md` (2025-11-21 experiments).

---

### 3. Experiments: Training Window and Data Selection

Goal: Reduce MAE and improve bucket coverage by choosing better training/calibration windows on the rebuilt dataset.

Proposed variants (each with its own `run_id`):

1. **Full-history (baseline on rebuilt data)**  
   - Already done: `v1_rebuilt_prodconfig_20251121`
   - Train: 2022-10-01 → 2024-04-30  
   - Cal:   2024-05-01 → 2025-05-31  
   - Val:   2025-06-01 → 2025-11-18  
   - Use as a reference point only (not production).

2. **Recent-only history**
   - Train: 2023-10-24 → 2025-04-30 (drop 2022 season).  
   - Cal / Val: same as above.
   - Rationale: earlier seasons (2022–23) may have systematically different rotations/pace; model might benefit from focusing on the recent minutes distribution.

3. **Very recent (current pipeline only)**
   - Train: 2024-10-22 → 2025-04-13 (i.e., rebuilt 2024–25 window).  
   - Cal: 2025-04-14 → 2025-05-31.  
   - Val: 2025-06-01 → 2025-11-18.
   - Rationale: match the label/feature pipeline and current NBA dynamics as closely as possible, at the cost of fewer samples.

Implementation details:
- Use `python -m projections.models.minutes_lgbm` with:
  - `--data-root /home/daniel/projections-data`
  - `--features /home/daniel/projections-data/gold/features_minutes_v1`
  - `--artifact-root artifacts/minutes_v1`
  - Appropriate `--train-start/--train-end/--cal-start/--cal-end/--val-start/--val-end`.
  - Fixed conformal settings: `--conformal-buckets starter,p50bins --conformal-k 200 --conformal-mode two-sided --playable-min-p50 10.0`.

Evaluation:
- For each new run_id: compute `metrics.json` and run `sequential_backtest` over 2025-10–11 (and optionally over another month).
- Select the best window setup before tuning hyperparameters.

---

### 4. Experiments: Model & Calibration Tuning

Once a reasonable training window is chosen, explore:

1. **LightGBM hyperparameters (low-risk sweeps)**
   - Baseline: defaults in `minutes_lgbm.py` (e.g. `max_depth=7`, `n_estimators=500`, `learning_rate=0.03`).
   - Sweeps:
     - Shallower trees: `max_depth=5–6`, maybe `n_estimators=600–800`.
     - Slightly higher `learning_rate` for faster fit, with fewer trees.
   - Objective: improve MAE, especially for starters and higher-minute buckets, without destroying coverage.

2. **Conformal/coverage tuning**
   - `conformal_k`: experiment with 100, 150 vs baseline 200 to reduce over‑conservatism in P10.
   - `conformal_buckets`:
     - Compare `starter,p50bins` vs `starter,p50bins,injury_snapshot` if we want injury-aware tails.
   - Keep `coverage_tolerance` at 0.02 for production, but allow relaxed runs (`--allow-guard-failure`) for exploratory training.

3. **Return-from-injury handling**
   - We now expose `games_since_return`, `days_since_return`, `ramp_flag`, `restriction_flag` consistently.
   - Experiments:
     - Upweight or oversample rows with `games_since_return==0` in training.
     - Alternatively, add a simple post-model adjustment (e.g., a small positive shift for `games_since_return==0` starters) and then re-run conformal calibration.

Deliverable:
- One or two tuned bundles that beat the chosen baseline window on both MAE and coverage criteria.

---

### 5. Error Analysis & Sanity Checks

For any promising bundle:

- **Player-level case studies**
  - For a handful of players (e.g., Darius Garland in the recent slate), inspect:
    - Features: starter flags, status, games_since_return, trend minutes.
    - Predictions: p10/p50/p90 vs realized minutes.
  - Confirm behavior is reasonable for:
    - Starters with full history.
    - Bench players.
    - Recently-returned players.

- **Bucket sanity**
  - Use `metrics.json` and backtest to verify:
    - Starters `>18` minutes: MAE and coverage acceptable.
    - Bench `10–20` minutes: not systematically under/over.
    - Status buckets (OUT/Q/PROB/UNK/AVAIL) behave as expected.

---

### 6. Deployment Strategy

1. **Shadow mode**
   - Wire a new bundle (e.g., `v2_minutes_...`) into a shadow scoring path:
     - Run alongside `v1_full_calibration` and write outputs to a separate `prediction_logs_minutes_v1` path or additional columns.
   - Compare daily distributions, per-game team totals, and key players vs current live.

2. **Promotion**
   - Once the new model:
     - Beats `v1_full_calibration` on MAE and coverage.
     - Looks sane in shadow mode on several slates.
   - Update:
     - `config/minutes_current_run.json` (if used) to point to the new bundle.
     - Systemd env (`LIVE_BUNDLE_DIR`) as needed.
   - Keep `v1_full_calibration` around for quick rollback.

3. **Monitoring**
   - Extend `projections.cli.check_health` or add a small checker to:
     - Monitor daily p10/p90 coverage.
     - Flag days with extreme team totals (e.g., far from 240 when reconciliation is enabled).
     - Log basic drift metrics (mean/std of key features).

---

### 7. Summary of Concrete Next Steps

1. Choose and run **2–3 training-window configs** on the rebuilt features (Section 3).
2. For the best window, run a small **hyperparameter + conformal sweep** (Section 4).
3. For top 1–2 candidates:
   - Compare `metrics.json` against `v1_full_calibration`.
   - Run `sequential_backtest` and sanity-check key players/games (Section 5).
4. Put the leading candidate into **shadow mode**, then promote if it clearly dominates the baseline on your agreed metrics (Section 6).
