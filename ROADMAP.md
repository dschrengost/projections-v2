# projections-v2 — Minutes V1 ROADMAP (Smoke Slice → V1)
Version: 1.0 — 2025-11-08
Owner: projections‑v2

**Goal:** land a *real-data* smoke slice (one calendar month), minimal features, models, and reconciliation with hard anti‑leak gates. Each item is a small PR with tight acceptance criteria so agents don’t blow context.

**Reference:** See `minutes_v1_spec.md` for full contracts and definitions.

---

## Scope for Smoke Slice
- **Window:** 2024‑12‑01 → 2024‑12‑31 (2024‑25 season)
- **Outputs required by the end of PR‑3:**
  - Bronze/Silver: `injuries_raw`, `odds_raw`, `schedule`, `roster_nightly`, `injuries_snapshot`, `odds_snapshot`
  - Labels (immutable): `data/labels/season=2024/boxscore_labels.parquet` (hash stored)
  - Gold features: `data/gold/features_minutes_v1/season=2024/month=12/*.parquet`
  - Models + preds + reconciliation + metrics + HTML report for the window

---

## PR‑1 — “Smoke slice (real data) with snapshots + anti‑leak gates”
**Title:** `PR: smoke-slice-dec2024 — snapshots + anti-leak + labels freeze`

**Motivation:** Synthetic is fine for unit tests, not for provenance/leakage. This PR proves the pipeline works on real data for a bounded month.

### Deliverables
- **Bronze:** `data/bronze/injuries_raw/season=2024/date=YYYY-MM-DD/injuries.parquet`, `data/bronze/odds_raw/season=2024/date=YYYY-MM-DD/odds.parquet`
- **Silver:**  
  - `data/silver/schedule/*.parquet`  
  - `data/silver/injuries_snapshot/season=2024/month=12/*.parquet` (latest `as_of_ts ≤ tip_ts`)  
  - `data/silver/odds_snapshot/season=2024/month=12/*.parquet` (same rule)  
  - `data/silver/roster_nightly/season=2024/month=12/*.parquet`
- **Labels (frozen):** `data/labels/season=2024/boxscore_labels.parquet` + stored content hash file `boxscore_labels.hash`
- **Tests:**  
  - `tests/test_snapshot_selection.py` (property tests)  
  - `tests/test_anti_leak_asof.py` (assert all `*_as_of_ts ≤ tip_ts`)  
  - `tests/test_pk_uniqueness.py` (PKs from spec §1)  
  - `tests/test_schema_contract.py` (pandera/pydantic schema validation)
- **Coverage report:** `reports/minutes_v1/2024-12/coverage.csv` with:  
  - `% games with injuries snapshot` (target ≥ 95%)  
  - `% games with odds snapshot` (target ≥ 95%)  
  - `% active players with a feature row` (will be N/A until PR‑2; include games/players counts now)

### Touchpoints / Paths
- `src/projections/utils/asof_join.py`
- `src/projections/etl/snapshots/{injuries.py,odds.py}`
- `src/projections/etl/{schedule.py,roster_nightly.py}`
- `src/projections/labels/freeze.py`
- `.github/workflows/minutes_v1.yml` (CI: run tests)

### Acceptance Criteria
- All snapshot selections **deterministic** and obey `latest as_of_ts ≤ tip_ts`.
- Anti‑leak tests pass on Dec‑2024 slice (no feature/snapshot timestamps after tip).
- Labels parquet present, hash stored; re-read yields unchanged hash.
- PK uniqueness & schema tests pass.

### Suggested Commands
```bash
# Build snapshots for Dec-2024
uv run python -m projections.etl.snapshots.injuries --start 2024-12-01 --end 2024-12-31
uv run python -m projections.etl.snapshots.odds --start 2024-12-01 --end 2024-12-31

# Freeze labels (season-level, still valid for month)
uv run python -m projections.labels.freeze --season 2024 --out data/labels/season=2024/boxscore_labels.parquet
```

---

## PR‑2 — “Minimal feature set on smoke slice”
**Title:** `PR: features-core-dec2024 — availability/role/rest/trend/env/depth/return/coach`

**Motivation:** Ship only high‑signal, low‑risk features to validate end‑to‑end training.

### Deliverables
- Feature modules:
  - `src/projections/features/availability.py` (status → prior_play_prob; OUT/Q/PROB flags)
  - `src/projections/features/role.py` (recent_start_pct_10, starter_last_game)
  - `src/projections/features/rest.py` (days_since_last, b2b, 3in4, 4in6, min_last1/3/5, sum_min_7d)
  - `src/projections/features/trend.py` (roll_mean_3/5/10, roll_iqr_5, z_vs_10) — **asof, no same-day leak**
  - `src/projections/features/game_env.py` (home_flag, spread_home, total, blowout_index)
  - `src/projections/features/depth.py` (available_G/W/B, same_archetype_overlap)
  - `src/projections/features/return_ramp.py` (days_since_return, games_since_return, restriction/ramp flags via YAML)
  - `src/projections/features/coach.py` (coach_tenure_days, team_minutes_dispersion_prior)
- Assembler:
  - `src/projections/pipelines/build_features_minutes_v1.py`
- Output:
  - `data/gold/features_minutes_v1/season=2024/month=12/*.parquet`
- Tests:
  - Unit fixtures for each module verifying math and timestamp rules
  - End‑to‑end assertion: **100%** of rows have `feature_as_of_ts ≤ tip_ts`
  - Schema check: exact columns per spec §3 (fail on missing/extra)

### Acceptance Criteria
- Feature parquet exists for Dec‑2024; row coverage ≥ 95% of active player‑games.
- Provenance columns present (`injury_as_of_ts`, `odds_as_of_ts`, `feature_as_of_ts`).
- All feature tests pass; assembler enforces anti‑leak rule.

### Suggested Commands
```bash
uv run python -m projections.pipelines.build_features_minutes_v1 --start 2024-12-01 --end 2024-12-31
```

---

## PR‑3 — “Train, quantiles + conformal, reconciliation, and report (smoke)”
**Title:** `PR: train-dec2024 — ridge canary, LGBM p50+quantiles, conformal, L2 reconcile, report`

**Motivation:** Demonstrate real MAE, calibrated bands, and team‑total consistency on a real month.

### Deliverables
- **Baseline:** `src/projections/models/minutes_baseline.py` (ridge/EN) → `artifacts/minutes_v1/<run_id>/metrics.json`
- **Primary:** `src/projections/models/minutes_lgbm.py` (p50) with feature list hash & params in `meta.json`
- **Quantiles:** pinball (0.1/0.9) + conformal calibration → calibrated P10/P90
- **Reconciliation:** `src/projections/post/reconcile_minutes.py` + tests (`tests/test_reconcile_minutes.py`)
- **Predictions:** `data/preds/minutes_v1/2024-12/minutes_pred.parquet` with:
  - keys, raw p50, reconciled p50, calibrated p10/p90, provenance, run_id
- **Metrics:** `reports/minutes_v1/2024-12/metrics.csv` (MAE overall/starters/bench, P(|err|>6), coverage)
- **HTML Summary:** `reports/minutes_v1/2024-12/summary.html`

### Acceptance Criteria
- LGBM **beats** ridge on validation MAE (Dec‑2024 split: first ~3 weeks train, last ~1 week val).
- Conformalized P10≈0.10 and P90≈0.90 within ±2% on val.
- Reconciliation unit tests pass; team sums within ±0.01 of 240; no cap violations; L2 error decreases.
- `metrics.csv` and `summary.html` produced; summary opens locally.

### Suggested Commands
```bash
# Train (smoke split)
uv run python -m projections.models.minutes_baseline --start 2024-12-01 --end 2024-12-24 --val-end 2024-12-31 --run-id v1_dec_smoke_baseline
uv run python -m projections.models.minutes_lgbm --start 2024-12-01 --end 2024-12-24 --val-end 2024-12-31 --run-id v1_dec_smoke_lgbm

# Reusable YAML-driven train
uv run python -m projections.models.minutes_lgbm --config config/minutes_training_example.yaml

# Score & reconcile
uv run python -m projections.cli.score_minutes --start 2024-12-01 --end 2024-12-31 --run-id v1_dec_smoke_lgbm

# Emit metrics & HTML
uv run python -m projections.metrics.minutes_metrics --month 2024-12
```

---

## Optional Follow‑Ups (after PR‑3)
1. **Teammate‑dependency deltas (shrunk)** — `src/projections/features/teammate_deltas.py` (ridge‑shrunk; emit only if effective N≥K).  
2. **Blowout multipliers learning** — Bound α/β and show validation improvement on high‑spread bucket.  
3. **Ablations switch** — `--drop-group availability|role|rest|trend|env|depth|return|coach` → write `ablations.csv` with ΔMAE.

---

## PR Template (paste into GitHub)
```
### Summary
<what this PR does in one paragraph>

### Scope
- [ ] Limited to Dec-2024 smoke slice
- [ ] No changes to unrelated modules

### Outputs
- [ ] Files written to the expected paths (listed)
- [ ] Artifacts include run_id and feature list hash where applicable

### Tests
- [ ] Unit tests added
- [ ] Anti-leak and snapshot selection tests pass
- [ ] PK uniqueness & schema validations pass

### Acceptance Criteria
- <copy exact AC from the section above>
```

---

## Artifact Tree (expected after PR‑3)
```
data/
  bronze/
    injuries_raw/season=2024/*.parquet
    odds_raw/season=2024/*.parquet
  silver/
    schedule/*.parquet
    injuries_snapshot/season=2024/month=12/*.parquet
    odds_snapshot/season=2024/month=12/*.parquet
    roster_nightly/season=2024/month=12/*.parquet
  labels/
    season=2024/boxscore_labels.parquet
    season=2024/boxscore_labels.hash
  gold/
    features_minutes_v1/season=2024/month=12/*.parquet
  preds/
    minutes_v1/2024-12/minutes_pred.parquet
reports/
  minutes_v1/2024-12/metrics.csv
  minutes_v1/2024-12/summary.html
artifacts/
  minutes_v1/<run_id>/...
```

---

## Guardrails (do not merge if violated)
- Any `*_as_of_ts > tip_ts` in features or snapshots
- Labels parquet mutated after hash stored
- Missing provenance columns (`as_of_ts`, `feature_as_of_ts`, `run_id`, feature list hash)
- CI fails on schema or anti‑leak tests
