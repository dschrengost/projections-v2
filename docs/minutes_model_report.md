# NBA Minutes Model: Data Coverage, Feature Priorities, and Modeling Plan

## Summary
- We now have reliable sources for injuries (official hourly PDFs), box scores (NBA.com JSON), vegas lines (odds), and schedules (league + daily scoreboard) across 2022–23, 2023–24, 2024–25, and the current season. This is sufficient to construct an initial, production‑oriented minutes model.
- The biggest lift remaining is feature engineering/label hygiene and a few small data gaps (play‑by‑play‑derived signals, roster/coach/arena metadata) that we can fill with lightweight scrapes/lookups.

## Feature Importance (Most → Least)
1. Availability & Status
   - Player OUT/Questionable/Probable; DNP; ramp/management flags extracted from injury reports.
   - Rationale: availability drives binary minutes and upper bounds.
2. Teammate‑Dependency Deltas
   - Per‑player minutes change when specific teammates are OUT (pair/group deltas).
   - Rationale: rotation elasticity is highly teammate‑conditional.
3. Starter/Closer Role Signals
   - Recent starts %; propensity to be in the closing five; typical stint lengths.
   - Rationale: role determines baseline minutes distribution.
4. Rest/Load & Recent Workload
   - Days since last game; back‑to‑back, 3‑in‑4, 4‑in‑6; minutes in last 1/3/5 games and last 7 days.
   - Rationale: fatigue and coaching rest patterns cap minutes.
5. Recent Minutes Trend & Volatility
   - Rolling mean/median/stdev/IQR of minutes; change vs 10‑game baseline.
   - Rationale: captures stability vs experimentation.
6. Game Environment & Blowout Risk
   - Implied total/spread transforms; home/away; pace proxy; travel/time‑zone shift.
   - Rationale: blowouts compress starters, expand bench.
7. Depth/Availability by Archetype
   - Count of available G/W/BIG by team; overlap in archetype with target player.
   - Rationale: role scarcity expands minutes.
8. Return‑From‑Injury Ramp Flags
   - Days since return; first 1–3 games back window; reason‑based heuristics.
   - Rationale: medically managed minutes.
9. Coaching Tendencies / Coach Change
   - Coach tenure; rotation tightness (minutes dispersion) under coach.
10. Foul‑Trouble Propensity
   - Early fouls (≥2 in 1Q; ≥3 by half) rate; opponent draw‑foul.
11. Garbage‑Time Sensitivity
   - Share of minutes in garbage windows historically; team garbage rate.
12. Roster/Transactions Events
   - First 1–3 games post trade/signing; rotation shake‑ups.
13. Player Profile
   - Age, height, position archetype; career injury frequency (seasonal count).

## Data Gaps To Fill (Targeted)
- Play‑By‑Play JSON (NBA.com)
  - Needed for: closing five frequency, foul‑trouble timing, garbage‑time detection, stint lengths.
  - Plan: use `playbyplayv3` per game; derive features offline with rolling windows.
- Roster/Bio Metadata
  - Needed for: age/height/position; nightly active bodies by archetype.
  - Plan: players directory endpoint + daily scoreboard actives; cache nightly.
- Arena Time Zones / Travel Heuristics
  - Needed for: travel/time‑zone shift between games.
  - Plan: static mapping of arenas→tz; compute shifts from schedule.
- Coach Tenure Map
  - Needed for: coach change flags and coach‑level dispersion priors.
  - Plan: small curated CSV; update quarterly.
- Transactions (Optional for V1)
  - Needed for: post‑trade/post‑signing adjustment flags.
  - Plan: light scrape of NBA transactions feed; fallback detection via first appearance.

## Dataset Build Plan
1. Labeling
   - Unit row: player‑game.
   - Label: official minutes from post‑game box scores.
   - Leakage control: only use features available prior to tip; rolling stats exclude current game.
2. Feature Store
   - Materialize features per player‑game: availability; role; rest/load; trend/volatility; vegas transforms; depth; ramp; coach; foul/garbage summaries.
   - Windows: 3/5/10 games and 7/14 days; clip outliers.
3. Splits & CV
   - Time‑based train/val/test by weeks/months; team‑aware folds to reduce leakage.
   - Metrics: minutes MAE; absolute error buckets (e.g., >6 min); optional quantile calibration.

## Modeling Plan
- Baseline
  - Ridge/Elastic Net regression on engineered features (fast, interpretable sanity check).
- Primary Model (V1)
  - Gradient Boosting Regression (XGBoost/LightGBM) with monotone/interaction controls where appropriate.
  - Optional two‑stage: (a) classifier for “low‑minutes” (<10) vs “normal” and (b) regressor for conditional minutes.
  - Optional quantile model for P10/P50/P90 minutes bands.
- Post‑Processing Layer
  - Team‑level reconciliation to typical game minutes (≈240) while respecting availability; soft rescaling with caps.
  - Blowout adjustment: attenuate starters when large spread and high blowout probability.
- Extensions (V2)
  - Hierarchical shrinkage by coach/team/position; stacking with simple GLM for bias correction.
  - Mixture‑of‑experts by role (starter vs bench; guard/wing/big).

## Engineering & Production Plan
1. ETL & Caching
   - Daily pulls: latest injury report + vegas + schedule; refresh roster/players cache.
   - Historical backfills by season via CLI.
2. Feature Pipeline
   - Deterministic transformations; artifact versioning by season; tests for leakage and schema drift.
3. Training & Validation
   - Time‑split CV; periodic retraining (weekly) during season; drift monitoring.
4. Inference
   - Pre‑game scoring job: produce minutes projections per player‑game; persist JSON/parquet.
5. Monitoring
   - Track MAE over rolling windows; alert on data gaps (e.g., missing injury snapshot) and schema changes.

## Open Questions / Assumptions
- Overtime handling: V1 reconciles to 240 minutes; we can detect OT ex‑post for evaluation but won’t predict it.
- Injury text normalization: maintain a small keyword map (ramp, restriction) to stabilize heuristics.
- Coach map curation cadence and source of truth.

---

Prepared for: projections‑v2 minutes modeling kickoff.

