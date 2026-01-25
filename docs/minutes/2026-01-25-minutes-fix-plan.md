# Minutes Fix Plan (Transformer-First) — 2026-01-25

This document is the living plan to get minutes predictions back to “production-grade” after the recent model promotion.
The intent is to keep us aligned on **what we’re fixing**, **why it broke**, and **how we’ll ship a durable solution**.

## 0) TL;DR

- Make the **rotation-set transformer** the canonical minutes allocator (p50 + rotation realism) instead of relying on caps/reconciliation to shape outputs.
- Run a hard **training↔inference parity audit** (features, joins, semantics, imputations) and add **guardrails/observability** that detect drift and silent fallbacks.
- Fix **availability/injuries** semantics so `play_prob` is trustworthy (DNP-CD and OUT handling).
- Fix **odds + schedule/team metadata propagation** so the dashboard shows team names and Vegas lines correctly.
- Update the simulator to generate minutes worlds in **model space** (logits/share space + game script / garbage-time mechanism) so deep-bench minutes happen only in the worlds where they should.

## 1) Current Symptoms (as observed)

- **Minutes predictions are too flat**: most “active” players get non-trivial minutes; top-end looks suppressed.
- **No projected starters**: output looks like “rotation + deep bench” rather than a plausible starting 5 + rotation.
- **Team names show as IDs** in the dashboard (team identity join/regression).
- **No Vegas lines** visible (odds attach/regression).
- **Too many DNP-CD candidates get meaningful minutes**; injuries/availability may be broken.
- Prefect flow sometimes fails in sim with `KeyError: 'game_date'` (artifact schema regression).

## 2) Working Assumptions / Constraints

- We treat this as a **pipeline + parity** problem first, not “model is bad” by default.
- We want to minimize arbitrary caps and reconciliation that override the model signal.
- Production is driven by config pointers under `config/*_current_run.json` and the Prefect orchestrator `prefect_flows/live_nba_pipeline.py`.
- Any code/config changes should be shipped via branch → commit → PR, with a safe rollback path.

## 3) What We Know So Far (high-confidence findings)

### 3.1 The rotation-set transformer itself is not necessarily “flat”
When the rotation-set model is run directly on built live features, it can produce a plausible sparse rotation and non-flat minutes totals that sum to 240 per team.

### 3.2 It can be bypassed (or suppressed) by pipeline selection + guardrails
There are two common ways a good allocator becomes “flat” in production:

1) **Selection bypass**: the pipeline uses another scorer (e.g. RMH) and never calls rotation-set.
2) **Guardrail suppression**: the overlay/blend/fallback logic de-weights or overrides rotation-set predictions due to a “coverage” heuristic or missing fields.

Concrete pointers:
- `prefect_flows/live_nba_pipeline.py` decides which minutes scoring path is active.
- `projections/cli/score_minutes_rotation_set_v1.py` applies the overlay and guardrails.
- `projections/rotation/guardrails.py` can blend/fallback aggressively if it thinks inputs are missing or unreliable.

## 4) Target End-State Architecture

We want minutes to come from a probabilistic process that is realistic and sim-friendly:

1) **Player pool + metadata** (schedule/team/opponent/positions) is correct.
2) **Availability model** outputs `play_prob` and status features that match training semantics.
3) **Rotation allocator** (transformer) outputs a sparse, team-normalized minutes mean (`minutes_p50` / `minutes_mean`) that already satisfies team totals.
4) **World generation** samples minutes in model space and adds “blowout/garbage-time” behavior explicitly, rather than via global flattening.

### Why transformer-first?
Minutes are fundamentally a *team allocation* problem with strong within-team constraints and substitution covariance. A set-based model that operates at the (team, game) set level and normalizes to 240 minutes is structurally aligned with the domain.

## 5) Workstreams

### Workstream A — Confirm what production is actually running
**Outcome**: We can answer “which model produced this slate” deterministically.

Checklist:
- Trace `config/*_current_run.json` usage in the live flow.
- Confirm whether RMH minutes scoring is enabled and whether it bypasses rotation-set scoring.
- Confirm the minutes artifact being served to the dashboard:
  - `artifacts/minutes_v1/daily/<date>/run=<id>/effective_minutes.parquet`
  - `gold/projections_minutes_v1/game_date=<date>/run=<id>/...`
- Verify minutes artifacts contain required schema for downstream consumers, including `game_date`.

Artifacts to add (if missing):
- A clear `minutes_alloc_mode` / `minutes_model_name` / `bundle_id` stamp in minutes outputs.
- A short diagnostics JSON written next to `minutes.parquet` summarizing:
  - join coverage rates (odds, injuries, starters)
  - guardrail triggers and fallback counts
  - team sum checks and rotation size metrics

### Workstream B — Training↔Inference parity audit (hard gate)
**Outcome**: We can prove inference inputs match what the transformer was trained on.

For the transformer `model_dir` (see `config/rotation_set_minutes_live.json`):
- Load `feature_columns.json` and compare to live-built features.
- Validate:
  - Column presence
  - dtypes (categorical vs numeric, ints vs floats)
  - missingness patterns (systematic NaNs indicate join regression)
  - value ranges (e.g., spread range, totals, priors)
  - key semantics (`team_id`, `opponent_id`, `game_id` normalization)

Audit outputs (must be written and kept for incident response):
- `missing_columns` (required by model but not present)
- `extra_columns` (present at inference but not used; ok but track)
- `drift_report`:
  - % NaN by column
  - summary stats by column (p01/p50/p99)
  - compare to training reference stats if available

Recommended tooling:
- Extend/standardize `scripts/diagnostics/rotation_set_minutes_live_pathology.py` to emit a parity report artifact.

### Workstream C — Fix schedule/team metadata propagation (dashboard “IDs” regression)
**Outcome**: Dashboard shows proper team names/codes, opponent, home/away, etc.

Likely root causes:
- Schedule join missing/failed: `silver/schedule/.../schedule.parquet`
- Team dimension not attached or overwritten incorrectly
- Wrong keys used (e.g., `team_id` vs `team_code` vs `tricode`)

Concrete checks:
- Verify schedule partitions exist for the target date/season:
  - `silver/schedule/season=YYYY/month=MM/schedule.parquet`
- Ensure minutes outputs carry human-readable team fields the dashboard expects (or the API attaches them reliably).

### Workstream D — Fix odds (Vegas lines) end-to-end
**Outcome**: Vegas lines exist in minutes features, minutes predictions, and dashboard API payloads.

Checklist:
- Validate silver odds snapshots exist and are non-empty:
  - `silver/odds_snapshot/season=YYYY/month=MM/odds_snapshot.parquet`
- Validate `spread_home`, `total`, and `odds_as_of_ts` are attached during feature building.
- Verify these columns propagate to `minutes.parquet` / `effective_minutes.parquet` and are consumed in the API/dashboard.

Related code:
- `projections/features/game_env.py` (odds attach)
- `projections/etl/odds.py` (odds raw + snapshot)

### Workstream E — Fix availability/injuries semantics (DNP-CD + OUT realism)
**Outcome**: `play_prob` and status flags match training semantics and suppress DNP/CD appropriately.

Checklist:
- Verify injury snapshot ingestion is current and correctly joined.
- Verify the meaning of “missing injury record” vs “healthy” is consistent.
- Ensure pre-tip status snapshots are used consistently (avoid leaking post-tip status).
- Validate DNP-history features are computed correctly and are actually used by the model at inference.

Deliverables:
- A “play_prob audit” report per slate:
  - distribution of `play_prob` by status bucket
  - top DNP-CD risk players and their assigned play_prob/minutes
  - count of players with `play_prob > x` but `minutes_p50 < y` (and vice versa)

### Workstream F — Make transformer the canonical allocator (remove suppression)
**Outcome**: Minutes p50 reflect transformer signal; guardrails only protect against true pathologies.

Steps:
1) Ensure the live scoring path calls the transformer when enabled (no bypass).
2) Replace any “coverage” heuristic that incorrectly triggers blend/fallback.
3) Reduce “blend-to-baseline” usage:
   - Blending should be a last-resort fallback, not the default pre-lineup behavior.
4) Keep only safety constraints that do not flatten:
   - sum-to-240 (already handled by model)
   - non-negativity
   - hard-zero OUT players (via play_prob/status)
   - minimal feasibility fallback when feature rows are missing

### Workstream G — Probabilistic worlds without global flattening
**Outcome**: The sim gets realistic variance and deep bench minutes in blowout worlds without “everyone gets 5+”.

Principles:
- Sample in **model space** (gate/share logits + low-rank team factors), then normalize to 240.
- Model “deep bench minutes” via an explicit **game script / garbage-time pool**:
  - In close games: pool ≈ 0
  - In blowouts: pool material; star minutes down, bench minutes up

Implementation options (ordered by time-to-value):
1) **Two-pool allocator** (recommended):
   - Transformer predicts competitive allocation.
   - Separate learned (or structured) allocator distributes a garbage-time pool to bench players.
2) **Script-conditioned transformer**:
   - Add script latent (margin bucket / blowout risk) as an input and/or mixture head.
3) **Stint/lineup process** (long-term “gold standard”):
   - Model substitution events and stint durations; minutes fall out naturally.

Calibration:
- Fit spread→margin distribution from historical outcomes (see `scripts/sim_v2/calibrate_vegas_env.py`).
- Fit margin→garbage-time pool on historical minutes (deep bench minutes vs |margin|).

### Workstream H — Evaluation + Success Criteria
**Outcome**: We have objective metrics to confirm we’re better.

Core metrics (minutes):
- MAE / pinball loss on minutes (p50 and quantiles if we emit them)
- Team sum integrity (mean and tail deviations from 240)
- Rotation realism:
  - `top8_share`, `K(minutes>0)`, `K(minutes>=2)`, entropy
  - starter vs bench distributions

Availability metrics:
- OUT recall (players who are OUT should have near-zero `play_prob` and minutes)
- DNP-CD false positives (players who should be near-zero minutes but aren’t)

DFS utility metrics:
- downstream sim stability (variance realism)
- contest-level backtests: ROI/exposure sanity against historical slates

## 6) Concrete Near-Term Milestones

### Milestone 1 — Stop the bleeding (1–2 days)
- Prove which model is producing production minutes.
- Restore missing team/odds fields end-to-end.
- Ensure injuries feed is current and `play_prob` isn’t silently defaulting to 1.0.
- Remove any guardrail logic that forces broad blending/fallback in normal conditions.

### Milestone 2 — Transformer canonical minutes (3–7 days)
- Transformer writes `minutes.p50` as the canonical minutes allocator in the standard artifact location.
- Add parity + coverage diagnostics next to artifacts.
- Ship a config-controlled fallback to baseline only when required columns are missing.

### Milestone 3 — Model-native worlds (1–2 weeks)
- Implement logit-space noise + garbage-time pool worlds.
- Calibrate game scripts and garbage-time pool against historical outcomes.
- Feed worlds directly into sim and derive quantiles for dashboards.

## 7) Key Files / Entry Points

- Live pipeline orchestrator: `prefect_flows/live_nba_pipeline.py`
- Rotation-set config: `config/rotation_set_minutes_live.json`
- RMH config (if in use): `config/rmh_current_run.json`
- Rotation-set scoring entry: `projections/cli/score_minutes_rotation_set_v1.py`
- Rotation-set model: `projections/rotation/set_model.py`
- Guardrails: `projections/rotation/guardrails.py`
- Minutes sim live runner: `scripts/sim_v2/run_sim_live.py`
- Sim worlds generator: `scripts/sim_v2/generate_worlds_fpts_v2.py`
- Game scripts module: `projections/sim_v2/game_script.py`
- Odds attach: `projections/features/game_env.py`

## 8) Open Questions (to resolve explicitly)

- What should be the “contract” for starters when no confirmed lineup exists?
  - (A) Predict a probabilistic starter flag, or
  - (B) Derive starter priors from historical + coach tendencies + roster/injury context.
- Should the transformer emit `play_prob` itself (multi-task), or do we keep `play_prob` separate and treat it as an upstream dependency?
- What is the acceptable fallback behavior when injuries/odds feeds are missing?
  - (A) fail-fast in production, or
  - (B) degrade gracefully but loudly (artifact diagnostics + alerting)?

## 9) Appendix: Suggested Debug Commands

These are “do not lose the plot” commands for incident response:

- Minutes artifact provenance (per date):
  - Inspect `artifacts/minutes_v1/daily/<date>/run=<id>/effective_minutes.parquet`
  - Verify `game_date`, `team_id`, odds columns, and model stamps.

- Rotation-set live pathology (example):
  - `uv run python -m scripts.diagnostics.rotation_set_minutes_live_pathology --date 2026-01-24 --run-id <run_id>`

- Vegas environment calibration helper:
  - `uv run python -m scripts.sim_v2.calibrate_vegas_env --help`

