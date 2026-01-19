# Audit Findings: `play_prob` provenance + ownership/dupe sensitivity

This is an evidence-first audit with minimal, knobbed patches. All diagnostics are read-only scripts under `scripts/diagnostics/`.

## A) Where does `play_prob` come from, and does it affect worlds?

### A1) Provenance (live pipeline wiring)

**Primary producer (minutes scoring):**
- `projections/cli/score_minutes_v1.py:_score_rows` → `projections/models/minutes_lgbm.py:predict_play_probability`
- Output location (live minutes): `<DATA_ROOT>/artifacts/minutes_v1/daily/<DATE>/run=<RUN_ID>/{minutes.parquet,effective_minutes.parquet}`

**Carried into unified projections:**
- `projections/cli/finalize_projections.py` (merges minutes + sim outputs) writes:
  - `<DATA_ROOT>/artifacts/projections/<DATE>/run=<RUN_ID>/projections.parquet`

**Consumed by worlds generator:**
- `scripts/sim_v2/generate_worlds_fpts_v2.py:main`
  - if `profile_cfg.use_play_prob_masking=True`, samples `active_mask ~ Bernoulli(play_prob)` per player/world.

**Hard evidence (single-slate trace, includes per-stage values):**
```bash
uv run python scripts/diagnostics/audit_play_prob_provenance.py \
  --date 2026-01-16 \
  --data-root /home/daniel/projections-data \
  --sim-output-root /home/daniel/projections-data/artifacts/debug/sim_v2_worlds_fpts_v2 \
  --sim-run-id dnp_fix \
  --output-csv /tmp/audit_play_prob_provenance_2026-01-16_dnp_fix.csv
```
Key printed diagnostics from that run:
- `play_prob head: ok (n=210)` (LGBM head computed; not defaulting due to missing artifacts)
- OUT players (e.g. Zach Collins): `play_prob_stage6_sim_input=0.0000` and `worlds_dnp_rate=1.0000`
- Q player (Ivica Zubac): `play_prob_stage6_sim_input=0.9667` and `worlds_dnp_rate=0.0318` (expected `1 - 0.9667 = 0.0333`)

### A2) Is DNP masking non-trivial in worlds?

**It is non-trivial when `0 < play_prob < 1`, but most players are hard 0/1 on the audited slate.**

Evidence (worlds integrity payload written by sim):
- `.../game_date=2026-01-16/run=dnp_fix/metrics.json`
  - `worlds_integrity.play_prob.n_one = 150`, `n_zero = 54` (only `6` players had `0 < play_prob < 1` on this slate)

**Top150 fragility diagnostic (from the saved contest_sim build + worlds_matrix):**
- The provenance script prints:
  - `Top150 P(any zero-score player) distribution ... mean=0.0015 p50=0.0000 p90=0.0000`
  - This is consistent with “most selected lineups avoid the few low-play-prob players”.

### A3) Bug found: rates noise could break DNP=0 semantics (masking appeared inert)

**Symptom (pre-fix artifacts):** OUT / inactive players showed non-trivial non-zero FPTS mass even with `play_prob_masking=True`, so `worlds_dnp_rate` did **not** match `(1 - play_prob)`.

Repro against the existing sim artifact for `2026-01-16`:
```bash
uv run python scripts/diagnostics/audit_play_prob_provenance.py \
  --date 2026-01-16 \
  --data-root /home/daniel/projections-data \
  --output-csv /tmp/audit_play_prob_provenance_2026-01-16_prod.csv
```
Key printed lines from that run:
- Zach Collins (OUT): `play_prob_stage6_sim_input=0.0000` but `worlds_dnp_rate=0.1655` (should be `1.0`)
- Ivica Zubac (Q): `play_prob_stage6_sim_input=0.9667` but `worlds_dnp_rate=0.0054` (expected `~0.0333`)
- Immanuel Quickley (play_prob forced to 0 by effective layer): `worlds_dnp_rate=0.1671` (should be `1.0`)

**Fix (minimal, hard guarantee):**
- `scripts/sim_v2/generate_worlds_fpts_v2.py:main`
  - hard-mask `stat_totals` and `fpts_chunk`/`stat_box` by `active_mask` so inactive players are exactly 0 FPTS.

**Post-fix evidence:** re-run sim to a debug output root and re-audit:
```bash
uv run python -m scripts.sim_v2.generate_worlds_fpts_v2 \
  --start-date 2026-01-16 --end-date 2026-01-16 \
  --n-worlds 5000 --profile sim_v3 \
  --data-root /home/daniel/projections-data \
  --output-root /home/daniel/projections-data/artifacts/debug/sim_v2_worlds_fpts_v2 \
  --run-id dnp_fix --minutes-run-id 20260117T045504Z --rates-run-id 20260117T045504Z

uv run python scripts/diagnostics/audit_play_prob_provenance.py \
  --date 2026-01-16 \
  --data-root /home/daniel/projections-data \
  --sim-output-root /home/daniel/projections-data/artifacts/debug/sim_v2_worlds_fpts_v2 \
  --sim-run-id dnp_fix \
  --output-csv /tmp/audit_play_prob_provenance_2026-01-16_dnp_fix.csv
```
Key printed lines from the post-fix audit:
- Zach Collins (OUT): `worlds_dnp_rate=1.0000`
- Ivica Zubac (Q): `worlds_dnp_rate=0.0318` vs expected `0.0333`

---

## B) How sensitive are lineup rankings / ROI proxies to ownership + dupe penalty?

### B1) Where ownership enters the pipeline (file/function list)

**Contest sim (selection + expected dupes):**
- Ownership load:
  - `projections/api/contest_sim_api.py:_load_player_ownership` (from unified projections `pred_own_pct` or `silver/ownership_predictions`)
- Dupe penalty model:
  - `projections/contest_sim/dupe_penalty.py:compute_batch_dupe_penalties` (binning on lineup sum ownership; outputs `E[1/K]` in `(0,1]`)
- Selection score:
  - `projections/contest_sim/contest_sim_service.py:run_contest_simulation` (uses `dupe_penalty` and `rank_mode` to compute `select_score`)

**Opponent field generation (generated_field mode):**
- `projections/contest_sim/field_library_manager.py:load_or_build_field_library`
- `projections/contest_sim/field_library_quickbuild.py:build_quickbuild_field_library`

**Optimizer (separate from contest sim):**
- Player pool carries `own_proj` (from `pred_own_pct`) into solver input:
  - `projections/api/optimizer_service.py:build_player_pool`
- Ownership penalties exist in optimizer backends when enabled by config:
  - `projections/optimizer/cpsat_solver.py` (ownership penalty hooks)

### B2) Knobs added (minimal conditionals)

**`ownership_mode` (contest sim):** `off | dupe_only | field_only | full`
- Implemented in:
  - `projections/api/contest_sim_api.py` (request parsing + field library cache separation)
  - `projections/contest_sim/contest_sim_service.py:run_contest_simulation` (dupe penalties bypassed when disabled)
  - `projections/contest_sim/field_library_quickbuild.py:build_quickbuild_field_library` (skip ownership features when disabled)

### B3) Backtest-style ablation (re-rank the *same* lineup set)

Command used (writes both per-date CSV and aggregate markdown):
```bash
uv run python scripts/diagnostics/ownership_ablation_backtest.py \
  --start 2026-01-08 --end 2026-01-16 \
  --data-root /home/daniel/projections-data \
  --worlds-sample 2000 \
  --output-csv /tmp/own_ablation.csv \
  --output-md /tmp/own_ablation.md
```

Key aggregate results (mean over 7 slates; baseline = `full/current`):
- Disabling dupe penalty (`off/current`) changes the selected top150 drastically:
  - `jaccard_vs_full_current ≈ 0.1862`
  - selected lineups have higher `mean_mean` and `tail_score_mean` (but are much more duplicated under the dupe model; see below).
- Switching to multiplicative dupe handling (`full/tail_times_dupe`) is mostly stable vs current:
  - `jaccard_vs_full_current ≈ 0.9467`

Duplication “pressure” signal (computed under the dupe model):
- `full/current`: `dupe_penalty_mean ≈ 0.7309`
- `full/tail_only` (ignores dupe in ranking): `dupe_penalty_mean ≈ 0.3890` (much higher expected duplication)

---

## C) Is `select_score` over-penalizing high-mean lineups due to dupe scaling?

### C1) Rank modes implemented

Implemented in `projections/contest_sim/contest_sim_service.py:run_contest_simulation`:
- `tail_only`: `select_score = tail_score`
- `tail_times_dupe`: `select_score = tail_score * dupe_penalty`
- `current` (existing): `select_score = tail_score - (1 - dupe_penalty) * mean`

### C2) Hard evidence: `current` can violate dominance

Toy diagnostic + unit test:
```bash
uv run python scripts/diagnostics/select_score_toy.py
uv run pytest -q tests/contest_sim/test_select_score_toy.py
```

The test constructs a plausible example where lineup A dominates B on `(mean, p90, ucv90)` with equal `dupe_penalty`,
but `current` ranks A below B because the penalty term scales with `mean`.

---

## Recommended default “safe mode” (based on this audit)

- If you want minimal behavior change but remove the dominance failure mode: use `rank_mode=tail_times_dupe` (keeps most of the current top150; Jaccard ≈ 0.95 in the ablation).
- If you want to remove ownership/dupe model influence entirely (at the cost of more duplication risk): use `ownership_mode=off` + `rank_mode=tail_only` and rely on diversification constraints instead.

## Next actions (low-effort, high-signal)

1) Run the ablation over a larger historical range and (if available) join to realized lineup scores/contest results to quantify ROI changes (not just stability metrics).
2) If ownership is kept: add a lightweight report of `sum_own` bins for selected top150 to see where the dupe model concentrates penalties.
3) If `play_prob` mass is mostly `{0,1}` on most slates: treat play_prob as a “hard availability” signal and focus uncertainty elsewhere (minutes/rates), or invest in improving the play_prob head/calibration.
