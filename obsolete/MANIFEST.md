# Obsolete Files Manifest

This folder contains files that are no longer actively used in the projections-v2 system.
Archived on: 2025-12-08

## Why These Files Were Archived

### fpts_v1 (Fantasy Points V1 Prediction)
**Status:** Replaced by sim_v2

The original fantasy points prediction system used a LightGBM model to predict DK fantasy points directly.
This was deprecated when sim_v2 was introduced, which generates Monte Carlo simulations and provides
`dk_fpts_mean` as a more robust prediction with full distributions.

**Archived files:**
- `code/fpts_v1/` - Core module (datasets, eval, scoring, production)
- `code/score_fpts_v1.py`, `eval_fpts_v1.py`, `backfill_fpts_v1.py` - CLI commands
- `scripts/fpts/` - Training and debugging scripts
- `tests/fpts/` - All fpts_v1 tests
- `artifacts/fpts_lgbm/` - Trained models (v1 and v2)
- `artifacts/fpts_eval/` - Evaluation reports
- `configs/fpts_current_run.json` - Runtime config

### sim_v1 (Simulator V1)
**Status:** Replaced by sim_v2

The original simulator for generating fantasy points worlds.

**Archived files:**
- `code/sim_v1/` - Core module (residuals, sampler)
- `scripts/sim_v1/` - Calibration and debugging scripts
- `scripts/sim_old/` - Old noise parameter computation scripts

### minutes_nn (Neural Network Experiment)
**Status:** Experimental, never production

An experimental neural network approach for minutes prediction. Never made it to production.

**Archived files:**
- `scripts/minutes_nn/` - Training and evaluation scripts
- `tests/test_minutes_nn_smoke.py` - Smoke test

### Documentation (Historical)
**Archived files:**
- `docs/projections/fpts_per_min_v0.md` - Original fpts spec
- `docs/minutes_v1_bad_fold.md` - Historical issue with cross-validation
- `docs/NOTES-11-18.md` - Scratch notes from calibration session
- `docs/scaffold.md` - Original project structure planning doc
- `docs/pipeline_hardening_report.md` - Dated pipeline hardening analysis

### Logs & Debug Scripts
**Archived files:**
- `logs/` - Old backfill logs and JVM crash dumps
- `code/inspect_fpts*.py` - Debug scripts for fpts data

### Tests
**Archived files:**
- `tests/test_minutes_api_fpts.py` - FPTS API test (depends on deprecated fpts_v1)

---

## Active Systems (NOT archived)

The following systems remain active and in production:
- **minutes_v1** - Minutes prediction (LightGBM quantile regression)
- **rates_v1** - Per-minute stat rate predictions
- **sim_v2** - Monte Carlo FPTS simulator (replacement for fpts_v1)
- **optimizer** - Lineup optimizer (CP-SAT solver)
- **ownership_v1** - Ownership prediction

---

## Restoring Files

If you need to restore any of these files:
```bash
# Example: restore fpts_v1 module
mv obsolete/code/fpts_v1 projections/

# Example: restore a CLI command
mv obsolete/code/score_fpts_v1.py projections/cli/
```
