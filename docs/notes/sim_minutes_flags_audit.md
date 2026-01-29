# Sim Minutes Flags Audit (sim_v2 / sim_v3)

Date: 2026-01-29

This doc inventories the knobs that influence **minutes world generation** in the `sim_v2` pipeline (production uses the `sim_v3` profile via `prefect_flows/live_nba_pipeline.py` → `scripts.sim_v2.run_sim_live` → `scripts/sim_v2/generate_worlds_fpts_v2.py`).

## Production Path (Current)

- Prefect flow: `prefect_flows/live_nba_pipeline.py` runs sim with `sim_profile="sim_v3"`.
- Sim entrypoint: `scripts/sim_v2/run_sim_live.py` calls `scripts/sim_v2/generate_worlds_fpts_v2.py`.
- Profile config: `config/sim_v2_profiles.json` → `"profiles"."sim_v3"`.
- Note: RotAlloc codepaths exist in-repo, but the `nba-live-pipeline` Prefect flow does **not** invoke RotAlloc in the sim stage as of 2026-01-29.

## Profile Config Flags (config/sim_v2_profiles.json)

All flags below are loaded into `projections/sim_v2/config.py:SimV2Profile` via `load_sim_v2_profile()`.

### Availability / Active Mask

- `min_play_prob`
  - Location: `config/sim_v2_profiles.json` → profile root; loaded in `projections/sim_v2/config.py`
  - Default: `0.05` (code default), but profile-specific
  - Effect: filters rows out of the minutes input *before* sim (removes low play-prob players entirely)
  - Used in prod: `sim_v3` sets `0.0`

- `use_play_prob_masking`
  - Location: profile root; loaded in `projections/sim_v2/config.py`
  - Default: `true`
  - Effect:
    - `true`: sample `active_mask ~ Bernoulli(play_prob)` per player per world
    - `false`: treat all players with `play_prob > 0` as active (worlds become conditional-on-playing)
  - Used in prod: `sim_v3` sets `true`

- `minutes_availability_policy.*` (NEW)
  - Location: profile root → `minutes_availability_policy`; loaded in `projections/sim_v2/config.py:MinutesAvailabilityPolicyConfig`
  - Defaults: `enabled=false`, `play_prob_floor=0.99`, `rotation_lock_top_k=8`, `rotation_lock_minutes_threshold=20.0`
  - Effect: optional transform `p_eff = f(p_raw)`; sim samples actives using `p_eff` (raw play_prob unchanged)
  - Used in prod: `sim_v3` sets `enabled=false` (wired, off by default)

- `play_prob_policy.*` (NEW, preferred)
  - Location: profile root → `play_prob_policy`; loaded in `projections/sim_v2/config.py:PlayProbPolicyConfig`
  - Defaults: `enabled=false`, `rotation_lock_floor=0.995`, `rotation_lock_min_cond_p50=18`, `rotation_lock_topk=8`, `probable_floor=0.90`
  - Effect: policy layer that:
    - forces OUT/INACTIVE/SUSPENDED to `play_prob_eff=0`
    - floors rotation locks who are not on the injury report to ~1.0 (but not exactly 1.0)
    - floors PROBABLE to at least 0.90
  - Used in prod: `sim_v3` sets `enabled=true` (and currently uses `rotation_lock_min_cond_p50=8.0`; see `docs/notes/play_prob_policy.md`)

### Team/World Feasibility (NEW)

- `minutes_feasibility.*`
  - Location: profile root → `minutes_feasibility`; loaded in `projections/sim_v2/config.py:MinutesFeasibilityConfig`
  - Defaults: `enabled=false`, `min_active_players=8`, `min_sum_demand=210.0`, `max_resample_attempts=10`
  - Effect: after availability sampling, enforce per-(team, world) feasibility by resampling team availability draws (bounded retries) and a deterministic promotion fallback.
  - Used in prod: `sim_v3` sets `enabled=true`

### Minutes Sampling Backend Selection

- `preserve_input_rotation`
  - Location: profile root; loaded in `projections/sim_v2/config.py`
  - Default: `false`
  - Effect: when `true`, sim skips rotation pruning/eligibility filtering and uses the structured minutes noise backend.
  - Used in prod: `sim_v3` sets `true`

- `minutes_worlds.mode`
  - Location: profile root → `minutes_worlds`; loaded in `projections/sim_v2/config.py:MinutesWorldsConfig`
  - Default: `"legacy"`
  - Effect:
    - `"legacy"`: uses structured minutes noise / game scripts / fallback sampling
    - `"model_space_v1"`: uses transformer/model-space minutes worlds sampler (PR5 backend)
  - Used in prod: `sim_v3` uses `"legacy"` (PR5 backend is off)

### Team-240 Reconciliation / Caps

- `minutes_absorption_caps.*` (NEW)
  - Location: profile root → `minutes_absorption_caps`; loaded in `projections/sim_v2/config.py:MinutesAbsorptionCapsConfig`
  - Defaults: `enabled=false`, rank buckets + delta caps
  - Effect: adds an increase-only cap during team-240 reconciliation so deep bench cannot absorb extreme minutes when availability is sparse.
  - Used in prod: `sim_v3` sets `enabled=true`

- Hard cap minutes (not a profile flag)
  - Location: `scripts/sim_v2/generate_worlds_fpts_v2.py` (`MINUTES_CAP_SIM_V3`)
  - Default: `41.0`
  - Effect: per-player hard cap used by the fast team-240 allocator (prevents >cap worlds)
  - Used in prod: yes (sim_v3 path)

### Rotation Pruning / Eligibility (Legacy / Potentially Unused in Prod)

- `rotation.max_size` / `rotation.protected_size`
  - Location: profile root → `rotation`; loaded in `projections/sim_v2/config.py`
  - Defaults: `max_size=None` (sim code uses legacy default 10), `protected_size=None`
  - Effect: caps rotation size in legacy reconciliation paths; `protected_size` protects a core from being squeezed.
  - Used in prod: likely **no** (prod uses `preserve_input_rotation=true`)

## Environment Variables (scripts/sim_v2)

- `PROJECTIONS_DATA_ROOT`
  - Location: `projections/paths.py` and subprocess env setup in Prefect flow
  - Default: `./data`
  - Effect: controls where live inputs + artifacts are read/written
  - Used in prod: yes

- `PROJECTIONS_SIM_AUDIT`
  - Location: `scripts/sim_v2/generate_worlds_fpts_v2.py`
  - Default: `0`
  - Effect: extra audit logs (team-sum err, tail projection warnings, etc.)
  - Used in prod: no (default off)

- `PROJECTIONS_SIM_DEV_ASSERTS`
  - Location: `scripts/sim_v2/generate_worlds_fpts_v2.py`
  - Default: `0`
  - Effect: raises if minutes go negative or team/world sums drift from 240 (dev-only)
  - Used in prod: no (default off)

- `PROJECTIONS_SIM_WRITE_MINUTES_MATRIX`
  - Location: `scripts/sim_v2/generate_worlds_fpts_v2.py`
  - Default: `0`
  - Effect: writes `minutes_matrix.parquet` alongside `worlds_matrix.parquet` for audits
  - Used in prod: no (default off)

- `PROJECTIONS_SIM_FAIL_ON_EXTREME_FPTS`
  - Location: `scripts/sim_v2/generate_worlds_fpts_v2.py`
  - Default: unset/false
  - Effect: fails hard if any active player-world has extreme FPTS (guardrail)
  - Used in prod: no (default off)

- `PROJECTIONS_SIM_FAIL_HARD`
  - Location: `scripts/sim_v2/generate_worlds_fpts_v2.py`
  - Default: `0`
  - Effect: escalates some warnings (e.g., worlds with zero active players) into hard failures
  - Used in prod: typically no (default off)

- `PROJECTIONS_SKIP_POINTER_WRITES`
  - Location: set by `prefect_flows/live_nba_pipeline.py`, honored by `scripts/sim_v2/run_sim_live.py`
  - Default: unset
  - Effect: prevents subprocess steps from updating `latest_run.json` pointers mid-run
  - Used in prod: yes (set to `1` inside the Prefect lock)

## Legacy / Risky Switches (Audit Notes)

- `PROJECTIONS_MINUTES_ALLOC_MODE`
  - Location: `scripts/sim_v2/generate_worlds_fpts_v2.py`
  - Default: unset
  - Effect: legacy override for minutes allocation modes (e.g., RotAlloc eligibility paths). Can silently change eligibility filtering and rotation sizing when set.
  - Used in prod: **no** (per current workflow; keep unset)

- `PROJECTIONS_ALLOW_LEGACY_FLAT_GOLD_READS`
  - Location: `scripts/sim_v2/generate_worlds_fpts_v2.py`
  - Default: unset/false
  - Effect: allows reading legacy flat gold artifacts instead of run-scoped inputs. Risky for reproducibility.
  - Used in prod: unknown (should remain off unless explicitly needed)
