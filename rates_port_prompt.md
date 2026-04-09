# Task: Port rates_v1 scoring into nba-dfs-v2 (refactor-as-you-go)

You are working in two repos:

- **Source repo (read-only):** `/home/daniel/projects/projections-v2`
- **Destination repo:** `/home/daniel/projects/nba-dfs-v2`

Goal: port the `rates_v1` scoring path (loader, preprocess, scorer) into `nba_dfs.rates`, against the existing promoted bundle. Refactor as you go to fit the new repo's conventions and to remove documented cruft. The success criterion is that you can load the current promoted bundle and score a feature DataFrame and get sensible LightGBM predictions for all 12 targets.

This task is **scoring only**. Do not port the live feature build (`build_rates_features_live.py`), the CLI (`score_rates_live.py`), the schema validation (`projections/rates_v1/schemas.py`), or the `current.py`/`production.py` selector wrappers. Those decisions are deferred and will be handled separately.

Background: `RATES_INVENTORY.md` in the source repo has the full rationale. Read it before starting if you haven't already. The minimum scoring file set identified there is three files: `loader.py`, `preprocess.py`, `score.py`. The current promoted bundle lives at `/home/daniel/projections-data/artifacts/rates_v1/runs/rates_v1_stage6_action_props_h75_propsfix_20260218_194555` and contains 12 LightGBM model files plus `feature_cols.json` and `meta.json`.

## Constraints

1. **Do not modify any file in `projections-v2`.** Read-only.
2. **Refactor as you go is permitted, but only for cleanups that would be obviously correct.** Renaming for clarity, dropping dead branches, replacing private-helper imports with their inlined equivalents, simplifying redundant abstractions. Do **not** redesign the algorithm, do **not** change the scoring math, do **not** change the bundle format the loader expects. If you find yourself making a judgment call about whether a change is in scope, stop and note it in `PORT_NOTES.md` instead of doing it.
3. **Do not port `schemas.py`.** The inventory documented that it lags the current promoted bundle (still says stage3 when bundle is stage6). The new repo will not carry this. The contract for what features the bundle expects comes from `feature_cols.json` inside the bundle itself, which is the actual runtime source of truth.
4. **Do not implement `--no-strict` zero-fill behavior.** The inventory flagged this as operationally convenient but dangerous. The new repo's scorer should fail loudly if features are missing, not silently zero-fill. This is one of the documented cleanups.
5. **Add `lightgbm` to `pyproject.toml`** as a new dependency. Pin reasonably (`lightgbm>=4.6,<5`).
6. **Add `NBA_DFS_RATES_BUNDLE_DIR` to `src/nba_dfs/config.py`** as a new env var, separate from `NBA_DFS_BUNDLE_DIR`. Default it to `/home/daniel/projections-data/artifacts/rates_v1/runs/rates_v1_stage6_action_props_h75_propsfix_20260218_194555`. Mirror the same pattern as the existing bundle dir: read at import time, raise a clear `FileNotFoundError` if the path doesn't exist.
7. **Do not touch `nba_dfs.minutes`, `nba_dfs.features`, or any other existing module** beyond `config.py` and `pyproject.toml`. The rates port lives entirely under `src/nba_dfs/rates/`.
8. **Do not install dependencies, run pytest, or execute scoring** as part of the port itself. The user will run validation separately. (But do confirm the scaffold parses by, e.g., `python -c "import ast; ast.parse(open('src/nba_dfs/rates/score.py').read())"` if you want — that doesn't require the deps to be installed.)

## Source files to read first

Before writing anything, read these in order and take notes:

1. `projections/rates_v1/loader.py` — bundle loading, `feature_cols.json`/`meta.json` parsing
2. `projections/rates_v1/preprocess.py` — fill-value parity helpers for odds and tracking
3. `projections/rates_v1/score.py` — the actual `predict_rates` function
4. `meta.json` and `feature_cols.json` from the current promoted bundle, so you understand the contract

If any of these files import something not on the inventory's clean-scoring file list (e.g., a private helper from `minutes_v1`, a script-module import, or anything from `rates_v1.schemas`), **stop and report** rather than improvising. The inventory said the scoring path is clean — if it isn't, that's important information.

Note in `PORT_NOTES.md` for each file:

- Public API: what functions/classes are imported by other modules in the scoring path
- Any imports that are *not* stdlib, lightgbm, numpy, pandas, or other rates_v1 scoring files
- Any code paths that look like dead branches (old feature_set support, unused config knobs, etc.) that are safe cleanup candidates
- Any code paths that look like they might be load-bearing in non-obvious ways (don't touch these)

## Destination structure

```
src/nba_dfs/rates/
├── __init__.py
├── README.md
├── loader.py
├── preprocess.py
└── score.py
```

Filenames mirror the source for diff-ability. Refactor *within* files as needed, but don't reorganize across files unless there's a strong reason and you've documented it.

## Refactoring scope (what's in, what's out)

**In scope (do these):**

- Fix import paths from `projections.rates_v1.X` → `nba_dfs.rates.X`
- Fix `projections.paths.data_path` import — the new repo doesn't have `projections.paths`. Replace with the appropriate use of `nba_dfs.config.RATES_BUNDLE_DIR`. If the loader currently takes a `run_id` and constructs a path from `data_path("artifacts", "rates_v1", "runs", run_id)`, change it to take a `bundle_dir: Path` directly. The selector indirection (run_id → path) is out of scope for this port and will be handled later if needed.
- Drop any `--no-strict` / `strict=False` zero-fill code paths. If the scorer encounters a missing feature column, it should raise a clear error.
- Drop dead `feature_set` branches if there are any (e.g., stage0/stage1/stage2 handling that's no longer reachable for the stage6 bundle). The bundle's `meta.json` records its own `feature_set`, but the runtime scorer doesn't need to support stages it will never see. **Only drop branches you can prove are unreachable** — check carefully before deleting.
- Inline any private helpers from elsewhere in `rates_v1` if it makes the file cleaner. Do not inline anything from outside `rates_v1`.
- Replace `print` statements with appropriate logging or removal (the inventory didn't flag any, but watch for them).
- Modernize type hints where they're missing or use `typing.List`/`typing.Dict` instead of `list`/`dict`. The new repo is Python 3.11 so use the modern style.
- Add a module docstring to each file briefly explaining its role.

**Out of scope (do NOT do these):**

- Do not change the LightGBM scoring math, the column rename pattern (raw target → `pred_*`), or the efficiency clamp values.
- Do not add new features, new config knobs, or new public functions.
- Do not change the bundle directory layout or file naming convention.
- Do not introduce new dependencies beyond `lightgbm` (which is already justified).
- Do not write a CLI wrapper. Scoring is a Python function call, not a command.
- Do not implement selector indirection (the run_id → bundle_dir mapping). The new scorer takes a `bundle_dir` directly.
- Do not implement the live feature build, even partially. If the scorer expects features to be present, it expects the *caller* to have built them.

## File-by-file requirements

### `src/nba_dfs/rates/__init__.py`

Re-export the public scoring function. Mirror the pattern from `nba_dfs/minutes/__init__.py`:

```python
"""Per-minute rate predictions from the rates_v1 LightGBM bundle.

Ported from projections-v2 with refactor-as-you-go cleanups.
See docs/PORT_NOTES.md for the porting log.
"""

from nba_dfs.rates.score import predict_rates
from nba_dfs.rates.loader import load_rates_bundle, RatesBundle  # or whatever the actual class is

__all__ = ["predict_rates", "load_rates_bundle", "RatesBundle"]
```

Use the actual public names from the source files. If `RatesBundle` is named something else, use the real name.

### `src/nba_dfs/rates/loader.py`

Load a rates bundle from a directory. Public API should be roughly:

- A `RatesBundle` (or equivalent) dataclass or namedtuple holding the loaded LightGBM boosters keyed by target name, the feature columns list from `feature_cols.json`, and the metadata dict from `meta.json`.
- A `load_rates_bundle(bundle_dir: Path) -> RatesBundle` function.

The loader should:

- Take a `Path` to a bundle directory directly. No selector lookup.
- Load all 12 `model_*.txt` files via `lightgbm.Booster(model_file=...)`.
- Load `feature_cols.json` and store the feature column list.
- Load `meta.json` and store it as a dict (or a typed object if the source already has one).
- Validate that all expected files are present and raise a clear error if any are missing.
- **Not** look up run_ids, **not** consult selector JSONs, **not** import from `projections.paths`.

If the source loader does something more elaborate (e.g., loading optional calibration files), keep that behavior but make sure the path resolution comes from the passed-in `bundle_dir`.

### `src/nba_dfs/rates/preprocess.py`

Train/serve parity helpers for fill values. Likely a small module of pure functions. Port largely as-is, with import path fixes. Check whether the helpers reference any values from `meta.json` — if so, the contract for how they're called probably involves passing in the meta dict, which is fine.

### `src/nba_dfs/rates/score.py`

The main scoring function. Should expose `predict_rates(features_df: pd.DataFrame, bundle: RatesBundle) -> pd.DataFrame` (or whatever the source signature is).

Required behavior:

- Validate that all `bundle.feature_columns` are present in `features_df`. If any are missing, raise `RatesScoringError` (a new exception class defined in this module) with a message listing the missing columns. **Do not** zero-fill silently.
- Apply preprocessing (fill values from meta, etc.) before scoring.
- Score each of the 12 targets with its booster.
- Return a DataFrame with the key columns (`game_id`, `team_id`, `player_id`, `game_date` if present) plus the 12 prediction columns.
- Use the `pred_*` naming convention from the source (`pred_fga2_per_min`, etc.) — that's the contract downstream consumers will eventually want.
- Apply the efficiency clamps from the source (`pred_fg2_pct` to `[0.30, 0.75]`, `pred_fg3_pct` to `[0.20, 0.55]`, `pred_ft_pct` to `[0.50, 0.95]`). Keep the exact same values.

### `src/nba_dfs/rates/README.md`

Document:

- What this module does: per-minute rate predictions from a promoted `rates_v1` LightGBM bundle.
- The 12 prediction columns it produces, grouped (rate targets vs efficiency targets).
- The bundle format it expects (`model_*.txt` files plus `feature_cols.json` and `meta.json`).
- That the bundle path is read from `NBA_DFS_RATES_BUNDLE_DIR` via `nba_dfs.config`.
- That the scorer requires the caller to provide a fully-populated feature DataFrame matching `bundle.feature_columns`. **No silent zero-filling.**
- The current promoted bundle path and its `feature_set` (`stage6_action_props`) and target list, copied from the inventory.
- Explicitly: this module is **scoring only**. The live feature build is not yet implemented. See `PORT_NOTES.md` for the open question.

### `src/nba_dfs/config.py`

Add `RATES_BUNDLE_DIR` next to the existing `BUNDLE_DIR`. Same pattern: read env var with a default, raise `FileNotFoundError` at import if the path doesn't exist. Update the module docstring if needed.

### `pyproject.toml`

Add `lightgbm>=4.6,<5` to the dependencies list. Keep alphabetical order if the existing list is alphabetized.

## Documentation requirements

### `docs/PORT_NOTES.md`

Append a new section:

```markdown
## Port: rates_v1 scoring path (refactor-as-you-go)

Date: <today>
Source: projections-v2 @ <git sha of source repo HEAD>
Files ported: 3 (loader.py, preprocess.py, score.py)
Bundle: rates_v1_stage6_action_props_h75_propsfix_20260218_194555

### Refactoring cleanups performed
<list each cleanup with a one-line justification, e.g.:
- Dropped --no-strict zero-fill code path: documented as dangerous in RATES_INVENTORY.md, missing features now raise RatesScoringError
- Inlined private helper _foo from rates_v1.bar into score.py since bar is not being ported
- Removed dead stage0-stage5 branches from feature_set dispatch since current bundle is stage6 and selector indirection is out of scope
>

### Out of scope (deferred)
- Live feature build (build_rates_features_live.py): the rates feature build will plug into nba_dfs.features once we determine how much overlap there is with the existing GTv2 feature set. See "Open question: rates feature build" below.
- Selector indirection (current.py, production.py): the new scorer takes a bundle_dir directly. If we need slate-by-slate selector lookup later, we'll add it then.
- schemas.py: documented as stale (validates stage3 contract while bundle is stage6). Not carried forward. The bundle's feature_cols.json is the runtime source of truth.
- score_rates_live.py CLI: not needed; the new repo treats scoring as a Python function.

### Anomalies encountered
<anything you noticed during reading or porting that's worth recording>

## Open question: rates feature build

We have not decided how rates features get built in the new repo. The old repo's build_rates_features_live.py has hidden coupling to minutes_v1 (private _parse_minutes_iso import), expects the older minutes feature contract as its starting point, and backfills from gold/rates_training_base.

In the new repo, the GTv2 live feature build (nba_dfs.features.gtv2_live) already produces a 679-column player feature set. The open question is: how much of the rates stage6 feature contract is already a subset of, or computable from, the GTv2 feature set?

Three possible outcomes when we investigate:

1. High overlap: rates features are a subset of GTv2 features plus the Action Network props join. Rates feature build collapses to a column selector + props join.
2. Partial overlap: most rates features are present but some require new builders. Manageable port.
3. Low overlap: rates needs its own feature pipeline. Another A/B decision (port build_rates_features_live.py vs rebuild from contract).

This investigation is the next task after the rates scoring port is validated. It does not block the scoring port itself.
```

### `src/nba_dfs/rates/README.md`

Already covered above.

## Validation step (Codex does NOT run this; user will)

After the port is committed, the user will:

1. `uv sync` to pick up `lightgbm`
2. `uv run python -c "from nba_dfs.rates import predict_rates, load_rates_bundle; from nba_dfs.config import RATES_BUNDLE_DIR; b = load_rates_bundle(RATES_BUNDLE_DIR); print(len(b.feature_columns), 'features')"` — sanity check that the bundle loads
3. Write a smoke test mirroring the minutes smoke test pattern

Codex should leave a placeholder `tests/test_rates_smoke.py` file marked `@pytest.mark.requires_data` and `pytest.skip(...)` with a docstring explaining what it'll eventually test, identical in spirit to the original `test_minutes_smoke.py` placeholder. **Do not** populate the synthetic feature DataFrame — that's a user task because constructing it requires reading `feature_cols.json` and understanding the contract, which is exactly the validation we want the user to do by hand.

## Commit

One commit:

```
Port rates_v1 scoring path (3 files, refactor-as-you-go)
```

Do not push.

## Stop conditions

Stop and report (do not improvise) if any of the following:

1. `loader.py`, `preprocess.py`, or `score.py` import from a module the inventory said the scoring path doesn't depend on.
2. The bundle directory doesn't exist or doesn't contain the expected files (`model_*.txt`, `feature_cols.json`, `meta.json`).
3. You can't tell whether a code branch is dead or load-bearing — leave it alone and note it.
4. You find a refactoring opportunity that would change the scoring math, even slightly. Math stays exactly as-is.
5. The source files turn out to be much larger or more coupled than the inventory described. The inventory is two days old; if reality has drifted, that's a finding, not a thing to power through.
6. You feel the urge to also port `schemas.py`, `current.py`, or `production.py`. They're explicitly out of scope.

When you stop, list specifically what you found and what you did NOT do.

## What success looks like

- 3 new files in `src/nba_dfs/rates/` (plus updated `__init__.py` and new `README.md`)
- `config.py` updated with `RATES_BUNDLE_DIR`
- `pyproject.toml` updated with `lightgbm` dependency
- `docs/PORT_NOTES.md` updated with port log and the open question section
- `tests/test_rates_smoke.py` placeholder
- One commit with the message above
- A clear report of what was refactored and why
- No code executed, no dependencies installed
