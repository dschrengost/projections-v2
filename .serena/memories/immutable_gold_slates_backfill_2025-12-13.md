## Immutable gold slates backfill (Phase 3.4)

Implemented historical slate freezing backfill CLI and a small freezer tweak to support early-season games.

### New CLI
- `scripts/backfill_slates.py` is now a real Typer CLI (single command; invoked without subcommand).
- Loads silver schedule partitions under `<DATA_ROOT>/silver/schedule/season=YYYY/**.parquet` and iterates games in a date window.
- Calls `projections/cli/freeze_slates.py::_freeze_game_snapshot` for each game and snapshot type.
- Resumable by default (skips when both parquet + manifest exist); optional partial repair overwrite.
- Writes status JSON at `<out_root>/_backfill_slates_status.json` by default.

Key options:
- `--start/--end` or `--date`
- `--snapshot-type {lock,pretip,both}` (default `both`)
- `--dry-run/--no-dry-run` (default `dry-run`)
- `--resume/--no-resume`, `--force`, `--repair-partials/--no-repair-partials`
- `--allow-empty-history/--require-history` (default allow)

### Freezer tweak
- `projections/cli/freeze_slates.py` `_freeze_game_snapshot(..., require_history: bool = True)` now supports `require_history=False`.
- If roster time-travel yields empty at `lock` (common when roster snapshots only exist at tip), freezer falls back to a tip-time roster snapshot and records `inputs.roster_cutoff_fallback_used=true` and `inputs.live_labels_source=roster_nightly_tip_fallback`.
- Final fallback (when roster is entirely missing) uses boxscore labels to enumerate players but clears outcomes (minutes/starter flags) and records `inputs.live_labels_source=boxscore_labels`.
- Missing odds and missing roster no longer crash feature building:
  - `projections/features/game_env.py` defaults spread/total-derived features when odds are absent.
  - `projections/features/depth.py` defaults depth features when roster is absent.

Validation:
- `uv run ruff check projections/cli/freeze_slates.py scripts/backfill_slates.py tests/test_backfill_slates_cli.py`
- `uv run pytest -q tests/test_backfill_slates_cli.py`