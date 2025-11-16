You are an AI coding assistant working inside my `projections-v2` NBA DFS repo on a Linux machine.

I want you to **fix and unify the injuries and daily lineups pieces of the pipeline**, so that:

- Injuries ETL **no longer depends on manual JSON** and can scrape internally.
- Daily lineups have a real ETL + CLI that writes bronze + silver under the data root.
- `roster_nightly` can reliably use silver daily lineups in the live pipeline.
- All of this respects `/home/daniel/projections-data` as the external data root via `paths.get_data_root()`.

Work in small, reviewable steps. Before you change any code, print a short checklist of what you’re about to do.

Use the existing **pipeline inventory doc** as ground truth:

- `docs/pipeline/pipeline_inventory.md`

Key excerpts from that doc (do not re-quote, just use them):

- Injuries ETL:
  - `scrapers/nba_injuries.py` → `NBAInjuryScraper`
  - `projections/etl/injuries.py` currently expects `--injuries-json` (manual JSON) and writes:
    - bronze: `<data_root>/bronze/injuries_raw/season=<season>/date=<YYYY-MM-DD>/injuries.parquet`
    - silver: `<data_root>/silver/injuries_snapshot/season=<season>/month=<MM>/injuries_snapshot.parquet`
- Daily lineups:
  - `scrapers/nba_daily_lineups.py` → `NbaDailyLineupsScraper` + `normalize_daily_lineups()`
  - README claims there is a `python -m projections.scrape nba-daily-lineups ...` CLI, but `projections/scrape.py` does **not** implement it.
  - Intended silver layout: `<data_root>/silver/nba_daily_lineups/season=<season_start>/date=<YYYY-MM-DD>/lineups.parquet`
- Roster nightly:
  - `projections/etl/roster_nightly.py` uses schedule + optional roster bronze + optional daily lineups silver to build:
    - bronze: `<data_root>/bronze/roster_nightly_raw/season=<season>/date=<YYYY-MM-DD>/roster.parquet`
    - silver: `<data_root>/silver/roster_nightly/season=<season>/month=<MM>/roster.parquet`
  - It already has a `--lineups-dir` argument and expects silver daily lineups when available.

Across the repo, `projections/paths.get_data_root()` resolves `PROJECTIONS_DATA_ROOT` or falls back to `<repo>/data`.

-------------------------------------------------------------------------------
GOALS
-------------------------------------------------------------------------------

Implement the following, with minimal but clean refactors:

1. **Injuries ETL can scrape directly and no longer requires JSON.**
   - `projections/etl/injuries.py` should call `NBAInjuryScraper` internally for a given date window and season.
   - `--injuries-json` becomes *optional* (debug/override), not required.
   - The CLI becomes the canonical “scrape → bronze → silver” path for injuries in both live and backfill pipelines.

2. **Daily lineups have a proper ETL + CLI.**
   - New module: `projections/etl/daily_lineups.py` (or equivalent) that:
     - Calls `NbaDailyLineupsScraper` over a start/end date range.
     - Writes bronze and silver Parquets under `paths.get_data_root()`.
   - Restore/add a Typer CLI matching the README example for `nba-daily-lineups`.
   - Output layout should be consistent with the intention in the inventory doc.

3. **Roster nightly ETL uses daily lineups silver by default.**
   - Ensure `projections/etl/roster_nightly.py`:
     - Has a sensible default `lineups-dir` pointing at the new daily lineups silver path under the data root.
     - Treats missing daily lineups gracefully (empty join) but uses them automatically when present.

4. **All three CLIs are safe to use in a live pipeline.**
   - Everything takes `--data-root` (defaulting to `paths.get_data_root()`).
   - Idempotent: re-running for the same date range does not corrupt data.
   - Basic log messages for what’s being scraped/written and where.

Do **not** touch the minutes feature builders or `build_minutes_live` in this pass. This spec is only about injuries + daily lineups + roster_nightly wiring.

-------------------------------------------------------------------------------
STEP 1 – Reconfirm current behavior (read-only)
-------------------------------------------------------------------------------

1. Open and read (no changes yet):

   - `scrapers/nba_injuries.py`
   - `projections/etl/injuries.py`
   - `scrapers/nba_daily_lineups.py`
   - `projections/etl/roster_nightly.py`
   - `projections/scrape.py`
   - `projections/paths.py`
   - `docs/pipeline/pipeline_inventory.md`

2. Print a short summary in the Codex output of what you found specifically for:
   - How `NBAInjuryScraper` is currently used (if at all) in `etl.injuries`.
   - How `NbaDailyLineupsScraper` is currently used (if at all) anywhere.
   - What Typer commands `projections/scrape.py` currently exposes.

Do not modify files in this step.

-------------------------------------------------------------------------------
STEP 2 – Injuries ETL: internal scraping, no mandatory JSON
-------------------------------------------------------------------------------

Goal: Make `projections/etl/injuries.py` the canonical path: it should be able to scrape and persist injuries without an external JSON prerequisite.

Implement the following changes:

1. **Refactor injuries ETL to support two modes:**

   In `projections/etl/injuries.py`:

   - Keep the existing CLI entrypoint and core signature (Typer app or main function), but:
     - Add a new boolean flag: `--use-scraper / --no-use-scraper`, with default **True**.
     - Make `--injuries-json` **optional**. When `--use-scraper` is true and `--injuries-json` is not provided:
       - Instantiate and call `NBAInjuryScraper` from `scrapers.nba_injuries` for the given date window.
       - Construct an in-memory DataFrame from the scraped records.
       - Feed that directly into the bronze/silver pipeline (no JSON file needed).
   - If `--injuries-json` is provided:
     - Treat it as a source of raw injury records that bypasses the scraper.
     - Mark this path as “override/debug” in a comment and CLI help string.
   - Persist to the standard paths:
     - Bronze: `<data_root>/bronze/injuries_raw/season=<season>/date=<YYYY-MM-DD>/injuries.parquet`
     - Silver: `<data_root>/silver/injuries_snapshot/season=<season>/month=<MM>/injuries_snapshot.parquet`
     - Use `paths.get_data_root()` when `--data-root` is not supplied.

2. **Data root & partitioning:**

   - Ensure the ETL uses `paths.get_data_root()` as the default and respects an explicit `--data-root` parameter.
   - Write one parquet per day (overwriting partitions when rerunning) so reruns are idempotent and downstream globbing can recurse over `date=*`.

3. **Logging & idempotency:**

   - Before writing, log (via `print` or the project’s logger):
     - effective `data_root`,
     - season, month (or date window),
     - row counts for bronze and silver.
   - Writing logic should be idempotent per month:
     - Overwrite the target parquet file for that `season` + `month` instead of blindly appending, unless the current pattern in the project is different. If the existing code already uses overwrite semantics, keep that.

4. **Help text + docs nudge:**

   - Update CLI help strings for `--injuries-json` to clearly say it’s optional and intended for debugging or special offline flows.
   - Add a small comment near the main ETL entrypoint explaining that the intended usage is `use-scraper=True` with no JSON.

Do not yet add new Typer commands for injuries in `projections/scrape.py`. That will be a separate concern.

-------------------------------------------------------------------------------
STEP 3 – Daily lineups ETL + CLI
-------------------------------------------------------------------------------

Goal: Implement a proper ETL that uses `NbaDailyLineupsScraper` and writes bronze + silver under the data root, plus restore the CLI advertised in the README.

1. **Create ETL module: `projections/etl/daily_lineups.py`**

   - Implement a Typer app (or extend the existing ETL CLI pattern) with a main command, e.g.:

     - `uv run python -m projections.etl.daily_lineups run --start YYYY-MM-DD --end YYYY-MM-DD --season 2025 [--data-root PATH] [--timeout SECONDS]`

   - Behavior:

     - Resolve `data_root` via `paths.get_data_root()` if not explicitly provided.
     - For each date in `[start, end]`:
       - Use `NbaDailyLineupsScraper` (from `scrapers/nba_daily_lineups`) to fetch that day’s daily lineups.
       - Normalize using the existing `normalize_daily_lineups()` helper to get a DataFrame.
       - Write **bronze**:
         - e.g. `<data_root>/bronze/daily_lineups/season=<season>/date=<YYYY-MM-DD>/daily_lineups_raw.parquet`
       - Write **silver**:
         - e.g. `<data_root>/silver/nba_daily_lineups/season=<season>/date=<YYYY-MM-DD>/lineups.parquet`
     - If there are no lineups for a date (data source returns nothing):
       - Log that case and either write an empty partition or skip the write; choose whichever is consistent with other ETLs (roster/odds).
     - Add basic logging per date: start/end, rows scraped, bronze/silver paths written.

   - Use the same date parsing and Typer patterns as other ETLs (see `projections/etl/odds.py` or `schedule.py` for reference).

2. **Wire CLI into `projections/scrape.py` for debug/raw usage**

   - In `projections/scrape.py`, add a new Typer command `nba-daily-lineups` that:
     - Mirrors the README example but uses the new ETL under the hood.
     - Accepts:
       - `--start`, `--end`, `--season`
       - `--data-root`
       - Optional `--out` and `--pretty` flags to dump JSON for a given date range if needed.
     - Internally:
       - If `--out` is not provided, just call the ETL (`projections.etl.daily_lineups`) and exit.
       - If `--out` *is* provided:
         - Call the scraper directly for the requested date(s),
         - Write JSON to the specified path with optional pretty printing,
         - And still route through the ETL for bronze/silver writes, unless that’s too awkward. If needed, split into:
           - `nba-daily-lineups scrape-json ...`
           - `nba-daily-lineups etl ...`
           but keep naming consistent and simple.

   - Ensure CLI help strings clarify:
     - The ETL is the canonical path (bronze/silver),
     - JSON output is optional and mainly for inspection.

-------------------------------------------------------------------------------
STEP 4 – Roster nightly default wiring
-------------------------------------------------------------------------------

Goal: Make it easy for `roster_nightly` to pick up daily lineups without manual path tinkering.

1. **Update `projections/etl/roster_nightly.py`:**

   - Look at how it currently handles `--lineups-dir`.
   - Set its default to a function of `paths.get_data_root()` and the silver daily lineups layout you implemented, e.g.:

     - default `lineups-dir` = `<data_root>/silver/nba_daily_lineups`

   - Ensure that:

     - If the directory or partitions don’t exist, the ETL logs a message and proceeds with an empty join (no crash).
     - If it finds matching date/season partitions, it merges the projected/confirmed starter flags into the roster slice as intended.

2. **Keep CLI signature stable.**

   - Don’t remove `--lineups-dir`; just make the default smarter.
   - Make sure `--data-root` still applies and is used to derive the default `lineups-dir` when not explicitly supplied.

-------------------------------------------------------------------------------
STEP 5 – Smoke tests & usage examples
-------------------------------------------------------------------------------

Add minimal tests and docs to prove this wiring works.

1. **Tests (or at least testable scripts):**

   - If there’s a tests directory for ETLs, add small smoke tests that:
     - Instantiate the injuries ETL in “scrape mode” with a tiny date window and mocked scraper (or stubbed HTTP client).
     - Instantiate the daily lineups ETL with a stubbed scraper.
     - Assert that:
       - The expected folders under a temp `data_root` exist.
       - Parquet files have at least the expected core columns (e.g. `player_id`, `team_id`, `status`).

   - If the repo doesn’t currently use pytest for ETLs, add at least a small internal “manual test” script under `scripts/` that can be used to sanity check the new CLIs.

2. **Docs / examples:**

   - Update or create `docs/pipeline/live_pipeline_usage.md` or modify `pipeline_inventory.md` to add a short “How to run” snippet:

     - Injuries (live / scrape mode):

       - `uv run python -m projections.etl.injuries --start 2025-11-16 --end 2025-11-16 --season 2025 --month 11 --data-root /home/daniel/projections-data`

     - Daily lineups:

       - `uv run python -m projections.etl.daily_lineups run --start 2025-11-16 --end 2025-11-16 --season 2025 --data-root /home/daniel/projections-data`

     - Roster nightly leveraging lineups:

       - `uv run python -m projections.etl.roster_nightly --start 2025-11-16 --end 2025-11-16 --season 2025 --month 11 --data-root /home/daniel/projections-data`

   - Make sure these examples actually match the code you implemented.

-------------------------------------------------------------------------------
CONSTRAINTS & STYLE
-------------------------------------------------------------------------------

- Follow the existing code style and CLI patterns (Typer usage, logging style, etc.).
- Keep changes incremental and well-commented, especially where behavior changes (e.g., `--injuries-json` now optional).
- Preserve backward compatibility where reasonable:
  - Don’t break existing ETL callers that pass `--injuries-json`.
  - Don’t rename or remove existing CLIs unless absolutely necessary.
- Prefer clarity over cleverness: small helper functions for path construction, date loops, and logging are better than big monolithic blocks.

Start by printing your step checklist and a brief re-summarization of the current behavior from Step 1, then proceed through Steps 2–5.
