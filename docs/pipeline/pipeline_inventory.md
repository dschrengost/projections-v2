# Pipeline Inventory

Status snapshot of the current scrapers/ingestors, how they persist bronze/silver/gold data, and how the existing minutes pipeline consumes those artifacts.

## Boxscore scraper (NBA.com liveData)

- **Implementation:** `scrapers/nba_boxscore.py` defines `NbaComBoxScoreScraper`, which uses `scrapers.nba_schedule.NbaScheduleScraper` to locate completed games and normalizes each game's team/player JSON payloads into dataclasses.
- **CLI / entrypoint:** `uv run python -m projections.etl.boxscores --start YYYY-MM-DD --end YYYY-MM-DD --season 2025 [--data-root PATH] [--timeout SECONDS]` (`projections/etl/boxscores.py` via Typer).
- **Arguments:** start/end date window, `--season` label (partition + hash naming), optional `--data-root` (defaults to `paths.get_data_root()`), and an HTTP timeout.
- **Outputs / storage:** Instead of a bronze table, the ETL converts the scraped data directly into `boxscore_labels` and writes Parquet to `<data_root>/labels/season=<season>/boxscore_labels.parquet`, updating the immutability hash through `freeze_boxscore_labels`. Format = Parquet partitioned by season.
- **Downstream usage:** `projections/cli/build_minutes_live.py` loads these labels for live minutes inference (`labels_default = data_root / "labels" / f"season={season}/boxscore_labels.parquet"`). `projections/cli/build_minutes_roles.py`, archetype delta builders, and `projections/minutes_v1/smoke_dataset.py` also expect the frozen labels. This file is effectively the “gold” label set for both live predictions and offline training.
- **Notes / assumptions:** There is no persisted “bronze” layer for raw box score JSON. The scraper always queries NBA.com live, so historical backfills must be re-scraped (or supplied via saved JSON to `SmokeDatasetBuilder`). Pipeline callers must ensure they run this after games finish to keep labels fresh.

## Injury scraper (NBA.com PDF reports)

- **Implementation:** `scrapers/nba_injuries.py` exposes `NBAInjuryScraper`, which downloads the league-issued PDF injury grids every 30 minutes, parses them with Tabula (`TabulaTableReader`), and yields `InjuryRecord` rows. `projections/etl/injuries.py` is the bronze/silver builder.
- **CLI / entrypoint:** `uv run python -m projections.etl.injuries --injuries-json PATH --schedule 'glob' --start YYYY-MM-DD --end YYYY-MM-DD --season 2025 --month 11 [--data-root PATH] [...]`
  - There is *not* currently a Typer command wired up for `python -m projections.scrape injuries`; the README references this but `projections/scrape.py` only exposes the Oddstrader command. Today we either run a one-off script that calls `NBAInjuryScraper` and saves JSON, or rely on the historical JSON that `SmokeDatasetBuilder` consumes under `data/raw/`.
- **Arguments:** JSON input path, optional schedule parquet glob(s) (falls back to live NBA API), start/end, season/month partitions, destination overrides, and timeout knobs for both roster enrichment and schedule fetches.
- **Outputs / storage:** Bronze Parquet at `<data_root>/bronze/injuries_raw/season=<season>/injuries_<mon>.parquet` (month slugged filename). Silver snapshot at `<data_root>/silver/injuries_snapshot/season=<season>/month=<MM>/injuries_snapshot.parquet`.
- **Downstream usage:** Live minutes builder expects the silver snapshot per season (`projections/cli/build_minutes_live.py` loads `<data_root>/silver/injuries_snapshot/season=<season>`). `SmokeDatasetBuilder` writes/reads the same schema. Any gold features that touch `injury_as_of_ts` or availability flags depend on this bronze→silver pipeline being current.
- **Notes / assumptions:** Manual JSON step breaks the “single command” flow, and Tabula requires Java + `jpype1`. `TeamResolver` uses schedule parquet (or API fallback) to normalize team IDs before writing. Bronze is month-partitioned while silver is season/month folders with fixed filenames.

## Daily lineup scraper (NBA.com probable/confirmed starters)

- **Implementation:** `scrapers/nba_daily_lineups.py` provides `NbaDailyLineupsScraper` plus `normalize_daily_lineups()` that flattens the JSON feed into player-level records with lineup role/status flags and timestamps.
- **CLI / entrypoint:** README shows an example `uv run python -m projections.scrape nba-daily-lineups --date ... --out ...`, but that Typer command is not currently implemented in `projections/scrape.py`. There is no packaged CLI; to scrape today we must call the scraper module directly (e.g., via a notebook or ad-hoc script) and manually persist the JSON/Parquet.
- **Outputs / storage:** Normalized DataFrame contains columns such as `game_id`, `team_id`, `player_id`, lineup role/status, and timestamps. README indicates the intended default `silver` location is `<data_root>/silver/nba_daily_lineups/season=<season_start>/date=<YYYY-MM-DD>/lineups.parquet`, with optional raw JSON saved elsewhere if `--out` is provided.
- **Downstream usage:** `projections/etl/roster_nightly.py` can ingest these Parquets through `--lineups-dir` to enrich roster snapshots with `is_projected_starter`/`is_confirmed_starter` metadata. No other module reads them today because the scraper is not wired into any official ETL.
- **Notes / assumptions:** Missing CLI + orchestrator = no automated way to guarantee `silver/nba_daily_lineups` exists. The roster ETL tolerates their absence, defaulting to empty columns. This dataset should be treated as bronze (raw JSON) + silver (normalized Parquet) but only the normalization helper exists right now.

## Odds scraper (Oddstrader)

- **Implementation:** `scrapers/oddstrader.py` exposes `OddstraderScraper` plus dataclasses `EventOdds` and `MarketLine`. It calls Oddstrader's `current_lines` GraphQL endpoint and normalizes markets based on provider priority.
- **CLI / entrypoints:**
  - `uv run python -m projections.scrape oddstrader --start YYYY-MM-DD --end YYYY-MM-DD [--out PATH] [--pretty]` writes optional JSON dumps for inspection.
  - `uv run python -m projections.etl.odds --start ... --end ... --season 2025 --month 11 [--schedule 'glob'] [--data-root PATH] [...]` (`projections/etl/odds.py`) performs the bronze/silver writes directly (it calls the scraper internally, so no JSON prerequisite).
- **Arguments:** Date window, season/month partitions, optional schedule globs (with API fallback), destination overrides, Oddstrader timeout, and schedule timeout.
- **Outputs / storage:** Bronze Parquet `<data_root>/bronze/odds_raw/season=<season>/odds_<mon>.parquet`. Silver snapshot `<data_root>/silver/odds_snapshot/season=<season>/month=<MM>/odds_snapshot.parquet`.
- **Downstream usage:** Live minutes builder loads `<data_root>/silver/odds_snapshot/season=<season>` for per-game betting context. Bronze feeds `SmokeDatasetBuilder` and any backfill jobs. CLI also returns JSON when `--out` is set (debug only).
- **Notes / assumptions:** Bronze files are month-sliced but live builder expects whole-season directories when loading silver data, so the ETL must be idempotent and append-safe. There is no explicit ingestion of other sportsbooks yet—Oddstrader is the single source.

## Roster + player metadata (NBA.com player index → roster nightly snapshots)

- **Implementation:** `scrapers/nba_players.py` defines `NbaPlayersScraper`, which scrapes the NBA players landing page (`__NEXT_DATA__`) and returns `PlayerProfile` rows. `projections/etl/roster_nightly.py` orchestrates: it loads schedule parquet, optional bronze roster polls, optional daily lineup Parquets, and can fall back to live `NbaPlayersScraper` when no bronze inputs exist.
- **CLI / entrypoint:** `uv run python -m projections.etl.roster_nightly --schedule 'glob' --start YYYY-MM-DD --end YYYY-MM-DD --season 2025 --month 11 [--data-root PATH] [--roster 'glob'] [--lineups-dir PATH] [--scrape-missing/--no-scrape-missing] [--roster-timeout SECONDS] ...`.
- **Arguments:** Schedule inputs (globs or API fallback), optional roster bronze globs, date window, season/month, destination overrides, optional lineup directory (defaults to `<data_root>/silver/nba_daily_lineups`), and timeouts for roster + schedule fetches. `--scrape-missing` controls whether it polls the live NBA players page when parquet inputs are empty.
- **Outputs / storage:** Bronze `<data_root>/bronze/roster_nightly/season=<season>/month=<MM>/roster_raw.parquet`. Silver `<data_root>/silver/roster_nightly/season=<season>/month=<MM>/roster.parquet`. Schema tracked by `ROSTER_NIGHTLY_RAW_SCHEMA` and validated before write.
- **Downstream usage:** Live minutes builder loads season directories under `silver/roster_nightly/` (with fallback windows) to build the `roster_slice` for a slate. `SmokeDatasetBuilder` writes the same schema when building offline smoke slices. Lineup metadata (projected starters) only appears when the `nba_daily_lineups` normalization step has populated the corresponding silver folder.
- **Notes / assumptions:** Bronze is partitioned by season/month but not by day, so daily reruns overwrite the same `roster_raw.parquet`. The ETL enforces `as_of_ts <= tip_ts` and merges daily lineup info when available. Scraper requires schedule parquet for the window to know which teams to poll.

## Shared schedule + config context

- **Schedule ETL:** `projections/etl/schedule.py` provides `uv run python -m projections.etl.schedule --start --end --season [--data-root PATH] [--out PATH] [--timeout SECONDS]`. It writes month partitions under `<data_root>/silver/schedule/season=<season>/month=<MM>/schedule.parquet` (or a single parquet when `--out` is supplied). All scrapers besides boxscores depend on this silver layer via `load_schedule_data()` in `projections/etl/common.py`, which automatically falls back to hitting the NBA schedule API when the globs are empty.
- **Data root management:** `projections/paths.get_data_root()` resolves `PROJECTIONS_DATA_ROOT` or falls back to `<repo>/data`. Nothing in the repo currently hard-codes `/home/daniel/projections-data`; we rely on the environment variable or explicit CLI `--data-root` parameters to redirect writes to the external storage.

## Current end-to-end flow

1. **Offline smoke/backfill months**
   - `uv run python -m projections.cli.build_month YYYY MM [--end-year ... --end-month ...]` (`projections/cli/build_month.py`) loops over requested months.
   - For each month it invokes `SmokeDatasetBuilder` (`projections/minutes_v1/smoke_dataset.py`), which reads previously scraped JSON files (`data/raw/nba_injuries_*.json`, `.../oddstrader_*.json`, `.../nba_boxscores_*.json`, etc.) and materializes bronze + silver Parquets plus frozen labels. This bypasses the live scrapers entirely.
   - After bronze is refreshed, the CLI calls the minutes gold feature builder (`features_cli.main(...)`) to rebuild monthly features.

2. **Ad-hoc “live” updates**
   - Operators manually run each ETL in sequence:
     1. (Missing) Run a script or notebook that instantiates `NBAInjuryScraper` and saves JSON for the target day (expected by `--injuries-json`).
     2. `uv run python -m projections.etl.injuries ...` to convert that JSON into bronze/silver.
     3. `uv run python -m projections.etl.odds ...` to refresh odds bronze/silver for the slate window.
     4. `uv run python -m projections.etl.roster_nightly ...` (optionally pointing at `silver/nba_daily_lineups` if that data exists).
     5. `uv run python -m projections.etl.boxscores ...` (typically for yesterday) to keep `boxscore_labels.parquet` up to date.
     6. `uv run python -m projections.cli.build_minutes_live --date YYYY-MM-DD [--data-root ... --out-root ...]` to generate live features / minutes for the slate. This command reads the silver tables produced above and writes outputs under `<out_root>/minutes_live/<YYYY-MM-DD>/`.
   - There is no single “live pipeline” CLI; each step is triggered manually, and some required scrapers (injuries, daily lineups) lack packaged commands altogether.

## Pain points & inconsistencies

1. **No turnkey injury or lineup scrapers:** `projections/scrape.py` only exposes the Oddstrader command; the README instructions for `python -m projections.scrape injuries|nba-daily-lineups` do not match the code, forcing ad-hoc scripts for those data sources.
2. **Manual JSON dependency for injuries:** `projections/etl.injuries` requires `--injuries-json`, while other ETLs fetch data directly. This breaks the “single CLI” story and prevents a straight scrape→bronze run.
3. **Lineups data is optional/undefined:** There is an expectation of `<data_root>/silver/nba_daily_lineups/...`, but no official CLI produces it. As a result, `roster_nightly` usually runs without projected/confirmed starter info.
4. **Boxscore “bronze” is missing:** `projections/etl.boxscores` writes straight into the season-level `labels` folder, so we cannot re-derive labels from a canonical bronze dataset or re-run transforms without hitting NBA.com again.
5. **Data root defaults still point inside the repo:** All CLIs fall back to `<repo>/data` unless `PROJECTIONS_DATA_ROOT` is set or `--data-root` is passed. The desired `/home/daniel/projections-data` location is not codified anywhere yet.
6. **No single live pipeline orchestrator:** Operators must run 5–6 commands (plus a bespoke injury scrape) in the right order. `projections/cli/build_month` orchestrates smoke/backfill builds but still assumes pre-scraped JSON rather than calling the live scrapers.
7. **Bronze partitioning varies by source:** injuries/odds write `season=<season>/file_per_month`, roster writes `season=<season>/month=<MM>/roster_raw.parquet`, and there is no bronze folder at all for boxscores/lineups. We'll need to normalize layouts when we standardize on `/home/daniel/projections-data`.
