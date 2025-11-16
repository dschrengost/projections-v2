# Pipeline Inventory

Status snapshot of the current scrapers/ingestors, how they persist bronze/silver/gold data, and how the existing minutes pipeline consumes those artifacts.

## Bronze storage contract

- **Layout:** Every bronze dataset writes to `<data_root>/bronze/<dataset>/season=<season>/date=<YYYY-MM-DD>/<filename>.parquet`.
- **Filenames:** Defaults live in `projections/etl/storage.py` (e.g., `injuries.parquet`, `odds.parquet`, `roster.parquet`, `daily_lineups_raw.parquet`, `boxscores_raw.parquet`).
- **Metadata:** Partition DataFrames always contain `date`, `season_start` (or `season`), `ingested_ts`, and `source` columns in addition to dataset-specific fields or JSON payloads (`payload_json` for scrapers that persist the raw API response). Callers can override the bronze root per ETL via `--bronze-root`.

## Boxscore scraper (NBA.com liveData)

- **Implementation:** `scrapers/nba_boxscore.py` defines `NbaComBoxScoreScraper`, which uses `scrapers.nba_schedule.NbaScheduleScraper` to locate completed games and normalizes each game's team/player JSON payloads into dataclasses.
- **CLI / entrypoint:** `uv run python -m projections.etl.boxscores --start YYYY-MM-DD --end YYYY-MM-DD --season 2025 [--data-root PATH] [--timeout SECONDS]` (`projections/etl/boxscores.py` via Typer).
- **Arguments:** start/end date window, `--season` label (partition + hash naming), optional `--data-root` (defaults to `paths.get_data_root()`), and an HTTP timeout.
- **Outputs / storage:** Bronze JSON snapshots at `<data_root>/bronze/boxscores_raw/season=<season>/date=<YYYY-MM-DD>/boxscores_raw.parquet` (one row per day storing the serialized NBA.com payload). Silver/gold labels at `<data_root>/labels/season=<season>/boxscore_labels.parquet`, updated via `freeze_boxscore_labels`.
- **Downstream usage:** `projections/cli/build_minutes_live.py` loads these labels for live minutes inference (`labels_default = data_root / "labels" / f"season={season}/boxscore_labels.parquet"`). `projections/cli/build_minutes_roles.py`, archetype delta builders, and `projections/minutes_v1/smoke_dataset.py` also expect the frozen labels. This file is effectively the “gold” label set for both live predictions and offline training.
- **Notes / assumptions:** There is no persisted “bronze” layer for raw box score JSON. The scraper always queries NBA.com live, so historical backfills must be re-scraped (or supplied via saved JSON to `SmokeDatasetBuilder`). Pipeline callers must ensure they run this after games finish to keep labels fresh.

## Injury scraper (NBA.com PDF reports)

- **Implementation:** `scrapers/nba_injuries.py` exposes `NBAInjuryScraper`, which downloads the league-issued PDF injury grids every 30 minutes, parses them with Tabula (`TabulaTableReader`), and yields `InjuryRecord` rows. `projections/etl/injuries.py` is the bronze/silver builder.
- **CLI / entrypoint:** `uv run python -m projections.etl.injuries --start YYYY-MM-DD --end YYYY-MM-DD --season 2025 --month 11 [--data-root PATH] [--schedule 'glob'] [--use-scraper/--no-use-scraper] [--injuries-json PATH] [--timeout SECONDS] [--injury-timeout SECONDS]`
- **Arguments:** Optional schedule parquet glob(s) (falls back to live NBA API), date window, season/month partitions, destination overrides, Typer flags for `NBAInjuryScraper` (`--use-scraper` + `--injury-timeout`), and optional `--injuries-json` overrides for debugging/offline sources.
- **Outputs / storage:** Bronze Parquets partitioned by day at `<data_root>/bronze/injuries_raw/season=<season>/date=<YYYY-MM-DD>/injuries.parquet`. Silver snapshot at `<data_root>/silver/injuries_snapshot/season=<season>/month=<MM>/injuries_snapshot.parquet`.
- **Downstream usage:** Live minutes builder expects the silver snapshot per season (`projections/cli/build_minutes_live.py` loads `<data_root>/silver/injuries_snapshot/season=<season>`). `SmokeDatasetBuilder` writes/reads the same schema. Any gold features that touch `injury_as_of_ts` or availability flags depend on this bronze→silver pipeline being current.
- **Notes / assumptions:** Default behavior scrapes directly (no JSON prerequisite). `--injuries-json` provides a manual override path when re-processing historical outputs or debugging. Tabula requires Java + `jpype1`. Bronze is idempotent per day, while silver remains season/month folders with fixed filenames.

## Daily lineup scraper (NBA.com probable/confirmed starters)

- **Implementation:** `scrapers/nba_daily_lineups.py` provides `NbaDailyLineupsScraper` plus `normalize_daily_lineups()` that flattens the JSON feed into player-level records with lineup role/status flags and timestamps. `projections/etl/daily_lineups.py` runs the bronze→silver ETL.
- **CLI / entrypoint:** `uv run python -m projections.etl.daily_lineups run --start YYYY-MM-DD --end YYYY-MM-DD --season 2025 [--data-root PATH] [--timeout SECONDS]` writes bronze/silver partitions. For ad-hoc JSON dumps plus ETL, `uv run python -m projections.scrape nba-daily-lineups --start YYYY-MM-DD --end YYYY-MM-DD --season 2025 [--data-root PATH] [--out PATH] [--pretty]`.
- **Outputs / storage:** Bronze JSON snapshots at `<data_root>/bronze/daily_lineups/season=<season>/date=<YYYY-MM-DD>/daily_lineups_raw.parquet` (one row per day storing payload + metadata). Silver normalization at `<data_root>/silver/nba_daily_lineups/season=<season>/date=<YYYY-MM-DD>/lineups.parquet`.
- **Downstream usage:** `projections/etl/roster_nightly.py` inspects `<data_root>/silver/nba_daily_lineups` by default to enrich roster snapshots with projected/confirmed starter flags. Other minutes builders can point at the same directory.
- **Notes / assumptions:** The ETL writes day-level partitions and is idempotent per day. CLI handles fetching/logging; `--out` is optional for debugging payloads while keeping bronze/silver current.

## Odds scraper (Oddstrader)

- **Implementation:** `scrapers/oddstrader.py` exposes `OddstraderScraper` plus dataclasses `EventOdds` and `MarketLine`. It calls Oddstrader's `current_lines` GraphQL endpoint and normalizes markets based on provider priority.
- **CLI / entrypoints:**
  - `uv run python -m projections.scrape oddstrader --start YYYY-MM-DD --end YYYY-MM-DD [--out PATH] [--pretty]` writes optional JSON dumps for inspection.
  - `uv run python -m projections.etl.odds --start ... --end ... --season 2025 --month 11 [--schedule 'glob'] [--data-root PATH] [...]` (`projections/etl/odds.py`) performs the bronze/silver writes directly (it calls the scraper internally, so no JSON prerequisite).
- **Arguments:** Date window, season/month partitions, optional schedule globs (with API fallback), destination overrides, Oddstrader timeout, and schedule timeout.
- **Outputs / storage:** Bronze `<data_root>/bronze/odds_raw/season=<season>/date=<YYYY-MM-DD>/odds.parquet` (rows partitioned by the normalized `as_of_ts`). Silver snapshot `<data_root>/silver/odds_snapshot/season=<season>/month=<MM>/odds_snapshot.parquet`.
- **Downstream usage:** Live minutes builder loads `<data_root>/silver/odds_snapshot/season=<season>` for per-game betting context. Bronze feeds `SmokeDatasetBuilder` and any backfill jobs. CLI also returns JSON when `--out` is set (debug only).
- **Notes / assumptions:** Bronze files are month-sliced but live builder expects whole-season directories when loading silver data, so the ETL must be idempotent and append-safe. There is no explicit ingestion of other sportsbooks yet—Oddstrader is the single source.

## Roster + player metadata (NBA.com player index → roster nightly snapshots)

- **Implementation:** `scrapers/nba_players.py` defines `NbaPlayersScraper`, which scrapes the NBA players landing page (`__NEXT_DATA__`) and returns `PlayerProfile` rows. `projections/etl/roster_nightly.py` orchestrates: it loads schedule parquet, optional bronze roster polls, optional daily lineup Parquets, and can fall back to live `NbaPlayersScraper` when no bronze inputs exist.
- **CLI / entrypoint:** `uv run python -m projections.etl.roster_nightly --schedule 'glob' --start YYYY-MM-DD --end YYYY-MM-DD --season 2025 --month 11 [--data-root PATH] [--roster 'glob'] [--lineups-dir PATH] [--scrape-missing/--no-scrape-missing] [--roster-timeout SECONDS] ...`.
- **Arguments:** Schedule inputs (globs or API fallback), optional roster bronze globs, date window, season/month, destination overrides, optional lineup directory (defaults to `<data_root>/silver/nba_daily_lineups`), and timeouts for roster + schedule fetches. `--scrape-missing` controls whether it polls the live NBA players page when parquet inputs are empty.
- **Outputs / storage:** Bronze `<data_root>/bronze/roster_nightly_raw/season=<season>/date=<YYYY-MM-DD>/roster.parquet`. Silver `<data_root>/silver/roster_nightly/season=<season>/month=<MM>/roster.parquet`. Schema tracked by `ROSTER_NIGHTLY_RAW_SCHEMA` and validated before write.
- **Downstream usage:** Live minutes builder loads season directories under `silver/roster_nightly/` (with fallback windows) to build the `roster_slice` for a slate. `SmokeDatasetBuilder` writes the same schema when building offline smoke slices. Lineup metadata (projected starters) only appears when the `nba_daily_lineups` normalization step has populated the corresponding silver folder.
- **Notes / assumptions:** Bronze is partitioned by season/month but not by day, so daily reruns overwrite the same `roster_raw.parquet`. The ETL enforces `as_of_ts <= tip_ts` and merges daily lineup info when available. Scraper requires schedule parquet for the window to know which teams to poll.

## Live pipeline quick commands

Reference commands for a single slate (assumes `/home/daniel/projections-data` is mounted and schedule parquet globs are available when needed):

```bash
# Orchestrator (runs all stages sequentially)
uv run python -m projections.cli.live_pipeline run \
  --start 2025-11-16 --end 2025-11-16 \
  --season 2025 --month 11 \
  --data-root /home/daniel/projections-data

# Injuries ETL (scrape mode, bronze+silver)
uv run python -m projections.etl.injuries \
  --start 2025-11-16 --end 2025-11-16 \
  --season 2025 --month 11 \
  --data-root /home/daniel/projections-data \
  --use-scraper

# Daily lineups ETL (bronze+silver per day)
uv run python -m projections.etl.daily_lineups run \
  --start 2025-11-16 --end 2025-11-16 \
  --season 2025 \
  --data-root /home/daniel/projections-data

# Roster nightly with lineup enrichment, schedule parquet fallback optional
uv run python -m projections.etl.roster_nightly \
  --start 2025-11-16 --end 2025-11-16 \
  --season 2025 --month 11 \
  --data-root /home/daniel/projections-data \
  --schedule "/home/daniel/projections-data/silver/schedule/season=2025/*.parquet"
```

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
  1. `uv run python -m projections.etl.injuries ...` (scrapes directly, optional `--injuries-json` override).
  2. `uv run python -m projections.etl.daily_lineups run ...` (or `python -m projections.scrape nba-daily-lineups ...` when a JSON dump is desired).
  3. `uv run python -m projections.etl.odds ...` to refresh odds bronze/silver for the slate window.
  4. `uv run python -m projections.etl.roster_nightly ...` (defaults to ingesting `silver/nba_daily_lineups`).
  5. `uv run python -m projections.etl.boxscores ...` (typically for yesterday) to keep `boxscore_labels.parquet` up to date.
  6. `uv run python -m projections.cli.build_minutes_live --date YYYY-MM-DD [--data-root ... --out-root ...]` to generate live features / minutes for the slate. This command reads the silver tables produced above and writes outputs under `<out_root>/minutes_live/<YYYY-MM-DD>/`.
   - There is no single “live pipeline” CLI; each step is triggered manually even though dedicated ETLs now exist for injuries, daily lineups, odds, roster, etc.

## Pain points & inconsistencies

1. **No single live pipeline orchestrator:** Operators must run several ETLs manually (injuries, daily lineups, odds, roster, etc.). `projections/cli/build_month` handles smoke/backfill flows but still assumes pre-cached JSON and does not call the live scrapers.
2. **Data root defaults still point inside the repo:** All CLIs fall back to `<repo>/data` unless `PROJECTIONS_DATA_ROOT` is set or `--data-root` is passed. The desired `/home/daniel/projections-data` location is not codified anywhere yet.
3. **Bronze partitioning was unified for the primary scrapers, but historical backfills still need to be reprocessed to conform to the contract (and boxscore payloads only include the NBA.com JSON today). We'll need to run a one-time migration against `/home/daniel/projections-data` and ensure future “gold rebuilds” rely on bronze rather than cached JSON.**
