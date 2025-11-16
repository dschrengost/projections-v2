Short version: the inventory is super helpful—but if you just “wire it up” as-is you’re going to bake in some structural warts that will hurt once you’re backfilling seasons and iterating on labels/minutes.

Let me go through this systematically and push on a few assumptions.

⸻

1. Big picture: what’s good vs risky

Good / salvageable:
	•	You already have a clear contract for live minutes: it wants silver/schedule, silver/injuries_snapshot, silver/odds_snapshot, silver/roster_nightly + labels/boxscore_labels.parquet. That’s a nice, small dependency surface.
	•	Odds and roster ETLs are basically the pattern you want everywhere:
	•	Scraper called inside ETL.
	•	Bronze + silver written straight from the ETL.
	•	Date/season parameters + --data-root.
	•	paths.get_data_root() already gives you an indirection point; you’re not hard-coding repo paths everywhere.

Risky / scaling problems:
	1.	Boxscore “no bronze” and labels as pseudo-gold.
You’re treating labels/season=X/boxscore_labels.parquet as the canonical dataset. That’s fine until:
	•	You change how labels are built (e.g., different garbage-time cutoff, overtime handling, foul-out logic).
	•	You want to add new label fields derived from boxscores (usage, pace, etc.).
With no boxscore bronze, your only option is “hit NBA.com again and rebuild everything,” and if the API changes or rate-limits get tighter, you’re stuck.
	2.	Injuries ETL depending on manual JSON.
This is the opposite of robust:
	•	Humans running bespoke scripts to create JSON is how you get missing days and subtle schema drift.
	•	It also violates your own “bronze is the lake, no raw JSON” rule.
	3.	Lineups being “imaginary silver”.
The rest of the stack assumes there is silver/nba_daily_lineups/..., but:
	•	There’s no official CLI.
	•	Nothing in live flow actually guarantees those files exist.
So you’re designing downstream logic on top of data that, in practice, often doesn’t exist.
	4.	Two different worlds for “live” vs “smoke/backfill”.
	•	Live path: scrapers + ETLs + build_minutes_live, mostly reading from bronze/silver.
	•	Backfill path: build_month + SmokeDatasetBuilder, which eats JSON and re-implements parts of the bronze logic.
This is technical debt: two implementations of the same conceptual pipeline, plus different assumptions about partitioning and schema.
	5.	Partitioning is inconsistent and not obviously idempotent.
	•	Some bronze tables are season=X/file_per_month.
	•	Others are season=X/month=MM/file.parquet.
	•	Boxscores have no bronze at all and labels are one file per season.
It’s not obvious what happens if you re-run today only, or if you re-run a partial month. That’s a scaling issue once you introduce automation and backfill.

⸻

2. Assumptions I’d challenge

Assumption: “Labels parquet as gold is enough; we don’t need boxscore bronze.”

Counterpoint: you’re freezing one particular interpretation of boxscores forever.
	•	If you later discover a bug in how labels treat DNPs, or you want to add outcome-conditioned labels (e.g., “minutes in competitive time only”), you now have to:
	•	rely on NBA.com still serving pristine historical boxscores,
	•	and accept that re-scrapes might not match older states (e.g., retroactive stat corrections).
	•	Keeping a compact boxscore bronze (game/team/player level) is cheap storage compared to the value of being able to re-label for new model experiments.

I’d strongly recommend: keep labels as gold, but add a proper boxscore bronze and a thin “labels from boxscore bronze” builder.

⸻

Assumption: “We don’t need raw JSON at all.”

I get the instinct—you don’t want a swamp of .json files. But 0 JSON is probably too far the other way.

You don’t need production JSON, but you probably want:
	•	A debug flag on each scraper/ETL to optionally dump the raw payload for a small window (--debug-raw-out).
	•	Possibly a short retention raw bucket (e.g. last 7 days) if you start seeing upstream regressions.

That way your canonical long-term store is bronze, but you’re not blind if the parser starts failing.

⸻

Assumption: “Live first, backfill later” is safe with the current design.

It’s only safe if you insist on:
	•	The same code paths handling live and backfill (parameterized by date),
and
	•	You’re willing to refactor now so build_month stops using its own JSON-based path.

If you layer automation on top of the current design and then try to retrofit backfill, you’ll have to untangle two competing pipelines.

⸻

3. What I’d redesign (without burning the house down)

Here’s what I’d change structurally before going all-in on automation.

3.1. Establish a clean storage contract
	1.	Decide a universal bronze partitioning scheme
For all event-like sources (injuries, odds, lineups, roster, boxscores), pick something like:
	•	bronze/<source>/season=<season>/date=<YYYY-MM-DD>/part-*.parquet
You can still have month-level ETL jobs that process a whole month; they just write multiple date partitions.
	2.	Standardize silver: “snapshot tables by date/season”
E.g.:
	•	silver/schedule/season=<season>/date=...
	•	silver/injuries_snapshot/season=<season>/date=...
	•	silver/odds_snapshot/season=<season>/date=...
	•	silver/roster_nightly/season=<season>/date=...
	•	silver/nba_daily_lineups/season=<season>/date=...
The minutes builder can still glob whole seasons; it just aggregates date partitions.
	3.	Treat labels as gold, but build them from bronze
	•	Introduce bronze/boxscores/season=<season>/date=....
	•	Have etl.boxscores (or a new etl.boxscore_labels) read bronze and write:
	•	gold/labels/season=<season>/boxscore_labels.parquet.
That lets you:
	•	change label logic later,
	•	experiment with alternative label sets (e.g. boxscore_labels_v2.parquet) without touching the raw data.

⸻

3.2. Kill the manual JSON dependency

Injuries:
	•	Change projections/etl.injuries to be the canonical path:
	•	It calls NBAInjuryScraper internally (just like odds ETL).
	•	Writes bronze + silver directly.
	•	Keep --injuries-json only as an override/debug input, not the default.
	•	Update any README/docs that still talk about “scrape JSON then run ETL”.

Daily lineups:
	•	Add a real ETL module, e.g. projections/etl/daily_lineups.py:
	•	uv run python -m projections.etl.daily_lineups --start --end --season --data-root ...
	•	Internally: call NbaDailyLineupsScraper, normalize, write:
	•	bronze/daily_lineups/...
	•	silver/nba_daily_lineups/...
	•	Wire projections/scrape.py to expose a nba-daily-lineups debug CLI (mirroring oddstrader) if you still want JSON dumps.

Backfill:
	•	Refactor SmokeDatasetBuilder / build_month so it stops reading JSON directly and instead:
	•	Calls the same ETLs (injuries/odds/roster/lineups/boxscores) with a date range,
	•	Then builds features off bronze/silver.

That eliminates the dual-pipeline problem.

⸻

3.3. Normalize live vs backfill semantics

Design everything around a single abstraction:

run_etl(source, start_date, end_date, season, data_root)

Then:
	•	Live pipeline = run those ETLs for today (and maybe yesterday for labels).
	•	Backfill pipeline = same calls, just with a wider date range loop.

That gives you:
	•	One code path to debug.
	•	Same schemas and layouts for both live and historical data.
	•	A clear story when you stick a scheduler / Prefect / whatever on top.

⸻

4. Robustness & scalability knobs you’re currently missing

If you want this to scale beyond “Daniel babysits uv commands,” you’ll eventually want:
	1.	Run metadata & idempotency
	•	Each ETL should write a small metadata artifact per run:
	•	run_id, source, start/end, git SHA, scraper version, row counts, min/max timestamps, success/failure.
	•	Idempotent writes:
	•	Partition overwrite by date (or atomic rename) rather than blind append into the same file.
	•	That’s what lets you reason about partial failures and re-runs.
	2.	Backpressure/error tolerance
	•	Injuries / lineups / odds endpoints will sometimes fail or timeout.
	•	You want per-source retry logic and the ability to skip a non-critical source without killing the whole live pipeline.
	•	That’s much easier if each ETL has a clear “unit of work” = (source, date).
	3.	Versioned schemas
	•	Right now, schema changes are implicit. If you add a column, downstream code just discovers it at runtime.
	•	Long term, you probably want a small schema registry per table (even if it’s just a YAML file + tests), so you can:
	•	assert on expected columns,
	•	track v1/v2 migrations.

You don’t need to build all of that today, but you should design the storage layout and ETLs so adding those things later is straightforward.

⸻

5. Concrete next steps I’d give Codex

If you want to keep momentum and still avoid painting yourself into a corner, I’d line up work in this order:
	1.	Boxscore ETL rework (bronze + labels)
	•	Add bronze/boxscores with date partitioning.
	•	Change etl.boxscores to:
	•	write bronze first,
	•	then derive boxscore_labels.parquet from bronze.
	•	Keep the current labels path so build_minutes_live doesn’t break.
	2.	Injuries ETL cleanup
	•	Make projections/etl.injuries scrape internally by default.
	•	Deprecate --injuries-json or mark as optional/debug.
	•	Update docs to treat this as the canonical path.
	3.	Daily lineups ETL + CLI
	•	Implement etl.daily_lineups with bronze + silver.
	•	Hook it into roster_nightly via the already-intended silver/nba_daily_lineups path.
	4.	Align bronze partitioning on a date dimension
	•	Pick a standard pattern (season/date).
	•	Update injuries, odds, roster ETLs to conform (with backward-compat where needed).
	5.	Introduce a minimal “live pipeline” CLI
	•	Doesn’t need to be fancy; just:
	•	resolve data root,
	•	run schedule/injuries/odds/lineups/roster/boxscores for a given date,
	•	then call build_minutes_live.
	•	Make it idempotent per date.

Once those are in place, then I’d worry about a full backfill CLI that loops over date ranges and uses the same ETLs.
