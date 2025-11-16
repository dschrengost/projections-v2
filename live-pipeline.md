You are an AI coding assistant working inside my `projections-v2` NBA DFS repo on a Linux box.

## Context

- I’m building a fully automated *live* data pipeline for NBA projections.
- All long-term data storage lives outside the repo at:

  /home/daniel/projections-data

- We use a bronze/silver/gold mental model:
  - **Bronze** = our “data lake”: minimally processed, schema-stable tabular data (Parquet/Feather/CSV) that is the long-term source of truth.
  - **Silver/Gold** = cleaned/feature-ready tables and model inputs.
- We have multiple scrapers / ingestors, some of which are already partially wired:
  - Boxscore scraper (NBA.com / similar) – I know this currently writes directly to a bronze dataframe.
  - Injury scraper
  - Daily starting lineups / rotations / probable lineups scraper
  - OddsTrader (or similar) odds scraper
  - Roster / player metadata scraper
- We decided:
  - We **do not need to persist raw JSON** by default; bronze is our canonical “lake”.
  - However, it’s fine to keep optional short-term JSON dumps behind a debug flag if it helps with troubleshooting.

My priorities:
- **First:** get a robust, repeatable **live pipeline** running end-to-end off today’s data:
  - Scrape→bronze→any existing downstream transforms that power minutes/projections.
- **Later:** backfill historical data for model training using the same code paths (date-parameterized, no one-off scripts).

## Your overall goal

1. **Assess where we are right now** in terms of scrapers, bronze outputs, and any downstream transforms.
2. **Design a coherent live pipeline** that:
   - runs off a configurable data root (`/home/daniel/projections-data`),
   - can be invoked for “today” and also for an arbitrary date range (for future backfill),
   - uses bronze as the durable data layer.
3. **Implement the minimum set of changes** (config, CLIs, and light refactors) to make the *live* pipeline runnable with a single command I can stick behind `cron` or a systemd timer.

Work in small, reviewable steps. Before modifying code, print a short plan/checklist.

---

## Step 1: Recon / Inventory

1. Scan the repo for our scrapers and ingestors. In particular, look for anything related to:
   - boxscores
   - injuries
   - lineups / starting lineups / rotations
   - OddsTrader / odds / betting lines
   - rosters / player metadata

   Use filename and symbol searches (e.g. `scrape`, `cli`, `ingest`, `nba-com`, `injuries`, `oddstrader`, etc.) to locate:
   - Modules
   - CLIs / entrypoints
   - Any existing “pipeline” or “build” scripts (e.g. `build_month`, dataset builders, ETL scripts).

2. For **each** scraper you find (boxscore, injuries, lineups, odds, roster), produce a short summary:
   - File + function / CLI entrypoint
   - Expected arguments (date ranges, season, output paths, etc.)
   - Current output format (JSON vs Parquet vs something else)
   - Where it writes today (inside repo vs external path; folder layout)
   - Any assumptions about being run as a one-off vs pipeline component

3. Specifically answer:
   - For the **boxscore** scraper: where exactly is “bronze” written, in what format, and who reads it next?
   - Are there any existing silver/gold builders that already expect bronze data (e.g. minutes features, schedule features, odds features)?
   - Is there any existing “live pipeline” or “production” script, or is everything currently ad-hoc (e.g. scattered `uv run` commands)?

4. Write your findings into a new markdown doc in the repo, for example:

   docs/pipeline/pipeline_inventory.md

   This doc should have:
   - A section per scraper (boxscore, injuries, lineups, odds, roster)
   - A section describing **current** end-to-end flow (even if incomplete)
   - A quick list of pain points / inconsistencies you see (paths, formats, missing config, etc.)

Do **not** change code yet. Just add the inventory doc.

---

## Step 2: Design the live pipeline + storage layout

Based on the inventory, design a simple, consistent structure rooted at:

  /home/daniel/projections-data

Your design should cover:

1. **Directory layout under data root**  
   Propose a layout that works for live **and** backfill, e.g.:

   /home/daniel/projections-data/
     bronze/
       boxscores/...
       injuries/...
       lineups/...
       odds/...
       rosters/...
     silver/
       ...
     gold/
       ...

   Use what you discover in the repo to propose something that fits existing code as much as possible. Prefer Parquet for bronze writes where feasible.

2. **Config & path management**
   - Introduce a single source of truth for the data root and layer paths (e.g. a small Python config module or YAML file read at runtime).
   - Codify:
     - the data root (`/home/daniel/projections-data`),
     - per-source bronze locations,
     - any known silver/gold tables currently in use.

   The design should allow me to override the data root via environment variable or CLI flag for local experiments if needed.

3. **Live pipeline shape**
   Define what a **single live run** should do for a given `slate_date` (e.g. `2025-11-16`):

   - Fetch / update:
     - injuries for that date (and possibly surrounding window if needed),
     - boxscores (if there are games in progress / recently finished),
     - lineups for that date,
     - odds for that date,
     - rosters (daily snapshot or only when changed).
   - Write/merge into bronze under the data root with predictable partitioning (by date and/or season).
   - Optionally trigger any existing downstream transforms that are already wired (e.g. feature builders) **if** they are safe and quick enough.

   Also design it so the same functions can be used later for backfill by passing a date range instead of “today”.

4. Capture this design in a new doc, e.g.:

   docs/pipeline/live_pipeline_design.md

   This should include:
   - A simple diagram / bullet list of steps
   - The proposed directory layout
   - The key commands (conceptually) for `live run`, `backfill run`, etc.

Again: don’t write code yet. Just design and document.

---

## Step 3: Implementation – config + CLIs + light refactors

Once the inventory and design docs are written, implement the minimal set of changes to make the **live** pipeline runnable.

1. **Config and data root**
   - Add a small configuration layer for data paths, e.g. either:
     - a `config/data_paths.yaml` file plus a helper module to read it, or
     - a `projections/config/data_paths.py` module with sensible defaults.
   - Allow overriding the data root with an environment variable (e.g. `PROJECTIONS_DATA_ROOT`) or a `--data-root` CLI flag.
   - Ensure existing scrapers can use this config without breaking current behavior.

2. **Normalize scraper outputs**
   For each scraper (boxscore, injuries, lineups, odds, roster):

   - Add or standardize a `--data-root` (or equivalent) parameter so they can write under `/home/daniel/projections-data` instead of inside the repo.
   - Ensure they write bronze outputs in a consistent format (prefer Parquet) and partitioning scheme compatible with the design from Step 2.
   - If a scraper currently writes raw JSON:
     - Either:
       - replace it with direct bronze writes, or
       - add a flag like `--debug-raw-json-out` so JSON is optional and not part of the core pipeline.
   - Make sure any downstream consumers are updated to read from the new bronze paths.

   When refactoring, keep changes small and focused; don’t rewrite scraper logic unless strictly necessary for path/config consistency.

3. **Create a single “live pipeline” CLI entrypoint**

   Implement a CLI module (or extend an existing one) that can orchestrate the live run, for example:

   - `projections/cli/live_pipeline.py` with commands like:
     - `live-pipeline run-day --date YYYY-MM-DD`
     - `live-pipeline run-today`
   - Or adapt to whatever CLI style is already used in this repo (e.g. `uv run projections.cli.live_pipeline ...`).

   This CLI should:

   - Resolve the data root (default `/home/daniel/projections-data`, overrideable).
   - For the target date:
     - Call the injury scraper in “update” mode for that date.
     - Call the boxscore scraper as appropriate.
     - Call the daily lineup scraper.
     - Call the odds scraper.
     - Call the roster scraper (daily snapshot or a lighter update if appropriate).
   - Log what it’s doing and where it’s writing.
   - Be idempotent: re-running it for the same date should not corrupt data (e.g. use partition overwrite/merge semantics, not blind append).

   Don’t wire serious scheduling (cron/systemd) directly in code; just make sure a single `uv run` command exists that I can schedule myself.

4. **(Optional but ideal)** Wire in any already-existing downstream steps that are cheap and clearly part of the live path (e.g. updating a silver “today’s features” table) as separate subcommands, not as magic side effects.

---

## Step 4: Documentation and usage examples

1. Update or create a short operator guide, e.g.:

   docs/pipeline/live_pipeline_usage.md

   Include:

   - How to set up the data root (default + override).
   - Example commands for:
     - Running today’s live pipeline:
       - `uv run projections.cli.live_pipeline run-today`
     - Running for a specific date:
       - `uv run projections.cli.live_pipeline run-day --date 2025-11-16`
   - Notes on where to find bronze outputs for each data source.

2. At the end, print a concise summary of what you implemented and any obvious next steps for:
   - Backfilling historical data through the same pipeline.
   - Improving robustness (retry logic, alerting, CI checks).

---

## Style and constraints

- Follow existing code style and project structure.
- Prefer small, incremental patches over big rewrites.
- When unsure about naming or structure, infer from existing patterns in the repo.
- Keep everything focused on the **live pipeline first**; it’s fine to leave explicit `TODO` markers for full backfill orchestration.

Start by:
1. Printing the high-level checklist of the steps you plan to execute.
2. Then implement Step 1 (inventory doc), show me the resulting markdown outline, and proceed from there.
