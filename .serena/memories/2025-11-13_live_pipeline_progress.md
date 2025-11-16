# 2025-11-13 Minutes Live Pipeline Status

## Latest Run
- Run ID: `20251113T193000Z`
- Features: `data/live/features_minutes_v1/2025-11-13/run=20251113T193000Z/`
- Scores: `artifacts/minutes_v1/daily/2025-11-13/run=20251113T193000Z/`
- Built after ingesting fresh injury snapshots (bronze + silver) via the new `projections.etl.injuries` CLI.
- Builder validated NBA.com active roster and wrote `active_roster.parquet` + `inactive_players.csv`.
- Scoring CLI writes to run-scoped directories and updates `latest_run.json` pointers.

## Remaining Issues
1. **Dashboard still showing old slate**
   - FastAPI backend (uvicorn) runs on 0.0.0.0:8500 but still reads the historical gold artifact location.
   - Need to update the API to read `latest_run.json` + `artifacts/.../run=<run_id>/minutes.parquet` for the requested date.
2. **Schedule coverage**
   - Live builder logs “Dropping rows with missing schedule/tip_ts for games: …” because some historical preseason IDs remain unresolved. These are dropped and the run continues, but we should enrich schedule data further if we care about those games.
3. **Injury ingestion**
   - `projections.etl.injuries` now exists and can be run per day. It falls back to NBA schedule API if the silver parquet is stale. To refresh tomorrow, rerun the two commands:
     ```bash
     uv run python -m projections.scrape injuries --mode daily --date YYYY-MM-DD --out tmp_live/injuries_<date>.json --pretty
     uv run python -m projections.etl.injuries --injuries-json tmp_live/injuries_<date>.json --schedule data/silver/schedule/season=2025/month=<MM>/schedule.parquet --start YYYY-MM-DD --end YYYY-MM-DD --season 2025 --month <MM>
     ```

## Suggested Next Steps for 2025-11-14
1. Update the dashboard API to read from run-scoped artifacts (use `latest_run.json` under both `data/live/...` and `artifacts/...`).
2. Once the dashboard is updated, restart `uvicorn` + `npm run dev` so today’s run displays immediately.
3. For tomorrow’s slate, repeat the injury scrape + ETL commands, run `build_minutes_live` with `--history-days 30` + `--run-as-of-ts`, and score with `--mode live --run-id <timestamp>`.
4. (Optional) Add more schedule JSON fetches to cover any remaining preseason game IDs the builder is still dropping.

## Commands Run Today
- `uv run python -m projections.scrape injuries --mode daily --date 2025-11-13 --out tmp_live/injuries_2025-11-13.json --pretty`
- `uv run python -m projections.etl.injuries --injuries-json tmp_live/injuries_2025-11-13.json --schedule data/silver/schedule/season=2025/month=11/schedule.parquet --start 2025-11-13 --end 2025-11-13 --season 2025 --month 11`
- `uv run python -m projections.cli.build_minutes_live --date 2025-11-13 --run-as-of-ts 2025-11-13T19:30:00 --history-days 30 --roster-fallback-days 1`
- `uv run python -m projections.cli.score_minutes_v1 --date 2025-11-13 --mode live --run-id 20251113T193000Z`

Keep this memory handy for tomorrow’s work session so we can jump straight into updating the dashboard and refreshing the inputs for 11/14.