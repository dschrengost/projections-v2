# Minutes Dashboard (Local Workflow)

This describes the minimal commands for scoring a slate, starting the FastAPI backend,
and running the React dashboard (dev + production modes).

## 1. Score a daily slate

```bash
uv run python -m projections.cli.score_minutes_v1 --date YYYY-MM-DD
```

- Reads `config/minutes_current_run.json` to pick the latest `minutes_v1` bundle.
- Emits `minutes.parquet` + `summary.json` under `artifacts/minutes_v1/daily/YYYY-MM-DD/run=<run_id>/`
  and writes `latest_run.json` next to the run directories.
- Add `--end-date` for a short range or `--bundle-dir` to override the bundle. The FastAPI endpoints
  default to the run recorded in `latest_run.json`, or you can pass `?run_id=` to inspect older runs.

## 2. Backend API (FastAPI)

Serve JSON endpoints (`/api/minutes`, `/api/minutes/meta`) and, if available,
the built React assets from `web/minutes-dashboard/dist`.

```bash
uv run uvicorn projections.api.minutes_api:create_app --host 0.0.0.0 --port 8000
```

- If you use systemd, **only run one** of `minutes-dashboard.service` or `live-api.service` because both bind port 8501.
  Prefer `minutes-dashboard.service` (serves API + built UI); keep `live-api.service` disabled to avoid restarts.

- Uses `artifacts/minutes_v1/daily/` for data and `web/minutes-dashboard/dist/`
  for static files (build the frontend first—see step 4).
- Set env vars to override defaults:
  - `MINUTES_DAILY_ROOT=/custom/daily`
  - `MINUTES_DASHBOARD_DIST=/custom/dist`
  - `MINUTES_FPTS_ROOT=/path/to/gold/projections_fpts_v1` (optional; defaults to `PROJECTIONS_DATA_ROOT/gold/projections_fpts_v1`)
- If FPTS gold outputs exist, `/api/minutes` automatically includes `fpts_per_min_pred` /
  `proj_fpts` and `/api/minutes/meta` returns `fpts_meta` with model metadata.
- Minutes/FPTS gold artifacts (and `/api/minutes`) also surface the existing Vegas context columns:
  `spread_home`, `total`, `odds_as_of_ts`, `blowout_index`, `blowout_risk_score`, `close_game_score`,
  plus `team_implied_total` / `opponent_implied_total` when FPTS data is joined. Missing columns simply emit `null`.

## FPTS columns in the dashboard

- The table now includes a “Show FPTS columns” toggle that reveals:
  - `proj_fpts` — projected DraftKings fantasy points for each player.
  - `fpts_per_min_pred` — projected DraftKings fantasy points per minute.
- These values are hydrated from FPTS gold outputs written to
  `gold/projections_fpts_v1/date=YYYY-MM-DD/run=<fpts_run_id>/...`.
- The API resolves the active FPTS bundle via `config/fpts_current_run.json`
  (or any CLI overrides) and passes the run metadata to the dashboard so it can label
  which run produced the numbers.
- Production currently points to `fpts_lgbm_v2` (minutes_source=`actual`) via `config/fpts_current_run.json`,
  with the live pipeline writing gold files under `gold/projections_fpts_v1/`.
- When no FPTS gold exists for the requested date/run, the dashboard shows an
  “FPTS unavailable” badge and leaves the FPTS cells empty.

## 3. React dashboard (dev mode)

Hot-reload UI that proxies `/api/*` to the FastAPI dev server.

```bash
cd web/minutes-dashboard
npm install         # first run
npm run dev         # proxies to http://localhost:8000/api
```

Visit `http://localhost:5173` (default Vite port).

## 4. Build static assets (prod)

```bash
cd web/minutes-dashboard
npm install         # first run
npm run build       # outputs dist/
```

Now `uvicorn projections.api.minutes_api:create_app` will serve the bundled
UI at `/` while `/api/*` continues to return JSON.

## 5. Quick sanity script

```bash
uv run python scripts/check_minutes_daily.py --date YYYY-MM-DD
```

Prints row counts and per-team `minutes_p50` totals for a daily artifact.

## Minutes labels + eval inputs

- Gold minutes labels live under `<DATA_ROOT>/gold/labels_minutes_v1/season=YYYY/game_date=YYYY-MM-DD/labels.parquet`.
  The eval stack (`projections.minutes_v1.eval_live` + `projections.cli.eval_minutes_live`) joins these
  labels with prediction logs and the silver schedule to compute MAE + coverage metrics.
- Use the new backfill CLI to build any missing game-day partitions:

  ```bash
  PROJECTIONS_DATA_ROOT=/home/daniel/projections-data \
  uv run python -m projections.cli.backfill_minutes_labels \
    --start-date 2025-11-20 --end-date 2025-11-24
  ```

  This wrapper reuses the existing boxscore ETL (`projections.etl.boxscores`) and labels builder
  (`projections.cli.build_minutes_labels`) so we never hand-roll label logic again.
- For day-to-day upkeep, `scripts/run_post_slate_minutes_labels.sh` backfills “yesterday”
  (UTC-5) via the same CLI. Pair it with the example systemd unit/timer in `docs/systemd/`
  to keep labels fresh for live evaluations.
