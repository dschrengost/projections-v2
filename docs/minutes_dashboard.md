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

- Uses `artifacts/minutes_v1/daily/` for data and `web/minutes-dashboard/dist/`
  for static files (build the frontend first—see step 4).
- Set env vars to override defaults:
  - `MINUTES_DAILY_ROOT=/custom/daily`
  - `MINUTES_DASHBOARD_DIST=/custom/dist`

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
