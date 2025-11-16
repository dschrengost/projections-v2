# projections-v2

Maintainable scaffolding for an NBA player minutes prediction project. The layout follows data-science best practices (Cookiecutter-style) with clear separation between data storage, reusable Python modules, notebooks, configuration, and tests.

## Directory layout

```
.
├── config/          # YAML/TOML configs that drive experiments
├── data/
│   ├── raw/         # Immutable source data dumps
│   ├── external/    # Third-party data
│   ├── interim/     # Reusable cleaned datasets
│   └── processed/   # Feature matrices ready for modeling
├── models/          # Serialized model artifacts, predictions, logs
├── notebooks/       # Exploratory notebooks (EDA, prototyping)
├── projections/     # Python package containing reusable code
│   ├── data.py      # Raw/interim data handling utilities
│   ├── features.py  # Feature engineering helpers
│   ├── models/      # Classical + deep learning training code
│   ├── train.py     # Typer CLI entry-points for end-to-end runs
│   ├── evaluate.py  # Metric utilities
│   └── utils.py     # Shared helpers (config loading, seeds, paths)
├── tests/           # Pytest-based unit tests mirroring package layout
├── pyproject.toml   # Dependency + tool configuration (managed by uv)
└── uv.lock          # Generated lockfile (create via `uv lock`)
```

> Data root: When `PROJECTIONS_DATA_ROOT` is set, every CLI/script reads and writes under that absolute directory. Otherwise the repo-local `./data` folder shown above is used. See `projections.paths` if you need to inspect the resolved location programmatically.

## Getting started

1. [Install uv](https://github.com/astral-sh/uv) if it is not already available.
2. Create the virtual environment and install dependencies:
   ```bash
   uv sync
   ```
3. Activate the environment (uv will print the correct command, typically `source .venv/bin/activate`).

## Running experiments

1. Place your raw CSV (e.g., `nba_minutes.csv`) in `data/raw/`.
2. Adjust `config/settings.yaml` as needed (columns, rolling windows, model hyperparameters).
3. Execute the classical training pipeline:
   ```bash
   uv run python -m projections.train classical-minutes --raw-filename nba_minutes.csv
   ```
   Behind the scenes this will:
   - load raw data and create a cleaned interim artifact,
   - build rolling-window features,
   - split the data into train/validation partitions, and
   - train an XGBoost regressor while reporting validation metrics.

For deep-learning experiments, wire up PyTorch dataloaders and call:
```bash
uv run python -m projections.train deep-minutes --input-size <feature_count>
```

## Testing

Run the lightweight test suite before committing changes:
```bash
uv run pytest
```
Tests focus on deterministic, fast feedback (no large datasets required).

## Minutes V1 Quick Start

The `projections.minutes_v1` package implements the Quick Start tasks from `minutes_v1_spec.md`:

1. **Immutable labels** – use `freeze_boxscore_labels` to write `data/labels/season=YYYY/boxscore_labels.parquet`, and `load_frozen_labels` to read them without risk of mutation.
2. **Snapshot discipline** – `ensure_as_of_column` and `latest_pre_tip_snapshot` sanitize `as_of_ts` columns and pick the latest record with `as_of_ts ≤ tip_ts`.
3. **Feature builder** – `MinutesFeatureBuilder` assembles the spec’s availability, rest, trend, depth, archetype, coach, and provenance columns with strict `feature_as_of_ts`.
4. **Model stack** – `train_minutes_quickstart_models` fits the ridge baseline plus LightGBM p10/p50/p90 models, conformalizes intervals, and `predict_minutes` emits calibrated quantiles.
5. **L2 reconciliation** – `reconcile_team_minutes` and `reconcile_minutes` enforce the ≈240-minute team total with role-aware caps and blowout multipliers.
6. **Monitoring skeleton** – `compute_monitoring_snapshot` reports MAE, P(|err|>6), quantile coverage, and data freshness with rolling windows.

Example usage (inside `uv run python` or a notebook):

```python
from pathlib import Path
import pandas as pd
from projections.minutes_v1 import (
    MinutesFeatureBuilder,
    freeze_boxscore_labels,
    load_frozen_labels,
    predict_minutes,
    train_minutes_quickstart_models,
    reconcile_minutes,
    compute_monitoring_snapshot,
)

labels = load_frozen_labels(Path("data/labels"))
builder = MinutesFeatureBuilder(schedule_df, injuries_df, odds_df, roster_df, coach_df)
features = builder.build(labels)
artifacts = train_minutes_quickstart_models(features)
preds = predict_minutes(artifacts, features)
reconciled = reconcile_minutes(pd.concat([features, preds], axis=1))
monitoring = compute_monitoring_snapshot(pd.concat([features, preds], axis=1))
```

## Minutes dashboard (local)

The `minutes_v1` bundle ships with a repeatable daily scoring CLI plus a lightweight FastAPI + React viewer.

1. **Score a slate** (writes `artifacts/minutes_v1/daily/YYYY-MM-DD/`):
   ```bash
   uv run python -m projections.cli.score_minutes_v1 --date 2025-10-25
   ```
   Override `--bundle-dir` to point at a different trained run or add `--end-date` for a short range.

   Prefer YAML for repeatable runs:

   ```bash
   uv run python -m projections.cli.score_minutes_v1 --config config/minutes_scoring_example.yaml
   ```
   The YAML schema matches the CLI flags exactly, and any flag you pass on the command line overrides the config.

   Likewise, the production calibration file exposes a `minutes_training` block so you can retrain the December smoke run via:

   ```bash
   uv run python -m projections.models.minutes_lgbm --config config/calibration-prod.yaml
   ```

   **Live inference path:** build a same-day feature slice and score it without labels.

   ```bash
   uv run python -m projections.cli.build_minutes_live \
       --date 2025-11-13 \
       --run-as-of-ts 2025-11-13T15:00:00Z
   uv run python -m projections.cli.score_minutes_v1 \
       --date 2025-11-13 \
        --mode live \
        --run-id 20251113T150000Z
   ```
   The live builder reuses the MinutesFeatureBuilder pipeline with current schedule,
   injury, odds, roster snapshots, **and** an NBA.com active roster scrape for validation.
   Each run writes to `data/live/features_minutes_v1/YYYY-MM-DD/run=<run_as_of_ts>/` (including
   `active_roster.parquet` + any inactive player diffs), and the scoring CLI skips starter-label
   derivation when `--mode live` is set. `latest_run.json` under each date records which run the
   dashboard should surface. The FastAPI minutes API automatically reads this pointer (unless you
   pass `?run_id=` explicitly) so `/api/minutes` and `/api/minutes/meta` always return the newest run
   for the requested date—no manual file copying needed.

2. **Develop locally** (FastAPI + Vite with hot reload):
   ```bash
   uv run uvicorn projections.api.minutes_api:create_app --reload
   cd web/minutes-dashboard && npm install && npm run dev
   ```

3. **Serve the production build**:
   ```bash
   cd web/minutes-dashboard && npm install && npm run build
   uv run uvicorn projections.api.minutes_api:create_app --host 0.0.0.0 --port 8000
   ```

4. **Sanity-check a daily artifact**:
   ```bash
   uv run python scripts/check_minutes_daily.py --date 2025-10-25
   ```

Quick Start unit tests live under `tests/test_minutes_v1_*.py`; run them via:

```bash
uv run pytest tests/test_minutes_v1_quickstart.py tests/test_minutes_v1_modeling.py tests/test_minutes_v1_monitoring.py -q
```

### Environment prerequisites

Live scrapers require a few system-level dependencies in addition to the Python packages captured in `pyproject.toml`:

- Python ≥ 3.11 with this repo’s virtual environment (`uv sync`).
- Java 11+ (OpenJDK is fine) so `tabula-py` can parse the NBA injury PDFs.
- `PROJECTIONS_DATA_ROOT` pointing at the external data volume (default `/home/daniel/projections-data`).

Before scheduling ETLs via cron/systemd, run the environment check helper:

```bash
uv run python scripts/check_live_env.py
```

See `docs/pipeline/live_pipeline_setup.md` for the full setup guide.

### Live pipeline orchestrator

Once the environment is ready you can run the full scrape → bronze → silver loop with a single command:

```bash
uv run python -m projections.cli.live_pipeline run \
  --start 2025-11-16 --end 2025-11-16 \
  --season 2025 --month 11
```

Flags such as `--skip-injuries`, `--skip-lineups`, or `--run-roster/--skip-roster` let you toggle individual stages. Use `--schedule`/`--roster` globs when you already have parquet sources on disk; otherwise the ETLs fall back to the live NBA APIs.

### Real-Data Smoke Slice (Dec 2024)

To materialize the architect's reference slice (Dec‑2024), run:

```bash
uv run python -m projections.minutes_v1.smoke_dataset --start 2024-12-01 --end 2024-12-31
```

This command loads the season JSON dumps under `data/`, and emits:

- Bronze: `data/bronze/injuries_raw/season=2024/date=YYYY-MM-DD/injuries.parquet`, `data/bronze/odds_raw/season=2024/date=YYYY-MM-DD/odds.parquet`
- Bronze (roster): `data/bronze/roster_nightly_raw/season=2024/date=YYYY-MM-DD/roster.parquet`
- Silver: schedule/injuries/odds/roster snapshots under `data/silver/*/season=2024/month=12/`
- Immutable labels + hash: `data/labels/season=2024/boxscore_labels.parquet` and `boxscore_labels.hash`
- Coverage report: `reports/minutes_v1/2024-12/coverage.csv` (injuries+odds snapshot coverage, roster player counts, etc.)

Re-running the command overwrites the month’s artifacts so the smoke slice stays deterministic.

### Minutes V1 Calibration CLI

The calibrated p10/p90 workflow lives in `projections/cli/score_minutes.py` (scoring) and `projections/cli/sequential_backtest.py` (evaluation). Both CLIs accept the new asymmetric-cap and hysteresis flags. Example commands for the December 2024 slice:

```bash
# Score features with calibrated quantiles and rolling offsets
uv run projections/cli/score_minutes.py \
  --start 2024-12-01 --end 2024-12-31 \
  --run-id v1_dec_smoke_lgbm --season 2024 --month 12 \
  --use-buckets --use-global-p10-delta --use-global-p90-delta \
  --bucket-min-ess 360 --bucket-inherit \
  --global-recent-target-rows-p10 90 --global-recent-target-rows-p90 300 \
  --recent-target-rows 300 --recency-half-life-days 12 \
  --center-width --min-quantile-half-width 0.05 \
  --one-sided-days 7 --p10-hysteresis-lower 0.10 --p10-hysteresis-upper 0.11 \
  --delta-smoothing-alpha-p10 0.0 --delta-smoothing-alpha-p90 0.3 \
  --p10-floor-guard 0.11

# Run sequential backtest over the same window (produces reports/minutes_v1/2024-12/*)
uv run projections/cli/sequential_backtest.py \
  --start 2024-12-01 --end 2024-12-31 \
  --run-id v1_dec_smoke_lgbm --season 2024 --month 12 \
  --history-months 2 \
  --use-buckets --use-global-p10-delta --use-global-p90-delta \
  --bucket-min-ess 360 --bucket-inherit \
  --global-recent-target-rows-p10 90 --global-recent-target-rows-p90 300 \
  --center-width --min-quantile-half-width 0.05 \
  --one-sided-days 7 --p10-hysteresis-lower 0.10 --p10-hysteresis-upper 0.11 \
  --delta-smoothing-alpha-p10 0.0 --delta-smoothing-alpha-p90 0.3 \
  --p10-floor-guard 0.11
```

### Standalone Roster Snapshot Builder

When working with real roster feeds, convert bronze polling data into pre-tip snapshots (with an NBA.com fallback when raw files are missing) via:

```bash
uv run python -m projections.etl.roster_nightly \
  --schedule 'data/silver/schedule/season=2025/month=11/*.parquet' \
  --start 2025-11-14 --end 2025-11-14 \
  --season 2025 --month 11
```

Add one or more `--roster` globs when you already have bronze polling parquets; otherwise the CLI will scrape the live NBA.com active roster index for every scheduled team in the window. The command selects the latest `as_of_ts` ≤ `tip_ts` for each (`game_id`,`team_id`,`player_id`) pair using the same guardrails enforced during smoke-slice generation, then writes both `data/bronze/roster_nightly_raw/season=YYYY/date=YYYY-MM-DD/roster.parquet` and the normalized silver snapshot under `data/silver/roster_nightly/season=YYYY/month=MM/roster.parquet`.

### Standalone Injury Snapshot Builder

1. Scrape the official PDF snapshots for the desired window:
   ```bash
   uv run python -m projections.scrape injuries \
     --mode daily --date 2025-11-13 \
     --out tmp_live/injuries_2025-11-13.json --pretty
   ```
2. Convert the JSON into bronze/silver parquet outputs (falls back to the live NBA schedule API if your parquet slice is stale):
   ```bash
   uv run python -m projections.etl.injuries \
     --injuries-json tmp_live/injuries_2025-11-13.json \
     --schedule data/silver/schedule/season=2025/month=11/schedule.parquet \
     --start 2025-11-13 --end 2025-11-13 \
     --season 2025 --month 11
   ```

Outputs land under the standard partitions:

- `data/bronze/injuries_raw/season=YYYY/date=YYYY-MM-DD/injuries.parquet`
- `data/silver/injuries_snapshot/season=YYYY/month=MM/injuries_snapshot.parquet`

The ETL automatically enriches player IDs via the NBA.com roster scraper and normalizes game IDs / team IDs so the live builder can consume the snapshots immediately.

### Standalone Odds Snapshot Builder

Mirror the injury workflow for sportsbook data by scraping Oddstrader directly and emitting fresh bronze/silver snapshots:

```bash
uv run python -m projections.etl.odds \
  --start 2025-11-14 --end 2025-11-14 \
  --season 2025 --month 11 \
  --schedule 'data/silver/schedule/season=2025/month=11/*.parquet'
```

The ETL normalizes game IDs via `TeamResolver`, caps `as_of_ts` at `tip_ts`, and writes `data/bronze/odds_raw/season=YYYY/date=YYYY-MM-DD/odds.parquet` plus `data/silver/odds_snapshot/season=YYYY/month=MM/odds_snapshot.parquet`. Like the injury ETL, it falls back to the live NBA schedule API when your parquet slice is stale or missing.

### Standalone Daily Lineup Scraper

Pull the NBA.com lineup feed for a single slate, persist the raw JSON, and build bronze+silver Parquets in one shot:

```bash
uv run python -m projections.scrape nba-daily-lineups \
  --start 2025-11-14 --end 2025-11-14 \
  --season 2025 \
  --out data/raw/nba_daily_lineups/2025-11-14.json
```

The command materializes `data/bronze/daily_lineups/season=2025/date=2025-11-14/daily_lineups_raw.parquet` plus `data/silver/nba_daily_lineups/season=2025/date=2025-11-14/lineups.parquet`. These records power roster starter flags: use them to seed `data/silver/roster_nightly` and live features with `is_projected_starter` / `is_confirmed_starter`.

### Box Score Label Freezer

Automate the nightly gold refresh by scraping NBA.com liveData box scores and freezing immutable labels in one shot:

```bash
uv run python -m projections.etl.boxscores \
  --start 2025-11-14 --end 2025-11-14 \
  --season 2025
```

The CLI pulls every completed box score for the requested window, converts player stat lines into `boxscore_labels`, merges them with any existing season file, and reruns `freeze_boxscore_labels` so downstream jobs can read `data/labels/season=YYYY/boxscore_labels.parquet` without risking mutation.

### Schedule Refresh

Use the dedicated ETL to rebuild silver schedule partitions directly from the NBA API (no more manual parquet edits):

```bash
uv run python -m projections.etl.schedule \
  --start 2025-11-01 --end 2025-11-30 \
  --season 2025
```

The command writes `schedule.parquet` for each touched month under `data/silver/schedule/season=YYYY/month=MM/` and deduplicates by `game_id`, so downstream ETLs (injuries/odds/roster) always see the correct slate. Pass `--out` if you need to dump the range to a one-off parquet elsewhere.

### Month Builder (Bronze → Gold)

To build an entire month of bronze (injuries/odds/roster), silver snapshots, and gold features in one shot, run:

```bash
# Single month
uv run python -m projections.cli.build_month 2024 12

# Multi-month span (Dec 2024 → Feb 2025)
uv run python -m projections.cli.build_month 2024 12 --end-year 2025 --end-month 2
```

For each month in the requested range, the command:

1. Runs the `SmokeDatasetBuilder` (unless you pass `--skip-bronze`).
2. Invokes `projections.pipelines.build_features_minutes_v1` to populate `data/gold/features_minutes_v1/season=YYYY/month=MM/` (unless you pass `--skip-gold`).

Use `--data-root`, `--season`, or the various `--*-json` options to override defaults when targeting different seasons or custom raw paths.

These commands write predictions to `data/preds/minutes_v1/2024-12/` and diagnostics (rolling offsets, daily coverage, summary JSON) to `reports/minutes_v1/2024-12/`. Adjust the hysteresis/cap flags cautiously—CI guardrails expect month-level p10 in [0.09, 0.11], ≤3 low days after day 7, and width deltas under 15%.

## Next steps

- Populate `data/raw` with historical NBA minutes and flesh out the feature engineering logic.
- Expand `config/` with experiment-specific overrides (e.g., configs/classical.yaml).
- Add CI (e.g., GitHub Actions) to automatically run formatter + tests.
