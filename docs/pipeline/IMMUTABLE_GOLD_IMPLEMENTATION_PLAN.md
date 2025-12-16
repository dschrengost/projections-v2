# Immutable Gold Architecture - Implementation Plan

*data lives one level up from this repo in projections-data*

**Created:** 2025-12-08
**Status:** Draft
**Goal:** Enable perfect reproducibility of any historical slate from the last known data before tip

---

## Overview

This plan restructures the data pipeline to:
1. **Preserve time-series snapshots** in bronze (stop overwriting)
2. **Create immutable slate artifacts** in gold (per-game frozen state)
3. **Separate training data** from live pipeline
4. **Enable reliable backfill** for historical dates

---

## Key Decisions & Contracts (Updated)

This plan started as "write hourly partitions". After reviewing the current codebase + live systemd cadence, a few details need to be explicit to avoid accidentally *still* overwriting or breaking existing readers.

### Timestamp Semantics (as_of vs ingested)

- `as_of_ts` (**semantic time**) = when the upstream source says the info is effective/updated. This is the timestamp we use for "what was known by time T" filtering (training/backtests and slate freezing).
- `ingested_ts` (**observation time**) = when our pipeline captured the row. This is for audit, scraper-health monitoring, and debugging. We record it in gold manifests, but we do **not** use it as the primary selection key for "pre-tip" snapshots.

### Snapshot Selection Rule (Time Travel)

For any snapshot cutoff `T` (e.g., `lock_ts = tip_ts - 15m`, `pretip_ts = tip_ts`):

- **Filter:** prefer `as_of_ts <= T`. This is the invariant needed to avoid leakage in training/backtests.
- **Fallback:** only if a dataset cannot provide a meaningful `as_of_ts`, use `ingested_ts <= T` and record that fact explicitly in the gold manifest.
- **Deterministic ties:** when selecting "latest row", sort by `(as_of_ts, ingested_ts)` and take the last row per group.

### Live `run_as_of_ts` Guidance

`run_as_of_ts` is the *cutoff* that `build_minutes_live` uses when filtering silver snapshots. For live runs, prefer using the run wall-clock time (“now”) so faster feeds (odds) are not accidentally filtered out between slower feeds (injury PDFs). Per-game `tip_ts` bounds still cap any rows at tip to prevent post-tip leakage.

### Partition Timezone

- All `date=` and `hour=` partition keys in bronze are interpreted in **America/New_York (ET)** so they align with NBA "game date" conventions, tip windows, and the existing systemd schedule.
- Timestamps stored in parquet stay in UTC per the existing schemas (e.g. `as_of_ts` / `ingested_ts` are `datetime64[ns, UTC]`).

### What `date=` Means (Domain Date, ET)

To avoid UTC-midnight rollover bugs and to make retrieval ergonomic, `date=` is a **domain date** in ET:

- `injuries_raw`: injury report date in ET (derived from the NBA PDF report time).
- `odds_raw`: game date in ET (derived from schedule for each `game_id`, not from the odds line update timestamp).

### Bronze Append-Only Strategy (Per Dataset)

Bronze should be append-only for any dataset that can change over time.

- `injuries_raw`: **hourly** subpartitions (`hour=HH`) keyed off the NBA injury PDF `report_time` (ET). This is effectively idempotent: the PDF for a given `report_time` is a stable artifact, so rewriting the same hour is acceptable.
- `odds_raw`: **per-run** subpartitions (`run_ts=YYYYMMDDTHHMMSSZ`). Odds can change intra-hour and the live pipeline can run more frequently than hourly, so `hour=HH` alone would still overwrite.

Hourly downsampling is acceptable for injuries because the upstream artifact is effectively hourly already; for odds we should keep per-run snapshots at least until gold slates are frozen (Phase 3), then compact/prune safely via manifest references (Phase 5).

### Backward Compatibility / Rollout

To avoid breaking existing scripts/readers:

1. **Dual-write during Phase 1**: continue producing the current flat daily file (`date=.../<filename>`) as a **"latest view"** while also writing the new append-only subpartitions.
2. Update internal readers (notably `scripts/backfill_injuries.py`) to prefer the new layout.
3. Once stable, stop writing the legacy flat daily files (optional cleanup; not required for correctness).

### Expected Impact (What Will Change)

- Bronze layout gains new subfolders: `hour=*` (injuries) and `run_ts=*` (odds) under existing `date=*` partitions.
- Live ETLs (`projections/etl/injuries.py`, `projections/etl/odds.py`) change from overwrite-only to dual-write (new history + legacy latest view).
- Backfill scripts switch to writing the new append-only partitions (no more single daily overwrite artifacts).
- New gold "freeze slates" CLI + systemd timer is added; bronze/silver path contracts stay stable. Separately, we recommend live pipeline hardening (locks + run-id consistency) to reduce concurrency risk (see “Live Pipeline Trace & Hardening”).
- Training dataset builder gains an explicit `slate_snapshot_type` knob (default `pretip`).

### Gold Immutability Guard

Gold slates must be write-once by default:

- Fail if the target snapshot already exists (unless `--force` is passed).
- Always emit a manifest that records `snapshot_ts`, git SHA, and the specific input parquet paths (plus max `as_of_ts` / max `ingested_ts`) used to build the snapshot.

---

## Live Pipeline Trace & Hardening (Updated)

This plan should not break the existing live pipeline. That said, tracing the current systemd + scripts shows a few robustness issues that are worth addressing while we touch storage contracts.

### Current Live Entry Points (as of this repo)

- `systemd/live-score.timer` → `systemd/live-score.service` → `scripts/run_live_score.sh`
  - Runs: `projections.cli.live_pipeline` (injuries/odds/lineups/roster) → `build_minutes_live` → `score_minutes_v1` → `build_rates_features_live` → `score_rates_live` → `scripts.sim_v2.run_sim_live` → `score_ownership_live` → `finalize_projections`.
- `systemd/live-scrape.timer` → `scripts/run_live_scrape.sh` → `projections.cli.live_pipeline` only.
- `systemd/live-pipeline-*.timer` (hourly/evening/weekend) → `scripts/run_live_pipeline.sh` (scrape + optional minutes scoring).
- `systemd/live-rates.timer` → `scripts/run_live_rates.sh` (rates build/score).
- `systemd/live-sim.timer` → `scripts/run_live_sim.sh` (sim_v2).

### Compatibility Promise

Phase 1 bronze changes are **additive** and do not change any of the live silver outputs (`silver/*_snapshot/*.parquet`) that live scoring consumes. During rollout we also keep the legacy flat daily bronze parquet as a “latest view” so existing bronze readers keep working.

Concretely, this should *not* break:

- Existing scrapers / `projections.cli.live_pipeline` call sites (same CLI, same silver outputs).
- Existing systemd units that only care about silver outputs.
- Any code reading the legacy daily bronze parquet file directly (it still exists during dual-write rollout).

### Robustness Issues Found (Worth Fixing)

- **Overlapping systemd timers:** multiple services can run the same ETLs concurrently and write the same parquet paths.
- **Run-id drift:** `scripts/run_live_pipeline.sh` does not pass `--run-id` to `build_minutes_live` (features run_id defaults to `run_as_of_ts`) but passes a different run_id to `score_minutes_v1`.
- **Rates run-id mismatch:** `scripts/run_live_rates.sh` generates a fresh `RUN_ID` and passes it to `build_rates_features_live --strict`, but rates features default to the *minutes* run_id (so strict runs can fail when minutes features for that RUN_ID do not exist).
- **Silver snapshot write safety:** `projections/etl/odds.py` and `projections/etl/injuries.py` write silver snapshots in-place; with overlapping timers, this can cause partial reads / temporary regressions. Symptom: `spread_home`/`total` intermittently missing in minutes outputs until a subsequent run succeeds.
- **Sim minutes fallback:** `scripts/run_live_sim.sh` falls back to `${DATA_ROOT}/artifacts/minutes_v1/daily/...` but minutes artifacts are repo-local by default; it should consult `MINUTES_DAILY_ROOT` and/or use `latest_run.json` pointers.

### Recommended Hardening (Keep Pipeline Functional)

Option A (recommended): **Single orchestrator**

- Treat `scripts/run_live_score.sh` as the canonical live orchestrator.
- Disable redundant timers (`live-scrape`, `live-pipeline-*`, `live-rates`, `live-sim`) to eliminate duplicated work and reduce concurrency bugs.
- Add a single `flock` lock at the top of `scripts/run_live_score.sh` so no two runs overlap.
- Make snapshot writes resilient: atomic writes for silver parquets + “no-regress” semantics (never wipe `odds_snapshot`/`injuries_snapshot` coverage on a transient scrape/read failure).

Option B: **Keep separate timers (minimum operational change)**

- Add a shared `flock` lock (same lock file) to *all* live scripts (`run_live_score.sh`, `run_live_scrape.sh`, `run_live_pipeline.sh`, `run_live_rates.sh`, `run_live_sim.sh`) so they serialize.
- Fix run-id propagation: derive one `RUN_ID` per run and pass it through build → score → rates → sim.
- Make `run_live_rates.sh` resolve the latest minutes run id via `<DATA_ROOT>/live/features_minutes_v1/<date>/latest_run.json` (or accept `--run-id` explicitly), rather than generating a new one.
- Fix minutes artifact fallback in `run_live_sim.sh` (consult `MINUTES_DAILY_ROOT` and/or `latest_run.json` pointers).
- Strongly recommended: implement atomic writes (tmp + `os.replace`) for parquet and pointer JSON to avoid partial reads during concurrent readers (e.g., the new freeze-slates timer).

### Live Hardening Files (If Adopted)

- `scripts/run_live_score.sh`, `scripts/run_live_pipeline.sh`, `scripts/run_live_rates.sh`, `scripts/run_live_sim.sh`, optionally `scripts/run_live_scrape.sh`
- `systemd/live-score.service` and other `systemd/*.service` files (only if we change env vars or disable timers)

---

## Phase 1: Stop the Bleeding (Bronze Snapshot Partitions)

**Priority:** CRITICAL - Do this first to prevent future data loss
**Effort:** 1-2 days
**Risk:** Low (additive change, dual-write rollout)

### 1.1 Modify Bronze Storage Contract

**File:** `projections/etl/storage.py`

Current contract (already in repo) is daily partitions with a default filename per dataset:

```
<data_root>/bronze/<dataset>/season=<season>/date=<YYYY-MM-DD>/<filename>.parquet
```

We will extend this with **append-only subpartitions** while keeping the legacy flat file as a temporary "latest view":

```
<data_root>/bronze/<dataset>/season=<season>/date=<YYYY-MM-DD>/
  <filename>.parquet                            # legacy "latest view" (temporary; overwritten)
  hour=<HH>/<filename>.parquet                  # canonical history (injuries_raw)
  run_ts=<YYYYMMDDTHHMMSSZ>/<filename>.parquet  # canonical history (odds_raw)
```

```python
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd


def write_bronze_partition_hourly(
    frame: pd.DataFrame,
    *,
    dataset: str,
    data_root: Path,
    season: int,
    target_date: date,
    hour: int,
    bronze_root: Path | None = None,
    filename: str | None = None,
) -> BronzeWriteResult:
    """Write an hourly bronze subpartition (append-only semantics)."""
    day_dir = bronze_partition_dir(
        dataset,
        data_root=data_root,
        season=season,
        target_date=target_date,
        bronze_root=bronze_root,
    )
    output_name = filename or DEFAULT_BRONZE_FILENAMES.get(dataset, "data.parquet")
    destination = day_dir / f"hour={hour:02d}" / output_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return BronzeWriteResult(dataset=dataset, target_date=target_date, path=destination, rows=len(frame))


def write_bronze_partition_run(
    frame: pd.DataFrame,
    *,
    dataset: str,
    data_root: Path,
    season: int,
    target_date: date,
    run_ts: datetime,
    bronze_root: Path | None = None,
    filename: str | None = None,
) -> BronzeWriteResult:
    """Write a per-run bronze subpartition keyed by run_ts (append-only semantics)."""
    run_slug = ensure_datetime(run_ts).strftime("%Y%m%dT%H%M%SZ")
    day_dir = bronze_partition_dir(
        dataset,
        data_root=data_root,
        season=season,
        target_date=target_date,
        bronze_root=bronze_root,
    )
    output_name = filename or DEFAULT_BRONZE_FILENAMES.get(dataset, "data.parquet")
    destination = day_dir / f"run_ts={run_slug}" / output_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return BronzeWriteResult(dataset=dataset, target_date=target_date, path=destination, rows=len(frame))
```

**Strong recommendation:** make bronze parquet writes atomic (write to a temp file then `os.replace`) to avoid partial reads if any job crashes mid-write. This becomes more important once we add concurrent readers like the `freeze-slates` timer.

Also recommended for robustness: apply the same atomic-write pattern to **silver snapshot parquets** and pointer JSON files (`latest_run.json`). This directly addresses intermittent missing odds in live outputs when overlapping timers or partial writes occur.

### 1.2 Update Injuries ETL

**File:** `projections/etl/injuries.py`

Changes:

- Derive `target_date` and `hour` from `as_of_ts` **in ET**, not UTC (avoids date rollover bugs around midnight ET).
- Write the canonical history to `date=YYYY-MM-DD/hour=HH/<filename>.parquet` using `write_bronze_partition_hourly()`.
- **Dual-write (temporary)**: keep writing the existing flat daily file via `write_bronze_partition()` as a "latest view" so existing readers don't break.

```python
# In injuries ETL bronze persistence:
working = injuries_raw.copy()

# Partition keys are ET-based for stable "game day" semantics.
et_ts = working["as_of_ts"].dt.tz_convert("America/New_York")
working["_partition_date"] = et_ts.dt.normalize().dt.date
working["_partition_hour"] = et_ts.dt.hour

for partition_date, date_frame in working.groupby("_partition_date"):
    for hour, hour_frame in date_frame.groupby("_partition_hour"):
        storage.write_bronze_partition_hourly(
            hour_frame.drop(columns=["_partition_date", "_partition_hour"]),
            dataset="injuries_raw",
            data_root=data_root,
            season=season,
            target_date=partition_date,
            hour=int(hour),
            bronze_root=bronze_root_path,
        )

    # Transitional: continue writing the flat daily file as a "latest view".
    storage.write_bronze_partition(
        date_frame.drop(columns=["_partition_date", "_partition_hour"]),
        dataset="injuries_raw",
        data_root=data_root,
        season=season,
        target_date=partition_date,
        bronze_root=bronze_root_path,
    )
```

### 1.3 Update Odds ETL

**File:** `projections/etl/odds.py`

Odds can change intra-hour, and the live pipeline can run more frequently than hourly. To avoid overwriting within the same hour, store odds snapshots as **per-run** subpartitions:

- Canonical history: `date=<game_date ET>/run_ts=YYYYMMDDTHHMMSSZ/odds.parquet`
- Transitional "latest view": `date=<game_date ET>/odds.parquet`

Important: `date=` for odds is the **game date** (ET) derived from schedule for each `game_id`. Do **not** partition odds by `as_of_ts` calendar day — odds update timestamps can be on prior days.

```python
from datetime import datetime, timezone

run_ts = datetime.now(timezone.utc)

# Attach game_date (ET) to each odds row via schedule.
working = odds_raw.merge(schedule_df[["game_id", "game_date"]], on="game_id", how="left")
working["game_date"] = pd.to_datetime(working["game_date"]).dt.date

for game_date, day_frame in working.groupby("game_date"):
    payload = day_frame.drop(columns=["game_date"])
    storage.write_bronze_partition_run(
        payload,
        dataset="odds_raw",
        data_root=data_root,
        season=season,
        target_date=game_date,
        run_ts=run_ts,
        bronze_root=bronze_root_path,
    )

    # Transitional: keep the legacy flat daily file updated as a "latest view".
    storage.write_bronze_partition(
        payload,
        dataset="odds_raw",
        data_root=data_root,
        season=season,
        target_date=game_date,
        bronze_root=bronze_root_path,
    )
```

### 1.4 Update Read Functions

**File:** `projections/etl/storage.py`

Important: during Phase 1 we *dual-write* both the legacy flat daily parquet and the new `hour=*`/`run_ts=*` partitions. Any reader must avoid double-counting by **preferring history partitions** when they exist.

```python
def read_bronze_day(
    dataset: str,
    data_root: Path,
    season: int,
    target_date: date,
    *,
    include_runs: bool = True,
    prefer_history: bool = True,
) -> pd.DataFrame:
    """Read a day's bronze rows, preferring history partitions over the legacy flat file."""
    day_dir = bronze_partition_dir(
        dataset,
        data_root=data_root,
        season=season,
        target_date=target_date,
    )
    output_name = DEFAULT_BRONZE_FILENAMES.get(dataset, "data.parquet")

    hourly_paths = sorted(day_dir.glob(f"hour=*/{output_name}"))
    run_paths = sorted(day_dir.glob(f"run_ts=*/{output_name}")) if include_runs else []
    history_paths = hourly_paths + run_paths

    paths: list[Path] = []
    if prefer_history and history_paths:
        paths = history_paths
    else:
        flat_file = day_dir / output_name
        if flat_file.exists():
            paths = [flat_file]

    if not paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in paths]

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
```

**Call sites to update (to avoid double-counting during dual-write):**

- `projections/cli/build_minutes_live.py` backfill-mode injuries load (currently reads the entire `date=.../` directory recursively).
- `scripts/backfill_injuries.py` (and any other backfill tooling) should use the new helper rather than manual globs.

### 1.5 Retention Policy

Retention/compaction is valuable, but should be driven by **gold slates** once Phase 3 is stable. Recommendation:

- Keep full bronze history (hourly for injuries, per-run for odds) for a short rolling window (e.g., 30 days).
- After gold slates exist for a date range, prune bronze by keeping only snapshots referenced in gold manifests (or compact to an hourly summary) to save space without losing reproducibility.

Add cleanup script placeholder (Phase 5 implements the safe version):

**File:** `scripts/cleanup_bronze_partitions.py`

```python
def cleanup_old_bronze_partitions(data_root: Path, days_to_keep: int = 30):
    """Prune/compact bronze partitions older than N days (post-gold freeze)."""
    # 1) Identify days older than cutoff.
    # 2) If gold slates exist: keep only bronze snapshots referenced by manifests.
    # 3) Else: skip (don't delete data needed for future freezes).
```

---

## Phase 2: Backfill Historical Injuries

**Priority:** HIGH - Recover what we can(we can re-scrape what we are missing from NBA.com)
**Effort:** 2-3 days (mostly runtime)
**Risk:** Medium (NBA.com rate limits, missing PDFs)

### 2.1 Enhance Backfill Script

**File:** `scripts/backfill_injuries.py`

Current script fetches ~1 report per day and writes a single flat parquet (overwriting). Replace it with an hourly backfill that:

- Fetches **all available** hourly reports for a given ET game day (NBA publishes at `:30` each hour).
- Converts to the **same `injuries_raw` schema** as the live ETL (reuse the same resolver logic; avoid a separate player-name mapping).
- Writes to **hourly bronze subpartitions** via `write_bronze_partition_hourly()` (ET `date=` + ET `hour=`).
- **Dual-writes** the legacy flat daily file as a temporary "latest view" (same rollout as Phase 1 live ETL).
- Tracks per-day/per-hour success in a status JSON file so the job is resumable/idempotent.

```python
def backfill_all_hourly_reports(
    target_date: date,
    *,
    season: int,
    data_root: Path,
    dry_run: bool = True,
) -> BackfillResult:
    """Fetch all available hourly injury PDFs for an ET game day."""

    # NBA publishes reports at :30 each hour
    # Try all hours during the typical slate window (tune as needed).
    hours_to_try = range(8, 24)  # ET hours

    results = []
    with NBAInjuryScraper() as scraper:
        for hour in hours_to_try:
            report_time = datetime.combine(
                target_date,
                time(hour, 30),
                tzinfo=ET_TZ
            )
            try:
                if not scraper.report_exists(report_time):
                    continue
                records = scraper.fetch_report(report_time)

                # Convert to canonical injuries_raw format (same as live ETL).
                # Recommendation: share the normalization/resolver code with projections/etl/injuries.py
                # so backfill + live produce identical schemas and IDs.
                injuries_raw = build_injuries_raw_from_records(records, report_time, data_root=data_root)
                if not dry_run:
                    storage.write_bronze_partition_hourly(
                        injuries_raw,
                        dataset="injuries_raw",
                        data_root=data_root,
                        season=season,
                        target_date=target_date,
                        hour=hour,
                    )
                results.append((hour, len(records), "success"))
            except Exception as e:
                results.append((hour, 0, str(e)))

    return BackfillResult(date=target_date, hourly_results=results)
```

### 2.2 Backfill Execution Plan

```bash
# Phase 2a: Current season (2024-25) - highest priority
uv run python -m scripts.backfill_injuries backfill --start 2024-10-22 --end 2025-12-07 --no-dry-run

# Phase 2b: Previous season (2023-24) - for training data
uv run python -m scripts.backfill_injuries backfill --start 2023-10-24 --end 2024-04-14 --no-dry-run

# Phase 2c: 2022-23 season (optional, for more training data)
uv run python -m scripts.backfill_injuries backfill --start 2022-10-18 --end 2023-04-09 --no-dry-run
```

### 2.3 Backfill Status Tracking

Create status file to track progress:

```json
// <DATA_ROOT>/bronze/injuries_raw/_backfill_status.json
{
  "last_run": "2025-12-08T10:00:00Z",
  "dates_completed": ["2024-10-22", "2024-10-23", ...],
  "dates_failed": [{"date": "2024-11-15", "error": "PDF not found"}],
  "hours_fetched": 1234,
  "hours_missing": 56
}
```

---

## Phase 3: Gold Slate Artifacts

**Priority:** HIGH - Core reproducibility feature
**Effort:** 3-4 days
**Risk:** Low (new code path, doesn't modify existing)

### 3.1 Gold Slate Artifact Contract

**Important:** Gold slates must be built from **bronze raw history**, not from the current silver snapshots. Today the silver snapshots are "latest-only" views (good for live scoring), so they cannot reconstruct `lock` vs `pretip` states for the same game.

**Output layout (write-once)**

```
<DATA_ROOT>/gold/slates/
└── season=<season>/
    └── game_date=<YYYY-MM-DD>/
        └── game_id=<game_id>/
            ├── lock.parquet
            ├── pretip.parquet
            ├── manifest.lock.json
            └── manifest.pretip.json
```

**Snapshot semantics**

- `lock`: snapshot at `tip_ts - 15 minutes`
- `pretip`: snapshot at `tip_ts` (capped to tip; never after)

**What to store in parquet**

Recommended for "perfect reproducibility":

- Store the **exact Minutes V1 feature rows** used for scoring at `snapshot_ts` (player-level table).
- Reuse `FEATURES_MINUTES_V1_SCHEMA` where possible and add minimal metadata columns:
  - `snapshot_type` (`lock`/`pretip`)
  - `snapshot_ts` (UTC)
  - `frozen_at` (UTC; when we wrote the gold file)

This makes re-scoring fully reproducible even if feature code changes later; the manifest pins code + inputs for audit.

**File:** `projections/minutes_v1/schemas.py`

```python
# Add a thin wrapper schema for frozen slate features.
# Implementation can reuse FEATURES_MINUTES_V1_SCHEMA.columns + add snapshot metadata.
SLATE_FEATURES_MINUTES_V1_SCHEMA = TableSchema(
    name="slate_features_minutes_v1",
    columns=FEATURES_MINUTES_V1_SCHEMA.columns + (
        "snapshot_type",
        "snapshot_ts",
        "frozen_at",
    ),
    pandas_dtypes={...},
    primary_key=("game_id", "player_id", "snapshot_type"),
)
```

### 3.2 Slate Freeze CLI

**File:** `projections/cli/freeze_slates.py`

```python
"""Freeze immutable slate snapshots at lock time and pre-tip."""

@app.command()
def freeze(
    game_id: int = typer.Option(..., help="Game ID to freeze"),
    snapshot_type: str = typer.Option(..., help="'lock' or 'pretip'"),
    data_root: Path = typer.Option(...),
    out_root: Path = typer.Option(None),
    force: bool = typer.Option(False, "--force", help="Overwrite existing gold snapshot."),
) -> None:
    """Create an immutable gold snapshot for a single game."""

    # Load schedule (silver) to get tip_ts + game_date.
    schedule_row = load_schedule_row(data_root, game_id)
    tip_ts = schedule_row["tip_ts"]          # UTC
    game_date = schedule_row["game_date"]    # ET date key used by gold layout
    season = schedule_row["season"]

    # Determine snapshot timestamp (UTC).
    snapshot_ts = tip_ts - timedelta(minutes=15) if snapshot_type == "lock" else tip_ts

    # Read bronze raw history for the relevant ET date.
    # injuries_raw: hourly subpartitions (hour=HH)
    injuries_raw = storage.read_bronze_day("injuries_raw", data_root, season, game_date, include_runs=False)
    # odds_raw: per-run subpartitions (run_ts=...)
    odds_raw = storage.read_bronze_day("odds_raw", data_root, season, game_date, include_runs=True)

    # Select the latest rows with as_of_ts <= snapshot_ts (no leakage past snapshot_ts).
    injuries_at_ts = select_latest_before(
        injuries_raw,
        cutoff_ts=snapshot_ts,
        group_cols=["game_id", "player_id"],
        as_of_col="as_of_ts",
    )
    odds_at_ts = select_latest_before(
        odds_raw,
        cutoff_ts=snapshot_ts,
        group_cols=["game_id"],
        as_of_col="as_of_ts",
    )

    # Load roster (silver or bronze), time-travel to snapshot_ts.
    roster_raw = load_roster_history(data_root, season, game_date)
    roster_at_ts = select_latest_before(
        roster_raw,
        cutoff_ts=snapshot_ts,
        group_cols=["game_id", "player_id"],
        as_of_col="as_of_ts",
    )

    # Build frozen feature rows (recommended) OR a minimal state snapshot (fallback).
    # Recommended: reuse MinutesFeatureBuilder but inject the as-of-filtered snapshots above.
    frozen_features = build_minutes_features_at_snapshot(
        game_id=game_id,
        snapshot_ts=snapshot_ts,
        schedule_row=schedule_row,
        injuries_snapshot=injuries_at_ts,
        odds_snapshot=odds_at_ts,
        roster_snapshot=roster_at_ts,
        data_root=data_root,
    )

    # Write to gold (write-once by default).
    out_path = (out_root or data_root / "gold" / "slates") / f"season={season}" / f"game_date={game_date}" / f"game_id={game_id}"
    out_path.mkdir(parents=True, exist_ok=True)

    parquet_path = out_path / f"{snapshot_type}.parquet"
    manifest_path = out_path / f"manifest.{snapshot_type}.json"
    if parquet_path.exists() and not force:
        raise RuntimeError(f"Gold snapshot already exists at {parquet_path}. Use --force to overwrite.")

    frozen_features.to_parquet(parquet_path, index=False)

    # Write manifest with provenance (inputs + code version + summary stats).
    manifest = {
        "game_id": game_id,
        "season": season,
        "game_date": str(game_date),
        "tip_ts": tip_ts.isoformat(),
        "snapshot_type": snapshot_type,
        "snapshot_ts": snapshot_ts.isoformat(),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_rev_parse_head(),
        "row_count": int(len(frozen_features)),
        "inputs": {
            "injuries_raw_max_as_of_ts": _max_ts(injuries_at_ts, "as_of_ts"),
            "injuries_raw_max_ingested_ts": _max_ts(injuries_at_ts, "ingested_ts"),
            "odds_raw_max_as_of_ts": _max_ts(odds_at_ts, "as_of_ts"),
            "odds_raw_max_ingested_ts": _max_ts(odds_at_ts, "ingested_ts"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
```

### 3.3 Automated Slate Freezing

Replace the bash sketch with a small Python CLI that is easy to test and shares logic with the single-game freezer.

**File:** `projections/cli/freeze_slates.py`

Commands:

- `freeze-pending --lookahead-minutes 20 --snapshot-type lock`
- `freeze-pending --lookahead-minutes 5 --snapshot-type pretip`

```ini
[Unit]
Description=Freeze slate snapshots (lock/pretip)
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/daniel/projects/projections-v2
ExecStart=/home/daniel/.local/bin/uv run python -m projections.cli.freeze_slates freeze-pending
```

Add a corresponding timer (cadence similar to live-score):

**File:** `systemd/freeze-slates.timer`

```ini
[Unit]
Description=Freeze slates every 5 minutes

[Timer]
OnCalendar=*-*-* 08..23:*:00/5
Unit=freeze-slates.service
RandomizedDelaySec=30
Persistent=true

[Install]
WantedBy=timers.target
```

### 3.4 Backfill Historical Slates

**File:** `scripts/backfill_slates.py`

```python
"""Backfill gold/slates for historical games using bronze hourly data."""

def backfill_game_slate(
    game_id: int,
    tip_ts: datetime,
    data_root: Path,
) -> bool:
    """Create slate snapshots for a historical game."""

    # For lock snapshot: find bronze data with as_of_ts <= tip_ts - 15min
    lock_ts = tip_ts - timedelta(minutes=15)

    # Load bronze history for the game_date (ET partition key).
    injuries_bronze = load_all_hourly_injuries(data_root, season, game_date)
    odds_bronze = load_all_run_odds(data_root, season, game_date)

    # Filter to as_of_ts <= lock_ts, take latest per player
    injuries_at_lock = select_latest_before(injuries_bronze, lock_ts)

    # Same for odds and roster. If odds history is missing for a date, still write the slate but
    # record missing inputs explicitly in the manifest.
    # Build and write lock + pretip snapshots.
```

---

## Phase 4: Training Data Separation

**Priority:** MEDIUM - Enables clean experimentation
**Effort:** 2-3 days
**Risk:** Low

### 4.1 Training Dataset Structure

```
data/training/
├── datasets/
│   └── v1_20251208/
│       ├── features.parquet      # All features for date range
│       ├── labels.parquet        # Corresponding labels
│       ├── manifest.json         # Metadata
│       └── feature_hash.txt      # Hash of feature columns
│
└── splits/
    └── v1_20251208/
        ├── train_ids.parquet     # game_id, player_id for train
        ├── val_ids.parquet
        ├── test_ids.parquet
        └── split_config.json     # How split was created
```

### 4.2 Dataset Builder CLI

**File:** `projections/cli/build_training_dataset.py`

```python
@app.command()
def build(
    version: str = typer.Option(..., help="Dataset version (e.g., v1_20251208)"),
    start_date: datetime = typer.Option(...),
    end_date: datetime = typer.Option(...),
    data_root: Path = typer.Option(...),
    out_root: Path = typer.Option(None),
    use_slates: bool = typer.Option(True, help="Use gold/slates if available"),
    slate_snapshot_type: str = typer.Option(
        "pretip",
        help="Which slate snapshot to use when --use-slates (pretip recommended for no-leak training).",
    ),
) -> None:
    """Build a versioned training dataset from gold data."""

    out_path = (out_root or data_root / "training" / "datasets") / version
    out_path.mkdir(parents=True, exist_ok=True)

    # Load features from gold (or slates)
    if use_slates:
        features = load_features_from_slates(
            data_root,
            start_date,
            end_date,
            snapshot_type=slate_snapshot_type,
        )
    else:
        features = load_features_from_gold(data_root, start_date, end_date)

    # Load labels
    labels = load_labels(data_root, start_date, end_date)

    # Merge and validate
    dataset = features.merge(labels, on=["game_id", "player_id"], how="inner")

    # Write
    dataset.to_parquet(out_path / "features.parquet", index=False)

    # Write manifest
    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "row_count": len(dataset),
        "game_count": dataset["game_id"].nunique(),
        "player_count": dataset["player_id"].nunique(),
        "feature_columns": list(features.columns),
        "feature_hash": hash_columns(features.columns),
        "source": f"gold/slates/{slate_snapshot_type}" if use_slates else "gold/features_minutes_v1",
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
```

### 4.3 Split Generator

**File:** `projections/cli/create_splits.py`

```python
@app.command()
def create_temporal_split(
    dataset_version: str,
    train_end: datetime,
    val_end: datetime,
    data_root: Path,
) -> None:
    """Create temporal train/val/test split (no leakage)."""

    dataset_path = data_root / "training" / "datasets" / dataset_version
    df = pd.read_parquet(dataset_path / "features.parquet")

    df["game_date"] = pd.to_datetime(df["game_date"])

    train_mask = df["game_date"] <= train_end
    val_mask = (df["game_date"] > train_end) & (df["game_date"] <= val_end)
    test_mask = df["game_date"] > val_end

    splits_path = data_root / "training" / "splits" / dataset_version
    splits_path.mkdir(parents=True, exist_ok=True)

    df.loc[train_mask, ["game_id", "player_id"]].to_parquet(splits_path / "train_ids.parquet")
    df.loc[val_mask, ["game_id", "player_id"]].to_parquet(splits_path / "val_ids.parquet")
    df.loc[test_mask, ["game_id", "player_id"]].to_parquet(splits_path / "test_ids.parquet")
```

### 4.4 Training Script Update

**File:** `projections/train.py`

```python
@app.command()
def train_from_dataset(
    dataset_version: str = typer.Option(...),
    split_version: str = typer.Option(None, help="Defaults to dataset_version"),
    data_root: Path = typer.Option(...),
    output_dir: Path = typer.Option(...),
) -> None:
    """Train model from versioned dataset with reproducible splits."""

    split_version = split_version or dataset_version

    # Load dataset
    dataset = pd.read_parquet(data_root / "training" / "datasets" / dataset_version / "features.parquet")

    # Load splits
    splits_path = data_root / "training" / "splits" / split_version
    train_ids = pd.read_parquet(splits_path / "train_ids.parquet")
    val_ids = pd.read_parquet(splits_path / "val_ids.parquet")

    # Filter dataset
    train_df = dataset.merge(train_ids, on=["game_id", "player_id"])
    val_df = dataset.merge(val_ids, on=["game_id", "player_id"])

    # Train...
    # Save bundle with dataset_version in metadata
```

---

## Phase 5: Migration & Cleanup

**Priority:** LOW - After other phases stable
**Effort:** 1-2 days

### 5.1 Compact Historical Bronze

Once backfill is complete and slates are frozen:
- `injuries_raw`: optionally compact `hour=*` into the flat daily "latest view" for very old dates (or keep hourly indefinitely; size is modest).
- `odds_raw`: prune `run_ts=*` partitions that are **not referenced** by any gold slate manifest; optionally keep an hourly summary for long-term storage.
- Never delete any partition needed to reproduce an existing gold slate (gold manifests are the source of truth).

### 5.2 Archive Old Live Data

```bash
# Move old live features to archive
mv data/live/features_minutes_v1/2024-* /archive/live/
```

### 5.3 Update Documentation

- CLAUDE.md
- docs/pipeline/ETL_AUDIT_REPORT.md
- Add docs/pipeline/TRAINING_DATA.md

---

## Implementation Schedule

| Phase | Task | Effort | Dependencies |
|-------|------|--------|--------------|
| 1.1 | Bronze subpartition writers (hour + run) | 3 hours | None |
| 1.2 | Update injuries ETL (ET hour + dual-write) | 4 hours | 1.1 |
| 1.3 | Update odds ETL (per-run + dual-write) | 3 hours | 1.1 |
| 1.4 | Update read helpers + impacted scripts | 3 hours | 1.1 |
| 1.5 | Retention placeholder (defer to Phase 5) | 0.5 hours | 1.4 |
| **Phase 1 Total** | | **~1.5 days** | |
| 2.1 | Enhance backfill script | 4 hours | 1.2 |
| 2.2 | Run backfill (2024-25) | 8 hours runtime | 2.1 |
| 2.3 | Run backfill (2023-24) | 8 hours runtime | 2.1 |
| **Phase 2 Total** | | **~2 days** | |
| 3.1 | Gold slate contract + schema wrapper | 2 hours | None |
| 3.2 | Freeze CLI | 4 hours | 3.1, 1.4 |
| 3.3 | Automated freezing | 3 hours | 3.2 |
| 3.4 | Backfill historical slates | 4 hours | 3.2, 2.2 |
| **Phase 3 Total** | | **~2 days** | |
| 4.1-4.4 | Training separation | 8 hours | 3.4 |
| **Phase 4 Total** | | **~1 day** | |
| 5.1-5.3 | Migration & cleanup | 4 hours | All above |
| **Phase 5 Total** | | **~0.5 days** | |

**Total Estimated Effort:** ~7-8 working days

---

## Validation Checklist

After each phase, verify:

### Phase 1 Validation
- [ ] Injuries scrapes write to `hour=*` subpartitions (ET-based date/hour)
- [ ] Odds scrapes write to `run_ts=*` subpartitions (no intra-hour overwrite)
- [ ] Legacy flat daily files still written/readable during dual-write rollout
- [ ] Live pipeline continues working (silver outputs unchanged; bronze dual-write does not break readers)
- [ ] Backfill-mode injuries load (`build_minutes_live --backfill-mode`) avoids double-counting in `date=.../` partitions
- [ ] If adopting “Live Pipeline Hardening”: overlapping systemd runs serialize via `flock`

### Phase 2 Validation
- [ ] Historical PDFs fetched successfully
- [ ] Player ID resolution working
- [ ] Backfill status tracked

### Phase 3 Validation
- [ ] Lock snapshots created at T-15
- [ ] Pretip snapshots created at T-0
- [ ] Snapshot creation uses bronze history (not silver latest-only views)
- [ ] Gold snapshots are write-once by default (require `--force` to overwrite)
- [ ] Manifest records git SHA + input provenance (paths + max as_of/ingested timestamps)
- [ ] Can re-score minutes directly from frozen feature rows

### Phase 4 Validation
- [ ] Training dataset versioned correctly
- [ ] Splits are temporally correct (no leakage)
- [ ] Model training uses dataset version
- [ ] Experiment reproducible from manifest

---

## Rollback Plan

Each phase is additive and backward-compatible:

- **Phase 1:** Keep old `write_bronze_partition()` working; dual-write ensures legacy daily files remain available while new hourly/per-run history is added
- **Phase 2:** Backfill only adds data, doesn't modify existing
- **Phase 3:** Gold slates are new directory, doesn't affect existing gold
- **Phase 4:** Training datasets separate from live pipeline

If issues arise, simply revert to using old code paths while debugging.

---

## Open Questions

1. **Live pipeline hardening:** Consolidate on `live-score.timer` as the single orchestrator (recommended), or keep separate timers and implement shared locks + run-id plumbing fixes?

2. **Odds backfill:** Should we invest in a historical odds API subscription? (SportsDataIO ~$50/month)

3. **Retention policy:** How long to keep `odds_raw/run_ts=*` history before pruning to only gold-referenced snapshots? (Proposed: keep 30 days rolling; prune older by manifest references.)

4. **Odds selection policy:** If multiple books/markets exist, what is the deterministic "preferred" odds line per game at snapshot time?

5. **Training snapshot type:** Default training to `pretip` (recommended) or also publish a `lock` variant for ablation/earlier-decision use cases?

6. **Training dataset versioning:** Semantic versioning (v1.0.0) or date-based (v1_20251208)?

---

*Plan created by Claude Code - Ready for review and implementation*
