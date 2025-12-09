# Immutable Gold Architecture - Implementation Plan

**Created:** 2025-12-08
**Status:** Draft
**Goal:** Enable perfect reproducibility of any historical slate from the last known data before tip

---

## Overview

This plan restructures the data pipeline to:
1. **Preserve hourly snapshots** in bronze (stop overwriting)
2. **Create immutable slate artifacts** in gold (per-game frozen state)
3. **Separate training data** from live pipeline
4. **Enable reliable backfill** for historical dates

---

## Phase 1: Stop the Bleeding (Bronze Hourly Partitions)

**Priority:** CRITICAL - Do this first to prevent future data loss
**Effort:** 1-2 days
**Risk:** Low (additive change)

### 1.1 Modify Bronze Storage Contract

**File:** `projections/etl/storage.py`

```python
# Add new function for hourly partitioning
def write_bronze_partition_hourly(
    df: pd.DataFrame,
    dataset: str,
    data_root: Path,
    season: int,
    target_date: date,
    hour: int,
    bronze_root: Path | None = None,
) -> BronzeWriteResult:
    """Write bronze data with hourly partition for append-only semantics."""
    root = bronze_root or default_bronze_root(dataset, data_root)
    partition_path = (
        root
        / f"season={season}"
        / f"date={target_date.isoformat()}"
        / f"hour={hour:02d}"
    )
    partition_path.mkdir(parents=True, exist_ok=True)
    file_path = partition_path / f"{dataset}.parquet"
    df.to_parquet(file_path, index=False)
    return BronzeWriteResult(path=file_path, rows=len(df), target_date=target_date)
```

### 1.2 Update Injuries ETL

**File:** `projections/etl/injuries.py`

Changes:
- Extract hour from `as_of_ts`
- Call `write_bronze_partition_hourly()` instead of `write_bronze_partition()`
- Keep backward compatibility for reads (glob across hours)

```python
# In _persist_bronze():
for cursor in storage.iter_days(start_day, end_day):
    day_mask = bronze_normalized == cursor_utc
    if not day_mask.any():
        continue
    day_frame = bronze_df.loc[day_mask].copy()

    # NEW: Group by hour and write separate partitions
    day_frame["_hour"] = day_frame["as_of_ts"].dt.hour
    for hour, hour_frame in day_frame.groupby("_hour"):
        storage.write_bronze_partition_hourly(
            hour_frame.drop(columns=["_hour"]),
            dataset="injuries_raw",
            data_root=data_root,
            season=season,
            target_date=cursor.date(),
            hour=int(hour),
            bronze_root=bronze_root_path,
        )
```

### 1.3 Update Odds ETL

**File:** `projections/etl/odds.py`

Same pattern as injuries - partition by hour based on `as_of_ts`.

### 1.4 Update Read Functions

**File:** `projections/etl/storage.py`

```python
def read_bronze_day(
    dataset: str,
    data_root: Path,
    season: int,
    target_date: date,
) -> pd.DataFrame:
    """Read all hourly partitions for a given day."""
    root = default_bronze_root(dataset, data_root)
    day_path = root / f"season={season}" / f"date={target_date.isoformat()}"

    # Handle both old (flat) and new (hourly) formats
    flat_file = day_path / f"{dataset}.parquet"
    if flat_file.exists():
        return pd.read_parquet(flat_file)

    # Glob hourly partitions
    hourly_files = sorted(day_path.glob(f"hour=*/{dataset}.parquet"))
    if not hourly_files:
        return pd.DataFrame()

    frames = [pd.read_parquet(f) for f in hourly_files]
    return pd.concat(frames, ignore_index=True)
```

### 1.5 Retention Policy

Add cleanup script for bronze hourly data older than 30 days:

**File:** `scripts/cleanup_bronze_hourly.py`

```python
def cleanup_old_hourly_partitions(data_root: Path, days_to_keep: int = 30):
    """Compact hourly partitions older than N days into daily summaries."""
    cutoff = date.today() - timedelta(days=days_to_keep)
    # For dates before cutoff: merge hourly -> daily, delete hourly
    # This preserves the "latest pre-tip" snapshot while saving space
```

---

## Phase 2: Backfill Historical Injuries

**Priority:** HIGH - Recover what we can
**Effort:** 2-3 days (mostly runtime)
**Risk:** Medium (NBA.com rate limits, missing PDFs)

### 2.1 Enhance Backfill Script

**File:** `scripts/backfill_injuries.py`

Current script fetches 1 report per day (1 hour before tip). Enhance to:
- Fetch ALL available hourly reports for each game day
- Write to hourly bronze partitions
- Track which hours were successfully fetched

```python
def backfill_all_hourly_reports(
    target_date: date,
    data_root: Path,
    dry_run: bool = True,
) -> BackfillResult:
    """Fetch all available hourly injury PDFs for a date."""

    # NBA publishes reports at :30 each hour
    # Try all hours from 10:30 AM ET to 11:30 PM ET
    hours_to_try = range(10, 24)  # ET hours

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
                # Convert to bronze format and write
                bronze_df = records_to_bronze_df(records, report_time)
                if not dry_run:
                    storage.write_bronze_partition_hourly(
                        bronze_df,
                        dataset="injuries_raw",
                        hour=hour,
                        ...
                    )
                results.append((hour, len(records), "success"))
            except Exception as e:
                results.append((hour, 0, str(e)))

    return BackfillResult(date=target_date, hourly_results=results)
```

### 2.2 Backfill Execution Plan

```bash
# Phase 2a: Current season (2024-25) - highest priority
python -m scripts.backfill_injuries --start 2024-10-22 --end 2025-12-07 --no-dry-run

# Phase 2b: Previous season (2023-24) - for training data
python -m scripts.backfill_injuries --start 2023-10-24 --end 2024-04-14 --no-dry-run

# Phase 2c: 2022-23 season (optional, for more training data)
python -m scripts.backfill_injuries --start 2022-10-18 --end 2023-04-09 --no-dry-run
```

### 2.3 Backfill Status Tracking

Create status file to track progress:

```json
// data/bronze/injuries_raw/_backfill_status.json
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

### 3.1 Slate Schema Definition

**File:** `projections/minutes_v1/schemas.py`

```python
SLATE_SNAPSHOT_SCHEMA = TableSchema(
    name="slate_snapshot",
    columns=(
        # Identity
        "game_id",
        "game_date",
        "tip_ts",
        "snapshot_type",      # "lock" (T-15) or "pretip" (T-0)
        "snapshot_ts",        # When this snapshot was created

        # Injury state (per player)
        "player_id",
        "player_name",
        "team_id",
        "injury_status",
        "injury_as_of_ts",
        "restriction_flag",
        "ramp_flag",

        # Odds state (per game, denormalized)
        "spread_home",
        "total",
        "odds_as_of_ts",
        "odds_book",

        # Roster state
        "active_flag",
        "starter_flag",
        "is_projected_starter",
        "is_confirmed_starter",
        "roster_as_of_ts",

        # Derived features (subset for reproducibility)
        "roll_mean_5",
        "roll_mean_10",
        "days_since_last",
        "is_b2b",
    ),
    pandas_dtypes={...},
    primary_key=("game_id", "player_id", "snapshot_type"),
)
```

### 3.2 Slate Freeze CLI

**File:** `projections/cli/freeze_slate.py`

```python
"""Freeze immutable slate snapshots at lock time and pre-tip."""

@app.command()
def freeze(
    game_id: int = typer.Option(..., help="Game ID to freeze"),
    snapshot_type: str = typer.Option(..., help="'lock' or 'pretip'"),
    data_root: Path = typer.Option(...),
    out_root: Path = typer.Option(None),
) -> None:
    """Create immutable slate snapshot for a single game."""

    # Load schedule to get tip_ts
    schedule = load_schedule(data_root, game_id)
    tip_ts = schedule["tip_ts"]
    game_date = schedule["game_date"]

    # Determine snapshot timestamp
    if snapshot_type == "lock":
        snapshot_ts = tip_ts - timedelta(minutes=15)
    else:  # pretip
        snapshot_ts = tip_ts

    # Load silver data with as_of_ts <= snapshot_ts
    injuries = load_injuries_as_of(data_root, game_id, snapshot_ts)
    odds = load_odds_as_of(data_root, game_id, snapshot_ts)
    roster = load_roster_as_of(data_root, game_id, snapshot_ts)

    # Build slate snapshot
    slate = build_slate_snapshot(
        game_id=game_id,
        tip_ts=tip_ts,
        snapshot_type=snapshot_type,
        snapshot_ts=snapshot_ts,
        injuries=injuries,
        odds=odds,
        roster=roster,
    )

    # Write to gold
    out_path = (
        out_root or data_root / "gold" / "slates"
    ) / f"game_date={game_date}" / f"game_id={game_id}"
    out_path.mkdir(parents=True, exist_ok=True)

    slate.to_parquet(out_path / f"{snapshot_type}.parquet", index=False)

    # Write manifest
    manifest = {
        "game_id": game_id,
        "game_date": game_date.isoformat(),
        "tip_ts": tip_ts.isoformat(),
        "snapshot_type": snapshot_type,
        "snapshot_ts": snapshot_ts.isoformat(),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(slate),
        "injuries_as_of": injuries["as_of_ts"].max().isoformat() if not injuries.empty else None,
        "odds_as_of": odds["as_of_ts"].max().isoformat() if not odds.empty else None,
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
```

### 3.3 Automated Slate Freezing

**File:** `systemd/freeze-slates.service`

```ini
[Unit]
Description=Freeze slate snapshots at lock time
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/daniel/projects/projections-v2
ExecStart=/home/daniel/projects/projections-v2/scripts/freeze_pending_slates.sh
```

**File:** `scripts/freeze_pending_slates.sh`

```bash
#!/bin/bash
# Find games tipping in the next 20 minutes that don't have lock snapshots
# Freeze them

NOW_UTC=$(date -u +%s)
LOCK_WINDOW=1200  # 20 minutes

# Query schedule for games in window
# For each game without lock snapshot: freeze it
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

    # Load all hourly bronze partitions for the game date
    injuries_bronze = load_all_hourly_injuries(data_root, game_date)

    # Filter to as_of_ts <= lock_ts, take latest per player
    injuries_at_lock = select_latest_before(injuries_bronze, lock_ts)

    # Same for odds, roster
    # Build and write slate
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
) -> None:
    """Build a versioned training dataset from gold data."""

    out_path = (out_root or data_root / "training" / "datasets") / version
    out_path.mkdir(parents=True, exist_ok=True)

    # Load features from gold (or slates)
    if use_slates:
        features = load_features_from_slates(data_root, start_date, end_date)
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
        "source": "gold/slates" if use_slates else "gold/features_minutes_v1",
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
- Merge hourly partitions older than 30 days into daily
- Keep only "last pre-tip" snapshot for storage efficiency

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
| 1.1 | Bronze hourly storage function | 2 hours | None |
| 1.2 | Update injuries ETL | 4 hours | 1.1 |
| 1.3 | Update odds ETL | 2 hours | 1.1 |
| 1.4 | Update read functions | 2 hours | 1.1 |
| 1.5 | Retention policy script | 2 hours | 1.4 |
| **Phase 1 Total** | | **~1.5 days** | |
| 2.1 | Enhance backfill script | 4 hours | 1.2 |
| 2.2 | Run backfill (2024-25) | 8 hours runtime | 2.1 |
| 2.3 | Run backfill (2023-24) | 8 hours runtime | 2.1 |
| **Phase 2 Total** | | **~2 days** | |
| 3.1 | Slate schema | 1 hour | None |
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
- [ ] New scrapes write to hourly partitions
- [ ] Old flat files still readable
- [ ] Live pipeline works unchanged

### Phase 2 Validation
- [ ] Historical PDFs fetched successfully
- [ ] Player ID resolution working
- [ ] Backfill status tracked

### Phase 3 Validation
- [ ] Lock snapshots created at T-15
- [ ] Pretip snapshots created at T-0
- [ ] Manifest contains all metadata
- [ ] Can reconstruct features from slate alone

### Phase 4 Validation
- [ ] Training dataset versioned correctly
- [ ] Splits are temporally correct (no leakage)
- [ ] Model training uses dataset version
- [ ] Experiment reproducible from manifest

---

## Rollback Plan

Each phase is additive and backward-compatible:

- **Phase 1:** Keep old `write_bronze_partition()` working, new code writes hourly in parallel
- **Phase 2:** Backfill only adds data, doesn't modify existing
- **Phase 3:** Gold slates are new directory, doesn't affect existing gold
- **Phase 4:** Training datasets separate from live pipeline

If issues arise, simply revert to using old code paths while debugging.

---

## Open Questions

1. **Odds backfill:** Should we invest in a historical odds API subscription? (SportsDataIO ~$50/month)

2. **Retention policy:** How long to keep hourly granularity? (Proposed: 30 days)

3. **Slate features:** Which derived features to include in slate snapshots vs. recompute on demand?

4. **Training dataset versioning:** Semantic versioning (v1.0.0) or date-based (v1_20251208)?

---

*Plan created by Claude Code - Ready for review and implementation*
