# ETL Pipeline Audit Report

**Audit Date:** 2025-12-08
**Auditor:** Claude Code
**Scope:** Full pipeline trace - data sources, schemas, transformations, orchestration, gaps

---

## Executive Summary

The projections pipeline implements a **Medallion Architecture** (Bronze → Silver → Gold) with strong anti-leak enforcement. The system is well-designed for a solo project but has **critical gaps in data persistence** that prevent reliable slate reconstruction from historical snapshots.

### Key Findings

| Area | Status | Notes |
|------|--------|-------|
| Data Sources | Good | 9 scrapers covering all required inputs |
| Schema Enforcement | Good | Pandera validation, typed columns |
| Anti-Leak Controls | Excellent | Strict `as_of_ts ≤ tip_ts` throughout |
| Bronze Persistence | **GAP** | Daily partitions overwrite, no hourly retention |
| Silver Snapshots | Partial | Appended but filtered by latest-only selection |
| Gold Immutability | Good | Labels frozen, features monthly |
| Orchestration | Functional | systemd works but lacks observability |
| Reproducibility | **GAP** | Cannot reconstruct pre-tip state reliably |
| **Training Data** | **CRITICAL** | Historical rates_training_base severely incomplete (see below) |

### Critical Issue: Rates Training Base Incomplete

> [!CAUTION]
> **The `gold/rates_training_base` for 2022-2024 contains only ~20 rows/day instead of ~150+ expected.**
> This limits rates model training to ~6,000 rows when ~60,000+ should be available.

**Discovery:** During MLFlow integration, noticed train/cal/val split was 3,206 / 2,430 / 7,584 rows.
Investigation revealed the historical backfill captured only a fraction of players.

| Season | Rows | Dates | Avg Rows/Date | Expected |
|--------|------|-------|---------------|----------|
| 2022 | 2,873 | 144 | **20** | ~150 |
| 2023 | 3,206 | 163 | **20** | ~150 |
| 2024 | 3,293 | 166 | **20** | ~150 |
| 2025 | 6,721 | 44 | **153** | ✅ Correct |

**Root Cause:** The `build_training_base.py` script uses `game_date=` partition lookups, but historical odds/roster data is stored in `month=` partitions. Rows with missing joins were filtered out.

**Impact:**
- Rates models trained on ~10% of available data
- Feature distributions may be biased toward complete rows
- Model performance potentially degraded

**Remediation:**
1. Re-run `build_training_base.py` for 2023-10-01 to present (skip 2022-23 which lacks odds)
2. Verify partition path handling for month-partitioned sources
3. Target: ~60,000+ training rows for 2023-24 + 2024-25 seasons

---

## 1. Data Sources Inventory

### 1.1 External APIs / Scrapers

| Source | Scraper | Frequency | Data | Anti-Leak |
|--------|---------|-----------|------|-----------|
| NBA Injury PDFs | `scrapers/nba_injuries.py` | Hourly (:30) | Status, restrictions, return tracking | `as_of_ts` from PDF timestamp |
| NBA Schedule | `scrapers/nba_schedule.py` | Daily | Games, tip times, teams, venues | Static (tip_ts) |
| NBA Rosters | `scrapers/nba_players.py` | On-demand | Active players, teams, positions | `as_of_ts` from scrape time |
| NBA Daily Lineups | `scrapers/nba_daily_lineups.py` | Daily | Projected/confirmed starters | `lineup_timestamp` |
| NBA Box Scores | `scrapers/nba_boxscore.py` | Post-game | Actual minutes, stats | `label_frozen_ts` |
| Oddstrader | `scrapers/oddstrader.py` | Every 30min | Spread, total, book | `as_of_ts` from API |
| **DK Contests** | `scrapers/dk_contests/` | Post-contest | Lineups, ownership, payouts | Post-game only |
| **Linestar** | `scrapers/linestar/` | Daily | Projected ownership, FPTS projections | Pre-game |

### 1.2 DraftKings Contest Scrapers (Detail)

The `dk_contests/` directory contains a complete DK scraping ecosystem:

| Script | Purpose | Output |
|--------|---------|--------|
| `nba_gpp_scraper.py` | Discover available GPP contests | Contest metadata |
| `download_contest_results.py` | Download lineup CSVs for completed contests | `bronze/dk_contests/nba_gpp_data/{date}/results/` |
| `auth.py` | DK authentication/session management | Cookies/tokens |
| `build_ownership_data.py` | Aggregate lineups → player ownership | `bronze/dk_contests/ownership_by_slate/` |
| `prelock.py` | Capture pre-lock contest state | Pre-lock snapshots |
| `payouts_scraper.py` | Scrape payout structures | Prize pool data |
| `analyze_contests.py` | Analytics on contest results | Analysis reports |

**Important:** DK contest data is key for training the ownership model. The `build_ownership_data.py` script clusters contests by player overlap and computes entry-weighted average ownership per slate.

### 1.3 Linestar Scrapers (Detail)

The `scrapers/linestar/` directory contains historical Linestar data backfills:
- Projected ownership percentages
- FPTS projections
- Player exposure data

**Note:** Linestar data has a weak correlation (~0.11) with actual DK ownership. DK contest data (`ownership_by_slate/`) is preferred for ownership model training.

### 1.4 Source Dependencies

```
NBA.com Schedule API
    └─> tip_ts (anchor for all temporal filtering)

NBA.com Injury PDFs (hourly)
    └─> injuries_snapshot (selected: latest where as_of_ts ≤ tip_ts)

Oddstrader GraphQL API
    └─> odds_snapshot (selected: latest where as_of_ts ≤ tip_ts)

NBA.com Daily Lineups + Players API
    └─> roster_nightly (merged with starter signals)

NBA.com Box Scores (post-game)
    └─> labels/boxscore_labels.parquet (IMMUTABLE)

DraftKings (post-contest)
    └─> bronze/dk_contests/ownership_by_slate/ (ownership labels)

Linestar (daily, pre-game)
    └─> gold/ownership_training_base/ (training data, deprecated)
```

---

## 2. Data Flow Architecture

### 2.1 Bronze Layer

**Purpose:** Raw, unprocessed data as received from sources
**Path:** `data/bronze/<dataset>/season=<YYYY>/date=<YYYY-MM-DD>/`

| Dataset | File | Schema | Retention |
|---------|------|--------|-----------|
| injuries_raw | injuries.parquet | INJURIES_RAW_SCHEMA | Daily (overwrites) |
| odds_raw | odds.parquet | ODDS_RAW_SCHEMA | Daily (overwrites) |
| roster_nightly_raw | roster.parquet | ROSTER_NIGHTLY_RAW_SCHEMA | Daily (overwrites) |
| daily_lineups | daily_lineups_raw.parquet | Raw JSON payload | Daily (overwrites) |
| boxscores_raw | boxscores_raw.parquet | Raw JSON payload | Daily (appends) |

**CRITICAL GAP:** Bronze partitions are overwritten on each scrape. Hourly injury snapshots from the same day are lost.

### 2.2 Silver Layer

**Purpose:** Cleaned, normalized snapshots with temporal validation
**Path:** `data/silver/<dataset>/season=<YYYY>/month=<MM>/`

| Dataset | File | Schema | Selection Rule |
|---------|------|--------|----------------|
| injuries_snapshot | injuries_snapshot.parquet | INJURIES_SNAPSHOT_SCHEMA | Latest `as_of_ts ≤ tip_ts` per game/player |
| odds_snapshot | odds_snapshot.parquet | ODDS_SNAPSHOT_SCHEMA | Latest `as_of_ts ≤ tip_ts` per game |
| roster_nightly | roster.parquet | ROSTER_NIGHTLY_SCHEMA | Latest `as_of_ts` per game/player |
| schedule | schedule.parquet | SCHEDULE_SCHEMA | Static reference |
| nba_daily_lineups | lineups.parquet | Normalized lineup | Daily partition by date |

**Snapshot Selection Logic:**
```python
# From projections/etl/injuries.py
def select_injury_snapshot(bronze_df, schedule_df):
    # For each (game_id, player_id):
    # Select row with MAX(as_of_ts) WHERE as_of_ts <= tip_ts
```

**GAP:** Silver appends new snapshots but the selection function only picks the latest. Historical pre-tip snapshots are present but not easily queryable by run timestamp.

### 2.3 Gold Layer

**Purpose:** Analysis-ready features for training/inference
**Path:** `data/gold/features_minutes_v1/season=<YYYY>/month=<MM>/`

| Dataset | Purpose | Schema |
|---------|---------|--------|
| features_minutes_v1 | Training features | FEATURES_MINUTES_V1_SCHEMA (73 cols) |
| labels_minutes_v1 | Daily labels (from boxscores) | BOX_SCORE_LABELS_SCHEMA |
| projections_minutes_v1 | Live predictions (copied from artifacts) | Prediction output |
| prediction_logs_minutes | Audit trail of all predictions | Full feature + prediction |

### 2.4 Labels (Immutable)

**Purpose:** Frozen ground truth
**Path:** `data/labels/season=<YYYY>/boxscore_labels.parquet`

- Written once via `freeze_boxscore_labels()`
- Append-only with deduplication
- Contains: game_id, player_id, minutes (actual), starter_flag_label
- `label_frozen_ts` tracks capture time

---

## 3. Schema Definitions

### 3.1 Key Schemas (from `projections/minutes_v1/schemas.py`)

**Type Aliases:**
```python
STRING_DTYPE = "string[pyarrow]"
INT_DTYPE = "Int64"           # Nullable integer
FLOAT_DTYPE = "Float64"       # Nullable float
BOOL_DTYPE = "boolean"
UTC_TS = "datetime64[ns, UTC]"
NAIVE_TS = "datetime64[ns]"
```

**Primary Keys:**
| Schema | Primary Key |
|--------|-------------|
| INJURIES_SNAPSHOT | (game_id, player_id) |
| ODDS_SNAPSHOT | (game_id) |
| ROSTER_NIGHTLY | (team_id, game_date, player_id) |
| SCHEDULE | (game_id) |
| BOX_SCORE_LABELS | (game_id, player_id) |
| FEATURES_MINUTES_V1 | (game_id, player_id, team_id) |

### 3.2 Critical Timestamp Columns

| Column | Purpose | Validation |
|--------|---------|------------|
| `tip_ts` | Game tipoff (UTC) | Anchor for all temporal filtering |
| `as_of_ts` | Knowledge cutoff | Must be ≤ tip_ts for features |
| `ingested_ts` | When data was scraped | Audit only |
| `feature_as_of_ts` | Max of input as_of timestamps | Must be ≤ tip_ts |
| `label_frozen_ts` | When labels were captured | Audit only |

---

## 4. Orchestration Analysis

### 4.1 Systemd Timers

| Timer | Schedule | Service | Purpose |
|-------|----------|---------|---------|
| `live-scrape.timer` | :05, :35 (8am-11pm) | live-scrape.service | Scrape injuries/odds/roster |
| `live-score.timer` | Every 5 min (8am-11pm) | live-score.service | Full scoring pipeline |
| `live-boxscores.timer` | 3:30 AM daily | live-boxscores.service | Backfill box scores |
| `gold-features-daily.timer` | 10:00 AM daily | gold-features-daily.service | Build gold features |
| `nightly-eval.timer` | Daily | nightly-eval.service | Accuracy evaluation |
| `dk-ownership-aggregate.timer` | Daily | dk-ownership-aggregate.service | Aggregate DK ownership |

### 4.2 Live Scoring Flow (`run_live_score.sh`)

```
1. Determine phase (PRE_WINDOW / SLATE / POST)
   - PRE_WINDOW: >90min before first tip → run every 30min
   - SLATE: 90min before to last tip → run every 5min
   - POST: after all tips → skip

2. Scrape live data
   └─> projections.cli.live_pipeline
       ├─> injuries ETL
       ├─> daily_lineups ETL
       ├─> odds ETL
       └─> roster_nightly ETL

3. Build features
   └─> projections.cli.build_minutes_live
       ├─> Load historical labels
       ├─> Build live stubs for today
       ├─> Filter snapshots by run_as_of_ts
       └─> Run MinutesFeatureBuilder

4. Score model
   └─> projections.cli.score_minutes_v1
       ├─> Load LightGBM bundle
       ├─> Predict p10/p50/p90
       ├─> Apply conformal calibration
       └─> Optional: L2 reconciliation

5. Copy to gold
   └─> gold/projections_minutes_v1/game_date=<date>/

6. Build rates features
   └─> projections.cli.build_rates_features_live

7. Score rates
   └─> projections.cli.score_rates_live

8. Run simulation
   └─> scripts.sim_v2.run_sim_live (10k worlds)

9. Score ownership
   └─> projections.cli.score_ownership_live

10. Finalize unified projections
    └─> projections.cli.finalize_projections

11. Health checks
    └─> projections.cli.check_health
```

### 4.3 Timing Analysis

**Injury PDF Availability:**
- Published at :30 each hour (e.g., 5:30 PM ET for 5 PM data)
- Scrape runs at :05 and :35 → good coverage

**Live Score Timing:**
- Runs every 5 min during SLATE window
- Derives `run_as_of_ts` from latest injury snapshot
- This ensures temporal consistency

**GAP:** No alerting if scrapes fail. Pipeline continues with stale data.

---

## 5. Identified Gaps

### 5.1 CRITICAL: Bronze Overwrites

**Problem:** Bronze daily partitions are overwritten on each scrape.

```
data/bronze/injuries_raw/season=2025/date=2025-12-07/injuries.parquet
```

If you scrape at 5:35 PM and 6:35 PM, only the 6:35 PM data survives.

**Impact:**
- Cannot reconstruct the exact injury state from 5:35 PM
- Historical analysis limited to latest snapshot per day
- Backfills cannot reproduce pre-tip information state

**Recommendation:** Add hourly partitioning to bronze:
```
data/bronze/injuries_raw/season=2025/date=2025-12-07/hour=17/injuries.parquet
data/bronze/injuries_raw/season=2025/date=2025-12-07/hour=18/injuries.parquet
```

### 5.2 CRITICAL: Silver Snapshot Selection

**Problem:** Silver layer appends snapshots but `select_injury_snapshot()` only returns the latest pre-tip record.

**Impact:**
- You have the data but can't query "what was known at 5:35 PM?"
- Live runs work (they use current time) but backfills are approximate

**Recommendation:** Add explicit `run_as_of_ts` filtering to snapshot selection:
```python
def select_injury_snapshot(bronze_df, schedule_df, run_as_of_ts=None):
    if run_as_of_ts:
        # Filter to as_of_ts <= run_as_of_ts AND as_of_ts <= tip_ts
        # Return latest per (game_id, player_id) meeting both constraints
```

### 5.3 HIGH: No Pre-Tip Snapshot Freezing

**Problem:** There's no explicit "freeze" of the pre-tip state for each game.

**Impact:**
- Cannot guarantee reproducibility of slate projections
- Accuracy analysis may use post-hoc data

**Recommendation:** Create a frozen snapshot at T-15 minutes before each game tip:
```
data/gold/pretip_snapshots/game_date=2025-12-07/game_id=22500351/
├── injuries.parquet   (as_of_ts = tip_ts - 15min)
├── odds.parquet
├── roster.parquet
└── manifest.json      (tip_ts, snapshot_ts, run_id)
```

### 5.4 MEDIUM: Odds Not Persisted by Time

**Problem:** Similar to injuries, odds are selected by latest-only.

**Impact:**
- Line movement analysis not possible
- Cannot reconstruct "what was the line at 5 PM?"

**Recommendation:** Same as injuries - hourly bronze partitions + run_as_of_ts filtering.

### 5.5 MEDIUM: No Pipeline Status Dashboard

**Problem:** Pipeline status is written to `bronze/pipeline_status/` but no monitoring.

**Impact:**
- Silent failures
- Stale data served without warning

**Recommendation:** Add simple health endpoint:
```python
@app.get("/api/health")
def health():
    return {
        "last_scrape": last_injury_ts,
        "age_minutes": (now - last_injury_ts).minutes,
        "status": "healthy" if age < 60 else "stale"
    }
```

### 5.6 LOW: Artifact Accumulation

**Problem:** Live features accumulate (~100+ runs/day) without cleanup.

**Impact:**
- Disk space (minor, ~6MB/day for live features)
- Clutter

**Recommendation:** Add retention policy (keep last 24 hours of live features).

---

## 6. Reproducibility Assessment

### 6.1 What CAN Be Reproduced

| Artifact | Reproducible | Method |
|----------|--------------|--------|
| Labels | Yes | Immutable boxscore_labels.parquet |
| Gold Features (monthly) | Yes | Static monthly partitions |
| Final Predictions | Partially | Logged to prediction_logs_minutes |
| Model Bundles | Yes | Versioned in artifacts/minutes_lgbm/<run_id>/ |

### 6.2 What CANNOT Be Reproduced

| Artifact | Why |
|----------|-----|
| Pre-tip injury state | Bronze overwrites, silver latest-only |
| Pre-tip odds | Same as injuries |
| Exact live feature set | Trend features recomputed on each run |
| Run-specific roster | Only latest snapshot retained |

### 6.3 Reproducibility Roadmap

**Phase 1: Hourly Bronze Retention (1 week effort)**
```python
# In storage.py, add hour partition
def write_bronze_partition_hourly(df, dataset, date, hour, ...):
    path = f"bronze/{dataset}/season={season}/date={date}/hour={hour:02d}/"
```

**Phase 2: run_as_of_ts Filtering (2 days effort)**
```python
# Add to all snapshot selection functions
def select_injury_snapshot(bronze, schedule, run_as_of_ts=None):
    # ... existing logic ...
    if run_as_of_ts:
        mask &= bronze["as_of_ts"] <= run_as_of_ts
```

**Phase 3: Pre-Tip Freeze Job (3 days effort)**
```python
# New CLI command: freeze_pretip_snapshot
# Triggered by systemd at T-15 for each game
# Writes immutable snapshot to gold/pretip_snapshots/
```

---

## 7. Orchestration Recommendations

### 7.1 Systemd vs. Prefect

| Factor | Systemd | Prefect |
|--------|---------|---------|
| Complexity | Low | Medium |
| Dependencies | None | Python server |
| Monitoring | journalctl | Built-in UI |
| Retry Logic | Basic (Restart=) | Advanced |
| DAG Support | Manual | Native |
| Solo Project | Appropriate | Overkill |

**Recommendation:** Stay with systemd but add:
1. **Healthcheck endpoint** for monitoring
2. **Slack/Telegram alerts** on failure (OnFailure= already exists)
3. **Simple dashboard** showing last successful run times

### 7.2 Proposed Systemd Improvements

**Add health monitoring timer:**
```ini
# systemd/pipeline-health.timer
[Timer]
OnCalendar=*:0/10  # Every 10 minutes
Unit=pipeline-health.service

# systemd/pipeline-health.service
[Service]
ExecStart=/home/daniel/projects/projections-v2/scripts/check_pipeline_health.sh
```

**check_pipeline_health.sh:**
```bash
#!/bin/bash
# Check last injury scrape < 90 minutes old
# Check last score run < 15 minutes during SLATE
# Alert if stale
```

### 7.3 When to Consider Prefect

Move to Prefect/Dagster if:
- You need complex DAG dependencies
- You have multiple concurrent pipelines
- You need centralized logging across projects
- You want automatic retries with exponential backoff

For a solo project with 1 pipeline, systemd is appropriate.

---

## 8. Data Lineage Summary

```
NBA.com Injury PDFs (hourly)
    │
    ▼
bronze/injuries_raw/season={season}/date={date}/  [OVERWRITES - GAP]
    │
    ├─> Merge with existing silver
    ▼
silver/injuries_snapshot/season={season}/month={month}/  [APPENDS]
    │
    ├─> Filter: as_of_ts <= tip_ts (latest only) [GAP: no run_as_of_ts]
    ▼
MinutesFeatureBuilder._attach_injuries()
    │
    ▼
gold/features_minutes_v1/  [PERSISTED - Monthly]
    │
    ▼
artifacts/minutes_v1/daily/{date}/run={run_id}/  [PERSISTED - Per Run]
    │
    ▼
API /api/minutes  [SERVED]
```

---

## 9. Action Items

### Immediate (This Week)

1. **Add hourly partitioning to bronze injuries**
   - Modify `projections/etl/storage.py`
   - Update `projections/etl/injuries.py` to write hourly
   - Retention: 7 days of hourly data

2. **Add run_as_of_ts to snapshot selection**
   - Modify `select_injury_snapshot()` in `injuries.py`
   - Modify `latest_pre_tip_snapshot()` in `odds.py`
   - Enables reliable backfill

### Short-Term (This Month)

3. **Implement pre-tip freeze job**
   - New CLI: `projections.cli.freeze_pretip_snapshot`
   - Triggered at T-15 for each game
   - Writes to `gold/pretip_snapshots/`

4. **Add health monitoring**
   - `/api/health` endpoint
   - Systemd timer for health checks
   - Slack webhook on failure

### Long-Term (Optional)

5. **Consider Prefect** if complexity grows
6. **Add line movement tracking** (hourly odds retention)
7. **Build accuracy dashboard** from prediction_logs

---

## 10. Appendix: File Paths

### ETL Modules
```
projections/etl/
├── common.py          # Shared schedule/date utilities
├── storage.py         # Bronze path contracts
├── schedule.py        # Schedule ingestion
├── injuries.py        # Injury reports ETL
├── odds.py            # Betting lines ETL
├── roster_nightly.py  # Roster ETL
├── daily_lineups.py   # Lineups ETL
└── boxscores.py       # Game results ETL
```

### Scrapers
```
scrapers/
├── nba_schedule.py      # Schedule API
├── nba_injuries.py      # PDF injury reports
├── oddstrader.py        # Betting lines
├── nba_players.py       # Active rosters
├── nba_daily_lineups.py # Lineup projections
└── nba_boxscore.py      # Box scores
```

### Orchestration
```
systemd/
├── live-scrape.timer          # 30-min scraping
├── live-scrape.service
├── live-score.timer           # 5-min scoring
├── live-score.service
├── live-boxscores.timer       # Daily 3:30 AM
├── live-boxscores.service
├── gold-features-daily.timer  # Daily 10 AM
└── gold-features-daily.service

scripts/
├── run_live_scrape.sh
├── run_live_score.sh
├── run_live_pipeline.sh
├── run_gold_daily.sh
└── run_boxscores.sh
```

### Data Directories
```
data/
├── bronze/              # Raw extracted data (OVERWRITES - GAP)
├── silver/              # Normalized snapshots (APPENDS)
├── gold/                # Features + predictions (MONTHLY)
├── labels/              # Immutable box scores
├── live/                # Ephemeral live features
└── preds/               # Model predictions

artifacts/
└── minutes_v1/daily/    # Persisted predictions per run
```

---

*Report generated by Claude Code ETL Pipeline Audit*
