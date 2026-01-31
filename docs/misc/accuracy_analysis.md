# Historical Backfill & Accuracy Analysis

## Summary

Built a historical backfill pipeline to reconstruct past slates and measure prediction accuracy.

## November 2025 Results

| Metric | Value | Target |
|--------|-------|--------|
| **FPTS MAE** | 8.31 | <7.0 |
| **Coverage (p10-p90)** | 57.4% | 80% |
| **Coverage (p05-p95)** | 68.0% | 90% |
| **Roster Misses** | ~15/day | 0 |

## Key Findings

### 1. Coverage is Too Low
Uncertainty bands are too narrow - only 57% of actuals fall within p10-p90 range vs target 80%.

### 2. Roster Misses Driven by `min_last1 = 0`
Players who DNP'd their last game get predicted 0 minutes even when healthy:
- **Prosper**: status=AVAIL, min_last1=0 → predicted 0 min, played 26 min
- **Bogdanovic**: status=AVAIL, min_last1=0 → predicted 0 min, played 20 min

This is a **model calibration issue**, not an injury data issue.

### 3. Injury PDF Source Available
NBA publishes hourly PDFs at `ak-static.cms.nba.com`. Players NOT in PDF = healthy.

## New Scripts Created

| Script | Purpose |
|--------|---------|
| `scripts/analyze_accuracy.py` | Compare predictions vs box scores |
| `scripts/backfill_injuries.py` | Download historical injury PDFs |
| `scripts/backfill_season.sh` | Run full season backfill |

## Environment Variables Added

| Variable | Purpose |
|----------|---------|
| `LIVE_SKIP_SCRAPE=1` | Skip scraping, use existing bronze/silver |
| `LIVE_BACKFILL_MODE=1` | Relax roster age checks |
| `LIVE_LOCK_BUFFER_MINUTES=0` | Process all games regardless of tip time |

---

## Next Step: Gold Layer Design

### Problem
Current backfill relies on silver layer which gets overwritten. Need immutable historical data.

### Proposed Architecture

```
gold/
└── historical_snapshots/
    └── game_date=YYYY-MM-DD/
        └── tip_cutoff=HH:MM/
            ├── injuries.parquet     # Status as of cutoff
            ├── odds.parquet         # Lines as of cutoff  
            └── rosters.parquet      # Lineups as of cutoff
```

### Key Principles
1. **Immutable**: Once written, never modified
2. **Tip-keyed**: One snapshot per unique tip time window
3. **Reproducible**: Same gold data → same predictions

### Implementation Steps
1. Create `gold/historical_snapshots` schema
2. Build ETL to populate from bronze with correct timestamps
3. Modify `run_live_score.sh` to read from gold for backfill
4. Backfill all of 2024-25 season

---

## Open Issue: min_last1 = 0

### Behavior
Model predicts ~0 minutes when player's last game was DNP/rest.

### Examples from Nov 12
- Buddy Hield: min_last1=11.6, predicted well
- Prosper: min_last1=0.0, predicted 0 but played 26

### Potential Fixes
1. Add feature: `days_since_last_play`
2. Cap minimum prediction for healthy players
3. Weight injury status more than recent history
