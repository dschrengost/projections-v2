# Live Pipeline Hardening Plan

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PREFECT FLOWS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐   │
│  │  morning-daily   │      │  live-pipeline   │      │   nightly-eval   │   │
│  │    (8am ET)      │      │  (every 5 min)   │      │   (5:30am ET)    │   │
│  └────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘   │
│           │                         │                         │              │
│           ▼                         ▼                         ▼              │
│  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐    │
│  │ dk-salaries     │       │ run_live_scrape │       │ analyze_accuracy│    │
│  │ schedule-etl    │       │ run_live_score  │       └─────────────────┘    │
│  │ boxscores-etl   │       └─────────────────┘                              │
│  └─────────────────┘                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Flow: `live-pipeline-full`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: SCRAPE (run_live_scrape.sh → projections.cli.live_pipeline)         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │   injuries   │   │ daily_lineup │   │    odds      │   │roster_nightly│  │
│  │   (NBA PDF)  │   │ (stats.nba)  │   │ (oddstrader) │   │  (nba.com)   │  │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘  │
│         │                  │                  │                  │           │
│         ▼                  ▼                  ▼                  ▼           │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │bronze/silver │   │bronze/silver │   │bronze/silver │   │bronze/silver │  │
│  │injuries_*    │   │daily_lineups │   │  odds_*      │   │roster_nightly│  │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘  │
│                                                                              │
│  + ESPN injuries (fast fallback)                                             │
│  + Rotowire lineups (fast fallback)                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: SCORE (run_live_score.sh)                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  2a. build_minutes_live ──────────────────────────────────────────────────► │
│      │ Loads: roster, injuries, odds, schedule                               │
│      │ Outputs: live/features_minutes_v1/{date}/run={run_id}/features.parquet│
│      ▼                                                                       │
│  2b. score_minutes_v1 ────────────────────────────────────────────────────► │
│      │ Outputs: artifacts/minutes_v1/daily/{date}/run={run_id}/minutes.parquet│
│      │ Copies to: gold/projections_minutes_v1/game_date={date}/              │
│      ▼                                                                       │
│  2c. build_rates_features_live ───────────────────────────────────────────► │
│      │ Outputs: live/features_rates_v1/{date}/run={run_id}/                  │
│      ▼                                                                       │
│  2d. score_rates_live ────────────────────────────────────────────────────► │
│      │ Outputs: gold/rates_v1_live/game_date={date}/run={run_id}/            │
│      ▼                                                                       │
│  2e. run_sim_live (sim_v2) ───────────────────────────────────────────────► │
│      │ Outputs: gold/sim_v2/game_date={date}/run={run_id}/                   │
│      ▼                                                                       │
│  2f. score_ownership_live ────────────────────────────────────────────────► │
│      │ Outputs: gold/ownership_v1/game_date={date}/run={run_id}/             │
│      ▼                                                                       │
│  2g. finalize_projections ────────────────────────────────────────────────► │
│      │ Outputs: gold/projections_unified/{date}/run={run_id}/projections.parquet│
│      ▼                                                                       │
│  2h. validate + promote ──────────────────────────────────────────────────► │
│      │ Sets blessed_run.json if validation passes                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Known Failure Points

### 1. Data Not Available for Today's Games
| Issue | Root Cause | Current Behavior | Fix |
|-------|-----------|------------------|-----|
| Roster missing today's games | Existing roster file has old games only | Pipeline fails | ✅ FIXED: Now scrapes missing game_ids |
| Roster join fails on historical games | Left join includes games not in today's schedule | Pipeline fails | ✅ FIXED: Filter roster to target game_ids |
| DK salaries not fetched | `morning-daily` not scheduled or failed | Optimizer page empty | Need: Auto-trigger on first access |
| Schedule not updated | Schedule ETL didn't run | Can't determine games | Need: Validate schedule on startup |

### 2. External API Failures
| API | Failure Mode | Impact | Mitigation |
|-----|-------------|--------|------------|
| NBA injury PDF | Timeout/SSL | Missing injury data | Has ESPN fallback ✓ |
| stats.nba.com lineups | 404/timeout | Missing starter info | Has Rotowire fallback ✓ |
| Oddstrader | Timeout | Missing odds | Uses cached values ✓ |
| NBA.com roster | Timeout | Missing roster | **CRITICAL** - blocks pipeline |
| DraftKings API | Rate limit | Missing salaries | Retry with backoff |

### 3. Data Staleness
| Data | Max Age | Current Check | Issue |
|------|---------|---------------|-------|
| Roster | 18 hours | `roster_max_age_hours` | Can use stale data |
| Odds | None | None | Stale odds if API down |
| Injuries | None | None | Stale if PDF scrape fails |
| Schedule | None | None | Missing new games |

### 4. Pipeline Ordering Issues
| Dependency | Problem | Impact |
|------------|---------|--------|
| DK salaries before optimizer | Not enforced | Optimizer shows no players |
| Schedule before roster | Not enforced | Roster fails to join |
| Morning flow before live flow | Not enforced | Missing prerequisite data |

## Hardening Recommendations

### Phase 1: Critical Fixes (Immediate)

1. **Add DK Salaries to live-pipeline-full** (HIGH PRIORITY)
   ```python
   # In live_pipeline_full_flow, add before scrape:
   dk_result = run_dk_salaries_task(game_date=game_date)
   ```

2. **Add Schedule Validation at Start**
   - Check schedule has today's games before proceeding
   - If missing, fetch via `run_schedule_task`

3. **Improve Error Handling in Shell Script**
   - Each step should log success/failure clearly
   - Failed steps should not block downstream if data exists

### Phase 2: Resilience (This Week)

4. **Add Data Freshness Checks**
   ```python
   def check_data_freshness(game_date: str) -> dict:
       """Check all required data is fresh before pipeline run."""
       checks = {
           "schedule": check_schedule_has_games(game_date),
           "roster": check_roster_has_games(game_date),
           "dk_salaries": check_salaries_exist(game_date),
           "injuries": check_injuries_recent(game_date, max_age_hours=4),
           "odds": check_odds_recent(game_date, max_age_hours=2),
       }
       return checks
   ```

5. **Add Pre-Flight Check Flow**
   ```python
   @flow(name="preflight-check")
   def preflight_check_flow(game_date: str) -> bool:
       """Run before live pipeline to ensure all data is ready."""
       freshness = check_data_freshness(game_date)
       missing = [k for k, v in freshness.items() if not v]
       if missing:
           # Attempt to fetch missing data
           for item in missing:
               fetch_data(item, game_date)
       return all(freshness.values())
   ```

6. **Convert Shell Script to Python**
   - Move `run_live_score.sh` logic into Python tasks
   - Better error handling, logging, and recovery

### Phase 3: Observability (Next Week)

7. **Add Health Check Endpoint**
   ```python
   @app.get("/api/health/pipeline")
   def pipeline_health(date: str = None):
       """Return pipeline health status for date."""
       return {
           "schedule_ok": ...,
           "roster_ok": ...,
           "salaries_ok": ...,
           "last_run": ...,
           "next_run": ...,
       }
   ```

8. **Add Slack/Discord Alerts**
   - Alert on pipeline failure
   - Alert on stale data (>1 hour without update during slate)
   - Alert on validation failures

### Phase 4: Automation (Ongoing)

9. **Scheduled Flow Configuration**
   ```yaml
   # prefect.yaml
   deployments:
     - name: morning-daily
       schedule:
         cron: "0 8 * * *"  # 8am ET
         timezone: "America/New_York"

     - name: live-pipeline-full
       schedule:
         cron: "*/5 * * * *"  # Every 5 minutes
       parameters:
         run_sim: true

     - name: nightly-eval
       schedule:
         cron: "30 5 * * *"  # 5:30am ET
   ```

10. **Add Idempotency Guarantees**
    - Each step checks if output exists before running
    - Re-running is safe and fast

## Immediate Action Items

1. [ ] Add `run_dk_salaries_task` to `live_pipeline_full_flow`
2. [ ] Add schedule validation at pipeline start
3. [ ] Add pre-flight data freshness check
4. [ ] Convert critical shell script sections to Python
5. [ ] Add `/api/health/pipeline` endpoint
6. [ ] Set up Prefect schedules for all flows
7. [ ] Add alerting on failures
