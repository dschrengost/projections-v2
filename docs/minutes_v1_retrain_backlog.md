# Minutes V1 Model Retrain Backlog

Items to address in the next model retraining iteration.

## High Priority

### 1. DNP/Injury Tracking Features
**Issue**: Players returning from absence have low rolling averages because 0-minute games are included in calculations.

**Current workaround** (Dec 2025): Filter out 0-minute games in `build_minutes_live.py` when computing `roll_mean_*` features.

**Proposed for retrain**:
- Add `dnp_cd_flag` - Coach's decision / rest
- Add `dnp_injury_flag` - Injury-related DNP  
- Add `recent_dnp_count_7d` - Number of DNPs in last 7 days
- Source: Parse DNP reasons from boxscore data or injury reports

**Expected benefit**: Model can explicitly learn injury-return patterns (e.g., ramp-up after injury vs fresh after rest).

---

## Medium Priority

### 2. NBA Cup Game Schema
**Issue**: NBA Cup games have different `game_id` patterns (22500xxx vs 0022501xxx). Currently handled but not validated extensively.

**Action**: Audit NBA Cup game data to ensure features are computed correctly.

---

## Completed

- [x] Fix 0-minute games polluting `roll_mean_*` (workaround applied Dec 2025)
