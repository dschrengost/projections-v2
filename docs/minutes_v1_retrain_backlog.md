# Minutes V1 Model Retrain Backlog

Items to address in the next model retraining iteration.

---

## High Priority

### 0. Rotation Eligibility Model (P(in_rotation | ACTIVE))

**Fast fix (Dec 2025):** Add a conservative heuristic `rotation_prob` derived from pregame history features (recent minutes + starter history) and cap minutes quantiles for low `rotation_prob` players during scoring. This reduces “ACTIVE but 0 minutes” ghost allocations that can steal minutes from core DFS players during team-240 reconciliation.

**Future (recommended):** Replace the heuristic with a trained classifier for `P(in_rotation | ACTIVE)` (or `P(minutes > 0 | ACTIVE)` / `P(minutes > 4 | ACTIVE)`) and use it to gate minute allocation + stabilize rotation caps in both L2 reconciliation and sim.

### 1. DNP/Injury Label Inclusion in Training

**Root Cause (Dec 2025 investigation)**: Aaron Wiggins was projected 0 minutes despite being a 20+ mpg player because:
- He was injured Nov 8-28 (6 games with 0 minutes)
- These 0-minute games were included in boxscore labels with `starter_flag=1` (incorrect)
- His `roll_mean_10` dropped to 8.17 because DNP games polluted the average
- Training drops all `minutes > 0` games, so model never learned DNP patterns

**Current training behavior**:
```python
train_cond_df = train_df[train_df["plays_target"] == 1]  # Only games where player played
```
This trains the model on **conditional minutes only** ("if player plays, how many minutes?"). The model never sees DNP patterns.

**Proposed for retrain**:

1. **Add DNP reason labels to boxscore data**:
   - `dnp_injury` - player was on injury report / missed due to injury
   - `dnp_cd` - coach's decision (rest, rotation change, disciplinary)
   - Source: Cross-reference boxscore 0-minute games with injury report data

2. **Include DNP games in training** (with `minutes = 0`):
   - Model can learn that 0-minute games exist and when to predict them
   - Different patterns for injury-return vs CD-benched

3. **Add new features**:
   - `recent_dnp_injury_count_7d` - how many injury DNPs in last 7 days
   - `recent_dnp_cd_count_7d` - how many CD DNPs in last 7 days  
   - `games_since_injury_return` - games played since returning from injury stint
   - `is_post_injury_ramp` - flag for first 3-5 games after injury return

4. **Fix rolling average calculation** (already done Dec 2025):
   - Exclude 0-minute games from `roll_mean_*` so they reflect actual playing time

**Expected benefit**: 
- Model can predict 0 minutes when appropriate (not just conditional)
- Better handling of injury-return ramp-ups
- Distinguish "player is back but limited" vs "player is benched"

---

### 2. Teammate Injury Interaction Effects (HIGH VALUE)

**Motivation**: When a starter is out, someone absorbs those ~30 minutes. Currently we track `arch_delta_*` but it's incomplete.

**Proposed features**:
- `minutes_vacuum_team` - Total expected minutes of OUT teammates
- `minutes_vacuum_same_pos` - Minutes vacuum at player's position specifically
- `historical_minutes_boost_when_X_out` - Pre-computed: when player X was out, how much did player Y's minutes increase?
- `is_next_man_up` - Flag for primary backup to an OUT starter

**Data source**: Cross-reference injury data with historical boxscores

**Expected impact**: HIGH - This is likely one of the biggest missing signals. A bench player's minutes can double when a starter is out.

---

## Simulation / Distribution (Backlog)

### Game Scripts via Margin Residual (avoid double counting)

**Observation:** In the dashboard, `sim p50 minutes` often matches the Minutes LGBM `p50`. This is expected if the simulator samples minutes around the model median/mean (a symmetric distribution will preserve the p50), but it can still indicate we are not getting enough *correlated* variation (close-game vs blowout effects) in tails.

**Goal:** Improve realism/accuracy of tails and correlations (team stacks, bench blowout minutes, etc.) while avoiding “double counting” odds since `spread_home`/`total` are already features in the Minutes LGBM.

**Proposal (recommended): residual-driven scripts**

- Treat Minutes LGBM output as the **conditional center** (given injuries/role/odds).
- Drive game scripts using a sampled **margin residual** rather than raw spread:
  - `implied_margin = f(spread_home)` (linear is fine; already in `GameScriptConfig.spread_coef`)
  - Sample `epsilon ~ ResidualDist( spread, total, home/away, … )` from historical `actual_margin - implied_margin`
  - `margin = implied_margin + epsilon`
  - Classify script (close/comfortable/blowout) from `margin` and pick player quantile targets per role (starters vs bench).
- This preserves the LGBM’s mean signal from odds while letting simulation add **realistic uncertainty + correlated shifts**.

**Calibration / evaluation**

- Fit/calibrate `ResidualDist` (at minimum: `margin_std`; ideally heteroskedastic by spread/total).
- Tune quantile targets so **mean minutes stays stable** while tails/correlation improve (avoid systematic mean shifts unless justified).
- Evaluate:
  - Minutes quantile calibration (coverage / pinball loss / CRPS) by role/usage buckets.
  - Team-minute reconciliation interaction (does it wash out tail behavior?).
  - Tail accuracy on FPTS outcomes (especially bench blowout minutes + garbage time stats).

**Notes**

- Add/validate an overtime model (OT probability and minutes inflation) separately; OT is a major tail driver.
- If we want “what players actually played” in the dashboard, surface `minutes_sim_mean`, `minutes_sim_p10/p50/p90`, and `minutes_sim_std` directly rather than reusing the LGBM p50.

---

## New Feature Proposals

### 3. Pace/Tempo Features

**Motivation**: Higher pace = more possessions = different rotation patterns. Fast-paced games have more player substitutions due to fatigue.

**Proposed features**:
| Feature | Description | Source |
|---------|-------------|--------|
| `team_pace` | Team's average pace (possessions/48) | nba.com/stats |
| `opp_pace` | Opponent's average pace | nba.com/stats |
| `expected_total_possessions` | Predicted total possessions in game | pace * game time |
| `pace_differential` | team_pace - opp_pace | Derived |
| `is_high_pace_matchup` | Flag when combined pace > 210 | Derived |

**Expected impact**: MEDIUM - Affects overall game flow and potentially bench usage

---

### 4. Season Standings / Playoff Context

**Motivation**: Tanking teams rest veterans, playoff teams push. Late-season context matters hugely.

**Proposed features**:
| Feature | Description |
|---------|-------------|
| `team_win_pct` | Current season win % |
| `games_behind_playoff` | Games behind 8th seed (or 0 if in) |
| `games_ahead_of_lottery` | If tanking, GB from best lottery odds |
| `is_playoff_locked` | Flag: clinched playoff spot |
| `is_eliminated` | Flag: mathematically eliminated |
| `opp_win_pct` | Opponent's win % |
| `win_pct_differential` | Mismatch indicator |
| `days_until_season_end` | Countdown affects rest decisions |

**Data source**: NBA standings (easy to scrape/compute from schedule+results)

**Expected impact**: MEDIUM-HIGH - Tanking teams have very different rotation patterns

---

### 5. NBA Cup Game Flag

**Motivation**: NBA Cup (In-Season Tournament) has different stakes and potentially different rotation patterns.

**Proposed features**:
| Feature | Description |
|---------|-------------|
| `is_nba_cup_game` | Flag for In-Season Tournament games |
| `nba_cup_round` | Group stage vs knockout |
| `is_nba_cup_elimination` | Knockout stage = higher stakes |

**Data source**: Game ID pattern (22500xxx) or NBA API game_type field

**Expected impact**: LOW-MEDIUM - Limited games per season, but may have distinct patterns

---

### 6. Player Age

**Motivation**: Older players (34+) often get more rest, especially on B2Bs. Already in roster data but not used.

**Proposed features**:
| Feature | Description |
|---------|-------------|
| `player_age` | Current age in years |
| `is_veteran_35plus` | Load management flag |
| `age_x_b2b` | Interaction: age matters more on B2Bs |

**Data source**: Already in `roster_nightly.age`

**Expected impact**: MEDIUM - Clear pattern for veteran load management

---

### 7. Travel/Fatigue Features

**Motivation**: Cross-country travel, especially east-west, affects fatigue and minutes.

**Proposed features**:
| Feature | Description |
|---------|-------------|
| `travel_distance_miles` | Distance from previous game location |
| `time_zones_crossed` | Number of time zone changes |
| `is_altitude_game` | Flag for Denver games (5280 ft affects fatigue) |
| `cumulative_miles_7d` | Total travel in last 7 days |
| `direction_of_travel` | East-to-west is harder than west-to-east |
| `games_on_road_trip` | How many games into current road trip |

**Data source**: Arena locations → distance calculations

**Expected impact**: MEDIUM - Fatigue is real, especially for older players

---

### 8. Foul Tendency (STRONG PREDICTOR)

**Motivation**: High foul-rate players are more likely to sit with early foul trouble. This directly affects minutes.

**Proposed features**:
| Feature | Description |
|---------|-------------|
| `fouls_per_minute_season` | Foul rate |
| `fouls_per_game_last5` | Recent foul tendency |
| `foul_out_rate` | % of games with 6 fouls |
| `early_foul_trouble_pct` | % of games with 2+ fouls in 1st Q |

**Data source**: Boxscore data (we have this)

**Expected impact**: HIGH - Direct mechanism for reduced minutes. A player with 2 fouls in the 1st quarter often sits until halftime.

---

### 9. Recent Performance/Efficiency Signals

**Motivation**: Coaches adjust minutes based on who's "hot" or "cold". We have rolling minutes but not rolling performance.

**Proposed features**:
| Feature | Description |
|---------|-------------|
| `plus_minus_per_min_last5` | Recent +/- efficiency |
| `ts_pct_last5` | True shooting % last 5 games |
| `usage_rate_last5` | How much offense runs through player |
| `performance_trend` | Is efficiency going up or down? |

**Data source**: Boxscore data

**Expected impact**: MEDIUM - Hot players get extra run, cold players get pulled

---

### 10. Game Importance Composite

**Motivation**: Combine multiple signals into a single "game matters" score.

**Proposed formula**:
```python
game_importance = (
    2.0 * is_playoff_implications +
    1.5 * is_division_rivalry +
    1.0 * is_national_tv +
    1.0 * is_nba_cup_knockout +
    0.5 * is_home_game +
    -1.0 * is_b2b_vs_bad_team
)
```

**Expected impact**: MEDIUM - Composite may be more predictive than individual flags

---

### 11. Roster Volatility

**Motivation**: Roster churn = minutes uncertainty. New players, G-League call-ups, trades all create instability.

**Proposed features**:
| Feature | Description |
|---------|-------------|
| `days_since_roster_move` | Stability indicator |
| `roster_changes_30d` | Volatility indicator |
| `is_two_way_player` | May be sent down anytime |
| `games_with_team` | New players have less trust |

**Expected impact**: MEDIUM - Explains variance for fringe rotation players

---

### 12. Coach Tendencies

**Motivation**: Some coaches run 8-man rotations, others run 10-man. New coaches experiment more.

**Proposed features**:
| Feature | Description |
|---------|-------------|
| `coach_rotation_depth` | Avg players with 10+ min under this coach |
| `coach_tenure_games` | Games under current coach |
| `is_new_coach` | First 20 games with new coach |
| `coach_starter_minutes_avg` | Does coach ride starters hard or rest them? |

**Data source**: Coach tenure table + historical aggregates

**Expected impact**: MEDIUM - Explains team-level rotation differences

---

## Priority Matrix

| Feature | Data Difficulty | Expected Impact | Priority |
|---------|----------------|-----------------|----------|
| DNP labels + training | Medium | HIGH | 🔴 P0 |
| Teammate injury interaction | Medium | HIGH | 🔴 P0 |
| Foul tendency | Easy | HIGH | 🟡 P1 |
| Pace/tempo | Easy | MEDIUM | 🟡 P1 |
| Season standings | Easy | MEDIUM-HIGH | 🟡 P1 |
| Player age | Already have | MEDIUM | 🟡 P1 |
| Travel distance | Medium | MEDIUM | 🟢 P2 |
| NBA Cup flag | Easy | LOW-MEDIUM | 🟢 P2 |
| Recent performance | Easy | MEDIUM | 🟢 P2 |
| Game importance | Easy | MEDIUM | 🟢 P2 |
| Roster volatility | Medium | MEDIUM | 🟢 P2 |
| Coach tendencies | Medium | MEDIUM | 🟢 P2 |

---

## Completed Workarounds

- [x] Fix 0-minute games polluting `roll_mean_*` (workaround applied Dec 2025)
- [x] Change rotation threshold from 6 mpg to 12 mpg in sim (Dec 2025)
