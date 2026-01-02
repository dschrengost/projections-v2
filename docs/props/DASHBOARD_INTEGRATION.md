# Props Dashboard Integration Plan

## Overview
Add a Props page to the minutes dashboard showing player prop lines from multiple sportsbooks alongside our model predictions, with EV calculations and edge detection.

## Data Sources

### Props (from scraper)
- **Location**: `data/bronze/props/game_date=YYYY-MM-DD/props_*.parquet`
- **Fields**: `player_id`, `player_name`, `team`, `book`, `prop_type`, `line`, `over_odds`, `under_odds`
- **Books**: draftkings, fanduel, mgm, caesars, betrivers, hardrock, espnbet

### Predictions (from sim_v2 finalized projections)
- **Location**: `data/artifacts/projections/<date>/run=<ts>/projections.parquet`
- **Stat predictions**: `pts_mean`, `reb_mean`, `ast_mean`, `stl_mean`, `blk_mean`, `tov_mean`
- **For 3PT**: derive from `pred_fga3_per_min * minutes_p50 * pred_fg3_pct` (rates_v1)
- **Distributions**: `dk_fpts_p10/p50/p90` for uncertainty bounds

### Prop Type → Sim World Column Mapping
| Prop | World Column(s) | Notes |
|------|-----------------|-------|
| pts | `pts` | Direct from sim worlds |
| reb | `reb` | Direct from sim worlds |
| ast | `ast` | Direct from sim worlds |
| threes | `fgm3` | 3-point field goals made |
| blk | `blk` | Direct from sim worlds |
| stl | `stl` | Direct from sim worlds |
| turnovers | `tov` | Direct from sim worlds |
| ptsrebast | `pts + reb + ast` | Sum per world |
| ptsreb | `pts + reb` | Sum per world |
| ptsast | `pts + ast` | Sum per world |
| rebast | `reb + ast` | Sum per world |
| stlblk | `stl + blk` | Sum per world |

---

## Backend: `projections/api/props_api.py`

### Endpoints

#### `GET /api/props/lines`
Returns all prop lines for a date merged with predictions.

**Query params**: `date` (required), `prop_type` (optional filter), `min_edge` (optional)

**Response** (list of):
```python
{
  "player_id": "5110",
  "player_name": "Anthony Edwards",
  "team": "MIN",
  "opponent": "CHI",
  "prop_type": "pts",
  "prediction": 28.7,          # our model's mean
  "prediction_std": 8.2,       # standard deviation if available

  # Best line across books
  "best_over_line": 27.5,
  "best_over_odds": -108,
  "best_over_book": "fanduel",
  "best_under_line": 28.5,
  "best_under_odds": -105,
  "best_under_book": "draftkings",

  # EV calculations (for best lines)
  "over_implied_prob": 0.519,
  "over_true_prob": 0.542,     # P(stat > line) from our model
  "over_ev": 0.044,            # expected value per $1 bet
  "over_edge": "slight_over",  # edge classification

  "under_implied_prob": 0.512,
  "under_true_prob": 0.458,
  "under_ev": -0.106,
  "under_edge": "no_edge",

  # All book lines (for comparison view)
  "all_lines": [
    {"book": "draftkings", "line": 27.5, "over_odds": -115, "under_odds": -105},
    {"book": "fanduel", "line": 27.5, "over_odds": -108, "under_odds": -112},
    ...
  ]
}
```

#### `GET /api/props/summary`
High-level summary for date.

**Response**:
```python
{
  "date": "2025-12-29",
  "total_props": 5621,
  "players_with_props": 152,
  "props_with_edge": 47,       # |EV| > threshold
  "best_edges": [              # top 5 by EV
    {"player": "...", "prop": "pts", "side": "over", "ev": 0.12},
    ...
  ]
}
```

### EV Calculation Logic

**Using Monte Carlo world distributions** for accurate probability estimation:

```python
def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def load_sim_worlds(game_date: str, run_id: str) -> pd.DataFrame:
    """Load world-level stat samples from sim_v2.

    Returns DataFrame with columns: player_id, world_id, pts, reb, ast, stl, blk, tov, threes
    Each player has ~20K world samples.
    """
    worlds_path = f"data/artifacts/sim_v2/worlds_fpts_v2/game_date={game_date}/run={run_id}/"
    # Load and stack all world files
    ...

def calculate_true_prob_over(worlds_df: pd.DataFrame, player_id: str,
                              stat: str, line: float) -> float:
    """Calculate P(stat > line) from empirical world distribution.

    Uses actual Monte Carlo samples for accurate tail probabilities.
    """
    player_worlds = worlds_df[worlds_df['player_id'] == player_id][stat]
    return (player_worlds > line).mean()

def calculate_true_prob_under(worlds_df: pd.DataFrame, player_id: str,
                               stat: str, line: float) -> float:
    """Calculate P(stat < line) from empirical world distribution."""
    player_worlds = worlds_df[worlds_df['player_id'] == player_id][stat]
    return (player_worlds < line).mean()

def calculate_ev(true_prob: float, odds: int) -> float:
    """Calculate expected value per $1 risked."""
    if odds > 0:
        payout = odds / 100  # profit per $1 risked
    else:
        payout = 100 / abs(odds)
    # EV = P(win) * payout - P(lose) * stake
    return true_prob * payout - (1 - true_prob) * 1.0

# For combo props, sum the individual stat columns:
def get_combo_stat(worlds_df: pd.DataFrame, player_id: str, prop_type: str) -> pd.Series:
    """Get combined stat series for combo props."""
    player_worlds = worlds_df[worlds_df['player_id'] == player_id]
    if prop_type == 'ptsrebast':
        return player_worlds['pts'] + player_worlds['reb'] + player_worlds['ast']
    elif prop_type == 'ptsreb':
        return player_worlds['pts'] + player_worlds['reb']
    elif prop_type == 'ptsast':
        return player_worlds['pts'] + player_worlds['ast']
    elif prop_type == 'rebast':
        return player_worlds['reb'] + player_worlds['ast']
    elif prop_type == 'stlblk':
        return player_worlds['stl'] + player_worlds['blk']
    else:
        return player_worlds[prop_type]
```

**Caching strategy**: Load worlds once per date, cache in memory for the session.

### Edge Classification
- `strong_over`: over_ev > 0.10
- `slight_over`: 0.03 < over_ev <= 0.10
- `fair`: |ev| <= 0.03
- `slight_under`: 0.03 < under_ev <= 0.10
- `strong_under`: under_ev > 0.10

---

## Frontend: `web/minutes-dashboard/src/pages/PropsPage.tsx`

### Layout (follow OptimizerPage pattern)

```
┌─────────────────────────────────────────────────────────────────┐
│ Header: "Props & EV Analysis"          [Date picker] [Refresh] │
├─────────────────────────────────────────────────────────────────┤
│ Summary Bar: 152 players | 47 with edge | Top: Edwards PTS +12%│
├─────────────────────────────────────────────────────────────────┤
│ Filters: [Prop Type ▼] [Team ▼] [Min EV __] [Search ____]      │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Props Table (sortable columns)                              │ │
│ │ Player | Team | Prop | Pred | Line | O/U | Odds | EV | Edge │ │
│ │ ─────────────────────────────────────────────────────────── │ │
│ │ Edwards | MIN | PTS | 28.7 | 27.5 | O | -108 | +4.4% | ✓   │ │
│ │ Jokic   | DEN | REB | 13.2 | 12.5 | O | -115 | +3.1% | ✓   │ │
│ │ ...                                                         │ │
│ └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ [Expand row] → All Books Comparison                             │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ DraftKings: 27.5 O:-115 U:-105  ← best under               │ │
│ │ FanDuel:    27.5 O:-108 U:-112  ← best over                │ │
│ │ BetMGM:     28.5 O:-110 U:-110                              │ │
│ │ Caesars:    27.5 O:-120 U:+100                              │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

1. **Sortable table** - sort by EV, prediction, line difference, player name
2. **Prop type filter** - pts, reb, ast, threes, combos
3. **Edge filter** - show only props with EV > threshold
4. **Expandable rows** - click to see all book lines for comparison
5. **Best line highlighting** - green highlight on best over/under odds
6. **Edge badges** - color-coded (green = +EV, red = -EV, gray = fair)
7. **Line shopping** - show which book has best odds for each side

### Component Structure

```
src/
├── pages/PropsPage.tsx          # Main page component
├── api/props.ts                 # API client
└── components/
    └── PropsTable.tsx           # Reusable table (optional)
```

### API Client (`src/api/props.ts`)

```typescript
import { apiUrl } from './client'

export interface PropLine {
  player_id: string
  player_name: string
  team: string
  opponent: string
  prop_type: string
  prediction: number
  prediction_std?: number
  best_over_line: number
  best_over_odds: number
  best_over_book: string
  best_under_line: number
  best_under_odds: number
  best_under_book: string
  over_implied_prob: number
  over_true_prob: number
  over_ev: number
  over_edge: string
  under_implied_prob: number
  under_true_prob: number
  under_ev: number
  under_edge: string
  all_lines: BookLine[]
}

export interface BookLine {
  book: string
  line: number
  over_odds: number
  under_odds: number
}

export const getPropsLines = async (
  date: string,
  propType?: string,
  minEdge?: number
): Promise<PropLine[]> => {
  let url = `/api/props/lines?date=${date}`
  if (propType) url += `&prop_type=${propType}`
  if (minEdge) url += `&min_edge=${minEdge}`
  const res = await fetch(apiUrl(url))
  if (!res.ok) throw new Error(`${res.status}`)
  return res.json()
}
```

---

## Implementation Steps

### Phase 1: Backend API
1. Create `projections/api/props_api.py` with router
2. Implement `GET /api/props/lines` endpoint
3. Add EV calculation utilities (odds conversion, probability estimation)
4. Merge props data with predictions from projections parquet
5. Mount router in `minutes_api.py`
6. Test with curl/browser

### Phase 2: Frontend Page
1. Create `src/api/props.ts` API client
2. Create `src/pages/PropsPage.tsx` with basic table
3. Add to App.tsx routing (new tab)
4. Implement sorting and filtering
5. Add expandable rows for all-books view
6. Style with existing CSS patterns

### Phase 3: Enhancements
1. Add summary statistics bar
2. Add edge badges with color coding
3. Highlight best odds per side
4. Add prop type icons/labels
5. Consider caching props data (refreshes every few minutes)

---

## Files to Create/Modify

### New Files
- `projections/api/props_api.py` - FastAPI router
- `web/minutes-dashboard/src/api/props.ts` - API client
- `web/minutes-dashboard/src/pages/PropsPage.tsx` - Page component

### Modified Files
- `projections/api/minutes_api.py` - mount props router
- `web/minutes-dashboard/src/App.tsx` - add Props tab

---

## Future Ideas

1. **Historical tracking** - track how our edges performed over time
2. **Alerts** - notify when high-EV props appear
3. **Correlation awareness** - flag correlated props (same game, etc.)
4. **Kelly sizing** - suggest bet sizes based on edge and bankroll
5. **Line movement** - track how lines move throughout day
6. **Model calibration** - compare predicted vs actual hit rates
