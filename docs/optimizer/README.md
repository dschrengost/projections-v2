# QuickBuild Optimizer

Fast lineup generation for DraftKings NBA DFS using the dashboard optimizer tab.

## Overview

The QuickBuild optimizer generates thousands of unique lineups in seconds using a streaming multi-worker architecture with constraint-based filtering.

## Features

- **Fast Generation**: 5000+ lineups in ~3 seconds using parallel workers
- **Projection-based**: Uses unified projections (minutes + rates + sim FPTS)
- **Constraint Support**: Min/max salary, team limits, lock/ban players
- **Build Persistence**: Auto-saves builds to disk for later loading
- **Player Pool Filtering**: Search, sort, lock/unlock players
- **Lineup Filtering**: Filter by player name, sort by projection or salary

## Usage

### Dashboard UI

1. Navigate to the **Optimizer** tab in the dashboard
2. Select a date and slate (draft group)
3. Configure build settings in the sidebar:
   - **Max Lineups**: Target number of lineups
   - **Workers**: Parallel build workers (default: 4)
   - **Min Uniques**: Minimum unique players between lineups
   - **Team Limit**: Maximum players from same team
   - **Min/Max Salary**: Salary constraints
4. Lock/ban players using checkboxes in the player pool
5. Click **Generate Lineups**
6. Use the lineup filter bar to search/sort results
7. Click **Export CSV** for DraftKings upload

### Saved Builds

Builds automatically save to `projections-data/builds/optimizer/{game_date}/{job_id}.json`.

The **Saved Builds** panel appears after generating builds:
- View all saved builds for the current slate
- **Load** - reload lineups from a saved build
- **Delete** - remove a saved build

## API Endpoints

Base path: `/api/optimizer`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/slates?date=YYYY-MM-DD` | List available draft groups |
| GET | `/pool?date=&draft_group_id=` | Get player pool with projections |
| POST | `/build` | Start async build job |
| GET | `/build/{job_id}` | Poll job status |
| GET | `/build/{job_id}/lineups` | Get completed lineups |
| GET | `/build/{job_id}/export` | Export as DK-uploadable CSV (uses slot-specific draftableIds) |
| POST | `/export` | Export arbitrary lineups as DK CSV |
| GET | `/saved-builds?date=` | List saved builds |
| GET | `/saved-builds/{job_id}?date=` | Load saved build |
| DELETE | `/saved-builds/{job_id}?date=` | Delete saved build |

## Configuration

Config file: `config/optimizer.yaml`

```yaml
pool:
  max_pool: 10000
  builds: 4
  per_build: 3000
  min_uniq: 1

constraints:
  min_salary: null
  max_salary: 50000
  global_team_limit: 4

jobs:
  max_concurrent: 4
  job_ttl_minutes: 60
```

## Architecture

### Backend

- **`projections/api/optimizer_api.py`**: FastAPI router with endpoints
- **`projections/api/optimizer_service.py`**: Business logic, job store, persistence
- **`projections/optimizer/quick_build.py`**: Core lineup generator

### Frontend

- **`web/minutes-dashboard/src/pages/OptimizerPage.tsx`**: Main UI component
- **`web/minutes-dashboard/src/api/optimizer.ts`**: API client

### Data Flow

```
User Request → API → Build Player Pool → QuickBuild Workers → Dedupe → Save to Disk → Return Lineups
                    ↓
              Projections + Salaries + Positions merged
```

## Data Sources

- **Projections**: `artifacts/projections/unified_projections/` or `gold/projections_minutes_v1/`
- **Salaries**: `gold/dk_salaries/site=dk/game_date={date}/draft_group_id={id}/`
- **Saved Builds**: `projections-data/builds/optimizer/{game_date}/{job_id}.json`

## Slate Discovery

For past dates when DraftKings lobby API returns no contests, slates are discovered from existing salary data on disk.
