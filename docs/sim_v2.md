# Simulator v2 (sim_v2) Documentation

## Overview

The sim_v2 simulator generates Monte Carlo fantasy point (FPTS) projections for NBA DFS. It produces probabilistic forecasts with quantiles (p10, p50, p90, p95) that capture player upside and variance.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SIMULATOR PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐ │
│  │ minutes_v1   │───▶│  rates_v1    │───▶│   FPTS Computation    │ │
│  │ (p50 mins)   │    │ (per-min)    │    │   (DK scoring)        │ │
│  └──────────────┘    └──────────────┘    └───────────────────────┘ │
│         │                   │                       │               │
│         ▼                   ▼                       ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐ │
│  │ Game Scripts │    │ Rate Noise   │    │  World Aggregation    │ │
│  │ (margin adj) │    │ (t-dist)     │    │  (quantiles)          │ │
│  └──────────────┘    └──────────────┘    └───────────────────────┘ │
│                                                     │               │
│                                                     ▼               │
│                                          ┌───────────────────────┐ │
│                                          │  projections.parquet  │ │
│                                          │  (single output file) │ │
│                                          └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Minutes Predictions (`minutes_v1`)
- Quantile regression model predicting player minutes
- Outputs: `minutes_p10`, `minutes_p50`, `minutes_p90`
- Accounts for: injuries, rest, matchup, rotation

### 2. Rate Predictions (`rates_v1`)
- Per-minute stat rates (pts, reb, ast, etc.)
- Efficiency predictions (FG%, 3P%, FT%)
- Used to convert minutes → box score stats → FPTS

### 3. Game Scripts (`projections/sim_v2/game_script.py`)
- Samples game margins per world from Vegas spread
- Classifies margins into script categories
- **Shifts where we sample in the player's minutes distribution**
- Learned parameters from historical data:
  - `margin_std = 13.4` (residual variance)
  - `spread_coef = -0.726` (spread → expected margin)

**Quantile Targets by Script:**
| Script | Margin Range | Starter Quantile | Bench Quantile |
|--------|--------------|------------------|----------------|
| blowout_win | ≥+15 | p35 (less mins) | p55 |
| comfortable_win | +8 to +14 | p45 | p52 |
| close | -7 to +7 | **p65 (more mins)** | p48 |
| comfortable_loss | -14 to -8 | p50 | p50 |
| blowout_loss | ≤-15 | p30 (less mins) | p45 |

**Key Insight:** In close games, starters sample from p65 of their minutes distribution (more minutes). In blowouts, they sample from p30-p35 (less minutes, as they rest).

### 4. Noise Model
- Student-t distribution with `nu=5` degrees of freedom (fat tails)
- Scale parameter `k=0.65` controls variance relative to mean
- Applied per-stat to simulate game-to-game variance

## Configuration

### Profile Configuration (`config/sim_v2_profiles.json`)

```json
{
  "profiles": {
    "baseline": {
      "mean_source": "rates",
      "noise": {
        "k_default": 0.65,
        "nu": 5,
        "epsilon_dist": "student_t"
      },
      "game_script": {
        "enabled": true,
        "margin_std": 13.4,
        "spread_coef": -0.726
      },
      "enforce_team_240": true,
      "efficiency_scoring": true
    }
  }
}
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k_default` | 0.65 | Noise scale (higher = wider tails) |
| `nu` | 5 | Student-t df (lower = fatter tails) |
| `enforce_team_240` | true | Normalize team minutes to 240 |
| `game_script.enabled` | true | Apply game script adjustments |
| `min_play_prob` | 0.05 | Minimum play probability to include |

## Usage

### Live Simulation (Production)

```bash
# Run for today with 10k worlds
python -m scripts.sim_v2.run_sim_live \
  --run-date 2025-12-06 \
  --num-worlds 10000 \
  --profile baseline
```

### Batch Simulation (Backtesting)

```bash
# Run for date range
python -m scripts.sim_v2.generate_worlds_fpts_v2 \
  --start-date 2025-12-01 \
  --end-date 2025-12-06 \
  --n-worlds 50000 \
  --profile baseline
```

## Output Format

### `projections.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `game_date` | date | Game date |
| `game_id` | int | NBA game ID |
| `team_id` | int | Team ID |
| `player_id` | int | Player ID |
| `minutes_mean` | float | Expected minutes |
| `dk_fpts_mean` | float | Mean FPTS across worlds |
| `dk_fpts_std` | float | Std dev of FPTS |
| `dk_fpts_p05` | float | 5th percentile |
| `dk_fpts_p10` | float | 10th percentile |
| `dk_fpts_p25` | float | 25th percentile |
| `dk_fpts_p50` | float | Median |
| `dk_fpts_p75` | float | 75th percentile |
| `dk_fpts_p90` | float | 90th percentile |
| `dk_fpts_p95` | float | 95th percentile |
| `n_worlds` | int | Number of simulated worlds |
| `is_starter` | int | 1 if starter |
| `play_prob` | float | Probability of playing |

## Performance

### Benchmarks (206 players, 7 games)

| Worlds | Time | Throughput |
|--------|------|------------|
| 1,000 | 0.3s | 3,300/s |
| 10,000 | 3s | 3,300/s |
| 50,000 | 9s | 5,500/s |

### Resource Usage
- **Memory**: ~1-2 GB for 50k worlds
- **Disk**: 26 KB per date (single parquet file)
- **CPU**: Single-threaded (vectorized NumPy)

## Calibration

The simulator is calibrated such that actual outcomes fall within predicted quantiles at the expected rates:

| Metric | Target | Actual |
|--------|--------|--------|
| Below p10 | 10% | ~10% |
| Above p90 | 10% | ~11% |
| Above p95 | 5% | ~6% |
| Inside p10-p90 | 80% | ~79% |

## File Structure

```
projections/sim_v2/
├── config.py           # SimV2Profile dataclass and loader
├── game_script.py      # Game script sampling and adjustments
├── minutes_noise.py    # Minutes noise model
├── noise.py            # Rate noise model
└── residuals.py        # Residual sampling with team factors

scripts/sim_v2/
├── generate_worlds_fpts_v2.py   # Main world generation
├── run_sim_live.py              # Live pipeline wrapper
└── aggregate_worlds_to_projections.py  # Legacy aggregation (deprecated)

config/
└── sim_v2_profiles.json  # Profile configurations
```

## Extending the Simulator

### Adding New Features

1. **New noise parameters**: Add to `SimV2Profile` in `config.py`
2. **New adjustments**: Extend `game_script.py`
3. **New outputs**: Modify aggregation in `generate_worlds_fpts_v2.py`

### Custom Profiles

Create a new profile in `sim_v2_profiles.json`:

```json
{
  "profiles": {
    "high_variance": {
      "noise": {
        "k_default": 0.80,  // More variance
        "nu": 4             // Fatter tails
      }
    }
  }
}
```

## Teammate Correlations

### Empirical Findings (Dec 2025)

Analysis of ~6,000 player-games revealed:

| Metric | Correlation | Notes |
|--------|-------------|-------|
| Player residual vs Team-mean | **0.31** | When player over/underperforms, team moves together |
| Pairwise starter correlation | 0.04 | Low direct player-to-player correlation |
| Mixed (starter + bench) | -0.19 | Negative due to minutes substitution |

### Implementation

The simulator uses a **team shock** approach where all teammates receive the same noise adjustment per world:

```python
# For each target stat:
sigma_team = noise_params['sigma_team'] * team_sigma_scale   # ~0.34 from residuals
sigma_player = noise_params['sigma_player'] * player_sigma_scale

# Team shock (same for all teammates in a world)
team_shock = rng.normal(0, sigma_team)  
player_eps = rng.normal(0, sigma_player)  # Independent

stat_total = rate * minutes + team_shock + player_eps
```

### Tuning Knobs

| Parameter | Default | Effect |
|-----------|---------|--------|
| `team_sigma_scale` | 1.0 | Scales team-level correlation (reduce to <1 for less correlation) |
| `player_sigma_scale` | 1.0 | Scales individual variance |
| `rates_sigma_scale` | 1.0 | Overall noise scaling |

**Note:** The empirical 0.31 team-mean correlation aligns well with the `same_team_corr` values (~0.30-0.35) from rates residuals. Current settings appear well-calibrated. If correlations seem too high in practice, reduce `team_sigma_scale` to 0.7-0.8.

## Changelog

### 2025-12-07
- Enabled `rates_noise` in baseline profile (using Dec 5 residuals)
- Added `rates_noise_run_id` config option for explicit residual specification
- Documented teammate correlation analysis and calibration

### 2025-12-06
- Added game script feature with learned margin effects
- Refactored to in-memory aggregation (18x speedup)
- Tuned `k_default` from 0.35 to 0.65 for better tail calibration
- Output changed from per-world files to single `projections.parquet`
