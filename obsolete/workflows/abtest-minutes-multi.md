# Multi-Slate A/B/C Test for Minutes Allocators

## Overview

This framework compares three minutes allocation methods across multiple NBA game slates:

| Allocator | Description |
|-----------|-------------|
| **A** (SCALE_SHARES) | Share model predictions scaled to 240 per team |
| **B** (ROTALLOC) | Production allocator with rotation classifier + conditional means |
| **C** (SHARE_WITH_ROTALLOC_ELIGIBILITY) | Share predictions scaled WITHIN RotAlloc's eligible set only |

## Feature Contract Enforcement

The share model expects specific feature columns. Older feature files may have schema drift.

### Feature Contract Module
Located at: `projections/models/minute_share/feature_contract.py`

```python
from projections.models.minute_share import (
    load_expected_share_features,
    enforce_share_feature_contract,
)

# Load expected columns from bundle
expected = load_expected_share_features("artifacts/minute_share/minute_share_wfcv2_fold_20")

# Enforce contract (fills missing cols, creates NaN indicators)
df_fixed, report = enforce_share_feature_contract(df, expected)
```

### Rebuilding Features with Contract

For historical evaluation, rebuild features with contract enforcement:

```bash
uv run python scripts/rebuild_features_minutes_v1.py \
    --start-date 2025-11-01 --end-date 2025-12-31 \
    --share-bundle artifacts/minute_share/minute_share_wfcv2_fold_20 \
    --out-root /home/daniel/projections-data/live/features_minutes_v1
```

This writes contract-correct features to `<out_root>/<date>/run=<id>_contract/features.parquet`.

## Quality Tiers

To ensure trustworthy aggregates, slates are stratified by data quality:

| Tier | Criteria | Treatment |
|------|----------|-----------|
| **clean** | Passes integrity + missing_feature_frac ≤ 2% | Included in primary analysis |
| **degraded** | Passes integrity + 2% < missing_feature_frac ≤ 10% | Included but flagged |
| **skipped** | Failed integrity OR missing_feature_frac > 10% | Excluded, logged to skips.csv |

### Integrity Checks

A slate is skipped if:
- No games or teams
- Less than 2 teams per game (missing team data)
- Any team has < 8 players (incomplete roster)

## Usage

### Single Slate
```bash
uv run python scripts/abtest_minutes_allocators.py single \
    --game-date 2025-12-23
```

### Multi-Slate (Nov-Dec 2025)
```bash
uv run python scripts/abtest_minutes_allocators.py multi \
    --start-date 2025-11-01 \
    --end-date 2025-12-31
```

### Multi-Slate with Rebuilt Features
```bash
uv run python scripts/abtest_minutes_allocators.py multi \
    --start-date 2025-11-01 \
    --end-date 2025-12-31 \
    --features-root /home/daniel/projections-data/live/features_minutes_v1
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--bundle-dir` | From config | RotAlloc bundle directory |
| `--out-dir` | Auto-generated | Output directory |
| `--features-root` | None | Explicit features root (for rebuilt features) |
| `--cap-max` | 48.0 | Hard cap on minutes per player |
| `--k-core` | 8 | Core rotation size for RotAlloc |
| `--use-share-model/--no-share-model` | True | Use trained share model |
| `--parallel` | 1 | Number of parallel workers |

## Enabling Allocator C in Production

Allocator C is available behind a kill switch:

```bash
# Via environment variable
export PROJECTIONS_MINUTES_ALLOC_MODE=share_with_rotalloc_elig
uv run python -m projections.cli.score_minutes_v1 ...
```

Or in `config/minutes_current_run.json`:
```json
{
  "minutes_alloc_mode": "share_with_rotalloc_elig"
}
```

**Default behavior is unchanged** - only activates when explicitly set.

## Output Files

| File | Description |
|------|-------------|
| `aggregate_summary.json` | All aggregated metrics (all processed slates) |
| `aggregate_summary_clean.json` | Aggregated metrics (clean slates only) |
| `aggregate_by_slate.csv` | Per-slate comparison table |
| `aggregate_by_bucket.csv` | Per-bucket (0-10, 10-20, 20-30, 30+) MAE breakdown |
| `skips.csv` | Skipped slates with detailed reasons |
| `README.md` | Auto-generated interpretation and recommendations |

Each slate also gets a subdirectory with:
- `summary.json` - Slate-level metrics
- `players.parquet` - Player-level comparison

## Key Metrics

### Accuracy
- MAE / RMSE (overall and per-bucket)

### Realism
- **Top-5 sum**: Sum of top 5 player minutes per team (target: ~175)
- **Max minutes**: Maximum minutes allocated (target: ~40)
- **Gini coefficient**: Distribution evenness (lower = flatter)
- **Roster size**: Players receiving ≥1 minute

### Pathology Checks
- % slates with top5 < 150 (underestimating starters)
- % slates with max < 30 (too conservative)
- Bench crush: Mean minutes for highest bench player

## Interpreting Results

The auto-generated README.md includes:
1. **Summary table** comparing A/B/C on key metrics
2. **Win rates** showing % of slates each allocator wins
3. **Per-bucket analysis** identifying where each allocator excels
4. **Pathology checks** identifying problematic distributions
5. **Recommendation** based on accuracy + realism tradeoffs
