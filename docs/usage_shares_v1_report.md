# Usage Shares Model Report

This document describes the injury behavior report tool for usage shares models (LGBM and NN).

## Purpose

The report evaluates how well models handle **injury/vacancy reallocation**—the key behavior we care about beyond raw MAE improvements. When players are out, does the model correctly shift usage to the remaining players?

## Running the Report

```bash
uv run python -m scripts.usage_shares_v1.model_report \
  --data-root /home/daniel/projections-data \
  --run-id 20251215_220429 \
  --targets fga,tov \
  --backends baseline,lgbm,nn \
  --start-date 2024-11-01 \
  --end-date 2025-02-01 \
  --split val \
  --max-examples 25
```

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `--run-id` | (required) | Model artifacts run ID |
| `--targets` | `fga,tov` | Comma-separated targets to analyze |
| `--backends` | `baseline,lgbm,nn` | Backends to compare |
| `--split` | `val` | Data split: `train`, `val`, or `all` |
| `--val-days` | `30` | Days for validation split |
| `--topk` | `6` | Top players per team-game in examples |
| `--max-examples` | `25` | Maximum worst-case examples |
| `--counterfactual` | `false` | Run vacancy sensitivity analysis |
| `--out` | auto | JSON report output path |
| `--csv-out` | none | Optional CSV summary path |

## Report Sections

### 1. Global Metrics

Per target and backend:

| Metric | Description |
|--------|-------------|
| **MAE** | Mean absolute error of predicted shares vs true shares |
| **KL** | KL divergence from true to predicted distribution (per team-game, averaged) |
| **Top1** | Accuracy: is the predicted highest-share player actually the true highest? |
| **Top2** | Does the predicted top-2 contain the true top-2? |
| **Spearman** | Rank correlation between true and predicted shares within each team-game |

### 2. Vacancy-Bucketed Metrics

Metrics broken down by vacancy stress level:

- **vac_min_szn quantiles** (Q1-Q4): Season minutes vacated by injured players
- **vac_fga_szn quantiles**: Season FGA vacated
- **is_starter buckets**: Starter vs bench analysis
- **minutes_pred_p50 bins**: By projected playing time

This reveals whether models improve more in high-vacancy situations (where reallocation matters most).

### 3. Reallocation Sanity Score

For team-games in the **top 10% by vacancy**, we analyze who benefits from the model's share adjustments:

| Metric | Good Values | Interpretation |
|--------|-------------|----------------|
| **% Starter** | >60% | Top beneficiary of share increase is usually a starter |
| **% Top2 Min** | >50% | Top beneficiary is usually a top-2 minutes player on team |
| **Avg Rank** | <3.0 | Average minutes rank of beneficiary (1=highest minutes) |

Higher values mean the model correctly shifts usage to the players who would actually absorb the workload.

### 4. Worst-Case Examples

The N worst team-games by KL divergence, showing:

- Game/team/date identifiers
- Vacancy features for that game
- Top-K players with true vs predicted vs baseline shares

Use these to debug systematic model failures.

### 5. Counterfactual Sensitivity (Optional)

Tests whether models respond to vacancy features:

1. For high-vacancy games, artificially increase vacancy features by 1 std
2. Measure how shares change
3. Check if changes concentrate toward starters/high-minutes players

If models use vacancy features, shares should shift meaningfully. If changes are random or go to low-minutes players, the vacancy features may not be well-integrated.

## Output Files

### JSON Report

Full structured report at:
```
{data_root}/artifacts/usage_shares_v1/reports/{report_id}.json
```

Contains all metrics, bucketed results, reallocation details, and worst-case examples.

### CSV Summary (Optional)

Flat summary of global metrics for quick spreadsheet analysis.

## Interpreting Results

### Good Signs

- Models beat baseline on MAE/KL especially in **high-vacancy buckets**
- Reallocation sanity shows beneficiaries are usually starters with high minutes
- Worst-case examples are edge cases (very low/high vacancy, unusual lineups)

### Warning Signs

- Models do worse than baseline in high-vacancy buckets
- Reallocation sanity shows random beneficiary selection
- Worst-case examples show systematic failures (e.g., always underweighting starters)

## Baseline Definition

The baseline is `rate_weighted`:
```python
log_weight = log(season_{target}_per_min * minutes_pred_p50 + 0.5)
```

This represents the "null model": predict usage proportional to historical rate × expected minutes.
