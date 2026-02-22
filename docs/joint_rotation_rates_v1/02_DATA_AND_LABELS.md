# 02 Data And Labels

## Training Example Unit

Keep the same primary unit as rotation-set minutes:

- one example per `(game_id, team_id)` with variable number of player rows.

This preserves:

- set-model compatibility,
- team-level 240-minute constraints,
- existing train/eval splits built around team-games.

## Data Sources

1. Rotation/minutes training dataset:
- built by `scripts/rotation/build_rotation_train_dataset_v1.py`
- outputs `features.parquet` + `labels.parquet`.

2. Rates training base:
- built by `scripts/rates/build_training_base.py`
- includes per-minute labels and efficiency labels.

## Join Keys

Use strict key alignment:

- `game_id`,
- `team_id`,
- `player_id`,
- `game_date` (normalized date, when available).

Any key mismatch should be logged and excluded from supervised rates loss for that row.

## Label Groups

### Minutes/Rotation Labels

- minutes label from rotation/minutes dataset (`labels.parquet`).
- in-rotation binary label derived from minutes threshold (same as current workflow).

### Rate Labels

Per-minute:

- `fga2_per_min`,
- `fga3_per_min`,
- `fta_per_min`,
- `ast_per_min`,
- `tov_per_min`,
- `oreb_per_min`,
- `dreb_per_min`,
- `stl_per_min`,
- `blk_per_min`.

Efficiency:

- `fg2_pct_label`,
- `fg3_pct_label`,
- `ft_pct_label`.

## Loss Masks And Weights

1. Minutes loss mask:
- valid roster slots in team-game.

2. Rotation loss mask:
- valid + alloc-eligible slots (existing behavior).

3. Rates loss mask:
- valid slots with observed minutes above threshold (start with `minutes_actual >= 4` parity with rates base),
- optional smooth weighting by actual minutes to downweight noisy tiny-minute rows.

4. Efficiency loss mask:
- only where attempts are sufficient (existing label construction logic already encodes this).

## Anti-Leak Rules

Do not include any same-game post-tip columns in inputs (same policy as current rotation training):

- same-game rotation labels/stint-derived realized fields,
- actual stat outcomes for the target game,
- any columns whose availability is after `as_of_ts`.

Preserve existing timestamp integrity:

- enforce `feature_as_of_ts <= tip_ts`.

## Dataset Artifact Proposal

New dataset root (suggested):

- `$PROJECTIONS_DATA_ROOT/training/datasets/joint_rotation_rates_v1_<ts>/`

Contents:

- `features.parquet` (model inputs, one row per player-game-team),
- `labels_minutes.parquet`,
- `labels_rates.parquet`,
- `team_game_index.parquet` (grouping metadata),
- `manifest.json` (feature list, label coverage, filtering stats, date windows).

## Minimum Coverage Checks

Before training:

1. `% team-games with full minutes labels`.
2. `% player-rows with valid rates labels by target`.
3. Coverage in injury-heavy slices (vacated-minutes buckets).
4. Drift checks vs live feature distributions for critical features.
