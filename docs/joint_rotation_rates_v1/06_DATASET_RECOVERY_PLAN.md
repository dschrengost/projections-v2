# Joint Rotation/Rates v1 Dataset Recovery Plan (2026-02-23)

## Summary
Rebuild the rotation source dataset with full prior/volatility + Action props feature families, then rebuild the joint dataset from that source so we recover missing old signal groups without losing newer lineup/odds/game-context contract columns.

Locked decisions:
- Date scope: 2024-10-01 through 2026-02-15 (latest fully labeled day)
- Action props (`an_*`): included in main dataset

## Contract Requirements

### Must be present in recovered features
- PBP prior + volatility families:
  - `*_prior_{5,10,20}`
  - `*_std_prior_{5,10,20}`
  - `first_in_*`, `last_out_*`, `max_stint_*`, team depth/bench/starter volatility families
- Action props:
  - `action_props_as_of_ts`
  - `an_*` market/line/probability/availability fields
- Prop-implied minutes:
  - `an_implied_minutes`, `an_has_implied_minutes`, related prior fields

### Must still be present from new contract
- `lineup_available`
- `lineup_starter_announced`
- `vegas_total`, `vegas_spread`, `estimated_possessions`
- `vegas_total_missing`, `vegas_spread_missing`, `estimated_possessions_missing`

### Must be excluded from model input features
- `is_confirmed_starter`
- `first_in_time_real`
- `last_out_time_real`
- `time_unit_detected`

## Build Sequence

1. Build recovered rotation source dataset
```bash
uv run python scripts/rotation/build_rotation_train_dataset_v1.py \
  --minutes-dataset-dir /home/daniel/projections-data/training/datasets/v1_enriched_boxscore_20260218_livefill \
  --start-date 2024-10-01 \
  --end-date 2026-02-15 \
  --out-dir /home/daniel/projections-data/training/datasets/rotation_train_v1_pbp_priors_boxscore_oddsbf_actionprops_livefill_20260223T<ts>
```

2. Build joint dataset from recovered rotation source
```bash
uv run python scripts/rotation/build_joint_rotation_rates_dataset_v1.py \
  --rotation-dataset-dir /home/daniel/projections-data/training/datasets/rotation_train_v1_pbp_priors_boxscore_oddsbf_actionprops_livefill_20260223T<ts> \
  --start-date 2024-10-01 \
  --end-date 2026-02-15 \
  --out-dir /home/daniel/projections-data/training/datasets/joint_rotation_rates_v1_20260223T<ts>
```

3. Build a Rotowire-era parity slice for sensitivity checks (defer to eval harness phase)
```bash
uv run python scripts/rotation/build_joint_rotation_rates_dataset_v1.py \
  --rotation-dataset-dir /home/daniel/projections-data/training/datasets/rotation_train_v1_pbp_priors_boxscore_oddsbf_actionprops_livefill_20260223T<ts> \
  --start-date 2025-12-26 \
  --end-date 2026-02-15 \
  --out-dir /home/daniel/projections-data/training/datasets/joint_rotation_rates_v1_rotowire_era_20260223T<ts>
```
This subset aligns with the historical window where archived `silver/rotowire_lineups` exists.
Execution timing: build this subset when implementing the evaluation harness described in
`docs/joint_rotation_rates_v1/GAME_TRANSFORMER_SPEC.md` (Evaluation Plan / go-no-go diagnostics), not as a blocker for the base dataset build.

## Validation Checklist

1. Schema
- Recovered `an_*` columns are present.
- Recovered prior/volatility families are present.
- New lineup + game-context contract columns are present.
- Forbidden same-game leakage columns are absent.

2. Coverage
- `labels_boxscore_counts` join rate:
  - overall >= 0.93
  - completed-game subset >= 0.99
- `vegas_total` / `vegas_spread` coverage >= 0.90
- `estimated_possessions` final coverage = 1.00
- `lineup_available` coverage is tracked (expected ~0.30-0.40 with current historical lineup archives)
- `an_has_any_props` in expected range for this scope (~0.25 to 0.35)

3. Integrity
- Unique `(game_id, team_id, player_id, game_date)` keys.
- Row counts align across:
  - `features.parquet`
  - `labels_minutes.parquet`
  - `labels_rates.parquet`
  - `labels_boxscore_counts.parquet`

4. Comparison
- Compare against:
  - `/home/daniel/projections-data/training/datasets/joint_rotation_rates_v1_20260221T163500Z`
  - `/home/daniel/projections-data/training/datasets/joint_rotation_rates_v1_20260223T204243Z`
- Confirm key old feature families are recovered and new contract fields are retained.

5. Train/Serve parity diagnostics (required)
- Confirm training features include `lineup_available` and `lineup_starter_announced`.
- Confirm `lineup_starter_announced` is 0 whenever `lineup_available=0`.
- Report model performance by lineup state:
  - `lineup_available=0` vs `lineup_available=1`
  - `lineup_available=1, lineup_starter_announced=0` vs `=1`
- Run the same diagnostics on both:
  - full training window dataset
  - Rotowire-era parity slice dataset
Timing: run these diagnostics as part of the eval harness implementation from
`docs/joint_rotation_rates_v1/GAME_TRANSFORMER_SPEC.md`.

## Post-Dataset Parity Work (Required Before Promotion)

1. Training/evaluation slice reporting
- Add lineup-state slices to training/eval outputs from:
  - `scripts/rotation/train_joint_rotation_rates_model_v1.py`
  - `scripts/rotation/eval_joint_vs_current.py`
- Persist per-slice metrics in run artifacts (JSON + console summary).

2. Live calibration by lineup state
- In live scoring calibration, maintain separate correction terms for:
  - lineup unknown (`lineup_available=0`)
  - lineup known, non-starter (`lineup_available=1, lineup_starter_announced=0`)
  - lineup known, starter (`lineup_available=1, lineup_starter_announced=1`)

3. Production monitoring
- Add dashboard/ops checks for:
  - daily `lineup_available` rate
  - daily `lineup_starter_announced` rate
  - error drift by lineup state (minutes and key rates)
- Set alerts on abrupt shifts from trailing 14-day baseline.

## Assumptions
- Source minutes dataset: `v1_enriched_boxscore_20260218_livefill`
- End date `2026-02-15` is used because labels are complete through this day.
- Keep defaults enabled in rotation builder:
  - lineup backfill from `silver/nba_daily_lineups` ON
  - odds backfill ON
  - action props ON
  - prop-implied minutes ON
  - team-game validation ON
- Historical Rotowire lineups are unavailable before 2025-12-26, so full-window training cannot be pure Rotowire-parity.

## Contingency
If recovered prior/volatility families are unexpectedly missing, rebuild priors first:
```bash
uv run python scripts/rotation/build_rotation_priors_v1.py --clean --overwrite
```
Then rerun the rotation and joint dataset builds.
