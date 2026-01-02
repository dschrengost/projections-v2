# Rates v1 – North Star Document

**Goal:**  
Build a *practical, extensible* `rates_v1` module that predicts **per-minute mean rates** for key box-score stats using LightGBM, minutes, tracking, injuries, and Vegas context – and feeds a later sim layer that handles variance, tails, and correlations.

This doc is here to prevent us from drifting into 18 different architectures when we get tired or excited.

---

## 0. Design Principles

- **Separation of concerns**
  - Minutes model: predicts *minutes distributions*.
  - Rates model (`rates_v1`): predicts *per-minute means* for stats.
  - Sim: turns minutes + rates into *worlds* (distributions, tails, correlations).
- **GBM as v1, not destiny**
  - `rates_v1` is LightGBM-based, but its **interfaces and data** must be “deep-ready.”
  - Later `rates_v2` could be multi-task NN / flow / whatever, but must consume the same tables and emit the same outputs.
- **Means only**
  - `rates_v1` outputs **means**, never samples.
  - No QP, no “physics,” no worldgen in this module.
- **Data-first**
  - Clean, reproducible `rates_training_base` dataset is the real asset.
  - Tracking promotion is incremental (bronze → minimal silver → features), not a big bang.

---

## 1. Scope of `rates_v1`

### 1.1. Outputs

For each (player, game) on a slate, `rates_v1` returns **per-minute means**:

- `fga2_per_min_mean`  – 2PA per minute
- `fga3_per_min_mean`  – 3PA per minute
- `fta_per_min_mean`   – FTA per minute
- `ast_per_min_mean`
- `tov_per_min_mean`
- `oreb_per_min_mean`
- `dreb_per_min_mean`
- `stl_per_min_mean`
- `blk_per_min_mean`

These are **conditional on**:

- player role & history,
- minutes context,
- matchup & Vegas,
- injuries / usage vacated.

The sim will later convert these into **counts per world** via Poisson/NegBin, apply any team-budget corrections, and then convert to FPTS.

### 1.2. Explicit non-goals (for v1)

- No direct modeling of:
  - fantasy points,
  - usage% as a primary target,
  - touches/time-of-possession as outputs.
- No:
  - QP / optimization,
  - game scripts,
  - possession-level simulation.
- No deep models yet (NNs, flows). They’re a v2+ overlay.

---

## 2. Data Stack – Staged Plan

We don’t wait for “perfect tracking gold.” We build in stages.

### Stage 0 – `gold.rates_training_base` (no tracking)

Minimum viable dataset for training GBMs:

- One row = `(season, game_id, team_id, player_id)`.
- Derived targets: per-minute stats from boxscore + minutes labels.
- Features from:
  - minutes gold,
  - boxscore gold,
  - odds/context (spread, total, implied totals),
  - roster (position, starter flag),
  - injuries & usage vacated.

This is enough to train a baseline `rates_v1` with no tracking features.

### Stage 1 – `silver.tracking_game` (player-game tracking aggregates)

Promote tracking bronze → silver:

- Aggregate per-player, per-game tracking stats:
  - touches, time of possession, passes, drives, etc.
- Just enough structure to compute season/rolling averages as features.

### Stage 2 – Add tracking features into `rates_training_base`

- Join `silver.tracking_game` (or its rolling view) into `gold.rates_training_base`.
- Re-train `rates_v1` with tracking features:
  - `season_touches_per_min`,
  - `season_sec_per_touch`,
  - `season_drives_per_min`, etc.

### Stage 3 – (Later) Role clustering & fancy features

- Use tracking + box to cluster players into roles (Helio/Hub/Finisher/Connector).
- Add `role_cluster_id` (or one-hot) as a feature.
- Possibly add more nuanced opp defensive features.

---

## 3. Schema: `gold.rates_training_base`

**Table:** `gold.rates_training_base`  
**Grain:** one row per `(season, game_id, team_id, player_id)` where `minutes_actual > 0`.

### 3.1. Key columns

```text
season              int
game_id             string/int
game_date           date
team_id             string
opponent_id         string
player_id           string

3.2. Labels / targets

minutes_actual          float

fga2_per_min            float  # 2PA / minutes_actual
fga3_per_min            float  # 3PA / minutes_actual
fta_per_min             float
ast_per_min             float
tov_per_min             float
oreb_per_min            float
dreb_per_min            float
stl_per_min             float
blk_per_min             float

Where 2PA/3PA split is computed from boxscore (FGA and 3PA). If you don’t have explicit 2PA/3PA in gold yet, that’s a prerequisite.

3.3. Player / role features

position_primary        string   # e.g. "PG","SG","SF","PF","C"
position_flags_PG       int 0/1
position_flags_SG       int 0/1
position_flags_SF       int 0/1
position_flags_PF       int 0/1
position_flags_C        int 0/1

height_inches           float (optional)
weight_lbs              float (optional)

# Season or long-run role aggregates (from boxscore + tracking)
season_fga_per_min      float
season_3pa_per_min      float
season_fta_per_min      float
season_ast_per_min      float
season_tov_per_min      float
season_reb_per_min      float
season_stl_per_min      float
season_blk_per_min      float

# (After Stage 2: tracking-derived)
season_touches_per_min  float
season_sec_per_touch    float
season_drives_per_min   float

If some of these season aggregates aren’t available yet, mark them as TODOs and start with what you have.

3.4. Minutes / opportunity features

minutes_expected_p50    float   # from minutes_v1 model (as-of tip in training history)
minutes_spread          float   # p90 - p10 (from minutes_v1)
play_prob               float   # from minutes_v1

is_starter              int 0/1
bench_tier              int     # 0=starter, 1=primary bench, 2=deep bench (optional)

Training must use minutes predictions from a version of the minutes model that simulates as-of-game-day information (no leakage).

3.5. Game context / Vegas features

home_flag               int 0/1
days_rest               int (capped, e.g. 0–3+)

spread_close            float   # from odds; sign conventions must be consistent
total_close             float
team_itt                float   # implied team total
opp_itt                 float

season_pace_team        float
season_pace_opp         float
opp_def_rating          float
opp_def_3pa_allowed     float   # or 3PA allowed per game
opp_def_reb_rate        float   # opp REB% allowed
opp_def_ast_rate        float   # opp AST% allowed

These can come from a separate team-level stats table and joined by (season, opponent_id).

3.6. Injury / usage-vacated features

num_rotation_players_out       int
team_minutes_vacated           float  # vs some baseline rotation
team_usage_vacated             float  # sum over out players of (USG * minutes_share)

star_scorer_out_flag           int 0/1
primary_ballhandler_out_flag   int 0/1
starting_center_out_flag       int 0/1

These are computed using a long-run baseline per player (e.g., last season’s USG/MP, or rolling window).

3.7. Short-term form (optional initial pass)

last3_minutes_per_game         float
last3_fga_per_min              float
last3_3pa_per_min              float
last3_fta_per_min              float
last3_ast_per_min              float
last3_tov_per_min              float
last3_reb_per_min              float

These come from rolling windows over past games (excluding current game).

⸻

4. Schema: silver.tracking_game (Stage 1)

Table: silver.tracking_game
Grain: one row per (season, game_id, team_id, player_id).

4.1. Keys

season          int
game_id         string/int
team_id         string
player_id       string
game_date       date

4.2. Core tracking aggregates

minutes_tracking        float  # if provided; otherwise join minutes_actual for reference

touches                 float
time_of_poss_seconds    float
passes_made             float
drives                  float
# Add any others you actually get and trust

# convenience per-minute fields (optional but useful)
touches_per_min         float  # touches / minutes_tracking
top_per_min             float  # time_of_poss_seconds / 60 / minutes_tracking
passes_per_min          float
drives_per_min          float

Later, you’ll build rolling/season aggregates from this:
	•	season_touches_per_min, season_sec_per_touch, season_drives_per_min.

These live either in:
	•	a gold.tracking_rollups table, or
	•	are computed on the fly when building rates_training_base.

⸻

5. Model Training – rates_v1

5.1. Targets

We train one LightGBM regressor per target:

fga2_per_min
fga3_per_min
fta_per_min
ast_per_min
tov_per_min
oreb_per_min
dreb_per_min
stl_per_min
blk_per_min

Each model uses the same feature columns.

5.2. Train / cal / val splits

Time-based splits, e.g.:
	•	Train: seasons & dates up to some cutoff (e.g., 2018–2023-02-28).
	•	Cal: next window (e.g., 2023-03-01 to 2024-02-15).
	•	Val: most recent window (e.g., 2024-02-16 onward).

Principles:
	•	No leakage across time.
	•	Features for each row are constructed from data available before that game, consistent with live inference.

5.3. LightGBM parameters (initial defaults)

base_params = {
    "objective": "regression",
    "metric": "l2",
    "boosting_type": "gbdt",
    "num_leaves": 64,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "max_depth": -1,
    "lambda_l2": 1.0,
}

We can tweak per target later (e.g., more regularization for noisy ones like STL/BLK), but this is fine to start.

5.4. Training script sketch
	•	scripts/rates/build_training_base.py
	•	Produces gold.rates_training_base.parquet.
	•	scripts/rates/train_rates_v1.py
	•	Loads training base.
	•	Splits into train/cal/val.
	•	Trains one GBM per target.
	•	Saves models & metadata into artifacts/rates_v1/runs/<run_id>/.

⸻

6. Inference API – rates_v1

rates_v1 lives as a small library with a stable API.

6.1. Bundle loader

# projections/rates_v1/loader.py

from dataclasses import dataclass
import lightgbm as lgb

@dataclass
class RatesBundle:
    models: dict[str, lgb.Booster]
    feature_cols: list[str]
    meta: dict

def load_rates_bundle(run_id: str) -> RatesBundle:
    ...

6.2. Scorer

# projections/rates_v1/score.py

import pandas as pd

def predict_rates(df_features: pd.DataFrame, bundle: RatesBundle) -> pd.DataFrame:
    X = df_features[bundle.feature_cols].values
    out = {}
    for target, model in bundle.models.items():
        out[target + "_mean"] = model.predict(X)
    return pd.DataFrame(out, index=df_features.index)

Externally, the world only sees:
	•	Input: features DF.
	•	Output: per-minute mean columns.

How you got there (GBM vs deep, single vs multi-model) is internal.

6.3. Live feature builder (mirrors training)

# projections/rates_v1/build_features_live.py

def build_rates_features_for_slate(
    slate_games,
    as_of_ts,
    minutes_bundle,
    data_root,
) -> pd.DataFrame:
    """
    Returns df_features with rows aligned to (player, game) for the slate,
    with the same feature_cols expected by rates_v1.
    """
This function:
	•	Pulls:
	•	live minutes expectations (p50, p10, p90, play_prob),
	•	latest injuries/status at as_of_ts,
	•	current season aggregates (box + tracking),
	•	current Vegas lines.
	•	Constructs the same feature columns as rates_training_base.

⸻

7. How This Plays with Future Deep Models

To keep the door open for a deep rates_v2:
	•	Do NOT bake GBM-specific assumptions into:
	•	rates_training_base schema,
	•	predict_rates signature,
	•	sim inputs.
	•	A future rates_v2 can:
	•	read the same rates_training_base table,
	•	use the same feature set (maybe extended),
	•	expose the same outputs (per-minute means).

We can also use rates_v1 as a teacher:
	•	rates_v2 would predict residuals over rates_v1 means, improving structure while keeping GBM as a prior.

⸻

8. TODO / Implementation Checklist

Stage 0 – without tracking
	•	Define gold.rates_training_base schema (this doc).
	•	Implement scripts/rates/build_training_base.py.
	•	Implement scripts/rates/train_rates_v1.py:
	•	config for train/cal/val date ranges,
	•	LightGBM training loop over targets,
	•	artifact output (models + feature list + meta).
	•	Implement projections/rates_v1/loader.py & score.py.
	•	Implement projections/rates_v1/build_features_live.py (basic version without tracking).

Stage 1 – tracking silver
	•	Implement silver.tracking_game ETL:
	•	from bronze tracking endpoints,
	•	to per-player, per-game aggregates.
	•	Implement rolling tracking aggregates (season or N-game):
	•	season_touches_per_min, season_sec_per_touch, etc.
	•	Update build_training_base.py to join tracking rollups.
	•	Re-train rates_v1 with tracking-enhanced features → new run_id.
	•	Compare metrics vs baseline; choose production bundle.

Stage 2+ – extras
	•	Role clustering using tracking + box (Helio/Hub/Finisher/Connector).
	•	Add role_cluster_id to features.
	•	Explore more nuanced opponent features.
	•	Plan rates_v2 deep experiment (multi-task NN / residual over GBM).

⸻

This document is the guardrail:
	•	If we catch ourselves talking about possession Markov chains, heavy QP, or flows before gold.rates_training_base + rates_v1 + a basic sim exist, we’re off-plan.




