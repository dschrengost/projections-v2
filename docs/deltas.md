Feature: Archetype-Based Injury Deltas for Minutes Model

0. Goal / Non-Goals

Goal

Capture how a player’s minutes respond when archetype-level teammates are out (e.g., “backup C when starting C sits”) and expose this as stable, low-noise features for the minutes model.

Non-Goals (for this task)
	•	No strict (player, teammate) pair deltas.
	•	No modeling of usage, FGA, etc. This is minutes-only.
	•	No complex Bayesian shrinkage; we’ll use simple pooling + thresholds.

⸻

1. Data Dependencies

Use existing artifacts where possible:
	•	Labels
data/labels/season={season}/boxscore_labels.parquet
Required columns (or equivalent):
	•	game_id
	•	season
	•	team_id
	•	player_id
	•	minutes (actual minutes label)
	•	started_flag (bool/int, 1 if started)
	•	Optional but helpful: primary_position (e.g., “PG”, “SF”, “C”, etc.)
	•	Injury / availability snapshot at tip
Whatever you’re already using in the labels pipeline to mark OUT/available at tip, e.g.:
data/gold/injury_features/season={season}/injury_features.parquet
Required columns (per game, per player):
	•	game_id
	•	player_id
	•	team_id
	•	available_at_tip or status_at_tip (to derive OUT vs IN)

Assumption: labels + injury snapshot can be joined on (game_id, player_id).

⸻

2. Role Definition (Archetypes)

We define a role key per (season, player_id) that combines position group + starter/bench tier.

2.1 Position Group

From primary_position (or equivalent) define:
	•	Map:
	•	{"PG", "SG"} → "G"
	•	{"SF", "PF"} → "F"
	•	{"C"} → "C"
	•	Hybrids like "G-F", "F-C": take the first letter (or use a simple heuristic).

Output column:
	•	position_group ∈ {"G","F","C"}

If you don’t have primary_position, use whatever roster table you already have and mirror this mapping.

2.2 Starter Tier

Using season-level stats:
	•	For each (season, player_id):
	•	games_played = count games with any minutes > 0
	•	games_started = count games with started_flag == 1
	•	start_rate = games_started / games_played (guard against div by 0).

Define starter_tier:
	•	If games_played < 10: "unknown" (treated as bench in v1).
	•	Else:
	•	start_rate ≥ 0.6 → "starter"
	•	0.2 ≤ start_rate < 0.6 → "swing"
	•	start_rate < 0.2 → "bench"

Output column:
	•	starter_tier ∈ {"starter","swing","bench","unknown"}

2.3 Role Key

Combine:
	•	role_key = position_group + "_" + starter_tier
Examples: "G_starter", "F_bench", "C_swing".

Persist a roles artifact:
	•	File:
data/gold/minutes_roles/season={season}/roles.parquet
	•	Schema:
	•	season
	•	player_id
	•	position_group
	•	starter_tier
	•	role_key

This artifact is used by the archetype delta builder and can be reused elsewhere.

⸻

3. Archetype Delta Estimation (Offline)

We want to answer:

For players with role role_p, how do their minutes change when teammates with role role_t are missing?

3.1 Construct Game-Level Panel

For each game:
	1.	Join boxscore_labels with roles and injury availability.
	2.	For each (game_id, team_id):
	•	Build a list of teammates and their roles:
	•	For each player on the team:
	•	role_key
	•	available_at_tip (bool or derived from status)
	•	Compute, for each role_key on that team:
	•	available_players_role[role_key] = count of players with role_key and available_at_tip == True
	•	missing_players_role[role_key] = total players with that role_key on roster − available_players_role[role_key]

We don’t need perfect roster completeness; if you only see players in the boxscores + injury table, base counts on that.

3.2 Build Player-Game Rows with Role Context

For each player_id row in boxscore_labels:
	•	Attach:
	•	role_p = player’s role_key
	•	For each role_t seen on that team across the season:
	•	missing_role_t_count for this game (0, 1, 2, …)
	•	Store:
	•	minutes (label)
	•	game_id, team_id, season

We do not explode all roles into columns; instead, we aggregate by (role_p, role_t).

3.3 Aggregate by (role_p, role_t)

For each pair (role_p, role_t):
	•	Consider all player-game rows where the player had role role_p, and role_t appears on that player’s team at some point in the season.

Split into two subsets:
	•	Present: games where missing_role_t_count == 0
	•	Missing: games where missing_role_t_count ≥ 1

Compute:
	•	n_games_present
	•	n_games_missing
	•	mean_minutes_present = average minutes over present games
	•	mean_minutes_missing_any = average minutes over missing games
	•	Optionally:
	•	mean_missing_role_t_count over missing games

Define raw delta:
	•	delta_any = mean_minutes_missing_any - mean_minutes_present

Optionally scale to per missing player:
	•	delta_per_missing = delta_any / max(mean_missing_role_t_count, 1.0)

3.4 Guardrails / Thresholds

To avoid noisy nonsense:
	•	Only keep a (role_p, role_t) combo if:
	•	n_games_present ≥ min_present_games (default: 50)
	•	n_games_missing ≥ min_missing_games (default: 15)
	•	Clip deltas:
	•	delta_per_missing clipped to e.g. [-12, +12] minutes.

Expose thresholds via config:
	•	config/minutes_archetype_deltas.yaml:
	•	min_present_games
	•	min_missing_games
	•	max_abs_delta_per_missing

3.5 Output Artifact

File:
data/gold/features_minutes_v1/season={season}/archetype_deltas.parquet

Schema:
	•	season
	•	role_p (string, role_key)
	•	role_t (string, role_key)
	•	delta_per_missing (float32)
	•	delta_any (float32)
	•	n_games_present (int32)
	•	n_games_missing (int32)

This is the only thing the runtime feature builder needs to use.

⸻

4. Runtime Integration (Feature Builder)

We extend MinutesFeatureBuilder (or equivalent) to consume archetype_deltas.parquet and add a small set of scalar features per player-game.

4.1 Loading Deltas

At slate build:
	•	Load roles for current season:
roles.parquet
	•	Load archetype deltas:
archetype_deltas.parquet filtered to season.

Build an in-memory structure keyed by (role_p, role_t) → delta_per_missing.

4.2 Per Player-Game Feature Computation

For each player row (one per (game_id, team_id, player_id) in the feature builder):
	1.	Determine role_p from roles (fallback is allowed; see below).
	2.	For this game, compute team role counts:
	•	missing_role_t_count for all roles role_t present in archetype_deltas.
	•	At minimum:
	•	missing_same_position_count = sum of missing teammates with same position_group.
	3.	Compute archetype delta features:

Loop over all role_t where missing_role_t_count > 0:
	•	Look up delta_per_missing for (role_p, role_t):
	•	If not found, treat as 0.0.
	•	Contribution:
	•	contrib = delta_per_missing * missing_role_t_count.
	•	Aggregate:
	•	arch_delta_sum += contrib
	•	If role_t has same position_group as role_p, update:
	•	arch_delta_same_pos += contrib
	•	arch_missing_same_pos_count += missing_role_t_count

Final features (per row):
	•	arch_delta_sum (float32)
	•	arch_delta_same_pos (float32)
	•	arch_missing_same_pos_count (int32)
	•	arch_missing_total_count (int32) – total missing teammates across all roles, even those with zero delta.

Optionally also:
	•	arch_delta_max_role (max positive contrib across role_t)
	•	arch_delta_min_role (most negative contrib)

4.3 Handling Unknown / Sparse Roles

If role_p is missing (e.g., starter_tier = "unknown" or we lack primary_position):
	•	Set:
	•	arch_delta_sum = 0
	•	arch_delta_same_pos = 0
	•	Still compute:
	•	arch_missing_same_pos_count (if we can infer position_group)
	•	arch_missing_total_count

This way, you still get robust “how many teammates are out” signals even if the specific role mapping is weak.

⸻

5. CLI / Config

5.1 Build Roles

New CLI module:
	•	projections/cli/build_minutes_roles.py

Example command:

uv run python -m projections.cli.build_minutes_roles \
    --seasons 2022-23 2023-24 2024-25 \
    --labels-root data/labels \
    --out-root   data/gold/minutes_roles


Responsibilities:
	•	Load boxscore_labels.parquet per season.
	•	Compute position_group, starter_tier, role_key.
	•	Write roles.parquet per season.

5.2 Build Archetype Deltas

New CLI:
	•	projections/cli/build_archetype_deltas.py

Example:

uv run python -m projections.cli.build_archetype_deltas \
    --seasons 2022-23 2023-24 2024-25 \
    --labels-root        data/labels \
    --injury-root        data/gold/injury_features \
    --roles-root         data/gold/minutes_roles \
    --out-root           data/gold/features_minutes_v1 \
    --config             config/minutes_archetype_deltas.yaml

esponsibilities:
	•	For each season:
	•	Load labels, injury features, roles.
	•	Build per-game role missing counts.
	•	Aggregate (role_p, role_t) deltas.
	•	Apply thresholds + clipping from config.
	•	Write archetype_deltas.parquet.

5.3 Wiring into Minutes Feature Build

Update your existing minutes feature CLI (whatever you use; example):

uv run python -m projections.cli.build_minutes_features \
    --season 2024-25 \
    --labels-root        data/labels \
    --injury-root        data/gold/injury_features \
    --roles-root         data/gold/minutes_roles \
    --archetype-root     data/gold/features_minutes_v1 \
    --out-root           data/gold/features_minutes_v1


	•	Ensure MinutesFeatureBuilder:
	•	Loads roles.parquet and archetype_deltas.parquet.
	•	Adds the features described in §4 to the final training/inference matrix.

⸻

6. Tests / Acceptance Criteria

6.1 Unit Tests

Add tests under e.g. tests/features/test_archetype_deltas.py:
	1.	Role assignment sanity
	•	Synthetic dataset with known started_flag patterns and positions.
	•	Verify that:
	•	High start_rate → starter
	•	Low start_rate → bench
	•	Role keys are as expected.
	2.	Archetype delta aggregation
	•	Small synthetic season with:
	•	A simple team, e.g., "G_starter" and "G_bench".
	•	Construct games where the starter is present vs missing.
	•	Check that:
	•	delta_per_missing has the correct sign and approximate magnitude.
	•	n_games_present and n_games_missing respect thresholds.
	•	Unsatisfied thresholds drop rows as expected.
	3.	Runtime feature builder
	•	Synthetic injury + label + roles data for one slate.
	•	Verify that for a game where "C_starter" is missing:
	•	The "C_bench" rows get positive arch_delta_sum / arch_delta_same_pos.
	•	Unaffected positions (e.g. "G_bench") see near-zero contributions.
	•	Verify missing/sparse roles fall back to 0 deltas but still get count features.

6.2 Integration / Regression Checks
	•	Add a small CLI smoke test (in CI or locally):
	•	Run the two new CLIs for a tiny season subset.
	•	Check that:
	•	roles.parquet and archetype_deltas.parquet exist and have non-empty rows.
	•	Minutes features matrix for a known game includes the new columns.

6.3 Modeling Acceptance

On a held-out backtest window:
	•	Compare minutes model with vs without archetype features.
	•	Acceptance criteria (tunable, but directionally):
	•	Overall minutes MAE improved or unchanged.
	•	On “high injury” games (>=2 teammates out), MAE improves by a measurable margin.
	•	No visible blow-ups for bench players (no crazy outliers due to these features).

If archetype features clearly worsen metrics, treat that as a red flag for the builder (bugs in deltas or leakage) rather than just “model doesn’t like them”.
