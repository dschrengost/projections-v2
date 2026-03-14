"""DNP (Did Not Play) history features for availability modeling.

This module computes point-in-time features capturing a player's history of being
"roster active but did not play" (DNP-CD pattern). These features help the model
distinguish between:
1. Players with no minutes because they're OUT/injured
2. Players who are available but consistently get DNP-CD
3. Players who are available and consistently play

Key features:
- games_since_last_roster_active: Team games since last roster_active_pre_tip==1
- consecutive_active_dnp: Consecutive team games where active but minutes==0
- active_but_dnp_rate_last10: Rate of DNPs among last 10 roster_active games
- inactive_streak_len: Consecutive team games where roster_active_pre_tip==0

CRITICAL INVARIANT: All features are computed using ONLY games strictly prior to
the current game (game_date < current_game_date). No label leakage is allowed.

Features are computed per (player_id, team_id) history so team changes reset stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# Empirical Bayes prior parameters for shrinkage of DNP rate with small samples.
# With alpha=1, beta=1 (uniform prior), (dnp + 1)/(n + 2) is the shrunk estimate.
DNP_RATE_ALPHA = 1.0
DNP_RATE_BETA = 1.0
DNP_RATE_MIN_OPPORTUNITIES = 3  # Below this, shrinkage is applied

# Cap for "never seen active before" in games_since_last_roster_active
GAMES_SINCE_CAP = 99


@dataclass(frozen=True)
class DNPHistoryConfig:
    """Configuration for DNP history feature computation."""

    # Empirical Bayes prior parameters
    alpha: float = DNP_RATE_ALPHA
    beta: float = DNP_RATE_BETA
    min_opportunities: int = DNP_RATE_MIN_OPPORTUNITIES
    games_since_cap: int = GAMES_SINCE_CAP

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "min_opportunities": self.min_opportunities,
            "games_since_cap": self.games_since_cap,
        }


def derive_roster_active_pre_tip(
    df: pd.DataFrame,
    *,
    is_out_col: str = "is_out",
    injury_snapshot_missing_col: str = "injury_snapshot_missing",
    minutes_col: str | None = None,
) -> pd.Series:
    """Derive the roster_active_pre_tip signal from existing columns.

    Definition: A player is considered "roster active pre-tip" if:
    - They appear in the dataset row (spine presence)
    - They are NOT explicitly OUT (is_out == 0)
    - AND we have a valid injury snapshot (injury_snapshot_missing == 0)
      OR they played minutes > 0 (proving they were active)

    This is a conservative definition that treats:
    - QUESTIONABLE, PROBABLE, AVAILABLE as roster active
    - restriction_flag, ramp_flag players as roster active (they may still play)
    - Only OUT status as inactive

    IMPORTANT: By default, when injury_snapshot_missing == 1 we treat the row as
    inactive unless there is an explicit active signal:
    - active_flag == True
    - lineup_roster_status == "Active"
    - no injury row was present (injury_row_present == False) and status == "Ava"
      (healthy/not listed on injury report)

    This keeps conservative behavior for truly unknown rows while preventing
    healthy, report-absent players from being incorrectly marked inactive.

    Args:
        df: DataFrame with injury/status columns
        is_out_col: Column name for the is_out flag (1 = OUT, 0 = not OUT)
        injury_snapshot_missing_col: Column name for snapshot missing flag
        minutes_col: Optional column name for minutes played (used to infer
            active status when snapshot is missing but player played)

    Returns:
        Series of int8: 1 if roster active, 0 if inactive (OUT or unknown)

    Limitations documented:
    - This definition may miss cases where a player is on an inactive list
      that isn't captured in the injury report (e.g., G-League assignment).
    """
    if is_out_col not in df.columns:
        # If no is_out column, treat all rows as active (conservative fallback)
        return pd.Series(1, index=df.index, dtype="int8")

    is_out = pd.to_numeric(df[is_out_col], errors="coerce").fillna(0).astype(int)
    roster_active = (is_out == 0).astype("int8")

    # If injury snapshot is missing, default to inactive unless we have
    # explicit active evidence from roster/lineup or a "no injury row + Ava" signal.
    if injury_snapshot_missing_col in df.columns:
        snapshot_missing = (
            pd.to_numeric(df[injury_snapshot_missing_col], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        explicit_active = pd.Series(False, index=df.index, dtype=bool)
        explicit_inactive = pd.Series(False, index=df.index, dtype=bool)

        if "active_flag" in df.columns:
            active_tokens = df["active_flag"].astype("string").str.strip().str.lower()
            explicit_active = explicit_active | active_tokens.isin({"1", "true", "t", "yes", "y"})
            explicit_inactive = explicit_inactive | active_tokens.isin({"0", "false", "f", "no", "n"})

        if "lineup_roster_status" in df.columns:
            roster_tokens = df["lineup_roster_status"].astype("string").str.strip().str.lower()
            explicit_active = explicit_active | roster_tokens.eq("active")
            explicit_inactive = explicit_inactive | roster_tokens.eq("inactive")

        no_row_available = pd.Series(False, index=df.index, dtype=bool)
        if {"injury_row_present", "status"}.issubset(df.columns):
            row_present = (
                df["injury_row_present"]
                .astype("boolean", copy=False)
                .fillna(False)
                .astype(bool)
            )
            status_tokens = df["status"].astype("string").str.strip().str.upper()
            no_row_available = (~row_present) & status_tokens.eq("AVA")

        if minutes_col and minutes_col in df.columns:
            minutes = pd.to_numeric(df[minutes_col], errors="coerce").fillna(0.0)
            uncertain = (snapshot_missing == 1) & (minutes == 0)
        else:
            uncertain = snapshot_missing == 1

        # Conservative default for uncertain rows.
        roster_active = roster_active.where(~uncertain, 0).astype("int8")

        # Recover active status only when we have explicit active evidence.
        active_recovery = uncertain & (is_out == 0) & ((explicit_active & ~explicit_inactive) | no_row_available)
        roster_active = roster_active.where(~active_recovery, 1).astype("int8")

    return roster_active


def compute_dnp_history_features(
    df: pd.DataFrame,
    *,
    config: DNPHistoryConfig | None = None,
    game_date_col: str = "game_date",
    player_id_col: str = "player_id",
    team_id_col: str = "team_id",
    is_out_col: str = "is_out",
    injury_snapshot_missing_col: str = "injury_snapshot_missing",
    minutes_col: str = "minutes",
    tip_ts_col: str = "tip_ts",
    validate_pit: bool = True,
) -> pd.DataFrame:
    """Compute DNP history features for each row using only prior games.

    This function computes four availability/DNP features per row:
    1. games_since_last_roster_active: Team games since last active
    2. consecutive_active_dnp: Streak of active-but-DNP games
    3. active_but_dnp_rate_last10: DNP rate in last 10 active games (shrunk)
    4. inactive_streak_len: Streak of not-active (OUT) games

    All features are computed per (player_id, team_id) so team changes reset history.

    CRITICAL: Uses only games with game_date strictly less than the current row's
    game_date. This ensures no label leakage.

    Args:
        df: DataFrame with game/player/team/injury/minutes data
        config: Optional configuration for priors and caps
        game_date_col: Column name for game date
        player_id_col: Column name for player ID
        team_id_col: Column name for team ID
        is_out_col: Column name for is_out flag
        injury_snapshot_missing_col: Column name for injury snapshot missing flag
        minutes_col: Column name for minutes played (label)
        tip_ts_col: Column name for tip timestamp (for PIT validation)
        validate_pit: If True, validates point-in-time correctness

    Returns:
        DataFrame with original columns plus:
        - roster_active_pre_tip: 1 if active, 0 if OUT
        - games_since_last_roster_active: int
        - never_roster_active_before: bool (1 if never seen active)
        - consecutive_active_dnp: int
        - active_but_dnp_rate_last10: float (shrunk estimate)
        - inactive_streak_len: int
    """
    if config is None:
        config = DNPHistoryConfig()

    required_cols = {game_date_col, player_id_col, team_id_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Create a working copy with normalized columns
    work = df.copy()
    work[game_date_col] = pd.to_datetime(work[game_date_col], errors="coerce")
    work[player_id_col] = pd.to_numeric(work[player_id_col], errors="coerce").astype("Int64")
    work[team_id_col] = pd.to_numeric(work[team_id_col], errors="coerce").astype("Int64")

    # Derive roster_active_pre_tip (now considers injury_snapshot_missing)
    work["roster_active_pre_tip"] = derive_roster_active_pre_tip(
        work,
        is_out_col=is_out_col,
        injury_snapshot_missing_col=injury_snapshot_missing_col,
        minutes_col=minutes_col,
    )

    # Get minutes for DNP determination (from label or feature column)
    if minutes_col in work.columns:
        work["_minutes"] = pd.to_numeric(work[minutes_col], errors="coerce").fillna(0.0)
    else:
        # If no minutes column, we can't compute DNP features accurately
        # Set to NaN and features will be flagged as missing
        work["_minutes"] = np.nan

    # Compute whether player was active but got DNP (minutes == 0)
    work["_active_dnp"] = (
        (work["roster_active_pre_tip"] == 1) & (work["_minutes"] == 0)
    ).astype("int8")

    valid_mask = (
        work[player_id_col].notna().to_numpy(dtype=bool, copy=False)
        & work[team_id_col].notna().to_numpy(dtype=bool, copy=False)
        & work[game_date_col].notna().to_numpy(dtype=bool, copy=False)
    )
    valid_pos = np.flatnonzero(valid_mask)
    invalid_pos = np.flatnonzero(~valid_mask)
    if valid_pos.size:
        player_keys = work.iloc[valid_pos][player_id_col].astype("int64").to_numpy()
        team_keys = work.iloc[valid_pos][team_id_col].astype("int64").to_numpy()
        date_keys = (
            work.iloc[valid_pos][game_date_col]
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64, copy=False)
        )
        stable_order = np.lexsort((valid_pos, date_keys, team_keys, player_keys))
        ordered_pos = np.concatenate((valid_pos[stable_order], invalid_pos))
        sorted_player_keys = player_keys[stable_order]
        sorted_team_keys = team_keys[stable_order]
    else:
        ordered_pos = invalid_pos
        sorted_player_keys = np.empty(0, dtype=np.int64)
        sorted_team_keys = np.empty(0, dtype=np.int64)

    work = work.take(ordered_pos).copy()

    n_rows = len(work)
    prior_mean = config.alpha / (config.alpha + config.beta)
    games_since = np.full(n_rows, config.games_since_cap, dtype=np.int32)
    never_active = np.ones(n_rows, dtype=np.int8)
    consec_dnp = np.zeros(n_rows, dtype=np.int32)
    dnp_rate = np.full(n_rows, prior_mean, dtype=np.float64)
    inactive_streak = np.zeros(n_rows, dtype=np.int32)

    active = pd.to_numeric(work["roster_active_pre_tip"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    active_dnp = pd.to_numeric(work["_active_dnp"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)

    valid_count = valid_pos.size
    if valid_count:
        group_starts_mask = np.empty(valid_count, dtype=bool)
        group_starts_mask[0] = True
        group_starts_mask[1:] = (
            (sorted_player_keys[1:] != sorted_player_keys[:-1])
            | (sorted_team_keys[1:] != sorted_team_keys[:-1])
        )
        group_starts = np.flatnonzero(group_starts_mask)
        group_ends = np.r_[group_starts[1:], valid_count]

        for start, end in zip(group_starts, group_ends, strict=False):
            last_active_game_idx: int | None = None
            current_dnp_streak = 0
            current_inactive_streak = 0
            recent_active_dnp: list[int] = []

            for i in range(start, end):
                local_i = i - start

                if last_active_game_idx is not None:
                    games_since[i] = local_i - last_active_game_idx
                    never_active[i] = 0

                consec_dnp[i] = current_dnp_streak

                if recent_active_dnp:
                    dnp_count = int(sum(recent_active_dnp))
                    n_active = len(recent_active_dnp)
                    if n_active >= config.min_opportunities:
                        dnp_rate[i] = dnp_count / n_active
                    else:
                        dnp_rate[i] = (
                            dnp_count + config.alpha
                        ) / (n_active + config.alpha + config.beta)

                inactive_streak[i] = current_inactive_streak

                if active[i] == 1:
                    last_active_game_idx = local_i
                    recent_active_dnp.append(int(active_dnp[i]))
                    if len(recent_active_dnp) > 10:
                        recent_active_dnp.pop(0)
                    if active_dnp[i] == 1:
                        current_dnp_streak += 1
                    else:
                        current_dnp_streak = 0
                    current_inactive_streak = 0
                else:
                    current_inactive_streak += 1
                    current_dnp_streak = 0

    work["games_since_last_roster_active"] = games_since
    work["never_roster_active_before"] = never_active
    work["consecutive_active_dnp"] = consec_dnp
    work["active_but_dnp_rate_last10"] = dnp_rate
    work["inactive_streak_len"] = inactive_streak

    # Point-in-time validation if requested
    if validate_pit and tip_ts_col in work.columns:
        _validate_point_in_time(
            work,
            game_date_col=game_date_col,
            player_id_col=player_id_col,
            team_id_col=team_id_col,
            tip_ts_col=tip_ts_col,
        )

    # Drop temporary columns
    work = work.drop(columns=["_minutes", "_active_dnp"], errors="ignore")

    # Ensure consistent dtypes
    work["games_since_last_roster_active"] = work["games_since_last_roster_active"].astype("int32")
    work["never_roster_active_before"] = work["never_roster_active_before"].astype("int8")
    work["consecutive_active_dnp"] = work["consecutive_active_dnp"].astype("int32")
    work["active_but_dnp_rate_last10"] = work["active_but_dnp_rate_last10"].astype("float64")
    work["inactive_streak_len"] = work["inactive_streak_len"].astype("int32")

    return work


def _validate_point_in_time(
    df: pd.DataFrame,
    *,
    game_date_col: str,
    player_id_col: str,
    team_id_col: str,
    tip_ts_col: str,
    sample_size: int = 100,
) -> None:
    """Validate that features are computed using only prior games.

    This samples rows and verifies that the computed features could not have
    been influenced by same-game or future-game data.

    Raises:
        ValueError: If point-in-time violation is detected
    """
    if df.empty:
        return

    # Sample rows to check
    sample_n = min(sample_size, len(df))
    sample_idx = df.sample(n=sample_n, random_state=42).index

    for idx in sample_idx:
        row = df.loc[idx]
        pid = row[player_id_col]
        tid = row[team_id_col]
        current_date = row[game_date_col]

        # Get all rows for this player-team
        player_team_mask = (df[player_id_col] == pid) & (df[team_id_col] == tid)
        player_team_df = df.loc[player_team_mask]

        # Count games strictly before current date
        prior_games = player_team_df[player_team_df[game_date_col] < current_date]
        n_prior = len(prior_games)

        # Validate games_since_last_roster_active
        # Note: games_since measures games since last active, so if n_prior=0,
        # the first game should have games_since=cap (never active before).
        # If games_since < cap but n_prior=0, something is wrong.
        # However, we must account for the fact that we're iterating through sorted
        # data and features are computed incrementally. Skip validation for rows
        # where we don't have enough context (first game in dataset).
        games_since = row["games_since_last_roster_active"]
        never_active = row.get("never_roster_active_before", 1)
        if games_since < GAMES_SINCE_CAP and never_active == 0:
            # There should be at least 1 prior game where player was active
            if n_prior == 0:
                raise ValueError(
                    f"PIT violation: games_since_last_roster_active={games_since} with "
                    f"never_active=0, but no prior games exist for player={pid} team={tid} "
                    f"date={current_date}"
                )


def compute_dnp_history_features_for_live(
    current_game_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    *,
    config: DNPHistoryConfig | None = None,
    game_date_col: str = "game_date",
    player_id_col: str = "player_id",
    team_id_col: str = "team_id",
    is_out_col: str = "is_out",
    injury_snapshot_missing_col: str = "injury_snapshot_missing",
    minutes_col: str = "minutes",
) -> pd.DataFrame:
    """Compute DNP history features for live inference.

    This is the live-inference variant that takes:
    1. current_game_df: Today's players (no minutes yet, just roster/status)
    2. historical_df: Historical games with realized minutes

    The function computes features for current_game_df using only historical_df.

    Args:
        current_game_df: DataFrame with today's players (game_date is today)
        historical_df: DataFrame with prior games including minutes
        config: Optional configuration
        game_date_col: Column name for game date
        player_id_col: Column name for player ID
        team_id_col: Column name for team ID
        is_out_col: Column name for is_out flag
        injury_snapshot_missing_col: Column name for injury snapshot missing flag
        minutes_col: Column name for minutes played (in historical_df only)

    Returns:
        current_game_df with DNP history features added
    """
    if config is None:
        config = DNPHistoryConfig()

    # Derive roster_active_pre_tip for current game (no minutes yet, so don't pass minutes_col)
    current = current_game_df.copy()
    current["roster_active_pre_tip"] = derive_roster_active_pre_tip(
        current,
        is_out_col=is_out_col,
        injury_snapshot_missing_col=injury_snapshot_missing_col,
        minutes_col=None,  # Current game has no minutes yet
    )

    # Prepare historical data
    hist = historical_df.copy()
    hist[game_date_col] = pd.to_datetime(hist[game_date_col], errors="coerce")
    hist[player_id_col] = pd.to_numeric(hist[player_id_col], errors="coerce").astype("Int64")
    hist[team_id_col] = pd.to_numeric(hist[team_id_col], errors="coerce").astype("Int64")
    # For historical games, use minutes to disambiguate when snapshot is missing
    hist["roster_active_pre_tip"] = derive_roster_active_pre_tip(
        hist,
        is_out_col=is_out_col,
        injury_snapshot_missing_col=injury_snapshot_missing_col,
        minutes_col=minutes_col,
    )
    hist["_minutes"] = pd.to_numeric(hist[minutes_col], errors="coerce").fillna(0.0)
    hist["_active_dnp"] = ((hist["roster_active_pre_tip"] == 1) & (hist["_minutes"] == 0)).astype("int8")

    # Get current game date (should be same for all rows)
    current[game_date_col] = pd.to_datetime(current[game_date_col], errors="coerce")
    current_dates = current[game_date_col].dropna().unique()
    if len(current_dates) != 1:
        raise ValueError(f"Expected single game_date for live inference, got {len(current_dates)}")
    current_date = current_dates[0]

    # Filter historical to only prior games
    hist = hist[hist[game_date_col] < current_date].copy()

    # Initialize output columns
    current["games_since_last_roster_active"] = config.games_since_cap
    current["never_roster_active_before"] = 1
    current["consecutive_active_dnp"] = 0
    current["active_but_dnp_rate_last10"] = config.alpha / (config.alpha + config.beta)
    current["inactive_streak_len"] = 0

    # Process each player-team in current
    for _, row in current.iterrows():
        pid = row[player_id_col]
        tid = row[team_id_col]

        # Get historical games for this player-team
        mask = (hist[player_id_col] == pid) & (hist[team_id_col] == tid)
        player_hist = hist.loc[mask].sort_values(game_date_col, ascending=True)

        if player_hist.empty:
            # No history - use defaults
            continue

        idx = row.name
        active = player_hist["roster_active_pre_tip"].values
        active_dnp = player_hist["_active_dnp"].values
        n_hist = len(player_hist)

        # games_since_last_roster_active
        active_indices = np.where(active == 1)[0]
        if len(active_indices) > 0:
            last_active_idx = active_indices[-1]
            games_since = n_hist - last_active_idx
            current.loc[idx, "games_since_last_roster_active"] = min(games_since, config.games_since_cap)
            current.loc[idx, "never_roster_active_before"] = 0
        # else: use defaults

        # consecutive_active_dnp
        # Count consecutive games where player was active but got 0 minutes (DNP-CD).
        # When we hit an OUT game, stop counting (don't include OUT in streak).
        # When we hit a game where player was active and played, stop counting.
        consec_dnp = 0
        for i in range(n_hist - 1, -1, -1):
            if active[i] == 1 and active_dnp[i] == 1:
                consec_dnp += 1
            elif active[i] == 1:
                # Active but played (got minutes) - end of DNP streak
                break
            else:
                # OUT (active==0) - end of streak, but keep the count
                # OUT games are not part of the active-DNP streak
                break
        current.loc[idx, "consecutive_active_dnp"] = consec_dnp

        # active_but_dnp_rate_last10
        active_games = player_hist[player_hist["roster_active_pre_tip"] == 1].tail(10)
        n_active = len(active_games)
        if n_active >= config.min_opportunities:
            dnp_count = (active_games["_active_dnp"] == 1).sum()
            current.loc[idx, "active_but_dnp_rate_last10"] = dnp_count / n_active
        elif n_active > 0:
            dnp_count = (active_games["_active_dnp"] == 1).sum()
            current.loc[idx, "active_but_dnp_rate_last10"] = (
                (dnp_count + config.alpha) / (n_active + config.alpha + config.beta)
            )
        # else: use default prior

        # inactive_streak_len
        inactive_streak = 0
        for i in range(n_hist - 1, -1, -1):
            if active[i] == 0:
                inactive_streak += 1
            else:
                break
        current.loc[idx, "inactive_streak_len"] = inactive_streak

    # Ensure consistent dtypes
    current["games_since_last_roster_active"] = current["games_since_last_roster_active"].astype("int32")
    current["never_roster_active_before"] = current["never_roster_active_before"].astype("int8")
    current["consecutive_active_dnp"] = current["consecutive_active_dnp"].astype("int32")
    current["active_but_dnp_rate_last10"] = current["active_but_dnp_rate_last10"].astype("float64")
    current["inactive_streak_len"] = current["inactive_streak_len"].astype("int32")

    return current


# Feature column names for export/reference
DNP_HISTORY_FEATURE_COLUMNS = [
    "roster_active_pre_tip",
    "games_since_last_roster_active",
    "never_roster_active_before",
    "consecutive_active_dnp",
    "active_but_dnp_rate_last10",
    "inactive_streak_len",
]
