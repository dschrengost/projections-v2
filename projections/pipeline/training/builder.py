"""Core builder logic for the training dataset."""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
from typing import Optional

from projections.pipeline.training import loaders

logger = logging.getLogger(__name__)

def build_features(
    game_date: str,
    target_as_of_ts: Optional[datetime],
    season: str
) -> pd.DataFrame:
    """
    Build the training feature set for a single game_date.
    
    Args:
        game_date: YYYY-MM-DD
        target_as_of_ts: The point-in-time snapshot to simulate. 
                         If None, calculate "lock" time (approx 30m before tip).
        season: Season label (e.g., 2025).
        
    Returns:
        DataFrame joined and validated.
    """
    # 1. Load Spine (Roster + Schedule)
    # We load roster for the specific date
    roster = loaders.load_roster(game_date, game_date, season)
    if roster.empty:
        logger.warning(f"No roster found for {game_date}")
        return pd.DataFrame()
        
    schedule = loaders.load_schedule(game_date, game_date, season)
    if schedule.empty:
        logger.warning(f"No schedule found for {game_date}")
        return pd.DataFrame()

    # Join Schedule to Spine (game_id)
    # Roster has game_id, Schedule has game_id + tip_ts
    spine = pd.merge(
        roster,
        schedule[["game_id", "tip_ts", "home_team_id", "away_team_id", "home_team_tricode", "away_team_tricode"]],
        on="game_id",
        how="left"
    )
    
    # Enrich simple features
    spine["is_home"] = spine["team_id"] == spine["home_team_id"]
    spine["opponent_team_id"] = np.where(
        spine["is_home"], 
        spine["away_team_id"], 
        spine["home_team_id"]
    )
    spine["team_tricode"] = np.where(
        spine["is_home"],
        spine["home_team_tricode"],
        spine["away_team_tricode"]
    )
    
    # Ensure merge keys are numpy int64 (not nullable Int64) for merge_asof compatibility
    spine["game_id"] = spine["game_id"].astype("int64")
    spine["player_id"] = spine["player_id"].astype("int64")
    spine["team_id"] = spine["team_id"].astype("int64")
    
    # 2. Determine "As Of" Time (Per Game)
    # If target_as_of_ts is global (e.g. daily backfill), used as is?
    # Or 'lock' logic: for each game, 30m before tip.
    # User requirement: "Partitioned by asof". This implies the dataset is usually uniform in asof logic.
    # But different games tip at different times. 
    # If using "Lock" mode, we synthesize a per-row or per-game as-of.
    # The output schema has `feature_as_of_ts`.
    
    if target_as_of_ts is None:
        # "Lock" logic: 30 mins before tip_ts
        spine["feature_as_of_ts"] = spine["tip_ts"] - pd.Timedelta(minutes=30)
    else:
        # Explicit overrides
        spine["feature_as_of_ts"] = pd.Timestamp(target_as_of_ts).tz_localize("UTC" if pd.Timestamp(target_as_of_ts).tzinfo is None else None)

    # 3. ASOF Joins: Odds & Injuries
    # Load ALL available history for date range up to game_date + 1 (buffer)
    # Correctness: we rely on pd.merge_asof.
    # We must join on [game_id] + time for Odds.
    # We must join on [player_id] + time for Injuries.
    
    # Odds
    odds = loaders.load_odds(game_date, game_date, season)
    if not odds.empty:
        # Ensure ns precision
        odds["as_of_ts"] = odds["as_of_ts"].astype("datetime64[ns, UTC]")
        spine["feature_as_of_ts"] = spine["feature_as_of_ts"].astype("datetime64[ns, UTC]")
        
        odds = odds.sort_values("as_of_ts")
        spine = spine.sort_values("feature_as_of_ts") # spine must be sorted by on key
        
        spine = pd.merge_asof(
            spine,
            odds[["game_id", "as_of_ts", "spread_home", "total"]], # Add book preference logic if needed?
            left_on="feature_as_of_ts",
            right_on="as_of_ts",
            by="game_id",
            direction="backward",
            allow_exact_matches=True
        )
        # Drop the join timestamp if present
        if "as_of_ts" in spine.columns:
            spine = spine.drop(columns=["as_of_ts"])
        
    # Injuries
    injuries = loaders.load_injuries(game_date, game_date, season)
    if not injuries.empty:
        # Ensure ns precision for merge compatibility
        injuries["as_of_ts"] = injuries["as_of_ts"].astype("datetime64[ns, UTC]")
        spine["feature_as_of_ts"] = spine["feature_as_of_ts"].astype("datetime64[ns, UTC]")
        
        injuries = injuries.sort_values("as_of_ts")
        # DEBUG
        logger.info(f"Injuries nulls: {injuries[['player_id', 'as_of_ts']].isnull().sum()}")
        logger.info(f"Injuries dtypes: {injuries.dtypes}")
        
        spine = spine.sort_values("feature_as_of_ts")
        
        spine = pd.merge_asof(
            spine,
            injuries[["player_id", "as_of_ts", "status"]],
            left_on="feature_as_of_ts",
            right_on="as_of_ts",
            by="player_id",
            direction="backward",
            allow_exact_matches=True
        )
        spine = spine.rename(columns={"status": "injury_status"})
        if "as_of_ts" in spine.columns:
            spine = spine.drop(columns=["as_of_ts"])

    # 4. Salaries (Removed)
    # DraftKings salaries caused row duplication due to multi-slate overlaps.
    # We have removed them from the minutes model training data.
    # See task history for context.

        
    # 5. Labels (Boxscores) - STRICTLY LEAK SAFE
    # Labels are post-game.
    labels = loaders.load_labels_boxscores(game_date, game_date, season)
    if not labels.empty:
        spine = pd.merge(
            spine,
            labels[["game_id", "player_id", "minutes", "starter_flag"]],
            on=["game_id", "player_id"],
            how="left",
            suffixes=("", "_actual")
        )
        # Handle nullable comparison safely and ensure float
        spine["minutes"] = pd.to_numeric(spine["minutes"], errors="coerce").fillna(0.0)
        spine["played_flag"] = (spine["minutes"] > 0).astype(int)
        
    # 6. Detailed Stat Labels (FPTS Base)
    stats = loaders.load_labels_fpts(game_date, game_date, season)
    if not stats.empty:
        # Join on game_id, player_id
        # stats df has: game_id, player_id, pts, reb...
        
        stat_cols = ["game_id", "player_id", "pts", "reb", "ast", "stl", "blk", "tov", "fg3m", "dk_fpts_actual"]
        # Ensure cols exist
        available_cols = [c for c in stat_cols if c in stats.columns]
        
        spine = pd.merge(
            spine,
            stats[available_cols],
            on=["game_id", "player_id"],
            how="left",
            suffixes=("", "_fpts") # shouldn't clash
        )
        
    # 7. Fail-fast uniqueness assertion on primary key
    spine = assert_unique_primary_key(spine, game_date=game_date)
    
    # Final Schema shaping
    return spine


def assert_unique_primary_key(
    df: pd.DataFrame, 
    game_date: str = "",
    allow_dedup: bool = False,
) -> pd.DataFrame:
    """Assert (game_id, team_id, player_id) uniqueness.
    
    This is the last line of defense against join explosions or upstream 
    data quality issues. If duplicates are found:
    - Default: raise ValueError with compact debug dump (top offenders)
    - With allow_dedup=True: log warning and deduplicate deterministically
    
    Args:
        df: DataFrame to check
        game_date: For error context
        allow_dedup: If True, repair by keeping last occurrence (must be explicit)
        
    Returns:
        Original df if unique, or deduped df if allow_dedup=True
        
    Raises:
        ValueError: If duplicates found and allow_dedup=False
    """
    pk_cols = ["game_id", "team_id", "player_id"]
    
    # Check for required columns
    missing_pk = [c for c in pk_cols if c not in df.columns]
    if missing_pk:
        logger.warning(f"Cannot check PK uniqueness, missing columns: {missing_pk}")
        return df
        
    # Find duplicates
    dup_mask = df.duplicated(subset=pk_cols, keep=False)
    n_duplicates = dup_mask.sum()
    
    if n_duplicates == 0:
        return df
        
    # Build compact debug dump
    duplicates_df = df[dup_mask]
    dup_summary = (
        duplicates_df
        .groupby(pk_cols)
        .size()
        .reset_index(name="row_count")
        .sort_values("row_count", ascending=False)
        .head(10)
    )
    n_unique_keys = len(dup_summary)
    
    if allow_dedup:
        logger.warning(
            f"[{game_date}] Duplicate primary keys detected! "
            f"{n_duplicates} rows across {n_unique_keys} keys. "
            f"Deduplicating (keeping last)..."
        )
        logger.warning(f"Top offending groups:\n{dup_summary.to_string()}")
        
        # Deterministic dedup: keep last (consistent with lined_timestamp sort in roster)
        return df.drop_duplicates(subset=pk_cols, keep="last").reset_index(drop=True)
    else:
        error_msg = (
            f"[{game_date}] Duplicate primary keys detected! "
            f"{n_duplicates} rows across {n_unique_keys} unique (game_id, team_id, player_id) keys.\n"
            f"INVARIANT VIOLATION: The training data contract requires unique primary keys.\n"
            f"This is likely caused by:\n"
            f"  1. Multi-slate salary joins (should be removed)\n"
            f"  2. Upstream data quality issues in roster/labels\n"
            f"  3. Incorrect join logic\n"
            f"\nTop offending groups (showing up to 10):\n{dup_summary.to_string()}\n"
            f"\nTo investigate: re-run with --allow-dedup flag and check dedup_report.json"
        )
        raise ValueError(error_msg)
