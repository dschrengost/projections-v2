"""Data loaders for the training dataset pipeline using Pandas."""

import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import glob

from projections import paths

def load_schedule(
    start_date: str,
    end_date: str,
    season: str
) -> pd.DataFrame:
    """
    Load schedule data (Silver).
    Returns columns: [game_id, game_date, tip_ts, home_team_id, away_team_id, ...]
    """
    data_root = paths.get_data_root()
    pattern = str(data_root / "silver" / "schedule" / f"season={season}" / "month=*" / "schedule.parquet")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No schedule files found for season {season}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    
    # Filter by date range
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    
    mask = (df["game_date"] >= start_ts) & (df["game_date"] <= end_ts)
    return df.loc[mask].copy()

def load_roster(
    start_date: str,
    end_date: str,
    season: str
) -> pd.DataFrame:
    """
    Load roster nightly spine (Silver).
    Returns columns: [game_id, player_id, team_id, player_name, listed_pos, ...]
    """
    data_root = paths.get_data_root()
    # Roster is partitioned by season/month usually
    pattern = str(data_root / "silver" / "roster_nightly" / f"season={season}" / "month=*" / "roster.parquet")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No roster files found for season {season}")
        
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    
    # Filter by date
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    
    mask = (df["game_date"] >= start_ts) & (df["game_date"] <= end_ts)
    df = df.loc[mask].copy()

    # Deduplicate: Keep latest lineup_timestamp per game_id, player_id
    if "lineup_timestamp" in df.columns:
        df = df.sort_values("lineup_timestamp")
        df = df.drop_duplicates(subset=["game_id", "player_id"], keep="last")
        
    return df

def load_odds(
    start_date: str,
    end_date: str,
    season: str
) -> pd.DataFrame:
    """
    Load odds snapshots (Silver).
    Sorted by [game_id, as_of_ts] for ASOF joins.
    """
    data_root = paths.get_data_root()
    pattern = str(data_root / "silver" / "odds_snapshot" / f"season={season}" / "month=*" / "odds_snapshot.parquet")
    files = glob.glob(pattern)
    
    if not files:
        # Warning instead of error? Or just empty DF?
        return pd.DataFrame()
        
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    
    # Filter by date (approximate filter on ingestion or as_of, but rigorous join later)
    # Actually, odds snapshots map to games. But they are time-series.
    # We might need to keep more data to support proper as-of joins if tolerance is large?
    # For now, let's load all for season or filter loosely? 
    # Let's filter by as_of_ts range? 
    # Actually best to filter by join keys if possible, but here we just return the dataset sorted.
    
    start_ts = pd.to_datetime(start_date).tz_localize("UTC")
    # end_ts = pd.to_datetime(end_date).tz_localize("UTC") + pd.Timedelta(days=2) # Buffer
    
    # Ensure sorted for asof merge
    df = df.dropna(subset=["game_id"])
    df["game_id"] = df["game_id"].astype(int)
    df = df.sort_values("as_of_ts")
    return df

def load_injuries(
    start_date: str,
    end_date: str,
    season: str
) -> pd.DataFrame:
    """
    Load injury snapshots (Silver).
    Sorted by [as_of_ts] for ASOF joins.
    """
    data_root = paths.get_data_root()
    pattern = str(data_root / "silver" / "injuries_snapshot" / f"season={season}" / "month=*" / "injuries_snapshot.parquet")
    files = glob.glob(pattern)
    
    if not files:
        return pd.DataFrame()

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.dropna(subset=["player_id", "as_of_ts"])
    df["player_id"] = df["player_id"].astype(int)
    df = df.sort_values("as_of_ts")
    return df



def load_labels_boxscores(
    start_date: str,
    end_date: str,
    season: str
) -> pd.DataFrame:
    """
    Load boxscore labels (Gold/Labels).
    """
    data_root = paths.get_data_root()
    path = data_root / "labels" / f"season={season}" / "boxscore_labels.parquet"
    
    if not path.exists():
        raise FileNotFoundError(f"No labels found for season {season}")
        
    df = pd.read_parquet(path)
    
    # Parse minutes from ISO duration if needed
    if "minutes" in df.columns and pd.api.types.is_object_dtype(df["minutes"]):
        df["minutes"] = pd.to_timedelta(df["minutes"], errors="coerce").dt.total_seconds() / 60.0
    
    # Filter by date
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    
    mask = (df["game_date"] >= start_ts) & (df["game_date"] <= end_ts)
    df = df.loc[mask].copy()
    # NOTE: Uniqueness is now enforced in builder.assert_unique_primary_key()
    # Removing silent drop_duplicates to ensure join explosions are visible.
    return df

def load_labels_fpts(
    start_date: str,
    end_date: str,
    season: str = "2025"
) -> pd.DataFrame:
    """
    Load FPTS training base for detailed labels (Gold).
    """
    data_root = paths.get_data_root()
    base_dir = data_root / "gold" / "fpts_training_base" / f"season={season}"
    
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    
    frames = []
    
    current = start_ts
    while current <= end_ts:
        date_str = current.strftime("%Y-%m-%d")
        path = base_dir / f"game_date={date_str}" / "fpts_training_base.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
        current += pd.Timedelta(days=1)
            
    if not frames:
        return pd.DataFrame()
        
    df = pd.concat(frames, ignore_index=True)
    # NOTE: Uniqueness is now enforced in builder.assert_unique_primary_key()
    # Removing silent drop_duplicates to ensure join explosions are visible.
    return df
