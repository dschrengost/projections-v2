"""
Build ownership training base from Linestar scraped CSV files.

Combines projected and actual ownership data from:
    scrapers/linestar/backfill/season_{year}/dk/pid_{pid}_proj.csv
    scrapers/linestar/backfill/season_{year}/dk/pid_{pid}_actual.csv

Joins with injury data for context features.

Output:
    gold/ownership_training_base/ownership_training_base.parquet
"""

from __future__ import annotations

import argparse
import glob
import re
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from projections.paths import data_path


def parse_game_ticks(val: object) -> datetime | None:
    """Parse LineStar game_ticks field to datetime.
    
    Handles formats like:
    - /Date(1761078600000-0400)/  (milliseconds since epoch with timezone)
    - Unix timestamp (int)
    """
    if pd.isna(val):
        return None
    
    s = str(val)
    
    # /Date(1234567890000-0400)/ format
    m = re.search(r"/Date\((\d+)", s)
    if m:
        ms = int(m.group(1))
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    
    # Plain unix timestamp
    try:
        ts = int(float(s))
        if ts > 1e12:  # milliseconds
            return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        elif ts > 1e9:  # seconds
            return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (ValueError, TypeError):
        pass
    
    return None


def normalize_name(val: object) -> str:
    """Normalize player name for matching: strip accents, lowercase, trim."""
    if val is None or pd.isna(val):
        return ""
    normalized = unicodedata.normalize("NFKD", str(val))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.strip().lower()


def load_injuries(injuries_root: Path) -> pd.DataFrame:
    """Load all injury data from bronze layer."""
    inj_files = glob.glob(str(injuries_root / "season=*" / "date=*" / "injuries.parquet"))
    if not inj_files:
        print(f"  No injury files found in {injuries_root}")
        return pd.DataFrame()
    
    print(f"  Loading {len(inj_files)} injury files...")
    all_inj = []
    for f in inj_files:
        try:
            df = pd.read_parquet(f)
            all_inj.append(df)
        except Exception as e:
            print(f"    Warning: failed to read {f}: {e}")
    
    if not all_inj:
        return pd.DataFrame()
    
    inj = pd.concat(all_inj, ignore_index=True)
    
    # Normalize date to string, handling NaN
    inj["report_date_str"] = pd.to_datetime(inj["report_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    inj = inj.dropna(subset=["report_date_str"])
    
    # Normalize player_id to string
    inj["player_id"] = inj["player_id"].astype(str)
    
    print(f"  Loaded {len(inj):,} injury rows, date range: {inj['report_date_str'].min()} to {inj['report_date_str'].max()}")
    
    return inj


def compute_injury_features(merged: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Add injury-derived features to the merged ownership data."""
    if injuries.empty:
        print("  No injury data available, skipping injury features")
        merged["player_is_out"] = 0
        merged["player_is_questionable"] = 0
        merged["team_outs_count"] = 0
        return merged
    
    result = merged.copy()
    
    print(f"  Matching injuries to ownership data by name...")
    
    # Normalize injury names from "Last, First" to "first last"
    def normalize_injury_name(name):
        if pd.isna(name):
            return ""
        name = str(name).strip()
        if "," in name:
            parts = name.split(",", 1)
            if len(parts) == 2:
                name = f"{parts[1].strip()} {parts[0].strip()}"
        return normalize_name(name)
    
    injuries = injuries.copy()
    injuries["player_name_norm"] = injuries["player_name"].apply(normalize_injury_name)
    
    # Group injuries by (date, normalized_name) -> status
    player_status = injuries.groupby(["report_date_str", "player_name_norm"]).agg({
        "status": "first"  # Take first status if duplicates
    }).reset_index()
    
    # Create lookup key using normalized name
    result["_lookup_key"] = result["game_date"] + "_" + result["player_name_norm"]
    player_status["_lookup_key"] = player_status["report_date_str"] + "_" + player_status["player_name_norm"]
    
    status_map = dict(zip(player_status["_lookup_key"], player_status["status"]))
    result["_player_status"] = result["_lookup_key"].map(status_map)
    
    # Binary flags
    result["player_is_out"] = (result["_player_status"] == "OUT").astype(int)
    result["player_is_questionable"] = result["_player_status"].isin(["Q", "PROB", "GTD"]).astype(int)
    
    # Team-level: count OUTs per (game_date, team) from the merged data itself
    result["team_outs_count"] = result.groupby(["game_date", "team"])["player_is_out"].transform("sum")
    
    # Clean up temp columns
    result.drop(columns=["_lookup_key", "_player_status"], inplace=True, errors="ignore")
    
    # Stats
    print(f"  Players matched to injury data: {result['_player_status'].notna().sum() if '_player_status' in result.columns else (result['player_is_out'] > 0).sum() + (result['player_is_questionable'] > 0).sum():,}")
    print(f"  Players marked OUT: {result['player_is_out'].sum():,}")
    print(f"  Players marked Q/PROB/GTD: {result['player_is_questionable'].sum():,}")
    print(f"  Mean team OUTs per slate: {result.groupby('slate_id')['team_outs_count'].first().mean():.1f}")
    
    return result



def load_season_data(backfill_root: Path, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all proj and actual CSVs for a given season.
    
    Filters actual ownership to include ONLY GPP contests (excludes Cash/Showdown).
    """
    season_dir = backfill_root / f"season_{season}" / "dk"
    if not season_dir.exists():
        print(f"  Season {season} directory not found: {season_dir}")
        return pd.DataFrame(), pd.DataFrame()
    
    proj_files = sorted(season_dir.glob("pid_*_proj.csv"))
    actual_files = sorted(season_dir.glob("pid_*_actual.csv"))
    
    print(f"  Season {season}: {len(proj_files)} proj files, {len(actual_files)} actual files")
    
    proj_dfs = []
    for f in proj_files:
        try:
            df = pd.read_csv(f)
            proj_dfs.append(df)
        except Exception as e:
            print(f"    Warning: failed to read {f.name}: {e}")
    
    actual_dfs = []
    for f in actual_files:
        try:
            # Load actual ownership
            df_actual = pd.read_csv(f)
            
            # Load corresponding contests file
            # Filename format: pid_{slate_id}_actual.csv -> pid_{slate_id}_contests.csv
            contest_file = f.parent / f.name.replace("_actual.csv", "_contests.csv")
            
            if contest_file.exists():
                df_contests = pd.read_csv(contest_file)
                
                # Merge to get contest metadata
                # contest_id is the key
                if "contest_id" in df_actual.columns and "contest_id" in df_contests.columns:
                    merged = df_actual.merge(
                        df_contests[["contest_id", "contest_type", "contest_name"]],
                        on="contest_id",
                        how="left"
                    )
                    
                    # FILTERING LOGIC
                    # 1. Exclude Cash Games (Type 1 usually, sometimes 2/3 are also cash-like? 
                    #    Standard mapping: 1=H2H/5050/DoubleUp, 4=GPP/Tournament)
                    #    We will keep ONLY contest_type == 4 (Tournaments) or maybe others if we are sure.
                    #    Let's stick to excluding Type 1 (Cash) explicitly, or keeping Type 4.
                        # Ensure contest_id matching
                    df_actual["contest_id"] = df_actual["contest_id"].astype(str)
                    df_contests["contest_id"] = df_contests["contest_id"].astype(str)
                    
                    merged = df_actual.merge(df_contests, on="contest_id", how="left")
                    
                    if "entries" not in merged.columns:
                        print(f"    Error: 'entries' column missing in {f.name}. Columns: {merged.columns.tolist()}")
                        continue

                    # 2. Exclude Showdown / Tiers
                    #    Check contest_name for "Showdown" or "Tiers"
                    
                    # 3. Exclude Small Field Contests (e.g. < 100 entries)
                    #    To avoid 100% ownership outliers from small leagues
                    
                    is_gpp = (merged["contest_type"] == 4)
                    is_not_showdown = ~merged["contest_name"].str.contains("Showdown", case=False, na=False)
                    is_not_tiers = ~merged["contest_name"].str.contains("Tiers", case=False, na=False)
                    is_large_field = (merged["entries"] > 1000)
                    
                    filtered = merged[is_gpp & is_not_showdown & is_not_tiers & is_large_field].copy()
                    
                    if not filtered.empty:
                        # Keep metadata for debugging
                        cols_to_keep = [
                            "player_name", "team", "pos", "salary", "actual_own_pct", 
                            "game_id", "player_id", "salary_id", "contest_id", 
                            "entries", "contest_name", "slate_id", "game_date"
                        ]
                        # Only keep columns that exist
                        cols = [c for c in cols_to_keep if c in filtered.columns]
                        filtered = filtered[cols]
                        
                        # Add season here, as it's not part of the original actual_df schema
                        filtered["season"] = season
                        
                        actual_dfs.append(filtered)
                else:
                    # If no contest_id, can't filter, skip or warn? 
                    # For now, if we can't verify it's GPP, we might want to skip to be safe.
                    # But let's assume if contest file exists, we should use it.
                    pass
            else:
                # If no contest file, we can't filter. 
                # Decision: Skip these files to ensure purity? Or include?
                # Given the goal is "clean GPP data", skipping is safer.
                # print(f"    Warning: no contest file for {f.name}, skipping")
                pass

        except Exception as e:
            print(f"    Warning: failed to read {f.name}: {e}")
    
    proj_df = pd.concat(proj_dfs, ignore_index=True) if proj_dfs else pd.DataFrame()
    actual_df = pd.concat(actual_dfs, ignore_index=True) if actual_dfs else pd.DataFrame()
    
    if not actual_df.empty:
        print(f"    Filtered Actual Rows: {len(actual_df):,} (from {len(actual_files)} files)")
    
    return proj_df, actual_df


def build_training_base(
    backfill_root: Path,
    output_path: Path,
    injuries_root: Path | None = None,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Build unified ownership training base from Linestar data.
    
    Joins projected ownership with actual ownership on (slate_id, player_id).
    Optionally joins injury context features.
    """
    if seasons is None:
        # Auto-detect seasons
        seasons = []
        for d in sorted(backfill_root.glob("season_*")):
            try:
                year = int(d.name.split("_")[1])
                seasons.append(year)
            except (ValueError, IndexError):
                pass
    
    print(f"Loading seasons: {seasons}")
    
    all_proj = []
    all_actual = []
    
    for season in seasons:
        proj_df, actual_df = load_season_data(backfill_root, season)
        if not proj_df.empty:
            proj_df["season"] = season
            all_proj.append(proj_df)
        if not actual_df.empty:
            actual_df["season"] = season
            all_actual.append(actual_df)
    
    if not all_proj:
        raise ValueError("No projected ownership data found")
    
    proj = pd.concat(all_proj, ignore_index=True)
    actual = pd.concat(all_actual, ignore_index=True) if all_actual else pd.DataFrame()
    
    print(f"\nTotal: {len(proj):,} proj rows, {len(actual):,} actual rows")
    
    # Parse game_ticks to game_date
    proj["game_dt"] = proj["game_ticks"].apply(parse_game_ticks)
    proj["game_date"] = proj["game_dt"].apply(
        lambda x: x.strftime("%Y-%m-%d") if x else None
    )
    
    # Normalize slate_id and player_id for joining
    proj["slate_id"] = proj["slate_id"].astype(str)
    proj["player_id"] = proj["player_id"].astype(str)
    
    if not actual.empty:
        actual["slate_id"] = actual["slate_id"].astype(str)
        actual["player_id"] = actual["player_id"].astype(str)
        
        # Aggregate actual ownership by (slate_id, player_id) - may have multiple contests
        # We average ownership, and take representative metadata
        agg_dict = {"actual_own_pct": "mean"}
        if "entries" in actual.columns:
            agg_dict["entries"] = "mean"
        if "contest_name" in actual.columns:
            agg_dict["contest_name"] = "first"
            
        actual_agg = (
            actual.groupby(["slate_id", "player_id"])
            .agg(agg_dict)
            .reset_index()
        )
        print(f"Unique (slate, player) pairs in actual: {len(actual_agg):,}")
        
        # Join proj with actual
        merged = proj.merge(
            actual_agg,
            on=["slate_id", "player_id"],
            how="left",
        )
    else:
        merged = proj.copy()
        merged["actual_own_pct"] = None
    
    # Add normalized name for potential future matching
    merged["player_name_norm"] = merged["player_name"].apply(normalize_name)
    
    # Load and join injury features
    print("\n--- Injury Features ---")
    if injuries_root is None:
        injuries_root = data_path() / "bronze" / "injuries_raw"
    injuries = load_injuries(injuries_root)
    merged = compute_injury_features(merged, injuries)
    
    # Select and order columns
    output_cols = [
        # Identifiers
        "season",
        "slate_id",
        "game_date",
        "player_id",
        "player_name",
        "player_name_norm",
        "team",
        "pos",
        # Salaries
        "salary",
        # Projections
        "proj_fpts",
        "scored_fpts",
        "floor_fpts",
        "ceil_fpts",
        "conf",
        "value_per_k",
        "ppg",
        # Vegas context
        "matchup",
        "home_team",
        "away_team",
        "opp_rank",
        "opp_total",
        # Injury context
        "player_is_out",
        "player_is_questionable",
        "team_outs_count",
        # Ownership
        "proj_own_pct",
        "actual_own_pct",
        # Metadata
        "game_id",
        "salary_id",
        "entries",
        "contest_name",
    ]
    
    # Only include columns that exist
    output_cols = [c for c in output_cols if c in merged.columns]
    result = merged[output_cols].copy()
    
    # Filter out rows without valid ownership data
    has_ownership = result["actual_own_pct"].notna()
    print(f"\nRows with actual ownership: {has_ownership.sum():,} / {len(result):,}")
    
    # Keep only rows with actual ownership for training
    result = result[has_ownership].copy()
    
    # Filter out broken slates (e.g. max ownership > 98%)
    # This catches "Zone Special" or other weird contests with 100% ownerships
    slate_max_own = result.groupby("slate_id")["actual_own_pct"].max()
    broken_slates = slate_max_own[slate_max_own > 98.0].index.tolist()
    
    if broken_slates:
        print(f"\nDropping {len(broken_slates)} broken slates (max own > 98%): {broken_slates}")
        result = result[~result["slate_id"].isin(broken_slates)].copy()
    
    # Compute additional metrics for sanity checks
    print(f"\n--- Summary ---")
    print(f"Unique slates: {result['slate_id'].nunique():,}")
    print(f"Unique players: {result['player_id'].nunique():,}")
    print(f"Date range: {result['game_date'].min()} to {result['game_date'].max()}")
    print(f"Actual ownership: mean={result['actual_own_pct'].mean():.2f}%, "
          f"std={result['actual_own_pct'].std():.2f}%")
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"\nWrote {len(result):,} rows to {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Build ownership training base from Linestar data")
    parser.add_argument(
        "--backfill-root",
        type=Path,
        default=Path(__file__).parent.parent.parent / "scrapers" / "linestar" / "backfill",
        help="Path to Linestar backfill directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path (default: gold/ownership_training_base/ownership_training_base.parquet)",
    )
    parser.add_argument(
        "--injuries-root",
        type=Path,
        default=None,
        help="Path to bronze injuries_raw directory",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help="Specific seasons to include (e.g., 2023 2024 2025)",
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = data_path() / "gold" / "ownership_training_base" / "ownership_training_base.parquet"
    
    build_training_base(
        backfill_root=args.backfill_root,
        output_path=args.output,
        injuries_root=args.injuries_root,
        seasons=args.seasons,
    )


if __name__ == "__main__":
    main()
