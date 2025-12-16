"""CLI for finalizing unified projections artifact.

Merges minutes, sim outputs, and ownership predictions into a single
per-run parquet file with complete projection data.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from projections.paths import data_path


def _normalize_name(value: str | None) -> str:
    """Normalize player name for matching: fold Unicode diacritics, lowercase, strip.
    
    Handles European characters like Dončić -> doncic, Jokić -> jokic.
    """
    if not value:
        return ""
    # Fold Unicode (e.g., Dončić -> Doncic) before stripping non-alphanumerics
    normalized = unicodedata.normalize("NFKD", value)
    ascii_folded = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", ascii_folded.lower())


# Columns to include from each source
MINUTES_COLUMNS = [
    # Identity
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "team_tricode",
    "opponent_team_id",
    "opponent_team_name",
    "opponent_team_tricode",
    "game_id",
    "game_date",
    "tip_ts",
    # Status
    "starter_flag",
    "is_projected_starter",
    "is_confirmed_starter",
    "status",
    "play_prob",
    "pos_bucket",
    # Minutes projections
    "minutes_p10",
    "minutes_p50",
    "minutes_p90",
    "minutes_p10_cond",
    "minutes_p50_cond",
    "minutes_p90_cond",
    # Game context
    "spread_home",
    "total",
    "team_implied_total",
    "opponent_implied_total",
    "odds_as_of_ts",
]

SIM_COLUMNS = [
    # FPTS quantiles
    "dk_fpts_mean",
    "dk_fpts_std",
    "dk_fpts_p05",
    "dk_fpts_p10",
    "dk_fpts_p25",
    "dk_fpts_p50",
    "dk_fpts_p75",
    "dk_fpts_p90",
    "dk_fpts_p95",
    # Box score stats
    "pts_mean",
    "reb_mean",
    "ast_mean",
    "stl_mean",
    "blk_mean",
    "tov_mean",
    # Minutes simulation stats
    "minutes_mean",       # model p50 (reference)
    "minutes_sim_mean",   # cross-world average
    "minutes_sim_std",    # cross-world std
    "minutes_sim_p10",    # sim p10
    "minutes_sim_p50",    # sim p50
    "minutes_sim_p90",    # sim p90
    # Metadata
    "sim_profile",
    "n_worlds",
    "is_starter",  # sim's view of starter
]

OWNERSHIP_COLUMNS = [
    "pred_own_pct",
    "salary",
]


def _load_minutes(
    game_date: date,
    run_id: str,
) -> Optional[pd.DataFrame]:
    """Load minutes artifact for a specific run."""
    minutes_root = Path("/home/daniel/projects/projections-v2/artifacts/minutes_v1/daily")
    run_path = minutes_root / str(game_date) / f"run={run_id}" / "minutes.parquet"
    
    if not run_path.exists():
        print(f"[finalize] Minutes not found at {run_path}")
        return None
    
    df = pd.read_parquet(run_path)
    
    # Select available columns
    available = [c for c in MINUTES_COLUMNS if c in df.columns]
    return df[available].copy()


def _load_sim(
    game_date: date,
    data_root: Path,
) -> Optional[pd.DataFrame]:
    """Load sim projections."""
    sim_path = (
        data_root / "artifacts" / "sim_v2" / "projections"
        / f"game_date={game_date}" / "projections.parquet"
    )
    
    if not sim_path.exists():
        print(f"[finalize] Sim projections not found at {sim_path}")
        return None
    
    df = pd.read_parquet(sim_path)
    
    # Select available columns + player_id for join
    available = ["player_id"] + [c for c in SIM_COLUMNS if c in df.columns]
    return df[available].copy()


def _load_ownership(
    game_date: date,
    draft_group_id: str,
    data_root: Path,
) -> Optional[pd.DataFrame]:
    """Load ownership predictions for a specific slate."""
    # Try per-slate path first (new format)
    own_path = data_root / "silver" / "ownership_predictions" / str(game_date) / f"{draft_group_id}.parquet"
    
    if not own_path.exists():
        # Fall back to legacy format (single file)
        legacy_path = data_root / "silver" / "ownership_predictions" / f"{game_date}.parquet"
        if legacy_path.exists():
            print(f"[finalize] Using legacy ownership path: {legacy_path}")
            own_path = legacy_path
        else:
            print(f"[finalize] Ownership not found at {own_path}")
            return None
    
    df = pd.read_parquet(own_path)
    
    # Use player_name for joining (ownership uses DK player_id, not NBA)
    available = ["player_name"] + [c for c in OWNERSHIP_COLUMNS if c in df.columns]
    if "player_name" not in df.columns:
        return None
    return df[available].copy()


def _load_salaries(
    game_date: date,
    draft_group_id: str,
    data_root: Path,
) -> Optional[pd.DataFrame]:
    """Load DK salaries for a specific slate."""
    # Try specific draft group first
    salaries_path = data_root / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}" / f"draft_group_id={draft_group_id}" / "salaries.parquet"
    
    if salaries_path.exists():
        df = pd.read_parquet(salaries_path)
    else:
        # Fall back to finding any slate
        base = data_root / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}"
        if not base.exists():
            return None
        
        draft_group_dirs = sorted(base.glob("draft_group_id=*"))
        if not draft_group_dirs:
            return None
        
        # Use largest slate as fallback
        all_salaries = []
        for dg_dir in draft_group_dirs:
            parquet_path = dg_dir / "salaries.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                all_salaries.append(df)
        
        if not all_salaries:
            return None
        
        combined = pd.concat(all_salaries, ignore_index=True)
        main_dg = combined.groupby("draft_group_id").size().idxmax()
        df = combined[combined["draft_group_id"] == main_dg].copy()
        print(f"[finalize] Salary fallback: using slate {main_dg}")
    
    # Normalize columns
    if "display_name" in df.columns:
        df["player_name"] = df["display_name"]
    
    return df[["player_name", "salary"]].drop_duplicates("player_name")


def finalize_projections(
    game_date: date,
    run_id: str,
    draft_group_id: str,
    data_root: Path,
) -> Optional[Path]:
    """
    Merge minutes, sim, and ownership into unified projections artifact.
    
    Args:
        game_date: Game date
        run_id: Minutes run identifier
        draft_group_id: DraftKings draft group ID for ownership lookup
        data_root: Data root path
    
    Returns path to unified artifact or None if minutes unavailable.
    """
    # Load minutes (required - base of the join)
    minutes = _load_minutes(game_date, run_id)
    if minutes is None or minutes.empty:
        print(f"[finalize] No minutes for {game_date} run={run_id}")
        return None
    
    print(f"[finalize] Loaded {len(minutes)} players from minutes")
    unified = minutes.copy()
    
    # Load and merge sim projections
    sim = _load_sim(game_date, data_root)
    if sim is not None and not sim.empty:
        # Join on player_id
        unified = unified.merge(
            sim,
            on="player_id",
            how="left",
            suffixes=("", "_sim")
        )
        matched = unified["dk_fpts_mean"].notna().sum()
        print(f"[finalize] Merged {matched}/{len(unified)} sim projections")
    else:
        print("[finalize] No sim projections available")
    
    # Load ownership (join on player_name since DK uses different IDs)
    ownership = _load_ownership(game_date, draft_group_id, data_root)
    if ownership is not None and not ownership.empty:
        # Normalize names for matching (handles Unicode like Dončić -> doncic)
        unified["_name_norm"] = unified["player_name"].apply(_normalize_name)
        ownership["_name_norm"] = ownership["player_name"].apply(_normalize_name)
        
        ownership_cols = ["_name_norm"] + [c for c in OWNERSHIP_COLUMNS if c in ownership.columns]
        unified = unified.merge(
            ownership[ownership_cols],
            on="_name_norm",
            how="left",
            suffixes=("", "_own")
        )
        unified = unified.drop(columns=["_name_norm"])
        
        matched = unified["pred_own_pct"].notna().sum() if "pred_own_pct" in unified.columns else 0
        print(f"[finalize] Merged {matched}/{len(unified)} ownership predictions")
    else:
        print("[finalize] No ownership predictions available")
    
    # Load salaries if not already present
    if "salary" not in unified.columns:
        salaries = _load_salaries(game_date, draft_group_id, data_root)
        if salaries is not None:
            # Join on player_name (normalized handles Unicode like Dončić)
            unified["_name_norm"] = unified["player_name"].apply(_normalize_name)
            salaries["_name_norm"] = salaries["player_name"].apply(_normalize_name)
            unified = unified.merge(
                salaries[["_name_norm", "salary"]],
                on="_name_norm",
                how="left"
            )
            unified = unified.drop(columns=["_name_norm"])
            print(f"[finalize] Merged {unified['salary'].notna().sum()} salaries")
    
    # Compute value (FPTS per $1k salary)
    if "dk_fpts_mean" in unified.columns and "salary" in unified.columns:
        unified["value"] = (unified["dk_fpts_mean"] / unified["salary"] * 1000).round(2)
    
    # Compute is_locked based on whether tip_ts has passed
    if "tip_ts" in unified.columns:
        now = pd.Timestamp.now(tz="UTC")
        tip_ts = pd.to_datetime(unified["tip_ts"], utc=True, errors="coerce")
        unified["is_locked"] = tip_ts.notna() & (tip_ts <= now)
        locked_count = unified["is_locked"].sum()
        print(f"[finalize] Marked {locked_count}/{len(unified)} players as locked (tip_ts <= {now.isoformat()})")
    else:
        unified["is_locked"] = False
    
    # Write unified artifact
    out_dir = data_root / "artifacts" / "projections" / str(game_date) / f"run={run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "projections.parquet"
    unified.to_parquet(out_path, index=False)
    
    # Update latest_run.json
    latest_pointer = out_dir.parent / "latest_run.json"
    with open(latest_pointer, "w") as f:
        json.dump({"run_id": run_id}, f)
    
    print(f"[finalize] Saved unified projections ({len(unified)} players) to {out_path}")
    
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Finalize unified projections artifact")
    parser.add_argument("--date", required=True, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--draft-group-id", required=True, help="DraftKings draft group ID")
    parser.add_argument("--data-root", default=None, help="Data root path")
    args = parser.parse_args()
    
    game_date = date.fromisoformat(args.date)
    root = Path(args.data_root) if args.data_root else data_path()
    
    result = finalize_projections(game_date, args.run_id, args.draft_group_id, root)
    
    if result is None:
        print("[finalize] Failed to create unified projections")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
