"""CLI for scoring ownership predictions on live slates."""

from __future__ import annotations

import argparse
import re
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


def _normalize_name(value: str | None) -> str:
    """Normalize player name for matching: fold Unicode diacritics, lowercase, strip.
    
    Handles European characters like Dončić -> doncic, Jokić -> jokic, Matković -> matkovic.
    """
    if not value:
        return ""
    # Fold Unicode (e.g., Dončić -> Doncic) before stripping non-alphanumerics
    normalized = unicodedata.normalize("NFKD", value)
    ascii_folded = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", ascii_folded.lower())

from projections.ownership_v1.loader import load_ownership_bundle
from projections.ownership_v1.score import compute_ownership_features, predict_ownership
from projections.ownership_v1.schemas import (
    validate_raw_input,
    fill_optional_columns,
    prepare_model_input,
)
from projections.ownership_v1.calibration import (
    SoftmaxCalibrator,
    apply_calibration_with_mask,
    CalibrationParams,
)
from projections.paths import data_path


def _load_calibration_config() -> dict:
    """Load ownership calibration config from YAML."""
    config_path = Path(__file__).parent.parent.parent / "config" / "ownership_calibration.yaml"
    if not config_path.exists():
        return {"calibration": {"enabled": False}}
    
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_schedule_with_times(game_date: date, data_root: Path) -> pd.DataFrame:
    """Load schedule with game times for lock detection."""
    month = game_date.month
    year = game_date.year if game_date.month >= 10 else game_date.year  # Season year
    
    schedule_path = data_root / "silver" / "schedule" / f"season={year}" / f"month={month:02d}" / "schedule.parquet"
    if not schedule_path.exists():
        return pd.DataFrame()
    
    df = pd.read_parquet(schedule_path)
    df = df[df['game_date'] == str(game_date)]
    
    # Parse tip_local_ts to datetime
    if 'tip_local_ts' in df.columns:
        df['game_start'] = pd.to_datetime(df['tip_local_ts'])
    
    return df


def _get_locked_teams(schedule: pd.DataFrame, current_time: datetime) -> set:
    """Get set of team tricodes whose games have already started."""
    if schedule.empty or 'game_start' not in schedule.columns:
        return set()
    
    locked_teams = set()
    for _, row in schedule.iterrows():
        game_start = row['game_start']
        if pd.isna(game_start):
            continue
        # Make timezone-naive comparison
        if hasattr(game_start, 'tzinfo') and game_start.tzinfo is not None:
            game_start = game_start.replace(tzinfo=None)
        if current_time >= game_start:
            locked_teams.add(row['home_team_tricode'])
            locked_teams.add(row['away_team_tricode'])
    
    return locked_teams


def _load_locked_predictions(game_date: date, data_root: Path) -> Optional[pd.DataFrame]:
    """Load previously locked ownership predictions."""
    locked_path = data_root / "silver" / "ownership_predictions" / f"{game_date}_locked.parquet"
    if locked_path.exists():
        return pd.read_parquet(locked_path)
    return None


def _save_locked_predictions(df: pd.DataFrame, game_date: date, data_root: Path) -> None:
    """Save predictions to locked file (only if no locked file exists yet)."""
    locked_path = data_root / "silver" / "ownership_predictions" / f"{game_date}_locked.parquet"
    if not locked_path.exists():
        df.to_parquet(locked_path)
        print(f"[ownership] Saved locked predictions: {len(df)} players")


PRODUCTION_MODEL_RUN = "poc_013_chalk_5x"


def _load_dk_salaries(
    game_date: date,
    data_root: Path,
) -> Optional[pd.DataFrame]:
    """
    Load DK salaries from gold layer.
    
    Discovers draft groups for the date and loads the main slate (largest one).
    """
    # Gold path: gold/dk_salaries/site=dk/game_date=YYYY-MM-DD/draft_group_id=<id>/salaries.parquet
    base = data_root / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}"
    
    if not base.exists():
        print(f"[ownership] No salary data at {base}")
        return None
    
    # Find all draft groups for this date
    draft_group_dirs = sorted(base.glob("draft_group_id=*"))
    if not draft_group_dirs:
        print(f"[ownership] No draft groups found for {game_date}")
        return None
    
    # Load all and pick the largest (main slate)
    all_salaries = []
    for dg_dir in draft_group_dirs:
        parquet_path = dg_dir / "salaries.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            df["draft_group_id"] = dg_dir.name.split("=")[1]
            all_salaries.append(df)
    
    if not all_salaries:
        print(f"[ownership] No salary parquet files found for {game_date}")
        return None
    
    # Concatenate all slates
    combined = pd.concat(all_salaries, ignore_index=True)
    
    # For ownership predictions, we typically want the main slate
    # Pick the draft_group with most players
    main_dg = combined.groupby("draft_group_id").size().idxmax()
    main_slate = combined[combined["draft_group_id"] == main_dg].copy()
    
    print(f"[ownership] Loaded {len(main_slate)} players from main slate (dg={main_dg})")
    
    # Normalize column names for ownership model
    if "display_name" in main_slate.columns and "player_name" not in main_slate.columns:
        main_slate["player_name"] = main_slate["display_name"]
    if "positions" in main_slate.columns and "pos" not in main_slate.columns:
        main_slate["pos"] = main_slate["positions"].apply(lambda x: "/".join(x) if isinstance(x, list) else str(x))
    if "team_abbrev" in main_slate.columns and "team" not in main_slate.columns:
        main_slate["team"] = main_slate["team_abbrev"]
    if "dk_player_id" in main_slate.columns and "player_id" not in main_slate.columns:
        main_slate["player_id"] = main_slate["dk_player_id"]
    
    # Ensure salary is numeric (some draft groups may have object dtype from upstream)
    if "salary" in main_slate.columns:
        main_slate["salary"] = pd.to_numeric(main_slate["salary"], errors="coerce").fillna(0).astype(int)
    
    return main_slate


def _load_fpts_predictions(
    game_date: date,
    run_id: str,
    data_root: Path,
) -> Optional[pd.DataFrame]:
    """
    Load FPTS predictions from sim_v2 worlds output.
    
    The sim projections contain dk_fpts_mean from Monte Carlo worlds.
    """
    # Sim projections: artifacts/sim_v2/projections/game_date=YYYY-MM-DD/projections.parquet
    sim_path = (
        data_root / "artifacts" / "sim_v2" / "projections" 
        / f"game_date={game_date}" / "projections.parquet"
    )
    
    if not sim_path.exists():
        print(f"[ownership] No sim projections at {sim_path}")
        return None
    
    print(f"[ownership] Loading FPTS from sim_v2 worlds: {sim_path}")
    df = pd.read_parquet(sim_path)
    
    # Use dk_fpts_mean from worlds
    if "dk_fpts_mean" not in df.columns:
        print(f"[ownership] sim projections missing dk_fpts_mean column")
        return None
    
    print(f"[ownership] Loaded {len(df)} players from sim projections")
    
    # Return player_id and dk_fpts_mean
    # Note: sim uses NBA player_id, we'll need to map to DK later
    return df[["player_id", "dk_fpts_mean"]].rename(columns={"dk_fpts_mean": "pred_fpts"})


def _load_injuries(
    game_date: date,
    data_root: Path,
) -> pd.DataFrame:
    """Load injury data for the date."""
    season = game_date.year if game_date.month >= 10 else game_date.year
    month = game_date.month
    
    inj_path = (
        data_root / "silver" / "injuries_snapshot"
        / f"season={season}" / f"month={month:02d}" / "injuries_snapshot.parquet"
    )
    
    if inj_path.exists():
        df = pd.read_parquet(inj_path)
        # Filter to game date and latest snapshot
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        df = df[df["game_date"] == game_date]
        if not df.empty:
            return df
    
    return pd.DataFrame()


def score_ownership(
    game_date: date,
    run_id: str,
    data_root: Path,
    model_run: str = PRODUCTION_MODEL_RUN,
) -> Optional[pd.DataFrame]:
    """
    Score ownership predictions for a live slate.
    
    Returns DataFrame with player ownership predictions or None if data unavailable.
    """
    # Load model bundle
    try:
        bundle = load_ownership_bundle(model_run)
    except FileNotFoundError as e:
        print(f"[ownership] Model not found: {e}")
        return None
    
    # Load DK salaries
    salaries = _load_dk_salaries(game_date, data_root)
    if salaries is None or salaries.empty:
        print(f"[ownership] No salary data for {game_date}")
        return None
    
    # Load FPTS predictions from sim
    fpts = _load_fpts_predictions(game_date, run_id, data_root)
    
    if fpts is None or fpts.empty:
        print(f"[ownership] No FPTS predictions for {game_date}, using salary-based estimate")
        # Use salary as proxy for FPTS if no predictions available
        salaries["proj_fpts"] = salaries["salary"] / 200.0  # Rough conversion
    else:
        # Load minutes to get player_name -> NBA player_id mapping
        # This bridges DK's display_name to sim's player_id
        import json
        from pathlib import Path
        
        minutes_root = Path("/home/daniel/projects/projections-v2/artifacts/minutes_v1/daily") / str(game_date)
        latest_pointer = minutes_root / "latest_run.json"
        
        player_id_map = None
        if latest_pointer.exists():
            try:
                with open(latest_pointer) as f:
                    latest_run = json.load(f).get("run_id")
                minutes_path = minutes_root / f"run={latest_run}" / "minutes.parquet"
                if minutes_path.exists():
                    # Load player_id, player_name, and status (for OUT filtering)
                    cols_to_load = ["player_id", "player_name"]
                    minutes_df = pd.read_parquet(minutes_path)
                    # Add status column if it exists (for OUT filtering)
                    if "status" in minutes_df.columns:
                        cols_to_load.append("status")
                    minutes_df = minutes_df[[c for c in cols_to_load if c in minutes_df.columns]].copy()
                    
                    # Create name -> player_id mapping using normalized names
                    # This handles European characters like Dončić -> doncic
                    minutes_df["_name_norm"] = minutes_df["player_name"].apply(_normalize_name)
                    player_id_map = minutes_df.drop_duplicates("_name_norm").set_index("_name_norm")["player_id"]
                    
                    # Also create player_id -> status map for OUT filtering
                    status_map = None
                    if "status" in minutes_df.columns:
                        status_map = minutes_df.drop_duplicates("player_id").set_index("player_id")["status"]
                    print(f"[ownership] Loaded {len(player_id_map)} player mappings from minutes")
            except Exception as e:
                print(f"[ownership] Failed to load minutes for mapping: {e}")
        
        if player_id_map is not None:
            # Map DK display_name -> NBA player_id using normalized names
            salaries["_name_norm"] = salaries["player_name"].apply(_normalize_name)
            salaries["nba_player_id"] = salaries["_name_norm"].map(player_id_map)
            
            # Now join with sim FPTS on NBA player_id
            salaries = salaries.merge(
                fpts.rename(columns={"pred_fpts": "proj_fpts"}),
                left_on="nba_player_id",
                right_on="player_id",
                how="left",
                suffixes=("", "_sim")
            )
            salaries = salaries.drop(columns=["_name_norm", "nba_player_id"], errors="ignore")
            
            # Add injury status from minutes predictions for OUT filtering
            if status_map is not None:
                # Map player_id back to nba_player_id we got from name matching
                salaries["_temp_pid"] = salaries["player_name"].apply(_normalize_name).map(player_id_map)
                salaries["_injury_status"] = salaries["_temp_pid"].map(status_map)
                salaries = salaries.drop(columns=["_temp_pid"], errors="ignore")
            if "player_id_sim" in salaries.columns:
                salaries = salaries.drop(columns=["player_id_sim"])
            
            matched = salaries["proj_fpts"].notna().sum()
            print(f"[ownership] Matched {matched}/{len(salaries)} players via name→id mapping")
        else:
            # Fallback: use salary-based estimate
            print(f"[ownership] No player_id mapping available, using salary-based estimate")
            salaries["proj_fpts"] = salaries["salary"] / 200.0
    
    # Fill missing FPTS
    salaries["proj_fpts"] = salaries["proj_fpts"].fillna(salaries["salary"] / 200.0)
    
    # Filter OUT players BEFORE making predictions
    # Players who are OUT won't be in DK contests, so shouldn't receive a prediction
    if "_injury_status" in salaries.columns:
        out_mask = salaries["_injury_status"].str.upper() == "OUT"
        out_count = out_mask.sum()
        if out_count > 0:
            out_names = salaries.loc[out_mask, "player_name"].tolist()
            print(f"[ownership] Filtering {out_count} OUT players: {out_names[:5]}{'...' if out_count > 5 else ''}")
            salaries = salaries[~out_mask].copy()
        # Clean up the temp column
        salaries = salaries.drop(columns=["_injury_status"], errors="ignore")
    
    # Validate raw input
    missing = validate_raw_input(salaries)
    if missing:
        print(f"[ownership] Missing required columns: {missing}")
        return None
    
    # Fill optional enrichment columns (injuries, etc.)
    salaries = fill_optional_columns(salaries)
    
    # Compute features
    features = compute_ownership_features(
        salaries,
        proj_fpts_col="proj_fpts",
        salary_col="salary",
        pos_col="pos",
        slate_id_col=None,  # Treat as single slate
    )
    
    # Add slate-level features
    features["slate_size"] = len(features)
    features["salary_pct_of_max"] = features["salary"] / features["salary"].max()
    features["is_min_salary"] = (features["salary"] == features["salary"].min()).astype(int)
    min_salary = features["salary"].min()
    features["slate_near_min_count"] = (features["salary"] <= min_salary + 200).sum()
    
    # Prepare model input (strict feature selection)
    try:
        X = prepare_model_input(features, bundle.feature_cols)
    except KeyError as e:
        print(f"[ownership] Feature mismatch: {e}")
        return None
    
    # Predict
    predictions = predict_ownership(features, bundle)
    
    # Build output DataFrame
    output_cols = ["player_id", "player_name", "salary", "pos", "team"]
    output = salaries[[c for c in output_cols if c in salaries.columns]].copy()
    output["proj_fpts"] = features["proj_fpts"]
    output["pred_own_pct"] = predictions.values
    output["game_date"] = game_date
    output["run_id"] = run_id
    
    # Apply calibration if enabled
    config = _load_calibration_config()
    cal_cfg = config.get("calibration", {})
    
    if cal_cfg.get("enabled", False):
        print(f"[ownership] Applying calibration (sum before: {output['pred_own_pct'].sum():.1f}%)")
        
        try:
            # Load calibrator
            calibrator_path = data_path() / cal_cfg.get("calibrator_path", "artifacts/ownership_v1/calibrator.json")
            if not calibrator_path.exists():
                print(f"[ownership] Calibrator not found at {calibrator_path}, skipping")
            else:
                calibrator = SoftmaxCalibrator.load(calibrator_path)
                
                # Build structural zero mask
                # True = include in calibration, False = structural zero (set to 0)
                struct_cfg = cal_cfg.get("structural_zeros", {})
                mask = pd.Series(True, index=output.index)
                
                # Exclude OUT players (already filtered earlier, but double-check)
                if struct_cfg.get("exclude_out", True) and "_injury_status" in salaries.columns:
                    mask &= (salaries["_injury_status"].str.upper() != "OUT")
                
                # Exclude zero-minute players - check if we have this info
                if struct_cfg.get("exclude_zero_minutes", True) and "proj_minutes" in output.columns:
                    mask &= (output["proj_minutes"] > 0)
                
                # Exclude zero prediction (optional - default False)
                if struct_cfg.get("exclude_zero_prediction", False):
                    mask &= (output["pred_own_pct"] > 0)
                
                # Apply calibration with mask
                scores = output["pred_own_pct"].values
                calibrated = apply_calibration_with_mask(
                    scores, 
                    mask.values, 
                    calibrator.params
                )
                
                # Store both raw and calibrated
                output["pred_own_pct_raw"] = output["pred_own_pct"]
                output["pred_own_pct"] = calibrated * 100.0  # Convert to percent
                
                # Log metrics
                log_cfg = config.get("logging", {})
                if log_cfg.get("log_metrics", True):
                    n_zeros = (~mask).sum()
                    print(f"[ownership] Calibration: {n_zeros} structural zeros, "
                          f"sum after: {output['pred_own_pct'].sum():.1f}%")
                
        except Exception as e:
            print(f"[ownership] Calibration failed: {e}, using raw predictions")
    
    # Apply playable filter: zero out unplayable players
    play_cfg = config.get("playable_filter", {})
    if play_cfg.get("enabled", False):
        min_fpts = play_cfg.get("min_proj_fpts", 8.0)
        
        # Optional slate-aware threshold
        if play_cfg.get("slate_aware", False):
            baseline = play_cfg.get("baseline_slate_size", 80)
            scale = play_cfg.get("scale_per_player", 0.05)
            slate_size = len(output)
            min_fpts = min_fpts + max(0, (slate_size - baseline) * scale)
        
        # Apply filter using proj_fpts from features
        unplayable_mask = output["proj_fpts"] < min_fpts
        n_filtered = unplayable_mask.sum()
        
        if n_filtered > 0:
            output.loc[unplayable_mask, "pred_own_pct"] = 0.0
    
    # Lock persistence: stop re-scoring after first game starts
    lock_cfg = config.get("lock_persistence", {})
    if lock_cfg.get("enabled", True):  # Default to enabled
        current_time = datetime.now()
        
        # Load schedule to detect locked games
        schedule = _load_schedule_with_times(game_date, data_root)
        locked_teams = _get_locked_teams(schedule, current_time)
        
        # Load previously locked predictions (if exists)
        locked_preds = _load_locked_predictions(game_date, data_root)
        
        # First run (no games locked yet): save predictions for later use
        if locked_preds is None and not locked_teams:
            output["is_locked"] = False
            _save_locked_predictions(output, game_date, data_root)
            return output
        
        # If ANY game has started, stop re-scoring and return saved predictions
        if locked_teams:
            print(f"[ownership] Locked teams (game started): {sorted(locked_teams)}")
            
            if locked_preds is not None and not locked_preds.empty:
                # Return saved predictions - no more re-scoring
                locked_preds["is_locked"] = True
                print(f"[ownership] Returning {len(locked_preds)} saved predictions (re-scoring stopped)")
                return locked_preds
            else:
                # Edge case: games locked but no saved file
                # Save current predictions and mark as locked
                print(f"[ownership] WARNING: Games locked but no saved file. Saving current predictions.")
                output["is_locked"] = True
                _save_locked_predictions(output, game_date, data_root)
                return output
        
        # No games locked yet but we have a saved file - continue with fresh predictions
        output["is_locked"] = False
    else:
        output["is_locked"] = False
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Score ownership predictions")
    parser.add_argument("--date", required=True, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--data-root", default=None, help="Data root path")
    parser.add_argument("--model-run", default=PRODUCTION_MODEL_RUN, help="Model run ID")
    args = parser.parse_args()
    
    game_date = date.fromisoformat(args.date)
    root = Path(args.data_root) if args.data_root else data_path()
    
    result = score_ownership(game_date, args.run_id, root, args.model_run)
    
    if result is None:
        print("[ownership] No predictions generated")
        return 1
    
    # Save predictions
    out_dir = root / "silver" / "ownership_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{game_date}.parquet"
    result.to_parquet(out_path)
    
    print(f"[ownership] Saved {len(result)} predictions to {out_path}")
    print(f"[ownership] Top 5 by ownership:")
    print(result.nlargest(5, "pred_own_pct")[["player_name", "salary", "pred_own_pct"]].to_string())
    
    return 0


if __name__ == "__main__":
    exit(main())
