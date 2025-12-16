"""CLI for scoring ownership predictions on live slates."""

from __future__ import annotations

import argparse
from functools import lru_cache
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from projections.ownership_v1.calibration import (
    SoftmaxCalibrator,
    apply_calibration_with_mask,
)
from projections.ownership_v1.loader import load_ownership_bundle
from projections.ownership_v1.schemas import (
    fill_optional_columns,
    prepare_model_input,
    validate_raw_input,
)
from projections.ownership_v1.score import compute_ownership_features, predict_ownership
from projections.paths import data_path


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


def _load_locked_predictions(game_date: date, draft_group_id: str, data_root: Path) -> Optional[pd.DataFrame]:
    """Load previously locked ownership predictions for a specific slate."""
    locked_path = data_root / "silver" / "ownership_predictions" / str(game_date) / f"{draft_group_id}_locked.parquet"
    if locked_path.exists():
        return pd.read_parquet(locked_path)
    return None


def _save_locked_predictions(df: pd.DataFrame, game_date: date, draft_group_id: str, data_root: Path) -> None:
    """Save predictions to locked file for a specific slate."""
    out_dir = data_root / "silver" / "ownership_predictions" / str(game_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    locked_path = out_dir / f"{draft_group_id}_locked.parquet"
    if not locked_path.exists():
        df.to_parquet(locked_path)
        print(f"[ownership] Saved locked predictions for slate {draft_group_id}: {len(df)} players")


PRODUCTION_MODEL_RUN = "dk_only_v4"


def _load_all_slates(
    game_date: date,
    data_root: Path,
) -> dict[str, pd.DataFrame]:
    """
    Load all DK slates (draft groups) for a date.
    
    Returns dict of {draft_group_id: DataFrame with normalized columns}.
    """
    base = data_root / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}"
    
    if not base.exists():
        print(f"[ownership] No salary data at {base}")
        return {}
    
    draft_group_dirs = sorted(base.glob("draft_group_id=*"))
    if not draft_group_dirs:
        print(f"[ownership] No draft groups found for {game_date}")
        return {}
    
    slates = {}
    for dg_dir in draft_group_dirs:
        parquet_path = dg_dir / "salaries.parquet"
        if not parquet_path.exists():
            continue
        
        df = pd.read_parquet(parquet_path)
        dg_id = dg_dir.name.split("=")[1]
        df["draft_group_id"] = dg_id
        
        # Normalize column names for ownership model
        if "display_name" in df.columns and "player_name" not in df.columns:
            df["player_name"] = df["display_name"]
        if "positions" in df.columns and "pos" not in df.columns:
            df["pos"] = df["positions"].apply(lambda x: "/".join(x) if isinstance(x, list) else str(x))
        if "team_abbrev" in df.columns and "team" not in df.columns:
            df["team"] = df["team_abbrev"]
        if "dk_player_id" in df.columns and "player_id" not in df.columns:
            df["player_id"] = df["dk_player_id"]
        if "salary" in df.columns:
            df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)
        
        slates[dg_id] = df
    
    print(f"[ownership] Loaded {len(slates)} slates for {game_date}")
    for dg_id, df in slates.items():
        teams = df["team"].unique() if "team" in df.columns else []
        print(f"  - {dg_id}: {len(df)} players, teams: {list(teams)[:6]}{'...' if len(teams) > 6 else ''}")
    
    return slates


def _get_slate_first_game_time(
    slate_teams: set[str],
    schedule: pd.DataFrame,
) -> Optional[datetime]:
    """Get earliest game start time for teams in this slate."""
    if schedule.empty or "game_start" not in schedule.columns:
        return None
    
    earliest = None
    for _, row in schedule.iterrows():
        home = row.get("home_team_tricode")
        away = row.get("away_team_tricode")
        if home in slate_teams or away in slate_teams:
            game_start = row["game_start"]
            if pd.notna(game_start):
                if hasattr(game_start, "tzinfo") and game_start.tzinfo is not None:
                    game_start = game_start.replace(tzinfo=None)
                if earliest is None or game_start < earliest:
                    earliest = game_start
    
    return earliest


def _is_slate_locked(
    slate_teams: set[str],
    schedule: pd.DataFrame,
    current_time: datetime,
) -> bool:
    """Check if a slate's first game has already started."""
    first_game = _get_slate_first_game_time(slate_teams, schedule)
    if first_game is None:
        return False
    return current_time >= first_game


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
        print("[ownership] sim projections missing dk_fpts_mean column")
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
        # injuries_snapshot is keyed by report_date (not game_date).
        if "report_date" in df.columns:
            df["_report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
            df = df[df["_report_date"] == game_date].copy()
            df = df.drop(columns=["_report_date"], errors="ignore")
        if not df.empty:
            return df
    
    return pd.DataFrame()


@lru_cache(maxsize=32)
def _historical_ownership_map(
    *, game_date_iso: str, data_root_str: str, window: int = 10
) -> dict[str, float]:
    """Compute name_norm -> avg ownership over last N slates before game_date."""

    data_root = Path(data_root_str)
    path = (
        data_root
        / "bronze"
        / "dk_contests"
        / "ownership_by_slate"
        / "all_ownership.parquet"
    )
    if not path.exists():
        return {}

    df = pd.read_parquet(path)
    if df.empty or "Player" not in df.columns or "own_pct" not in df.columns:
        return {}

    df = df.copy()
    df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce").dt.date
    cutoff = date.fromisoformat(game_date_iso)
    df = df[df["game_date"].notna() & (df["game_date"] < cutoff)].copy()
    if df.empty:
        return {}

    df["_name_norm"] = df["Player"].astype(str).apply(_normalize_name)
    df["_own"] = pd.to_numeric(df["own_pct"], errors="coerce")
    df = df[df["_name_norm"].ne("") & df["_own"].notna()].copy()
    if df.empty:
        return {}

    # Ensure stable ordering within date.
    sort_cols = ["game_date"]
    if "slate_id" in df.columns:
        sort_cols.append("slate_id")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    def _tail_mean(g: pd.DataFrame) -> float:
        return float(g.tail(window)["_own"].mean())

    means = df.groupby("_name_norm", sort=False).apply(_tail_mean, include_groups=False)
    return means.to_dict()


def _attach_live_ownership_enrichment(
    salaries: pd.DataFrame,
    *,
    game_date: date,
    data_root: Path,
    minutes_team_map: dict[int, str] | None,
    nba_player_ids: pd.Series | None,
) -> pd.DataFrame:
    """Attach player_own_avg_10, player_is_questionable, team_outs_count."""

    working = salaries.copy()
    working["_name_norm"] = working["player_name"].astype(str).apply(_normalize_name)

    # Historical ownership baseline.
    hist = _historical_ownership_map(
        game_date_iso=game_date.isoformat(),
        data_root_str=str(data_root),
        window=10,
    )
    if hist:
        overall = float(pd.Series(hist.values()).mean()) if hist else 0.0
        working["player_own_avg_10"] = working["_name_norm"].map(hist).fillna(overall).astype(float)
    else:
        working["player_own_avg_10"] = 0.0

    # Injury enrichment.
    inj = _load_injuries(game_date, data_root)
    if inj.empty:
        working["player_is_questionable"] = 0
        working["team_outs_count"] = 0
        working = working.drop(columns=["_name_norm"], errors="ignore")
        return working

    inj = inj.copy()
    # Keep latest snapshot per player when multiple as_of_ts rows exist.
    if "as_of_ts" in inj.columns:
        inj["_as_of_ts"] = pd.to_datetime(inj["as_of_ts"], utc=True, errors="coerce")
        inj = inj.sort_values("_as_of_ts").dropna(subset=["player_id"]).drop_duplicates("player_id", keep="last")
        inj = inj.drop(columns=["_as_of_ts"], errors="ignore")

    status = inj.get("status")
    status_raw = inj.get("status_raw")
    status_u = status.astype(str).str.upper() if status is not None else pd.Series("", index=inj.index)
    status_raw_u = status_raw.astype(str).str.upper() if status_raw is not None else pd.Series("", index=inj.index)

    is_out = status_u.eq("OUT") | status_raw_u.eq("OUT")
    is_q = status_u.isin(["Q", "PROB"]) | status_raw_u.isin(["QUESTIONABLE", "PROBABLE", "DOUBTFUL"])
    inj["_is_q"] = is_q.astype(int)

    # Team outs by tricode (DK salary table uses tricodes).
    team_outs_count: dict[str, int] = {}
    if "team_id" in inj.columns and minutes_team_map:
        team_ids = pd.to_numeric(inj["team_id"], errors="coerce")
        inj["_team_tricode"] = team_ids.map(lambda v: minutes_team_map.get(int(v)) if pd.notna(v) else None)
        outs_by_team = inj.loc[is_out & inj["_team_tricode"].notna()].groupby("_team_tricode")["player_id"].count()
        team_outs_count = {str(k): int(v) for k, v in outs_by_team.to_dict().items()}
    working["team_outs_count"] = working.get("team", pd.Series("", index=working.index)).map(team_outs_count).fillna(0).astype(int)

    # Player questionable by NBA player_id when available, else by name_norm.
    q_by_pid: dict[int, int] = {}
    if "player_id" in inj.columns:
        pid_series = pd.to_numeric(inj["player_id"], errors="coerce").astype("Int64")
        q_by_pid = {int(pid): int(flag) for pid, flag in zip(pid_series.dropna().astype(int), is_q.astype(int))}

    if nba_player_ids is not None:
        pid_norm = pd.to_numeric(nba_player_ids, errors="coerce").astype("Int64")
        working["player_is_questionable"] = pid_norm.map(lambda v: q_by_pid.get(int(v), 0) if pd.notna(v) else 0).astype(int)
    else:
        inj["_name_norm"] = inj["player_name"].astype(str).apply(_normalize_name)
        q_by_name = inj.loc[inj["_name_norm"].ne("")].groupby("_name_norm")["_is_q"].max()
        working["player_is_questionable"] = working["_name_norm"].map(q_by_name.to_dict()).fillna(0).astype(int)

    working = working.drop(columns=["_name_norm"], errors="ignore")
    return working


def score_ownership(
    slate_df: pd.DataFrame,
    draft_group_id: str,
    game_date: date,
    run_id: str,
    data_root: Path,
    model_run: str = PRODUCTION_MODEL_RUN,
) -> Optional[pd.DataFrame]:
    """
    Score ownership predictions for a single slate.
    
    Args:
        slate_df: DataFrame with salary data for this slate (already normalized)
        draft_group_id: DraftKings draft group ID
        game_date: Game date
        run_id: Run identifier
        data_root: Data root path
        model_run: Ownership model run ID
    
    Returns DataFrame with player ownership predictions or None if data unavailable.
    """
    # Load model bundle
    try:
        bundle = load_ownership_bundle(model_run)
    except FileNotFoundError as e:
        print(f"[ownership] Model not found: {e}")
        return None
    
    # Use provided slate data
    salaries = slate_df.copy()
    if salaries.empty:
        print(f"[ownership] Empty slate data for {draft_group_id}")
        return None
    
    # Load FPTS predictions from sim
    fpts = _load_fpts_predictions(game_date, run_id, data_root)
    
    if fpts is None or fpts.empty:
        print(f"[ownership] No FPTS predictions for {game_date}, using salary-based estimate")
        # Use salary as proxy for FPTS if no predictions available
        salaries["proj_fpts"] = salaries["salary"] / 200.0  # Rough conversion
        salaries = _attach_live_ownership_enrichment(
            salaries,
            game_date=game_date,
            data_root=data_root,
            minutes_team_map=None,
            nba_player_ids=None,
        )
    else:
        # Load minutes to get player_name -> NBA player_id mapping
        # This bridges DK's display_name to sim's player_id
        import json
        from pathlib import Path
        
        minutes_root = Path("/home/daniel/projects/projections-v2/artifacts/minutes_v1/daily") / str(game_date)
        latest_pointer = minutes_root / "latest_run.json"
        
        player_id_map = None
        team_id_to_tricode: dict[int, str] | None = None
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
                    if "team_id" in minutes_df.columns:
                        cols_to_load.append("team_id")
                    if "team_tricode" in minutes_df.columns:
                        cols_to_load.append("team_tricode")
                    minutes_df = minutes_df[[c for c in cols_to_load if c in minutes_df.columns]].copy()
                    
                    # Create name -> player_id mapping using normalized names
                    # This handles European characters like Dončić -> doncic
                    minutes_df["_name_norm"] = minutes_df["player_name"].apply(_normalize_name)
                    player_id_map = minutes_df.drop_duplicates("_name_norm").set_index("_name_norm")["player_id"]
                    
                    # Also create player_id -> status map for OUT filtering
                    status_map = None
                    if "status" in minutes_df.columns:
                        status_map = minutes_df.drop_duplicates("player_id").set_index("player_id")["status"]
                    if {"team_id", "team_tricode"}.issubset(minutes_df.columns):
                        team_id_to_tricode = (
                            minutes_df[["team_id", "team_tricode"]]
                            .dropna()
                            .drop_duplicates("team_id")
                            .set_index("team_id")["team_tricode"]
                            .to_dict()
                        )
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
            salaries = salaries.drop(columns=["player_id_sim"], errors="ignore")
            
            # Add injury status from minutes predictions for OUT filtering
            if status_map is not None:
                # Map player_id back to nba_player_id we got from name matching
                salaries["_temp_pid"] = salaries["player_name"].apply(_normalize_name).map(player_id_map)
                salaries["_injury_status"] = salaries["_temp_pid"].map(status_map)
                salaries = salaries.drop(columns=["_temp_pid"], errors="ignore")
            
            matched = salaries["proj_fpts"].notna().sum()
            print(f"[ownership] Matched {matched}/{len(salaries)} players via name→id mapping")

            # Attach historical ownership + injury enrichment (uses nba_player_id when available).
            salaries = _attach_live_ownership_enrichment(
                salaries,
                game_date=game_date,
                data_root=data_root,
                minutes_team_map=team_id_to_tricode,
                nba_player_ids=salaries.get("nba_player_id"),
            )
            salaries = salaries.drop(columns=["_name_norm", "nba_player_id"], errors="ignore")
        else:
            # Fallback: use salary-based estimate
            print("[ownership] No player_id mapping available, using salary-based estimate")
            salaries["proj_fpts"] = salaries["salary"] / 200.0
            salaries = _attach_live_ownership_enrichment(
                salaries,
                game_date=game_date,
                data_root=data_root,
                minutes_team_map=None,
                nba_player_ids=None,
            )
    
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
        _ = prepare_model_input(features, bundle.feature_cols)
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
    
    # Add draft_group_id to output
    output["draft_group_id"] = draft_group_id
    output["is_locked"] = False
    
    return output


def score_all_slates(
    game_date: date,
    run_id: str,
    data_root: Path,
    model_run: str = PRODUCTION_MODEL_RUN,
) -> dict[str, pd.DataFrame]:
    """
    Score ownership predictions for all slates on a date.
    
    Handles per-slate lock detection: once a slate's first game starts,
    that slate returns cached predictions while other slates continue updating.
    
    Returns dict of {draft_group_id: predictions_df}.
    """
    # Load all slates
    slates = _load_all_slates(game_date, data_root)
    if not slates:
        print(f"[ownership] No slates found for {game_date}")
        return {}
    
    # Load schedule for lock detection
    schedule = _load_schedule_with_times(game_date, data_root)
    current_time = datetime.now()
    
    results = {}
    
    for dg_id, slate_df in slates.items():
        slate_teams = set(slate_df["team"].unique()) if "team" in slate_df.columns else set()
        
        # Check if this slate is locked
        if _is_slate_locked(slate_teams, schedule, current_time):
            first_game = _get_slate_first_game_time(slate_teams, schedule)
            print(f"[ownership] Slate {dg_id} is LOCKED (first game: {first_game})")
            
            # Try to load cached predictions
            cached = _load_locked_predictions(game_date, dg_id, data_root)
            if cached is not None and not cached.empty:
                cached["is_locked"] = True
                results[dg_id] = cached
                print(f"  -> Using cached predictions: {len(cached)} players")
                continue
            else:
                print("  -> WARNING: No cached predictions, scoring anyway")
        else:
            first_game = _get_slate_first_game_time(slate_teams, schedule)
            print(f"[ownership] Slate {dg_id} is UNLOCKED (first game: {first_game})")
        
        # Score this slate
        predictions = score_ownership(
            slate_df=slate_df,
            draft_group_id=dg_id,
            game_date=game_date,
            run_id=run_id,
            data_root=data_root,
            model_run=model_run,
        )
        
        if predictions is not None:
            results[dg_id] = predictions
            
            # Save for future lock
            _save_locked_predictions(predictions, game_date, dg_id, data_root)
    
    return results


def _save_slates_metadata(
    results: dict[str, pd.DataFrame],
    game_date: date,
    schedule: pd.DataFrame,
    data_root: Path,
) -> None:
    """Save slates.json metadata file."""
    import json
    
    current_time = datetime.now()
    slates_meta = {}
    
    for dg_id, df in results.items():
        teams = list(df["team"].unique()) if "team" in df.columns else []
        slate_teams = set(teams)
        first_game = _get_slate_first_game_time(slate_teams, schedule)
        is_locked = _is_slate_locked(slate_teams, schedule, current_time)
        
        slates_meta[dg_id] = {
            "player_count": len(df),
            "teams": teams,
            "first_game_time": first_game.isoformat() if first_game else None,
            "is_locked": is_locked,
        }
    
    out_dir = data_root / "silver" / "ownership_predictions" / str(game_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "slates.json", "w") as f:
        json.dump(slates_meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Score ownership predictions")
    parser.add_argument("--date", required=True, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--data-root", default=None, help="Data root path")
    parser.add_argument("--model-run", default=PRODUCTION_MODEL_RUN, help="Model run ID")
    args = parser.parse_args()
    
    game_date = date.fromisoformat(args.date)
    root = Path(args.data_root) if args.data_root else data_path()
    
    # Score all slates
    results = score_all_slates(game_date, args.run_id, root, args.model_run)
    
    if not results:
        print("[ownership] No predictions generated")
        return 1
    
    # Save per-slate predictions
    out_dir = root / "silver" / "ownership_predictions" / str(game_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for dg_id, df in results.items():
        out_path = out_dir / f"{dg_id}.parquet"
        df.to_parquet(out_path)
        print(f"[ownership] Saved slate {dg_id}: {len(df)} predictions -> {out_path}")
    
    # Save slates metadata
    schedule = _load_schedule_with_times(game_date, root)
    _save_slates_metadata(results, game_date, schedule, root)
    
    # Print summary for largest slate
    main_dg = max(results.keys(), key=lambda k: len(results[k]))
    main_df = results[main_dg]
    print(f"\n[ownership] Main slate ({main_dg}) top 5 by ownership:")
    print(main_df.nlargest(5, "pred_own_pct")[["player_name", "salary", "pred_own_pct"]].to_string())
    
    return 0


if __name__ == "__main__":
    exit(main())
