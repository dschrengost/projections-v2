"""
Projection Diagnostics API - Serves feature and projection diagnostics to dashboard.
"""

from __future__ import annotations

import json
from datetime import date, datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from projections import paths

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])

LATEST_POINTER = "latest_run.json"
PINNED_POINTER = "pinned_run.json"
PROJECTIONS_FILENAME = "projections.parquet"

# Expected ranges for anomaly detection
EXPECTED_RANGES = {
    "season_fga_per_min": (0.1, 1.2),
    "season_3pa_per_min": (0.0, 0.6),
    "season_fta_per_min": (0.0, 0.4),
    "season_ast_per_min": (0.0, 0.5),
    "season_tov_per_min": (0.0, 0.2),
    "season_reb_per_min": (0.0, 0.6),
    "season_stl_per_min": (0.0, 0.1),
    "season_blk_per_min": (0.0, 0.15),
    "minutes_pred_p50": (0, 42),
}

ELITE_PLAYERS = {
    203999: "Nikola Jokic",
    201142: "Kevin Durant",
    203507: "Giannis Antetokounmpo",
    1629029: "Luka Doncic",
    203954: "Joel Embiid",
    201935: "James Harden",
    2544: "LeBron James",
    1628369: "Jayson Tatum",
    1630567: "Anthony Edwards",
    1629630: "Ja Morant",
}


def _load_latest_features(data_root: Path, game_date: date) -> pd.DataFrame:
    """Load latest rates features for a date."""
    feature_dir = data_root / "live" / "features_rates_v1" / game_date.isoformat()
    if not feature_dir.exists():
        return pd.DataFrame()
    
    latest_pointer = feature_dir / "latest_run.json"
    if latest_pointer.exists():
        with open(latest_pointer) as f:
            run_id = json.load(f).get("run_id")
        run_dir = feature_dir / f"run={run_id}"
    else:
        runs = sorted(feature_dir.glob("run=*"))
        if not runs:
            return pd.DataFrame()
        run_dir = runs[-1]
    
    parquet_path = run_dir / "features.parquet"
    if not parquet_path.exists():
        return pd.DataFrame()
    
    return pd.read_parquet(parquet_path)


def _resolve_unified_run_dir(day_dir: Path, *, run_id: str | None) -> tuple[Path | None, str | None]:
    if not day_dir.exists():
        return None, None

    if run_id is not None:
        candidate = day_dir / f"run={run_id}"
        return (candidate, run_id) if candidate.exists() else (None, None)

    pinned_pointer = day_dir / PINNED_POINTER
    if pinned_pointer.exists():
        try:
            payload = json.loads(pinned_pointer.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        pinned = payload.get("run_id") if isinstance(payload, dict) else None
        if pinned:
            candidate = day_dir / f"run={pinned}"
            if candidate.exists():
                return candidate, str(pinned)

    latest_pointer = day_dir / LATEST_POINTER
    if latest_pointer.exists():
        try:
            payload = json.loads(latest_pointer.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        latest = payload.get("run_id") if isinstance(payload, dict) else None
        if latest:
            candidate = day_dir / f"run={latest}"
            if candidate.exists():
                return candidate, str(latest)

    run_dirs = sorted([p for p in day_dir.glob("run=*") if p.is_dir()], reverse=True)
    if run_dirs:
        run_dir = run_dirs[0]
        resolved_run_id = run_dir.name.split("=", 1)[1] if run_dir.name.startswith("run=") else None
        return run_dir, resolved_run_id
    return None, None


def _load_unified_projections(
    data_root: Path,
    game_date: date,
    *,
    run_id: str | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """Load projections for a date from the unified projections artifact (source of truth)."""
    day_dir = data_root / "artifacts" / "projections" / game_date.isoformat()
    run_dir, resolved_run_id = _resolve_unified_run_dir(day_dir, run_id=run_id)
    if run_dir is None:
        return pd.DataFrame(), None
    parquet_path = run_dir / PROJECTIONS_FILENAME
    if not parquet_path.exists():
        return pd.DataFrame(), resolved_run_id
    return pd.read_parquet(parquet_path), resolved_run_id


def _check_anomalies(df: pd.DataFrame) -> list[dict]:
    """Check features against expected ranges."""
    anomalies = []
    for col, (min_val, max_val) in EXPECTED_RANGES.items():
        if col not in df.columns:
            continue
        
        below = df[df[col] < min_val]
        above = df[df[col] > max_val]
        
        if len(above) > 0:
            anomalies.append({
                "feature": col,
                "issue": "above_max",
                "expected_max": max_val,
                "count": len(above),
                "severity": "high" if max_val > 0 and df[col].max() > max_val * 5 else "medium",
            })
        
        # Only flag below_min if there are non-zero values expected
        if len(below) > 0 and min_val > 0:
            anomalies.append({
                "feature": col,
                "issue": "below_min",
                "expected_min": min_val,
                "count": len(below),
                "severity": "low",
            })
    
    return anomalies


@router.get("/feature-summary")
async def get_feature_summary(
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
):
    """Get feature summary statistics for a date."""
    try:
        game_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")
    
    data_root = paths.data_path()
    features = _load_latest_features(data_root, game_date)
    
    if features.empty:
        raise HTTPException(status_code=404, detail=f"No features found for {date}")
    
    # Compute summary stats
    key_features = [
        "minutes_pred_p50", "is_starter",
        "season_fga_per_min", "season_ast_per_min", "season_reb_per_min",
        "season_stl_per_min", "season_blk_per_min", "season_tov_per_min",
    ]
    
    stats = {}
    for feat in key_features:
        if feat in features.columns:
            col = features[feat]
            stats[feat] = {
                "min": float(col.min()) if pd.notna(col.min()) else None,
                "mean": float(col.mean()) if pd.notna(col.mean()) else None,
                "max": float(col.max()) if pd.notna(col.max()) else None,
                "missing": int(col.isna().sum()),
            }
    
    anomalies = _check_anomalies(features)
    
    return {
        "date": date,
        "row_count": len(features),
        "starters": int((features.get("is_starter", pd.Series()) == 1).sum()),
        "stats": stats,
        "anomalies": anomalies,
        "has_anomalies": len([a for a in anomalies if a["severity"] == "high"]) > 0,
    }


@router.get("/projections")
async def get_projections(
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    top_n: int = Query(20, description="Number of top players to return"),
    run_id: str | None = Query(None, description="Optional projections run_id (defaults to pinned/latest)."),
):
    """Get sim projections for a date."""
    try:
        game_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")
    
    data_root = paths.data_path()
    
    # Load features and projections
    features = _load_latest_features(data_root, game_date)
    projections, resolved_run_id = _load_unified_projections(data_root, game_date, run_id=run_id)
    if run_id is not None and resolved_run_id is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found for {date}")
    
    if projections.empty:
        raise HTTPException(status_code=404, detail=f"No projections found for {date}")
    
    # Merge with features for season stats
    if not features.empty:
        feature_cols = ["player_id", "season_fga_per_min", "season_ast_per_min", 
                       "season_reb_per_min", "minutes_pred_p50"]
        feature_cols = [c for c in feature_cols if c in features.columns]
        projections = projections.merge(
            features[feature_cols],
            on="player_id",
            how="left"
        )
    
    # Add player names for known elite players
    projections["player_name"] = projections["player_id"].map(
        lambda x: ELITE_PLAYERS.get(int(x), None)
    )
    
    # Sort by fpts_mean
    fpts_col = "dk_fpts_mean" if "dk_fpts_mean" in projections.columns else "fpts_mean"
    projections = projections.sort_values(fpts_col, ascending=False)
    
    # Select columns for response
    response_cols = [
        "player_id", "player_name", "team_id", "is_starter",
        "minutes_mean", "minutes_sim_mean", "minutes_pred_p50",
        "dk_fpts_mean", "dk_fpts_p50", "dk_fpts_p10", "dk_fpts_p90",
        "pts_mean", "reb_mean", "ast_mean", "stl_mean", "blk_mean", "tov_mean",
        "season_fga_per_min", "season_ast_per_min", "season_reb_per_min",
    ]
    response_cols = [c for c in response_cols if c in projections.columns]
    
    top_players = projections.head(top_n)[response_cols].to_dict("records")
    
    # Convert NaN to None
    for row in top_players:
        for k, v in row.items():
            if pd.isna(v):
                row[k] = None
            elif isinstance(v, (np.integer, np.floating)):
                row[k] = float(v)
    
    return {
        "date": date,
        "run_id": resolved_run_id,
        "total_players": len(projections),
        "players": top_players,
    }


@router.get("/player/{player_id}")
async def get_player_details(
    player_id: int,
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    run_id: str | None = Query(None, description="Optional projections run_id (defaults to pinned/latest)."),
):
    """Get detailed projection and features for a specific player."""
    try:
        game_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")
    
    data_root = paths.data_path()
    
    features = _load_latest_features(data_root, game_date)
    projections, resolved_run_id = _load_unified_projections(data_root, game_date, run_id=run_id)
    if run_id is not None and resolved_run_id is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found for {date}")
    
    # Find player in features
    player_features = {}
    if not features.empty:
        player_row = features[features["player_id"] == player_id]
        if not player_row.empty:
            row = player_row.iloc[0]
            for col in features.columns:
                val = row[col]
                if pd.isna(val):
                    player_features[col] = None
                elif isinstance(val, (np.integer, np.floating)):
                    player_features[col] = float(val)
                else:
                    player_features[col] = val
    
    # Find player in projections
    player_projections = {}
    if not projections.empty:
        player_row = projections[projections["player_id"] == player_id]
        if not player_row.empty:
            row = player_row.iloc[0]
            for col in projections.columns:
                val = row[col]
                if pd.isna(val):
                    player_projections[col] = None
                elif isinstance(val, (np.integer, np.floating)):
                    player_projections[col] = float(val)
                else:
                    player_projections[col] = str(val)
    
    if not player_features and not player_projections:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found for {date}")
    
    return {
        "player_id": player_id,
        "player_name": ELITE_PLAYERS.get(player_id),
        "date": date,
        "run_id": resolved_run_id,
        "features": player_features,
        "projections": player_projections,
    }


@router.get("/historical-comparison")
async def get_historical_comparison(
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    lookback_days: int = Query(7, description="Number of days to compare"),
):
    """Compare today's features to historical averages."""
    try:
        game_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")
    
    data_root = paths.data_path()
    
    # Load today's features
    today_features = _load_latest_features(data_root, game_date)
    if today_features.empty:
        raise HTTPException(status_code=404, detail=f"No features found for {date}")
    
    # Load historical features
    historical_stats = []
    from datetime import timedelta
    for i in range(1, lookback_days + 1):
        hist_date = game_date - timedelta(days=i)
        hist_features = _load_latest_features(data_root, hist_date)
        if not hist_features.empty:
            historical_stats.append({
                "date": hist_date.isoformat(),
                "row_count": len(hist_features),
                "mean_minutes_p50": float(hist_features["minutes_pred_p50"].mean()) if "minutes_pred_p50" in hist_features else None,
                "mean_fga_per_min": float(hist_features["season_fga_per_min"].mean()) if "season_fga_per_min" in hist_features else None,
            })
    
    # Compare key metrics
    comparison = {}
    for feat in ["season_fga_per_min", "season_ast_per_min", "season_reb_per_min"]:
        if feat in today_features.columns:
            today_mean = float(today_features[feat].mean())
            hist_means = [s.get(f"mean_{feat.replace('season_', '')}", today_mean) for s in historical_stats]
            hist_means = [m for m in hist_means if m is not None]
            hist_avg = sum(hist_means) / len(hist_means) if hist_means else today_mean
            
            comparison[feat] = {
                "today": today_mean,
                "historical_avg": hist_avg,
                "diff_pct": (today_mean - hist_avg) / hist_avg * 100 if hist_avg else 0,
            }
    
    return {
        "date": date,
        "lookback_days": lookback_days,
        "today_row_count": len(today_features),
        "historical_dates": [s["date"] for s in historical_stats],
        "comparison": comparison,
    }
