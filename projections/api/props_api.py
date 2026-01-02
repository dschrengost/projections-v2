"""FastAPI router for player props and EV analysis.

Endpoints:
- GET /api/props/lines - All prop lines for a date merged with predictions
- GET /api/props/summary - High-level summary for a date
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from projections import paths

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/props", tags=["props"])

# Default paths
DEFAULT_DATA_ROOT = paths.data_path()
DEFAULT_PROPS_ROOT = paths.data_path("bronze", "props")
DEFAULT_SIM_ROOT = paths.data_path("artifacts", "sim_v2", "worlds_fpts_v2")
DEFAULT_PROJECTIONS_ROOT = paths.data_path("artifacts", "projections")

# Prop type to sim world column mapping
PROP_TO_WORLD_COLUMN: dict[str, str | list[str]] = {
    "pts": "pts",
    "reb": "reb",
    "ast": "ast",
    "threes": "fgm3",
    "blk": "blk",
    "stl": "stl",
    "turnovers": "tov",
    "ptsrebast": ["pts", "reb", "ast"],
    "ptsreb": ["pts", "reb"],
    "ptsast": ["pts", "ast"],
    "rebast": ["reb", "ast"],
    "stlblk": ["stl", "blk"],
}


# ---------- Pydantic Models ----------


class BookLine(BaseModel):
    """A single book's line for a prop."""

    book: str
    line: float
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None


class PropLineResponse(BaseModel):
    """Response model for a single prop line."""

    player_id: str
    player_name: str
    team: str
    opponent: str
    prop_type: str
    prediction: Optional[float] = None
    prediction_std: Optional[float] = None

    # Best line across books
    best_over_line: Optional[float] = None
    best_over_odds: Optional[int] = None
    best_over_book: Optional[str] = None
    best_under_line: Optional[float] = None
    best_under_odds: Optional[int] = None
    best_under_book: Optional[str] = None

    # EV calculations
    over_implied_prob: Optional[float] = None
    over_true_prob: Optional[float] = None
    over_ev: Optional[float] = None
    over_edge: Optional[str] = None

    under_implied_prob: Optional[float] = None
    under_true_prob: Optional[float] = None
    under_ev: Optional[float] = None
    under_edge: Optional[str] = None

    # All book lines
    all_lines: list[BookLine] = []


class BestEdge(BaseModel):
    """A top edge for the summary."""

    player: str
    prop: str
    side: str
    ev: float


class PropsSummaryResponse(BaseModel):
    """Response model for props summary."""

    date: str
    total_props: int
    players_with_props: int
    props_with_edge: int
    best_edges: list[BestEdge] = []


# ---------- EV Calculation Utilities ----------


def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def calculate_ev(true_prob: float, odds: int) -> float:
    """Calculate expected value per $1 risked."""
    if odds > 0:
        payout = odds / 100  # profit per $1 risked
    else:
        payout = 100 / abs(odds)
    # EV = P(win) * payout - P(lose) * stake
    return true_prob * payout - (1 - true_prob) * 1.0


def classify_edge(over_ev: float | None, under_ev: float | None) -> tuple[str, str]:
    """Classify edge based on EV thresholds.

    Returns (over_edge, under_edge) classification.
    """
    over_edge = "no_edge"
    under_edge = "no_edge"

    if over_ev is not None:
        if over_ev > 0.10:
            over_edge = "strong_over"
        elif over_ev > 0.03:
            over_edge = "slight_over"
        elif over_ev >= -0.03:
            over_edge = "fair"

    if under_ev is not None:
        if under_ev > 0.10:
            under_edge = "strong_under"
        elif under_ev > 0.03:
            under_edge = "slight_under"
        elif under_ev >= -0.03:
            under_edge = "fair"

    return over_edge, under_edge


# ---------- Data Loaders ----------


def _load_props(game_date: date, props_root: Path) -> pd.DataFrame | None:
    """Load props data from bronze layer for a date.

    Returns the most recent props file if multiple exist.
    """
    date_dir = props_root / f"game_date={game_date.isoformat()}"
    if not date_dir.exists():
        logger.warning("Props directory not found: %s", date_dir)
        return None

    # Find all props files and get the most recent
    props_files = sorted(date_dir.glob("props_*.parquet"), reverse=True)
    if not props_files:
        logger.warning("No props files found in %s", date_dir)
        return None

    latest_file = props_files[0]
    logger.info("Loading props from %s", latest_file)
    return pd.read_parquet(latest_file)


def _load_projections(
    game_date: date, data_root: Path, run_id: str | None = None
) -> pd.DataFrame | None:
    """Load projections for a date from the unified projections artifact.

    Uses the same loading function as the /api/minutes endpoint to ensure consistency.
    This includes pts_mean, reb_mean, ast_mean, dk_fpts_mean, etc.
    """
    try:
        from projections.api.minutes_api import _load_unified_projections
        df, _, _ = _load_unified_projections(game_date, run_id, data_root)
        if df is not None and not df.empty:
            logger.info("Loaded %d players from unified projections", len(df))
            return df
    except ImportError:
        logger.warning("Could not import _load_unified_projections from minutes_api")
    except Exception as e:
        logger.warning("Failed to load unified projections: %s", e)
    
    return None


def _load_sim_worlds(
    game_date: date, sim_root: Path, run_id: str | None = None
) -> pd.DataFrame | None:
    """Load sim_v2 world samples for probability calculations.

    Returns DataFrame with columns: player_id, world_id, pts, reb, ast, stl, blk, tov, fgm3
    """
    date_dir = sim_root / f"game_date={game_date.isoformat()}"
    if not date_dir.exists():
        logger.info("Sim worlds directory not found: %s", date_dir)
        return None

    # Find run directory
    if run_id:
        run_dir = date_dir / f"run={run_id}"
    else:
        run_dirs = sorted(
            [d for d in date_dir.iterdir() if d.is_dir() and d.name.startswith("run=")],
            reverse=True,
        )
        if not run_dirs:
            logger.info("No sim run directories in %s", date_dir)
            return None
        run_dir = run_dirs[0]

    if not run_dir.exists():
        logger.info("Sim run directory not found: %s", run_dir)
        return None

    # Look for worlds parquet files
    worlds_files = list(run_dir.glob("worlds*.parquet"))
    if not worlds_files:
        logger.info("No worlds files found in %s", run_dir)
        return None

    # Load and concatenate all worlds files
    dfs = []
    for wf in worlds_files:
        try:
            df = pd.read_parquet(wf)
            dfs.append(df)
        except Exception as e:
            logger.warning("Failed to load worlds file %s: %s", wf, e)

    if not dfs:
        return None

    worlds_df = pd.concat(dfs, ignore_index=True)
    
    # Check if player_id column exists
    if "player_id" not in worlds_df.columns:
        logger.info("Worlds data missing player_id column, available: %s", list(worlds_df.columns)[:10])
        return None
    
    logger.info("Loaded %d world samples for %d players", len(worlds_df), worlds_df["player_id"].nunique())
    return worlds_df


def _calculate_true_prob_over(
    worlds_df: pd.DataFrame | None,
    player_id: str,
    prop_type: str,
    line: float,
    prediction: float | None = None,
    prediction_std: float | None = None,
) -> float | None:
    """Calculate P(stat > line) from empirical world distribution or normal approximation."""
    col_spec = PROP_TO_WORLD_COLUMN.get(prop_type)
    if col_spec is None:
        return None

    if worlds_df is not None:
        # Try to get world samples for this player
        player_worlds = worlds_df[worlds_df["player_id"].astype(str) == str(player_id)]

        if not player_worlds.empty:
            # Get the stat values
            if isinstance(col_spec, list):
                # Combo prop - sum columns
                stat_values = sum(
                    player_worlds[col].fillna(0) for col in col_spec if col in player_worlds.columns
                )
            else:
                if col_spec not in player_worlds.columns:
                    return None
                stat_values = player_worlds[col_spec]

            if len(stat_values) > 0:
                return float((stat_values > line).mean())

    # Fall back to distribution-based approximation if we have prediction
    if prediction is not None and prediction >= 0:
        from scipy import stats
        
        # Use Poisson for discrete low-count stats (more appropriate than normal)
        # Common lines are 0.5, 1.5, 2.5, etc so we need P(X >= ceil(line))
        discrete_stats = {"blk", "stl", "turnovers", "threes"}
        
        if prop_type in discrete_stats:
            # Poisson: P(X > line) = P(X >= ceil(line)) = 1 - P(X < ceil(line)) = 1 - CDF(floor(line))
            # For line=0.5, we want P(X >= 1) = 1 - P(X=0)
            return float(1 - stats.poisson.cdf(int(line), mu=max(prediction, 0.01)))
        else:
            # Normal approximation for continuous-ish stats (pts, reb, ast)
            std = prediction_std if prediction_std and prediction_std > 0 else prediction * 0.3
            if std > 0:
                return float(1 - stats.norm.cdf(line, loc=prediction, scale=std))

    return None


def _calculate_true_prob_under(
    worlds_df: pd.DataFrame | None,
    player_id: str,
    prop_type: str,
    line: float,
    prediction: float | None = None,
    prediction_std: float | None = None,
) -> float | None:
    """Calculate P(stat < line) from empirical world distribution or normal approximation."""
    col_spec = PROP_TO_WORLD_COLUMN.get(prop_type)
    if col_spec is None:
        return None

    if worlds_df is not None:
        player_worlds = worlds_df[worlds_df["player_id"].astype(str) == str(player_id)]

        if not player_worlds.empty:
            if isinstance(col_spec, list):
                stat_values = sum(
                    player_worlds[col].fillna(0) for col in col_spec if col in player_worlds.columns
                )
            else:
                if col_spec not in player_worlds.columns:
                    return None
                stat_values = player_worlds[col_spec]

            if len(stat_values) > 0:
                return float((stat_values < line).mean())

    # Fall back to distribution-based approximation if we have prediction
    if prediction is not None and prediction >= 0:
        from scipy import stats
        
        # Use Poisson for discrete low-count stats (more appropriate than normal)
        discrete_stats = {"blk", "stl", "turnovers", "threes"}
        
        if prop_type in discrete_stats:
            # Poisson: P(X < line) = CDF(floor(line) - 1) for line like 0.5, 1.5
            # For line=0.5, we want P(X < 0.5) = P(X <= -1) = 0 (impossible)
            # Actually: P(X < 0.5) = P(X = 0) since X is integer
            k = int(line) - 1 if line == int(line) else int(line) - 1
            if k < 0:
                # P(X < 0.5) = P(X = 0) for Poisson
                return float(stats.poisson.pmf(0, mu=max(prediction, 0.01)))
            return float(stats.poisson.cdf(k, mu=max(prediction, 0.01)))
        else:
            # Normal approximation for continuous-ish stats (pts, reb, ast)
            std = prediction_std if prediction_std and prediction_std > 0 else prediction * 0.3
            if std > 0:
                return float(stats.norm.cdf(line, loc=prediction, scale=std))

    return None


def _get_prediction_for_prop(
    projections_df: pd.DataFrame | None,
    player_name: str,
    prop_type: str,
) -> tuple[float | None, float | None]:
    """Get prediction mean and std for a prop type from projections.

    Matches on player_name since props data uses different player IDs than projections.
    Returns (mean, std).
    """
    if projections_df is None:
        return None, None

    # Normalize player name for matching (lowercase, strip whitespace)
    target_name = player_name.strip().lower()
    
    # Find player row by name
    projections_df = projections_df.copy()
    projections_df["_name_lower"] = projections_df["player_name"].str.strip().str.lower()
    player_rows = projections_df[projections_df["_name_lower"] == target_name]
    if player_rows.empty:
        return None, None

    player = player_rows.iloc[0]

    # Map prop type to projection column - unified projections use pts_mean, reb_mean etc
    # (without sim_ prefix), but we also check sim_ prefixed versions as fallback
    col_mapping = {
        "pts": ("pts_mean", "pts_std"),
        "reb": ("reb_mean", "reb_std"),
        "ast": ("ast_mean", "ast_std"),
        "threes": ("fgm3_mean", "fgm3_std"),
        "blk": ("blk_mean", "blk_std"),
        "stl": ("stl_mean", "stl_std"),
        "turnovers": ("tov_mean", "tov_std"),
    }

    # Try primary columns first, then sim_ prefixed as fallback
    if prop_type in col_mapping:
        mean_col, std_col = col_mapping[prop_type]
        mean_val = player.get(mean_col)
        std_val = player.get(std_col)

        # Fall back to sim_ prefixed columns
        if pd.isna(mean_val):
            sim_mean_col = f"sim_{mean_col}"
            sim_std_col = f"sim_{std_col}"
            mean_val = player.get(sim_mean_col)
            std_val = player.get(sim_std_col)

        if pd.notna(mean_val):
            return float(mean_val), float(std_val) if pd.notna(std_val) else None

    # Handle combo props
    combo_mapping = {
        "ptsrebast": ["pts", "reb", "ast"],
        "ptsreb": ["pts", "reb"],
        "ptsast": ["pts", "ast"],
        "rebast": ["reb", "ast"],
        "stlblk": ["stl", "blk"],
    }

    if prop_type in combo_mapping:
        components = combo_mapping[prop_type]
        total_mean = 0.0
        total_var = 0.0
        valid = True

        for comp in components:
            mean, std = _get_prediction_for_prop(projections_df, player_name, comp)
            if mean is None:
                valid = False
                break
            total_mean += mean
            if std:
                total_var += std**2

        if valid:
            total_std = total_var**0.5 if total_var > 0 else None
            return total_mean, total_std

    return None, None


# ---------- Endpoints ----------


def _get_props_lines_impl(
    game_date: date,
    data_root: Path,
    prop_type: str | None = None,
    min_edge: float | None = None,
) -> list[PropLineResponse]:
    """Internal implementation for getting props lines.

    This is the core logic, separated from FastAPI Query handling.
    """
    # Load data
    props_root = data_root / "bronze" / "props"
    projections_root = data_root / "artifacts" / "projections"
    sim_root = data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2"

    props_df = _load_props(game_date, props_root)
    if props_df is None or props_df.empty:
        raise HTTPException(status_code=404, detail=f"No props data for {game_date}")

    projections_df = _load_projections(game_date, data_root)
    worlds_df = _load_sim_worlds(game_date, sim_root)

    # Filter by prop type if specified
    if prop_type:
        props_df = props_df[props_df["prop_type"] == prop_type]

    # Group by player + prop_type to aggregate across books
    grouped = props_df.groupby(["player_id", "player_name", "team", "opponent", "prop_type"])

    results: list[PropLineResponse] = []

    for (player_id, player_name, team, opponent, pt), group in grouped:
        # Get all book lines
        all_lines: list[BookLine] = []
        for _, row in group.iterrows():
            all_lines.append(
                BookLine(
                    book=row["book"],
                    line=row["line"],
                    over_odds=int(row["over_odds"]) if pd.notna(row["over_odds"]) else None,
                    under_odds=int(row["under_odds"]) if pd.notna(row["under_odds"]) else None,
                )
            )

        # Find best over and under odds
        over_lines = [bl for bl in all_lines if bl.over_odds is not None]
        under_lines = [bl for bl in all_lines if bl.under_odds is not None]

        best_over = max(over_lines, key=lambda x: x.over_odds) if over_lines else None
        best_under = max(under_lines, key=lambda x: x.under_odds) if under_lines else None

        # Get prediction
        prediction, prediction_std = _get_prediction_for_prop(projections_df, player_name, pt)

        # Calculate EV for best lines
        over_implied_prob = None
        over_true_prob = None
        over_ev = None
        under_implied_prob = None
        under_true_prob = None
        under_ev = None

        if best_over and best_over.over_odds:
            over_implied_prob = american_to_implied_prob(best_over.over_odds)
            over_true_prob = _calculate_true_prob_over(
                worlds_df, str(player_id), pt, best_over.line, prediction, prediction_std
            )
            if over_true_prob is not None:
                over_ev = calculate_ev(over_true_prob, best_over.over_odds)

        if best_under and best_under.under_odds:
            under_implied_prob = american_to_implied_prob(best_under.under_odds)
            under_true_prob = _calculate_true_prob_under(
                worlds_df, str(player_id), pt, best_under.line, prediction, prediction_std
            )
            if under_true_prob is not None:
                under_ev = calculate_ev(under_true_prob, best_under.under_odds)

        # Classify edge
        over_edge, under_edge = classify_edge(over_ev, under_ev)

        # Apply min_edge filter
        if min_edge is not None:
            max_ev = max(over_ev or -999, under_ev or -999)
            if max_ev < min_edge:
                continue

        results.append(
            PropLineResponse(
                player_id=str(player_id),
                player_name=player_name,
                team=team,
                opponent=opponent,
                prop_type=pt,
                prediction=prediction,
                prediction_std=prediction_std,
                best_over_line=best_over.line if best_over else None,
                best_over_odds=best_over.over_odds if best_over else None,
                best_over_book=best_over.book if best_over else None,
                best_under_line=best_under.line if best_under else None,
                best_under_odds=best_under.under_odds if best_under else None,
                best_under_book=best_under.book if best_under else None,
                over_implied_prob=over_implied_prob,
                over_true_prob=over_true_prob,
                over_ev=over_ev,
                over_edge=over_edge,
                under_implied_prob=under_implied_prob,
                under_true_prob=under_true_prob,
                under_ev=under_ev,
                under_edge=under_edge,
                all_lines=all_lines,
            )
        )

    # Sort by max EV descending
    results.sort(key=lambda x: max(x.over_ev or -999, x.under_ev or -999), reverse=True)

    return results


@router.get("/lines", response_model=list[PropLineResponse])
def get_props_lines(
    date: Optional[str] = Query(None, description="Game date YYYY-MM-DD"),
    prop_type: Optional[str] = Query(None, description="Filter by prop type"),
    min_edge: Optional[float] = Query(None, description="Minimum EV to include"),
    data_root: Path = DEFAULT_DATA_ROOT,
) -> list[PropLineResponse]:
    """Get all prop lines for a date merged with predictions and EV calculations."""
    # Parse date
    if date:
        try:
            game_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {date}")
    else:
        game_date = datetime.now(timezone.utc).date()

    return _get_props_lines_impl(game_date, data_root, prop_type, min_edge)


@router.get("/summary", response_model=PropsSummaryResponse)
def get_props_summary(
    date: Optional[str] = Query(None, description="Game date YYYY-MM-DD"),
    data_root: Path = DEFAULT_DATA_ROOT,
) -> PropsSummaryResponse:
    """Get high-level props summary for a date."""
    # Parse date
    if date:
        try:
            game_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {date}")
    else:
        game_date = datetime.now(timezone.utc).date()

    # Get all lines to compute summary
    try:
        lines = _get_props_lines_impl(game_date, data_root)
    except HTTPException:
        return PropsSummaryResponse(
            date=game_date.isoformat(),
            total_props=0,
            players_with_props=0,
            props_with_edge=0,
            best_edges=[],
        )

    # Count unique players
    unique_players = set(line.player_id for line in lines)

    # Count props with edge (EV > 3%)
    props_with_edge = sum(
        1
        for line in lines
        if (line.over_ev and line.over_ev > 0.03) or (line.under_ev and line.under_ev > 0.03)
    )

    # Get top 5 edges
    edge_list: list[tuple[str, str, str, float]] = []
    for line in lines:
        if line.over_ev and line.over_ev > 0.03:
            edge_list.append((line.player_name, line.prop_type, "over", line.over_ev))
        if line.under_ev and line.under_ev > 0.03:
            edge_list.append((line.player_name, line.prop_type, "under", line.under_ev))

    edge_list.sort(key=lambda x: x[3], reverse=True)
    best_edges = [
        BestEdge(player=e[0], prop=e[1], side=e[2], ev=e[3]) for e in edge_list[:5]
    ]

    return PropsSummaryResponse(
        date=game_date.isoformat(),
        total_props=len(lines),
        players_with_props=len(unique_players),
        props_with_edge=props_with_edge,
        best_edges=best_edges,
    )

