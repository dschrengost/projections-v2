"""Core contest simulation service."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from projections.paths import data_path

from .payouts import compute_expected_user_payouts_vectorized
from .payout_generator import generate_payout_tiers, load_config, get_field_size
from .scoring_models import (
    ContestConfig,
    LineupEVResult,
    SummaryStats,
    ContestSimResult,
)
from .dupe_penalty import compute_batch_dupe_penalties

__all__ = [
    "load_worlds_matrix",
    "score_lineups",
    "run_contest_simulation",
]

logger = logging.getLogger(__name__)


def load_worlds_matrix(
    game_date: str,
    data_root: Path | None = None,
    n_synthetic_worlds: int = 10000,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load or generate worlds matrix.

    Attempts to load from consolidated worlds_matrix.parquet first.
    Falls back to individual world files if available.
    If neither exists but projections.parquet has mean/std, generates
    synthetic worlds by sampling from per-player normal distributions.

    Parameters
    ----------
    game_date : str
        Game date in YYYY-MM-DD format
    data_root : Path | None
        Data root directory
    n_synthetic_worlds : int
        Number of worlds to generate if using synthetic mode
    seed : int
        Random seed for synthetic world generation

    Returns
    -------
    Tuple[np.ndarray, Dict[str, int]]
        worlds_matrix: (n_worlds, n_players) FPTS matrix
        player_index: {player_id -> column index}
    """
    if data_root is None:
        data_root = data_path()

    worlds_dir = data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2" / f"game_date={game_date}"

    if not worlds_dir.exists():
        raise FileNotFoundError(f"No worlds data for {game_date} at {worlds_dir}")

    # Try consolidated matrix first
    matrix_path = worlds_dir / "worlds_matrix.parquet"
    if matrix_path.exists():
        logger.info(f"Loading consolidated worlds matrix from {matrix_path}")
        matrix_df = pd.read_parquet(matrix_path)
        player_ids = list(matrix_df.columns)
        player_index = {str(pid): idx for idx, pid in enumerate(player_ids)}
        worlds_matrix = matrix_df.values.astype(np.float64)
        logger.info(f"Loaded worlds matrix shape={worlds_matrix.shape}")
        
        # Validating worlds matrix data
        max_val = np.max(worlds_matrix)
        if max_val > 1000:
            logger.error(f"CORRUPTION DETECTED in consolidated matrix: max value is {max_val}")
            
            # Identify bad columns
            col_maxes = np.max(worlds_matrix, axis=0)
            bad_col_indices = np.where(col_maxes > 1000)[0]
            
            # Reverse map to player IDs for logging
            idx_to_pid = {v: k for k, v in player_index.items()}
            for idx in bad_col_indices:
                pid = idx_to_pid.get(idx, "unknown")
                logger.error(f"Corrupted Player {pid} (idx={idx}): max={col_maxes[idx]}")
                
            # Clamp to reasonable max (e.g. 500)
            logger.warning("Clamping worlds matrix to max 500.0")
            worlds_matrix = np.minimum(worlds_matrix, 500.0)
            
        return worlds_matrix, player_index

    # Try aggregating individual world files
    world_files = sorted(worlds_dir.glob("world=*.parquet"))

    if world_files:
        logger.info(f"Aggregating {len(world_files)} world files from {worlds_dir}")
        # Load first file to get player index
        first_df = pd.read_parquet(world_files[0])
        player_ids = first_df["player_id"].astype(str).tolist()
        player_index = {pid: idx for idx, pid in enumerate(player_ids)}

        # Determine FPTS column
        fpts_col = "dk_fpts_world"
        if fpts_col not in first_df.columns:
            for alt in ["fpts_world", "dk_fpts", "fpts"]:
                if alt in first_df.columns:
                    fpts_col = alt
                    break
            else:
                raise ValueError(f"No FPTS column found in world files. Available: {list(first_df.columns)}")

        # Aggregate all worlds
        n_worlds = len(world_files)
        n_players = len(player_ids)
        worlds_matrix = np.zeros((n_worlds, n_players), dtype=np.float64)

        for idx, world_file in enumerate(world_files):
            df = pd.read_parquet(world_file)
            df = df.set_index("player_id")
            for pid_idx, pid in enumerate(player_ids):
                if str(pid) in df.index:
                    worlds_matrix[idx, pid_idx] = float(df.loc[str(pid), fpts_col])

        logger.info(f"Aggregated {n_worlds} world files, shape={worlds_matrix.shape}")
        return worlds_matrix, player_index

    # Fall back to generating synthetic worlds from projections.parquet
    proj_path = worlds_dir / "projections.parquet"
    if not proj_path.exists():
        raise FileNotFoundError(
            f"No world files or projections.parquet found in {worlds_dir}"
        )

    logger.info(f"Generating {n_synthetic_worlds} synthetic worlds from {proj_path}")
    proj_df = pd.read_parquet(proj_path)

    # Get player IDs and stats
    player_ids = proj_df["player_id"].astype(str).tolist()
    player_index = {pid: idx for idx, pid in enumerate(player_ids)}

    # Get mean and std for each player
    means = proj_df["dk_fpts_mean"].values.astype(np.float64)
    means = np.nan_to_num(means, nan=0.0)
    
    # Use dk_fpts_std if available, otherwise estimate from p90/p10 or use 20% of mean
    if "dk_fpts_std" in proj_df.columns:
        stds = proj_df["dk_fpts_std"].values.astype(np.float64)
        stds = np.where(np.isnan(stds), means * 0.2, stds)
    elif "dk_fpts_p90" in proj_df.columns and "dk_fpts_p10" in proj_df.columns:
        p10 = np.nan_to_num(proj_df["dk_fpts_p10"].values, nan=0.0)
        p90 = np.nan_to_num(proj_df["dk_fpts_p90"].values, nan=0.0)
        # For normal dist, p90 - p10 ≈ 2.56 * std
        stds = np.maximum((p90 - p10) / 2.56, means * 0.1).astype(np.float64)
    else:
        stds = (means * 0.2).astype(np.float64)
    
    # Ensure minimum std to avoid degenerate distributions
    stds = np.maximum(stds, 1.0)

    # Generate synthetic worlds
    rng = np.random.default_rng(seed)
    n_players = len(player_ids)
    
    # Sample from normal distribution for each player across all worlds
    # Shape: (n_synthetic_worlds, n_players)
    worlds_matrix = rng.normal(
        loc=means[np.newaxis, :],
        scale=stds[np.newaxis, :],
        size=(n_synthetic_worlds, n_players),
    )
    
    # Clip to reasonable bounds (no negative FPTS, cap at ~3 std above mean)
    worlds_matrix = np.maximum(worlds_matrix, 0.0)
    worlds_matrix = np.minimum(worlds_matrix, means + 4 * stds)

    # Validating worlds matrix data
    max_val = np.max(worlds_matrix)
    if max_val > 1000:
        logger.error(f"CORRUPTION DETECTED: max value in worlds matrix is {max_val}")
        
        # Identify bad columns
        col_maxes = np.max(worlds_matrix, axis=0)
        bad_col_indices = np.where(col_maxes > 1000)[0]
        
        # Reverse map to player IDs for logging
        idx_to_pid = {v: k for k, v in player_index.items()}
        for idx in bad_col_indices:
            pid = idx_to_pid.get(idx, "unknown")
            logger.error(f"Corrupted Player {pid} (idx={idx}): max={col_maxes[idx]}")
            
        # Clamp to reasonable max (e.g. 500)
        logger.warning("Clamping worlds matrix to max 500.0")
        worlds_matrix = np.minimum(worlds_matrix, 500.0)

    logger.info(
        f"Generated synthetic worlds: shape={worlds_matrix.shape}, "
        f"mean range=[{worlds_matrix.mean(axis=0).min():.1f}, {worlds_matrix.mean(axis=0).max():.1f}]"
    )
    return worlds_matrix, player_index


def score_lineups(
    lineups: List[List[str]],
    worlds_matrix: np.ndarray,
    player_index: Dict[str, int],
) -> np.ndarray:
    """Score each lineup across all worlds.

    Parameters
    ----------
    lineups : List[List[str]]
        List of lineups, each a list of 8 player_ids
    worlds_matrix : np.ndarray
        (n_worlds, n_players) FPTS matrix
    player_index : Dict[str, int]
        Mapping from player_id to column index

    Returns
    -------
    np.ndarray
        (n_lineups, n_worlds) matrix of lineup FPTS totals
    """
    n_lineups = len(lineups)
    n_worlds = worlds_matrix.shape[0]

    lineup_scores = np.zeros((n_lineups, n_worlds), dtype=np.float64)

    for lu_idx, lineup in enumerate(lineups):
        for pid in lineup:
            pid_str = str(pid)
            if pid_str in player_index:
                col_idx = player_index[pid_str]
                lineup_scores[lu_idx, :] += worlds_matrix[:, col_idx]
            else:
                logger.warning(f"Player {pid_str} not found in worlds matrix")

    return lineup_scores


def run_contest_simulation(
    lineups: List[List[str]],
    game_date: str,
    archetype: str = "medium",
    field_size_bucket: str = "medium",
    field_size_override: int | None = None,
    entry_fee: float = 3.0,
    weights: List[int] | None = None,
    data_root: Path | None = None,
    player_ownership: Optional[Dict[str, float]] = None,
    entry_max: int = 150,
) -> ContestSimResult:
    """Run full contest simulation.

    Lineups compete against each other (self-play mode).

    Parameters
    ----------
    lineups : List[List[str]]
        List of lineups to simulate, each a list of player_ids
    game_date : str
        Game date in YYYY-MM-DD format
    archetype : str
        Payout archetype: "top_heavy", "medium", "flat"
    field_size_bucket : str
        Field size bucket: "small", "medium", "massive"
    field_size_override : int | None
        Exact field size, overrides bucket default
    entry_fee : float
        Entry fee per lineup
    weights : List[int] | None
        Entry counts per lineup (default: 1 each)
    data_root : Path | None
        Data root directory
    player_ownership : Optional[Dict[str, float]]
        Player ID -> ownership % mapping for dupe penalty calculation.
        If None, dupe penalties are not applied.
    entry_max : int
        Max entries per user (for dupe penalty binning)

    Returns
    -------
    ContestSimResult
        Complete simulation results
    """
    config = load_config()

    # Determine field size
    if field_size_override is not None:
        field_size = field_size_override
    else:
        field_size = get_field_size(field_size_bucket, config)

    # Load worlds and score lineups
    worlds_matrix, player_index = load_worlds_matrix(game_date, data_root)
    lineup_scores = score_lineups(lineups, worlds_matrix, player_index)

    n_lineups = len(lineups)
    n_worlds = worlds_matrix.shape[0]

    # Set weights (default 1 per lineup)
    if weights is None:
        weights = [1] * n_lineups

    # Scale weights to match field size
    # In self-play, we treat all lineups as both "user" and "field"
    total_entries = sum(weights)
    if total_entries <= 0:
        total_entries = n_lineups
        weights = [1] * n_lineups

    scale_factor = field_size / total_entries
    scaled_weights = [max(1, int(round(w * scale_factor))) for w in weights]

    # Generate payout tiers
    defaults = config.get("defaults", {})
    rake = float(defaults.get("rake", 0.12))
    payout_tiers = generate_payout_tiers(archetype, field_size, entry_fee, config)

    contest_config = ContestConfig(
        field_size=field_size,
        entry_fee=entry_fee,
        archetype=archetype,
        rake=rake,
    )

    logger.info(
        f"Running contest sim: {n_lineups} lineups, {n_worlds} worlds, "
        f"field_size={field_size}, archetype={archetype}"
    )

    # In self-play mode, user and field are the same
    # We use all lineups as both user and field
    payout_result = compute_expected_user_payouts_vectorized(
        user_scores=lineup_scores,
        field_scores=lineup_scores,
        user_weights=scaled_weights,
        field_weights=scaled_weights,
        payout_tiers=payout_tiers,
    )

    # Compute score distribution stats per lineup
    lineup_means = lineup_scores.mean(axis=1)
    lineup_stds = lineup_scores.std(axis=1)
    lineup_p90 = np.percentile(lineup_scores, 90, axis=1)
    lineup_p95 = np.percentile(lineup_scores, 95, axis=1)

    # Compute dupe penalties if ownership data is provided
    if player_ownership:
        logger.info("Computing dupe penalties with ownership data")
        dupe_penalties = compute_batch_dupe_penalties(
            lineups=lineups,
            player_ownership=player_ownership,
            field_size=field_size,
            entry_max=entry_max,
        )
    else:
        # No penalty if ownership not provided
        dupe_penalties = [1.0] * n_lineups

    # Build results
    results: List[LineupEVResult] = []
    for idx in range(n_lineups):
        ev = float(payout_result.expected_payouts[idx])
        dupe_penalty = dupe_penalties[idx]
        adjusted_payout = ev * dupe_penalty
        result = LineupEVResult(
            lineup_id=idx,
            player_ids=lineups[idx],
            mean=float(lineup_means[idx]),
            std=float(lineup_stds[idx]),
            p90=float(lineup_p90[idx]),
            p95=float(lineup_p95[idx]),
            expected_payout=ev,
            expected_value=ev - entry_fee,
            roi=(ev - entry_fee) / entry_fee if entry_fee > 0 else 0.0,
            win_rate=float(payout_result.win_rates[idx]),
            top_1pct_rate=float(payout_result.top_1pct_rates[idx]),
            top_5pct_rate=float(payout_result.top_5pct_rates[idx]),
            top_10pct_rate=float(payout_result.top_10pct_rates[idx]),
            cash_rate=float(payout_result.cash_rates[idx]),
            dupe_penalty=dupe_penalty,
            adjusted_expected_payout=adjusted_payout,
        )
        results.append(result)

    # Compute summary stats
    evs = [r.expected_value for r in results]
    rois = [r.roi for r in results]
    win_rates = [r.win_rate for r in results]
    top1_rates = [r.top_1pct_rate for r in results]

    stats = SummaryStats(
        lineup_count=n_lineups,
        worlds_count=n_worlds,
        avg_ev=float(np.mean(evs)) if evs else 0.0,
        avg_roi=float(np.mean(rois)) if rois else 0.0,
        positive_ev_count=sum(1 for ev in evs if ev > 0),
        best_ev_lineup_id=int(np.argmax(evs)) if evs else 0,
        best_win_rate_lineup_id=int(np.argmax(win_rates)) if win_rates else 0,
        best_top1pct_lineup_id=int(np.argmax(top1_rates)) if top1_rates else 0,
    )

    return ContestSimResult(
        results=results,
        config=contest_config,
        stats=stats,
    )
