"""Core contest simulation service."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

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
from .weights import drop_zero_weight_items, scale_integer_weights_to_target

__all__ = [
    "load_worlds_matrix",
    "score_lineups",
    "run_contest_simulation",
]

logger = logging.getLogger(__name__)

WorldsSource = Literal["gtv2", "sim_v2", "auto"]


def _resolve_worlds_root(
    game_date: str,
    *,
    data_root: Path,
    worlds_source: WorldsSource,
) -> Path:
    sim_v2_root = data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2" / f"game_date={game_date}"
    gtv2_root = data_root / "artifacts" / "gtv2_worlds" / f"game_date={game_date}"

    if worlds_source == "gtv2":
        if gtv2_root.exists():
            return gtv2_root
        raise FileNotFoundError(f"No gtv2 worlds data for {game_date} at {gtv2_root}")

    if worlds_source == "sim_v2":
        if sim_v2_root.exists():
            return sim_v2_root
        raise FileNotFoundError(f"No sim_v2 worlds data for {game_date} at {sim_v2_root}")

    if gtv2_root.exists():
        return gtv2_root
    if sim_v2_root.exists():
        return sim_v2_root

    raise FileNotFoundError(f"No worlds data for {game_date} at {gtv2_root} or {sim_v2_root}")


def _resolve_worlds_dir(root_dir: Path, run_id: str | None) -> Path:
    import json

    if run_id:
        candidate = root_dir / f"run={run_id}"
        if candidate.exists():
            return candidate

    pointer = root_dir / "latest_run.json"
    if pointer.exists():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            latest = payload.get("run_id")
        except Exception:
            latest = None
        if latest:
            candidate = root_dir / f"run={latest}"
            if candidate.exists():
                return candidate

    run_dirs = sorted(
        [p for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("run=")],
        reverse=True,
    )
    if run_dirs:
        return run_dirs[0]
    return root_dir


def _load_gtv2_worlds_matrix(worlds_dir: Path) -> Tuple[np.ndarray, Dict[str, int]]:
    worlds_path = worlds_dir / "worlds.parquet"
    if not worlds_path.exists():
        raise FileNotFoundError(f"No worlds.parquet found in {worlds_dir}")

    logger.info("Loading GTV2 worlds from %s", worlds_path)
    df = pd.read_parquet(worlds_path, columns=["world_idx", "player_id", "dk_fpts"])
    if df.empty:
        raise ValueError(f"GTV2 worlds parquet is empty: {worlds_path}")

    df["world_idx"] = pd.to_numeric(df["world_idx"], errors="coerce").astype("Int64")
    df["player_id"] = df["player_id"].astype(str)
    df["dk_fpts"] = pd.to_numeric(df["dk_fpts"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["world_idx"])
    if df.empty:
        raise ValueError(f"GTV2 worlds parquet has no valid world_idx rows: {worlds_path}")

    try:
        player_ids = [str(pid) for pid in sorted(df["player_id"].astype(int).unique().tolist())]
    except Exception:
        player_ids = sorted(df["player_id"].unique().tolist())

    world_ids = sorted(df["world_idx"].astype(int).unique().tolist())
    player_index = {pid: idx for idx, pid in enumerate(player_ids)}
    world_index = {wid: idx for idx, wid in enumerate(world_ids)}
    worlds_matrix = np.zeros((len(world_ids), len(player_ids)), dtype=np.float64)

    row_idx = df["world_idx"].astype(int).map(world_index).to_numpy(dtype=np.int64, copy=False)
    col_idx = df["player_id"].map(player_index).to_numpy(dtype=np.int64, copy=False)
    values = df["dk_fpts"].to_numpy(dtype=np.float64, copy=False)
    worlds_matrix[row_idx, col_idx] = values

    logger.info("Loaded GTV2 worlds matrix shape=%s from %s", worlds_matrix.shape, worlds_path)
    return worlds_matrix, player_index


def load_worlds_matrix(
    game_date: str,
    data_root: Path | None = None,
    run_id: str | None = None,
    worlds_source: WorldsSource = "gtv2",
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
    worlds_source : {"gtv2", "sim_v2", "auto"}
        Which worlds family to load. Live paths should use ``gtv2`` so missing
        generative worlds fail loudly. Backtests may opt into ``sim_v2``.
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

    base_dir = _resolve_worlds_root(game_date, data_root=data_root, worlds_source=worlds_source)
    worlds_dir = _resolve_worlds_dir(base_dir, run_id)

    if "gtv2_worlds" in base_dir.parts or (worlds_dir / "worlds.parquet").exists():
        return _load_gtv2_worlds_matrix(worlds_dir)

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
        logger.info("Aggregating %d world files from %s", len(world_files), worlds_dir)
        # Load first file to resolve schema + FPTS column.
        first_df = pd.read_parquet(world_files[0])

        # Determine FPTS column
        fpts_col = "dk_fpts_world"
        if fpts_col not in first_df.columns:
            for alt in ["fpts_world", "dk_fpts", "fpts"]:
                if alt in first_df.columns:
                    fpts_col = alt
                    break
            else:
                raise ValueError(f"No FPTS column found in world files. Available: {list(first_df.columns)}")

        # Build a union of player_ids across all world files. Older runs often emit sparse
        # per-world files (only players who played / received minutes), so using just the
        # first world's player_ids silently drops players present in later worlds.
        player_id_set: set[str] = set()
        for world_file in world_files:
            ids = pd.read_parquet(world_file, columns=["player_id"])["player_id"]
            player_id_set.update(ids.astype(str).tolist())
        if not player_id_set:
            raise ValueError(f"No player_id values found while aggregating world files under {worlds_dir}")

        # Deterministic ordering (prefer numeric sort when possible).
        try:
            player_ids = [str(pid) for pid in sorted((int(p) for p in player_id_set))]
        except Exception:
            player_ids = sorted(player_id_set)
        player_index = {pid: idx for idx, pid in enumerate(player_ids)}

        n_worlds = len(world_files)
        n_players = len(player_ids)
        worlds_matrix = np.zeros((n_worlds, n_players), dtype=np.float64)

        any_positive = False
        for world_idx, world_file in enumerate(world_files):
            df = pd.read_parquet(world_file, columns=["player_id", fpts_col])
            if df.empty:
                continue
            df["player_id"] = df["player_id"].astype(str)
            df[fpts_col] = pd.to_numeric(df[fpts_col], errors="coerce").fillna(0.0).astype(float)
            if not any_positive and (df[fpts_col] > 0).any():
                any_positive = True

            cols = df["player_id"].map(player_index)
            valid = cols.notna()
            if not valid.any():
                continue
            col_idx = cols[valid].astype(int).to_numpy()
            worlds_matrix[world_idx, col_idx] = df.loc[valid, fpts_col].to_numpy(dtype=np.float64, copy=False)

        max_val = float(np.max(worlds_matrix)) if worlds_matrix.size else 0.0
        if any_positive and max_val <= 0.0:
            raise RuntimeError(
                "Aggregated worlds matrix is all zeros despite positive per-world values. "
                "This usually indicates a player_id dtype mismatch or an unexpected world file schema."
            )

        logger.info("Aggregated %d world files, shape=%s", n_worlds, worlds_matrix.shape)
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

    # Synthetic worlds are a last-resort fallback. Treat DNP/inactive as 0 via play_prob
    # by sampling conditional-on-playing scores then zeroing out inactive worlds.
    if "dk_fpts_mean" not in proj_df.columns:
        raise FileNotFoundError(f"projections.parquet missing dk_fpts_mean at {proj_path}")

    means_cond = proj_df["dk_fpts_mean"].values.astype(np.float64)
    means_cond = np.nan_to_num(means_cond, nan=0.0)
    
    # Use dk_fpts_std if available, otherwise estimate from p90/p10 or use 20% of mean
    if "dk_fpts_std" in proj_df.columns:
        stds = proj_df["dk_fpts_std"].values.astype(np.float64)
        stds = np.where(np.isnan(stds), means_cond * 0.2, stds)
    elif "dk_fpts_p90" in proj_df.columns and "dk_fpts_p10" in proj_df.columns:
        p10 = np.nan_to_num(proj_df["dk_fpts_p10"].values, nan=0.0)
        p90 = np.nan_to_num(proj_df["dk_fpts_p90"].values, nan=0.0)
        # For normal dist, p90 - p10 ≈ 2.56 * std
        stds = np.maximum((p90 - p10) / 2.56, means_cond * 0.1).astype(np.float64)
    else:
        stds = (means_cond * 0.2).astype(np.float64)
    
    # Ensure minimum std to avoid degenerate distributions
    stds = np.maximum(stds, 1.0)

    # Generate synthetic worlds
    rng = np.random.default_rng(seed)
    n_players = len(player_ids)
    
    # Sample from normal distribution for each player across all worlds
    # Shape: (n_synthetic_worlds, n_players)
    if "play_prob" in proj_df.columns:
        play_prob = pd.to_numeric(proj_df["play_prob"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
        play_prob = np.clip(play_prob, 0.0, 1.0)
    else:
        play_prob = np.ones(n_players, dtype=np.float64)
    active_mask = rng.random(size=(n_synthetic_worlds, n_players)) < play_prob[np.newaxis, :]
    sampled = rng.normal(
        loc=means_cond[np.newaxis, :],
        scale=stds[np.newaxis, :],
        size=(n_synthetic_worlds, n_players),
    )
    
    # Clip to reasonable bounds (no negative FPTS, cap at ~4 std above mean_cond).
    sampled = np.maximum(sampled, 0.0)
    sampled = np.minimum(sampled, means_cond + 4 * stds)
    worlds_matrix = np.where(active_mask, sampled, 0.0)

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
    n_worlds = int(worlds_matrix.shape[0])

    lineup_scores = np.zeros((n_lineups, n_worlds), dtype=np.float64)

    # Fast path: convert each lineup to column indices once, then use NumPy gathers.
    # We intentionally avoid a fully-vectorized 3D gather (worlds x lineups x 8)
    # to keep memory bounded for K=1k–10k and W=10k.
    for lu_idx, lineup in enumerate(lineups):
        indices: List[int] = []
        missing: List[str] = []
        for pid in lineup:
            pid_str = str(pid).strip()
            if not pid_str:
                continue
            col = player_index.get(pid_str)
            if col is None:
                missing.append(pid_str)
                continue
            indices.append(int(col))

        if missing:
            # Log once per lineup (not once per player per lineup).
            logger.warning("Players not found in worlds matrix: %s", ",".join(missing))
        if not indices:
            continue

        cols = np.asarray(indices, dtype=np.int64)
        lineup_scores[lu_idx, :] = np.take(worlds_matrix, cols, axis=1).sum(axis=1)

    return lineup_scores


def run_contest_simulation(
    *,
    user_lineups: List[List[str]],
    game_date: str,
    run_id: str | None = None,
    archetype: str = "medium",
    field_size_bucket: str = "medium",
    field_size_override: int | None = None,
    entry_fee: float = 3.0,
    user_weights: List[int] | None = None,
    field_lineups: List[List[str]] | None = None,
    field_weights: List[int] | None = None,
    data_root: Path | None = None,
    player_ownership: Optional[Dict[str, float]] = None,
    entry_max: int = 150,
    ownership_mode: str = "full",
    rank_mode: str = "current",
    worlds_source: WorldsSource = "gtv2",
) -> ContestSimResult:
    """Run a contest simulation of user lineups against an opponent field.

    Parameters
    ----------
    user_lineups : List[List[str]]
        List of user lineups to simulate, each a list of player_ids.
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
    user_weights : List[int] | None
        Entry counts per user lineup (default: 1 each).
    field_lineups : List[List[str]] | None
        Opponent field lineups. When omitted, defaults to self-play (field lineups
        are the same as user lineups).
    field_weights : List[int] | None
        Entry counts per field lineup. If provided, will be scaled so that
        ``sum(field_weights) + sum(user_weights) == field_size``.
    data_root : Path | None
        Data root directory
    player_ownership : Optional[Dict[str, float]]
        Player ID -> ownership % mapping for dupe penalty calculation.
        If None, dupe penalties are not applied.
    entry_max : int
        Max entries per user (for dupe penalty binning)
    ownership_mode : str
        Ownership usage: full|dupe_only|field_only|off.
        - full/dupe_only: allow ownership-based dupe penalties when player_ownership is provided
        - field_only/off: dupe penalties disabled (treated as 1.0)
    rank_mode : str
        select_score formulation:
          - current: tail_score - (1 - dupe_penalty) * mean  (existing behavior)
          - tail_only: tail_score
          - tail_times_dupe: tail_score * dupe_penalty
    worlds_source : {"gtv2", "sim_v2", "auto"}
        Which worlds family to use for lineup scoring.

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

    n_user_lineups = len(user_lineups)
    if n_user_lineups <= 0:
        raise ValueError("user_lineups must be non-empty")

    # Set user weights (default 1 per lineup)
    if user_weights is None:
        user_weights = [1] * n_user_lineups
    if len(user_weights) != n_user_lineups:
        raise ValueError("user_weights length must match user_lineups length")
    if any(w < 0 for w in user_weights):
        raise ValueError("user_weights must be non-negative integers")

    user_total_entries = int(sum(max(0, int(w)) for w in user_weights))
    if user_total_entries <= 0:
        raise ValueError("Sum of user_weights must be positive")

    # Used for dupe-penalty logic: if the caller provides an explicit opponent
    # field, exact lineup duplication can be modeled directly via field weights.
    field_provided = field_lineups is not None

    # Self-play default: field is drawn from the user lineup set.
    if field_lineups is None:
        field_lineups = user_lineups
        base_field_weights = user_weights if field_weights is None else field_weights
    else:
        base_field_weights = field_weights if field_weights is not None else [1] * len(field_lineups)

    if len(base_field_weights) != len(field_lineups):
        raise ValueError("field_weights length must match field_lineups length")
    if any(w < 0 for w in base_field_weights):
        raise ValueError("field_weights must be non-negative integers")

    target_field_entries = int(field_size - user_total_entries)
    if target_field_entries <= 0:
        raise ValueError(
            f"User entries ({user_total_entries}) must be less than contest field_size ({field_size})"
        )

    scaled_field_weights = scale_integer_weights_to_target(
        base_field_weights,
        target_field_entries,
        min_weight=1 if target_field_entries >= len(field_lineups) else 0,
    )
    field_lineups, scaled_field_weights = drop_zero_weight_items(field_lineups, scaled_field_weights)
    if not field_lineups or sum(scaled_field_weights) <= 0:
        raise ValueError("Field weights must sum to a positive value after scaling")

    # Load worlds and score both user + field lineups
    sim_run_id: str | None = None
    try:
        # Prefer the sim_run_id referenced by the unified projections bundle so the
        # contest sim uses the same worlds as the projections shown/optimized.
        from projections.projections_bundle import load_unified_projections_df

        bundle = load_unified_projections_df(game_date, run_id=run_id, data_root=data_root)
        if "sim_run_id" in bundle.df.columns:
            vals = bundle.df["sim_run_id"].dropna().astype(str).unique().tolist()
            sim_run_id = vals[0] if len(vals) == 1 else None
    except Exception:
        sim_run_id = None

    worlds_matrix, player_index = load_worlds_matrix(
        game_date,
        data_root,
        run_id=sim_run_id or run_id,
        worlds_source=worlds_source,
    )
    user_scores = score_lineups(user_lineups, worlds_matrix, player_index)
    field_scores = score_lineups(field_lineups, worlds_matrix, player_index)

    n_worlds = worlds_matrix.shape[0]

    # Generate payout tiers
    defaults = config.get("defaults", {})
    rake = float(defaults.get("rake", 0.15))
    payout_tiers = generate_payout_tiers(archetype, field_size, entry_fee, config)

    contest_config = ContestConfig(
        field_size=field_size,
        entry_fee=entry_fee,
        archetype=archetype,
        rake=rake,
    )

    logger.info(
        "Running contest sim: user_lineups=%d field_lineups=%d worlds=%d field_size=%d archetype=%s",
        n_user_lineups,
        len(field_lineups),
        n_worlds,
        field_size,
        archetype,
    )

    payout_result = compute_expected_user_payouts_vectorized(
        user_scores=user_scores,
        field_scores=field_scores,
        user_weights=user_weights,
        field_weights=scaled_field_weights,
        payout_tiers=payout_tiers,
        workers=min(22, os.cpu_count() or 1),
        compute_field_side=False,
    )

    # Compute score distribution stats per lineup
    lineup_means = user_scores.mean(axis=1)
    lineup_stds = user_scores.std(axis=1)
    lineup_p90 = np.percentile(user_scores, 90, axis=1)
    lineup_p95 = np.percentile(user_scores, 95, axis=1)
    lineup_p10 = np.percentile(user_scores, 10, axis=1)

    # Compute UCVaR90: mean of top 10% scores (above p90) for each lineup
    # Vectorized: for each lineup, mask scores >= p90 and compute mean
    lineup_ucv90 = np.zeros(n_user_lineups, dtype=np.float64)
    lineup_cvar10 = np.zeros(n_user_lineups, dtype=np.float64)
    lineup_lcb95 = lineup_means - 1.96 * lineup_stds
    for idx in range(n_user_lineups):
        scores = user_scores[idx]
        q90 = lineup_p90[idx]
        q10 = lineup_p10[idx]
        tail_mask = scores >= q90
        downside_mask = scores <= q10
        if tail_mask.any():
            lineup_ucv90[idx] = scores[tail_mask].mean()
        else:
            lineup_ucv90[idx] = q90  # fallback if no scores above threshold
        if downside_mask.any():
            lineup_cvar10[idx] = scores[downside_mask].mean()
        else:
            lineup_cvar10[idx] = q10

    # Compute dupe penalties if ownership data is provided.
    #
    # Note: the payout engine already tie-splits when a user lineup matches a
    # lineup present in the modeled field (field_equal_weight > 0). In that case
    # we disable the ownership-based dupe penalty to avoid double-counting.
    ownership_mode_n = str(ownership_mode or "full").strip().lower()
    rank_mode_n = str(rank_mode or "current").strip().lower()
    if ownership_mode_n == "off":
        rank_mode_n = "tail_only"

    use_dupe_penalty = ownership_mode_n in {"full", "dupe_only"} and bool(player_ownership)
    if use_dupe_penalty:
        logger.info("Computing dupe penalties with ownership data")
        dupe_penalties = compute_batch_dupe_penalties(
            lineups=user_lineups,
            player_ownership=player_ownership,
            field_size=field_size,
            entry_max=entry_max,
        )
        dupe_penalty_disabled_for_matches = 0
        if field_provided:
            field_key_to_weight: Dict[Tuple[str, ...], int] = {}
            for lineup, weight in zip(field_lineups, scaled_field_weights):
                key = tuple(sorted(str(p).strip() for p in lineup if str(p).strip()))
                if not key:
                    continue
                field_key_to_weight[key] = field_key_to_weight.get(key, 0) + int(weight)

            for idx, lineup in enumerate(user_lineups):
                key = tuple(sorted(str(p).strip() for p in lineup if str(p).strip()))
                if field_key_to_weight.get(key, 0) > 0:
                    dupe_penalties[idx] = 1.0
                    dupe_penalty_disabled_for_matches += 1
    else:
        # No penalty if ownership not provided
        dupe_penalties = [1.0] * n_user_lineups
        dupe_penalty_disabled_for_matches = 0

    # Build results
    # Tail score config (could be exposed as params if needed)
    tail_weight_p90 = 0.6
    tail_weight_ucv = 0.4

    results: List[LineupEVResult] = []
    for idx in range(n_user_lineups):
        unadjusted_payout = float(payout_result.expected_payouts[idx])
        dupe_penalty = dupe_penalties[idx]
        expected_payout = unadjusted_payout * dupe_penalty

        p90_val = float(lineup_p90[idx])
        ucv90_val = float(lineup_ucv90[idx])
        cvar10_val = float(lineup_cvar10[idx])
        lcb95_val = float(lineup_lcb95[idx])
        robust_floor_val = min(lcb95_val, cvar10_val)
        tail_score_val = tail_weight_p90 * p90_val + tail_weight_ucv * ucv90_val
        # select_score subtracts the "penalty" portion: (1 - dupe_penalty) scaled
        # We use the raw tail_score - penalty_impact where penalty_impact reflects
        # the fractional reduction. The dupe_penalty is in [0,1] where 1=no penalty.
        # A dupe_penalty of 0.8 means we lose 20% due to duplication.
        # We scale the penalty by mean FPTS to make it comparable to tail_score units.
        if rank_mode_n == "tail_only":
            select_score_val = tail_score_val
        elif rank_mode_n == "tail_times_dupe":
            select_score_val = tail_score_val * float(dupe_penalty)
        elif rank_mode_n == "current":
            penalty_impact = (1.0 - dupe_penalty) * float(lineup_means[idx])
            select_score_val = tail_score_val - penalty_impact
        else:
            raise ValueError(f"Invalid rank_mode: {rank_mode!r}")

        result = LineupEVResult(
            lineup_id=idx,
            player_ids=user_lineups[idx],
            mean=float(lineup_means[idx]),
            std=float(lineup_stds[idx]),
            p90=p90_val,
            p95=float(lineup_p95[idx]),
            expected_payout=expected_payout,
            expected_value=expected_payout - entry_fee,
            roi=(expected_payout - entry_fee) / entry_fee if entry_fee > 0 else 0.0,
            win_rate=float(payout_result.win_rates[idx]),
            top_1pct_rate=float(payout_result.top_1pct_rates[idx]),
            top_5pct_rate=float(payout_result.top_5pct_rates[idx]),
            top_10pct_rate=float(payout_result.top_10pct_rates[idx]),
            cash_rate=float(payout_result.cash_rates[idx]),
            dupe_penalty=dupe_penalty,
            unadjusted_expected_payout=unadjusted_payout,
            adjusted_expected_payout=expected_payout,
            ucv90=ucv90_val,
            tail_score=tail_score_val,
            select_score=select_score_val,
            score_lcb95=lcb95_val,
            score_cvar10=cvar10_val,
            robust_floor=robust_floor_val,
        )
        results.append(result)

    # Compute summary stats
    evs = [r.expected_value for r in results]
    rois = [r.roi for r in results]
    win_rates = [r.win_rate for r in results]
    top1_rates = [r.top_1pct_rate for r in results]

    # Log tail metrics summary for debugging
    ucv90_vals = [r.ucv90 for r in results if r.ucv90 is not None]
    tail_scores = [r.tail_score for r in results if r.tail_score is not None]
    select_scores = [r.select_score for r in results if r.select_score is not None]
    if ucv90_vals:
        logger.info(
            "Tail metrics: ucv90=[%.1f, %.1f, %.1f], tail_score=[%.1f, %.1f, %.1f], "
            "select_score=[%.1f, %.1f, %.1f] (min/med/max)",
            min(ucv90_vals), float(np.median(ucv90_vals)), max(ucv90_vals),
            min(tail_scores), float(np.median(tail_scores)), max(tail_scores),
            min(select_scores), float(np.median(select_scores)), max(select_scores),
        )

    debug_payload: dict[str, object] = {
        "user_total_entries": user_total_entries,
        "field_total_entries": int(sum(scaled_field_weights)),
        "total_entries": int(user_total_entries + sum(scaled_field_weights)),
        "field_unique_k": int(len(field_lineups)),
        "dupe_penalty_disabled_for_field_matches": int(dupe_penalty_disabled_for_matches),
        "ownership_mode": ownership_mode_n,
        "rank_mode": rank_mode_n,
        "worlds_source": worlds_source,
    }

    # Optional DNP diagnostics for selection debugging (off by default).
    # Enable with: PROJECTIONS_CONTEST_SIM_DEBUG_TOPK_ANY_ZERO=1
    if os.environ.get("PROJECTIONS_CONTEST_SIM_DEBUG_TOPK_ANY_ZERO", "").strip().lower() in {"1", "true", "yes"}:
        try:
            top_k = min(int(entry_max), int(n_user_lineups))
            if top_k > 0 and worlds_matrix.size > 0:
                scores_arr = np.asarray(
                    [r.select_score if r.select_score is not None else float("-inf") for r in results],
                    dtype=np.float64,
                )
                if np.isfinite(scores_arr).any():
                    order = np.argsort(-scores_arr)
                    sort_key = "select_score"
                else:
                    ev_arr = np.asarray(evs, dtype=np.float64)
                    order = np.argsort(-ev_arr)
                    sort_key = "expected_value"

                top_idx = order[:top_k]
                zero_rates: list[float] = []
                missing_player_count = 0
                for i in top_idx:
                    lineup = user_lineups[int(i)]
                    raw_pids = [str(pid).strip() for pid in lineup if str(pid).strip()]
                    cols = [player_index.get(pid) for pid in raw_pids]
                    if any(c is None for c in cols):
                        missing_player_count += 1
                    cols_i = np.asarray([int(c) for c in cols if c is not None], dtype=np.int64)
                    if cols_i.size == 0:
                        continue
                    sub = np.take(worlds_matrix, cols_i, axis=1)
                    any_zero = np.any(sub == 0.0, axis=1)
                    zero_rates.append(float(np.mean(any_zero)))

                if zero_rates:
                    debug_payload["top_k"] = int(top_k)
                    debug_payload["top_k_sort_key"] = sort_key
                    debug_payload["top_k_p_any_zero_player"] = {
                        "mean": float(np.mean(zero_rates)),
                        "p50": float(np.median(zero_rates)),
                        "p90": float(np.percentile(zero_rates, 90)),
                        "missing_player_lineups": int(missing_player_count),
                    }
        except Exception as exc:
            logger.warning("Failed to compute DNP diagnostics: %s", exc)

    stats = SummaryStats(
        lineup_count=n_user_lineups,
        worlds_count=n_worlds,
        avg_ev=float(np.mean(evs)) if evs else 0.0,
        avg_roi=float(np.mean(rois)) if rois else 0.0,
        positive_ev_count=sum(1 for ev in evs if ev > 0),
        best_ev_lineup_id=int(np.argmax(evs)) if evs else 0,
        best_win_rate_lineup_id=int(np.argmax(win_rates)) if win_rates else 0,
        best_top1pct_lineup_id=int(np.argmax(top1_rates)) if top1_rates else 0,
        debug=debug_payload,
    )

    return ContestSimResult(
        results=results,
        config=contest_config,
        stats=stats,
    )
