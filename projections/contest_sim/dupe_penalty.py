"""Dupe penalty module for contest simulation.

Provides E[1/K] penalty estimation for lineup scoring based on ownership features.
Uses pre-trained model from gold/dupe_model.json.
"""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from projections.paths import data_path

logger = logging.getLogger(__name__)

# Default model path
DEFAULT_MODEL_PATH = data_path() / "gold" / "dupe_model.json"


def compute_e_inv_k(lambda_hat: float) -> float:
    """Compute E[1/K] = (1 - e^(-λ)) / λ for Poisson model.
    
    When K = 1 + D where D ~ Poisson(λ), this gives the expected split factor.
    """
    if lambda_hat <= 0:
        return 1.0
    if lambda_hat < 1e-6:
        return 1.0 - lambda_hat / 2
    return (1 - math.exp(-lambda_hat)) / lambda_hat


@lru_cache(maxsize=1)
def load_dupe_model(model_path: Optional[Path] = None) -> Dict:
    """Load dupe penalty model from JSON.
    
    Returns empty dict if model file doesn't exist (graceful degradation).
    """
    path = model_path or DEFAULT_MODEL_PATH
    if not path.exists():
        logger.warning(f"Dupe model not found at {path}, penalties disabled")
        return {}
    
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dupe model: {e}")
        return {}


def get_field_bucket(field_size: int) -> str:
    """Get field size bucket label."""
    if field_size < 3000:
        return "<3k"
    elif field_size < 10000:
        return "3k-10k"
    elif field_size < 50000:
        return "10k-50k"
    else:
        return "50k+"


def get_entry_max_bucket(entry_max: int) -> str:
    """Get entry max bucket label."""
    if entry_max <= 1:
        return "single"
    elif entry_max <= 20:
        return "low_multi"
    elif entry_max <= 150:
        return "high_multi"
    else:
        return "max_multi"


def get_sum_own_bin(sum_own: float, bin_size: int = 10) -> int:
    """Get sum ownership bin start (e.g., 80 for 80-90%)."""
    return int(sum_own / bin_size) * bin_size


def compute_lineup_dupe_penalty(
    lineup_players: List[str],
    player_ownership: Dict[str, float],
    field_size: int = 10000,
    entry_max: int = 150,
    model: Optional[Dict] = None,
) -> float:
    """Compute E[1/K] dupe penalty for a lineup.
    
    Args:
        lineup_players: List of player IDs or names in the lineup
        player_ownership: Dict mapping player ID/name -> ownership percentage
        field_size: Contest field size
        entry_max: Max entries per user
        model: Pre-loaded model dict (uses cached model if None)
    
    Returns:
        E[1/K] in range (0, 1]. Returns 1.0 if no penalty (unique lineup expected).
    """
    if model is None:
        model = load_dupe_model()
    
    if not model:
        return 1.0  # No penalty if model unavailable
    
    # Compute ownership features
    ownerships = [player_ownership.get(str(p), 0) for p in lineup_players]
    sum_own = sum(ownerships)
    
    # Get bin key
    meta = model.get("_meta", {})
    bin_size = meta.get("bin_size", 10)
    
    field_bucket = get_field_bucket(field_size)
    entry_max_bucket = get_entry_max_bucket(entry_max)
    sum_own_bin = get_sum_own_bin(sum_own, bin_size)
    
    bin_key = f"{field_bucket}|{entry_max_bucket}|{sum_own_bin}"
    
    # Look up bin stats
    bin_stats = model.get(bin_key)
    
    if bin_stats is not None:
        return bin_stats.get("e_inv_k", 1.0)
    
    # Fallback to global if bin not found
    global_e_inv_k = meta.get("global_e_inv_k", 1.0)
    logger.debug(f"Bin {bin_key} not found, using global E[1/K]={global_e_inv_k:.3f}")
    return global_e_inv_k


def compute_batch_dupe_penalties(
    lineups: List[List[str]],
    player_ownership: Dict[str, float],
    field_size: int = 10000,
    entry_max: int = 150,
    model_path: Optional[Path] = None,
) -> List[float]:
    """Compute E[1/K] penalties for multiple lineups.
    
    Args:
        lineups: List of lineups, each a list of player IDs
        player_ownership: Dict mapping player ID -> ownership percentage
        field_size: Contest field size
        entry_max: Max entries per user
        model_path: Optional path to model JSON
    
    Returns:
        List of E[1/K] penalties, same length as lineups
    """
    model = load_dupe_model(model_path)
    
    penalties = []
    for lineup in lineups:
        penalty = compute_lineup_dupe_penalty(
            lineup_players=lineup,
            player_ownership=player_ownership,
            field_size=field_size,
            entry_max=entry_max,
            model=model,
        )
        penalties.append(penalty)
    
    return penalties
