"""Game script prediction for simulator.

Predicts game margin distribution from Vegas spread and samples
scripts per world to shift player minutes quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class GameScriptConfig:
    """Configuration for game script adjustments."""
    
    # Residual standard deviation for margin sampling
    margin_std: float = 13.4
    
    # Spread coefficient (margin ≈ -coef * spread)
    spread_coef: float = -0.726

    # Noise added to per-world quantile targets (in quantile space).
    quantile_noise_std: float = 0.08
    
    # Script thresholds (from team's perspective)
    blowout_threshold: int = 15
    comfortable_threshold: int = 8
    
    # Quantile shifts by script and role
    # Format: {script: (starter_quantile, bench_quantile)}
    # 0.5 = sample at median, 0.7 = sample at p70, etc.
    quantile_targets: dict[str, tuple[float, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.quantile_targets:
            # Target quantiles based on game script
            # Close games: starters draw from higher quantile (more minutes)
            # Blowouts: starters draw from lower quantile (less minutes)
            self.quantile_targets = {
                "blowout_win": (0.35, 0.55),      # Starters rest, bench plays more
                "comfortable_win": (0.45, 0.52),  # Slight reduction for starters
                "close": (0.65, 0.48),            # Starters play more, bench less
                "comfortable_loss": (0.50, 0.50), # Baseline
                "blowout_loss": (0.30, 0.45),     # Starters pulled, bench garbage time
            }
    
    @classmethod
    def from_profile_config(cls, config: dict) -> "GameScriptConfig":
        """Load config from profile game_script dict."""
        quantile_targets = {}
        if "quantile_targets" in config:
            for script, targets in config["quantile_targets"].items():
                starter = targets.get("starter", 0.5)
                bench = targets.get("bench", 0.5)
                quantile_targets[script] = (starter, bench)
        
        return cls(
            margin_std=config.get("margin_std", 13.4),
            spread_coef=config.get("spread_coef", -0.726),
            quantile_noise_std=float(config.get("quantile_noise_std", 0.08)),
            quantile_targets=quantile_targets,
        )


def classify_script(margin: float, config: GameScriptConfig) -> str:
    """Classify game margin into a script category."""
    if margin >= config.blowout_threshold:
        return "blowout_win"
    if margin >= config.comfortable_threshold:
        return "comfortable_win"
    if margin >= -config.comfortable_threshold + 1:
        return "close"
    if margin >= -config.blowout_threshold + 1:
        return "comfortable_loss"
    return "blowout_loss"


def predict_margin_distribution(
    spread: float,
    config: GameScriptConfig,
) -> tuple[float, float]:
    """
    Predict margin distribution parameters.
    
    Args:
        spread: Team's spread (negative = favored)
        config: Game script configuration
        
    Returns:
        (mean_margin, std_margin) for sampling
    """
    mean_margin = config.spread_coef * spread
    return mean_margin, config.margin_std


def sample_minutes_with_scripts(
    minutes_p10: np.ndarray,
    minutes_p50: np.ndarray,
    minutes_p90: np.ndarray,
    is_starter: np.ndarray,
    game_ids: np.ndarray,
    team_ids: np.ndarray,
    spreads_home: np.ndarray,
    home_team_ids: dict[int, int],
    n_worlds: int,
    config: GameScriptConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample minutes per world based on game script.
    
    The game script determines which quantile of the player's minutes
    distribution we sample from. Close games → higher quantile for starters.
    
    Args:
        minutes_p10, p50, p90: Per-player minute quantiles
        is_starter: 1 if starter, 0 if bench
        game_ids, team_ids: Player game/team arrays
        spreads_home: Home team spread for each player's game
        home_team_ids: game_id -> home_team_id mapping
        n_worlds: Number of simulation worlds
        config: Game script configuration
        rng: Random generator
        
    Returns:
        minutes_worlds: shape (n_worlds, n_players)
    """
    n_players = len(minutes_p50)
    
    # Estimate player-level distribution from quantiles.
    #
    # Use an asymmetric split-normal centered at p50:
    #   For q <= 0.5: x = p50 + sigma_low  * z
    #   For q >= 0.5: x = p50 + sigma_high * z
    #
    # where z = Φ^{-1}(q). Choose sigma_low/high so that:
    #   p10 = p50 + sigma_low  * z10
    #   p90 = p50 + sigma_high * z90
    z90 = 1.2815515655446004  # stats.norm.ppf(0.90)
    p50 = minutes_p50
    p10 = np.minimum(minutes_p10, p50)
    p90 = np.maximum(minutes_p90, p50)
    sigma_low = np.maximum((p50 - p10) / z90, 0.5)
    sigma_high = np.maximum((p90 - p50) / z90, 0.5)
    
    # Get unique games and sample margins
    unique_games = {}  # (game_id, team_id) -> team_spread
    for i in range(n_players):
        gid = int(game_ids[i])
        tid = int(team_ids[i])
        spread_home = spreads_home[i]
        
        if pd.isna(spread_home):
            continue
        
        home_tid = home_team_ids.get(gid)
        is_home = (tid == home_tid) if home_tid is not None else True
        team_spread = spread_home if is_home else -spread_home
        
        key = (gid, tid)
        if key not in unique_games:
            unique_games[key] = team_spread
    
    # Sample margins per game per world
    game_margins = {}  # (game_id, team_id) -> array of margins per world
    for (gid, tid), team_spread in unique_games.items():
        mean_margin = config.spread_coef * team_spread
        margins = rng.normal(mean_margin, config.margin_std, size=n_worlds)
        game_margins[(gid, tid)] = margins
    
    # Build per-player per-world target quantiles
    target_quantiles = np.full((n_worlds, n_players), 0.5)  # Default to median
    
    for w in range(n_worlds):
        for i in range(n_players):
            gid = int(game_ids[i])
            tid = int(team_ids[i])
            key = (gid, tid)
            
            if key not in game_margins:
                continue
            
            margin = game_margins[key][w]
            script = classify_script(margin, config)
            starter = is_starter[i]
            
            if script in config.quantile_targets:
                starter_q, bench_q = config.quantile_targets[script]
                target_quantiles[w, i] = starter_q if starter else bench_q
    
    # Add some noise to target quantiles to avoid all same-script players
    # getting identical minutes
    quantile_noise = rng.normal(0, config.quantile_noise_std, size=target_quantiles.shape)
    target_quantiles = np.clip(target_quantiles + quantile_noise, 0.05, 0.95)
    
    # Sample minutes from player distribution at target quantile
    z_scores = stats.norm.ppf(target_quantiles)
    sigma = np.where(z_scores < 0.0, sigma_low[None, :], sigma_high[None, :])
    minutes_worlds = p50[None, :] + sigma * z_scores
    minutes_worlds = np.maximum(minutes_worlds, 0)
    
    return minutes_worlds


__all__ = [
    "GameScriptConfig",
    "classify_script", 
    "predict_margin_distribution",
    "sample_minutes_with_scripts",
]
