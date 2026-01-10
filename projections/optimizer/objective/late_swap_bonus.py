"""Late swap bonus for optimizer objective function.

Provides a small bonus to players in later-starting games to improve late swap
optionality. When two players have similar projections, the optimizer will prefer
the one in the later game since you have more time to react to news/injuries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LateSwapBonusConfig:
    """Configuration for late-starting game bonus in optimizer objective.

    Attributes:
        bonus_per_hour: Points bonus per hour after earliest game on slate.
            Typical values: 0.1-0.3. Higher values more aggressively favor
            late games at the expense of projection accuracy.
        max_bonus: Maximum total bonus to cap extreme late-game bias.
        enabled: Whether the bonus is applied. Off by default.
    """

    bonus_per_hour: float = 0.2
    max_bonus: float = 1.5
    enabled: bool = False

    def validate(self) -> None:
        """Validate configuration values."""
        if self.bonus_per_hour < 0:
            raise ValueError("bonus_per_hour must be >= 0")
        if self.max_bonus < 0:
            raise ValueError("max_bonus must be >= 0")


def build_late_swap_bonus(
    df: pd.DataFrame,
    cfg: LateSwapBonusConfig,
    game_start_col: str = "game_start_utc",
) -> np.ndarray:
    """Build late swap bonus vector for optimizer objective.

    Returns a positive bonus (to ADD to player weights) based on how many hours
    after the earliest game each player's game starts.

    Args:
        df: DataFrame with player data including game start times.
        cfg: Late swap bonus configuration.
        game_start_col: Column name containing game start times (ISO format or datetime).

    Returns:
        numpy array of bonus values (same length as df). Later games get higher values.
        Returns zeros if disabled or game times unavailable.
    """
    cfg.validate()

    n_players = len(df)

    if not cfg.enabled:
        return np.zeros(n_players, dtype=float)

    if game_start_col not in df.columns:
        return np.zeros(n_players, dtype=float)

    # Parse game start times to datetime
    starts = pd.to_datetime(df[game_start_col], errors="coerce", utc=True)

    # Find earliest game on the slate
    valid_starts = starts.dropna()
    if len(valid_starts) == 0:
        return np.zeros(n_players, dtype=float)

    earliest = valid_starts.min()

    # Calculate hours after earliest for each player
    hours_after = (starts - earliest).dt.total_seconds() / 3600.0
    hours_after = hours_after.fillna(0.0).to_numpy()

    # Apply bonus with cap
    bonus = np.clip(hours_after * cfg.bonus_per_hour, 0.0, cfg.max_bonus)

    return bonus
