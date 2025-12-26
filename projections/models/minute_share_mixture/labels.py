"""Labels module for minute share mixture model.

Defines the 5 minutes states for classification:
- S0: 0 minutes (DNP)
- S1: 1-10 minutes (garbage time)
- S2: 10-20 minutes (fringe rotation)
- S3: 20-30 minutes (core rotation)
- S4: 30+ minutes (starters)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# State boundaries (upper bound, exclusive for S1-S4)
STATE_BOUNDARIES = [0, 10, 20, 30, np.inf]  # S0 is exactly 0

# State names for display
STATE_NAMES = {
    0: "dnp",
    1: "garbage_time",    # 1-10
    2: "fringe_rotation", # 10-20
    3: "core_rotation",   # 20-30
    4: "starters",        # 30+
}

NUM_STATES = 5


def minutes_to_state(minutes: np.ndarray | pd.Series | float) -> np.ndarray | int:
    """Convert minutes to discrete state labels.
    
    The states are designed to target the 10-30 minute range where
    share-based allocators have the most trouble.
    
    Args:
        minutes: Raw minutes values (scalar, array, or series)
        
    Returns:
        State labels 0-4:
        - 0: DNP (exactly 0 minutes)
        - 1: 1-10 minutes
        - 2: 10-20 minutes
        - 3: 20-30 minutes
        - 4: 30+ minutes
    """
    is_scalar = np.isscalar(minutes)
    m = np.atleast_1d(np.asarray(minutes, dtype=np.float64))
    
    # Initialize all to S0 (DNP)
    states = np.zeros(len(m), dtype=np.int32)
    
    # S1: 0 < minutes <= 10
    states[(m > 0) & (m <= 10)] = 1
    
    # S2: 10 < minutes <= 20
    states[(m > 10) & (m <= 20)] = 2
    
    # S3: 20 < minutes <= 30
    states[(m > 20) & (m <= 30)] = 3
    
    # S4: minutes > 30
    states[m > 30] = 4
    
    if is_scalar:
        return int(states[0])
    return states


def state_to_bucket_name(state: int) -> str:
    """Convert state index to human-readable bucket name.
    
    Args:
        state: State index 0-4
        
    Returns:
        Human-readable name like "dnp", "fringe_rotation", etc.
    """
    return STATE_NAMES.get(state, f"unknown_{state}")


def state_to_minute_range(state: int) -> tuple[float, float]:
    """Get the minute range for a given state.
    
    Args:
        state: State index 0-4
        
    Returns:
        Tuple of (min_minutes, max_minutes) for the state.
        For S0 (DNP), returns (0, 0).
        For S4 (starters), returns (30, 48) for practical purposes.
    """
    if state == 0:
        return (0.0, 0.0)
    elif state == 1:
        return (0.0, 10.0)  # Note: excludes exactly 0
    elif state == 2:
        return (10.0, 20.0)
    elif state == 3:
        return (20.0, 30.0)
    elif state == 4:
        return (30.0, 48.0)  # Practical max
    else:
        raise ValueError(f"Invalid state: {state}")


def get_state_counts(minutes: np.ndarray | pd.Series) -> dict[int, int]:
    """Count players in each state.
    
    Args:
        minutes: Array of minute values
        
    Returns:
        Dict mapping state index to count
    """
    states = minutes_to_state(minutes)
    counts = {}
    for s in range(NUM_STATES):
        counts[s] = int((states == s).sum())
    return counts
