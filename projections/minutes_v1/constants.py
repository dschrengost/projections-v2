"""Constants shared across the Minutes V1 quick-start pipeline."""

from __future__ import annotations

from enum import Enum


class AvailabilityStatus(str, Enum):
    """Canonicalized availability states for injury snapshots."""

    OUT = "OUT"
    QUESTIONABLE = "Q"
    PROBABLE = "PROB"
    AVAILABLE = "AVAIL"
    UNKNOWN = "UNK"


STATUS_PRIORS: dict[AvailabilityStatus, float] = {
    AvailabilityStatus.OUT: 0.0,
    AvailabilityStatus.QUESTIONABLE: 0.55,
    AvailabilityStatus.PROBABLE: 0.78,
    AvailabilityStatus.AVAILABLE: 0.97,
    AvailabilityStatus.UNKNOWN: 0.82,
}

# Guard/Wing/Big archetype buckets used for depth counts.
ARCHETYPE_MAP: dict[str, str] = {
    "PG": "G",
    "SG": "G",
    "G": "G",
    "SF": "W",
    "PF": "B",
    "PF/C": "B",
    "C": "B",
    "F": "W",
    "F/C": "B",
    "G/F": "W",
}


# Default per-player caps used by the reconciliation module.
ROLE_MINUTES_CAPS: dict[str, float] = {
    "starter": 40.0,
    "bench": 30.0,
    "ramp": 22.0,
}
