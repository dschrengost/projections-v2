"""Generate payout tiers from archetype configuration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import yaml

from .scoring_models import PayoutTier

__all__ = ["generate_payout_tiers", "load_config", "get_field_size"]


def load_config(config_path: Path | None = None) -> dict:
    """Load contest sim configuration from YAML."""
    if config_path is None:
        config_path = Path(__file__).parent / "contest_sim.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_field_size(bucket: str, config: dict | None = None) -> int:
    """Get field size for a given bucket value or label.
    
    The bucket can be the numeric value as string (e.g., '5000') or the label.
    """
    if config is None:
        config = load_config()
    
    # field_sizes is a list of dicts with 'label' and 'value' keys
    field_sizes = config.get("field_sizes", [])
    
    # Try to match by value or label
    for fs in field_sizes:
        if str(fs.get("value")) == bucket or fs.get("label") == bucket:
            return int(fs.get("value", 25000))
    
    # If bucket looks like a number, use it directly
    try:
        return int(bucket)
    except (ValueError, TypeError):
        pass
    
    # Default fallback
    return 25000


def generate_payout_tiers(
    archetype_name: str,
    field_size: int,
    entry_fee: float = 3.0,
    config: dict | None = None,
) -> List[PayoutTier]:
    """Generate realistic DK-style payout tiers from archetype config.

    Parameters
    ----------
    archetype_name : str
        Archetype key: "top_heavy", "medium", or "flat"
    field_size : int
        Total number of entries in the contest
    entry_fee : float
        Entry fee per lineup
    config : dict | None
        Config dict, or None to load from default YAML

    Returns
    -------
    List[PayoutTier]
        Ordered list of payout tiers from 1st to last paying place
    """
    if config is None:
        config = load_config()

    defaults = config.get("defaults", {})
    rake = float(defaults.get("rake", 0.15))

    # payout_archetypes is a list of dicts with 'name', 'field_paid_pct', 'payout_table'
    archetypes = config.get("payout_archetypes", [])
    
    # Find matching archetype by name
    archetype = None
    for arch in archetypes:
        if arch.get("name") == archetype_name:
            archetype = arch
            break
    
    # Fallback to first archetype if not found
    if archetype is None and archetypes:
        archetype = archetypes[0]
    
    if archetype is None:
        # Use hard-coded defaults if no config
        archetype = {"first_place_pct": 0.20, "field_paid_pct": 20.0}
    
    # Use payout_table from config if available, otherwise generate
    if "payout_table" in archetype:
        raw_tiers = archetype["payout_table"]
        prize_pool = field_size * entry_fee * (1 - rake)

        # Calculate total reference payouts for scaling (only tiers that apply)
        reference_total = sum(
            tier["payout"] * (min(tier["end_place"], field_size) - tier["start_place"] + 1)
            for tier in raw_tiers
            if tier["start_place"] <= field_size
        )

        if reference_total <= 0:
            reference_total = 1.0  # Prevent division by zero

        # Scale payouts proportionally to actual prize pool
        tiers: List[PayoutTier] = []
        for tier in raw_tiers:
            if tier["start_place"] > field_size:
                continue  # Skip tiers beyond field size
            tiers.append(PayoutTier(
                start_place=tier["start_place"],
                end_place=min(tier["end_place"], field_size),
                payout=float(tier["payout"]) * (prize_pool / reference_total),
            ))
        return tiers

    first_place_pct = float(archetype.get("first_place_pct", 0.20))
    itm_pct = float(archetype.get("field_paid_pct", 20.0)) / 100.0
    decay_rate = float(archetype.get("decay_rate", 2.0))
    decay_rate = float(archetype.get("decay_rate", 2.0))

    # Calculate prize pool
    prize_pool = field_size * entry_fee * (1 - rake)

    # Number of places that pay
    itm_count = max(1, int(math.ceil(field_size * itm_pct)))

    # Generate tiers using exponential decay
    tiers: List[PayoutTier] = []

    if itm_count == 1:
        # Only 1st place pays
        tiers.append(PayoutTier(start_place=1, end_place=1, payout=prize_pool))
        return tiers

    # First place payout
    first_payout = prize_pool * first_place_pct
    remaining_pool = prize_pool - first_payout
    tiers.append(PayoutTier(start_place=1, end_place=1, payout=round(first_payout, 2)))

    # Generate decay curve for remaining places
    # Using power law decay: payout[i] = base * (1 / i^decay_rate)
    decay_weights = []
    for i in range(1, itm_count):
        weight = 1.0 / (i ** (1.0 / decay_rate))
        decay_weights.append(weight)

    total_weight = sum(decay_weights)
    if total_weight <= 0:
        total_weight = 1.0

    # Assign payouts to remaining positions
    payouts = []
    for weight in decay_weights:
        payout = (weight / total_weight) * remaining_pool
        payouts.append(payout)

    # Group similar payouts into tiers for efficiency
    current_payout = None
    tier_start = 2
    tier_end = 2

    for i, payout in enumerate(payouts):
        place = i + 2  # Places 2 through itm_count
        rounded_payout = _round_payout(payout)

        if current_payout is None:
            current_payout = rounded_payout
            tier_start = place
            tier_end = place
        elif abs(rounded_payout - current_payout) < 0.01:
            # Same payout, extend tier
            tier_end = place
        else:
            # Different payout, close current tier and start new one
            tiers.append(PayoutTier(
                start_place=tier_start,
                end_place=tier_end,
                payout=current_payout,
            ))
            current_payout = rounded_payout
            tier_start = place
            tier_end = place

    # Close final tier
    if current_payout is not None:
        tiers.append(PayoutTier(
            start_place=tier_start,
            end_place=tier_end,
            payout=current_payout,
        ))

    return tiers


def _round_payout(payout: float) -> float:
    """Round payout to realistic DK-style amounts."""
    if payout >= 100:
        return round(payout / 5) * 5  # Round to nearest $5
    elif payout >= 10:
        return round(payout, 0)  # Round to nearest $1
    elif payout >= 1:
        return round(payout, 2)  # Round to nearest cent
    else:
        return round(payout, 2)
