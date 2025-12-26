from __future__ import annotations

import pandas as pd

from projections.cli.score_minutes_v1 import _derive_rotalloc_minutes_tails


def test_rotalloc_tails_recenter_legacy_deltas() -> None:
    base_p10 = pd.Series([30.0, 10.0], index=[1, 2])
    base_p50 = pd.Series([35.0, 12.0], index=[1, 2])
    base_p90 = pd.Series([40.0, 18.0], index=[1, 2])
    new_p50 = pd.Series([37.0, 14.0], index=[1, 2])
    cap_eff = pd.Series([48.0, 48.0], index=[1, 2])

    p10_new, p90_new = _derive_rotalloc_minutes_tails(
        base_p10=base_p10,
        base_p50=base_p50,
        base_p90=base_p90,
        new_p50=new_p50,
        cap_eff=cap_eff,
    )

    # Preserve legacy deltas around p50: (p50-p10)=5 and (p90-p50)=5.
    assert float(p10_new.loc[1]) == 32.0
    assert float(p90_new.loc[1]) == 42.0

    assert float(p10_new.loc[2]) == 12.0
    assert float(p90_new.loc[2]) == 20.0


def test_rotalloc_tails_clamped_to_effective_cap() -> None:
    base_p10 = pd.Series([10.0])
    base_p50 = pd.Series([20.0])
    base_p90 = pd.Series([60.0])  # very wide legacy tail
    new_p50 = pd.Series([40.0])
    cap_eff = pd.Series([48.0])

    p10_new, p90_new = _derive_rotalloc_minutes_tails(
        base_p10=base_p10,
        base_p50=base_p50,
        base_p90=base_p90,
        new_p50=new_p50,
        cap_eff=cap_eff,
    )

    assert float(p10_new.iloc[0]) == 30.0
    assert float(p90_new.iloc[0]) == 48.0


def test_rotalloc_tails_zero_when_new_p50_is_zero() -> None:
    base_p10 = pd.Series([10.0])
    base_p50 = pd.Series([20.0])
    base_p90 = pd.Series([30.0])
    new_p50 = pd.Series([0.0])
    cap_eff = pd.Series([48.0])

    p10_new, p90_new = _derive_rotalloc_minutes_tails(
        base_p10=base_p10,
        base_p50=base_p50,
        base_p90=base_p90,
        new_p50=new_p50,
        cap_eff=cap_eff,
    )

    assert float(p10_new.iloc[0]) == 0.0
    assert float(p90_new.iloc[0]) == 0.0


def test_rotalloc_tails_not_fixed_multipliers() -> None:
    """Verify recentered tails preserve legacy spreads, NOT fixed multipliers (0.7/1.3)."""
    # Setup: legacy model with asymmetric tails (10 below, 15 above p50)
    base_p10 = pd.Series([20.0])  # delta_low = 30 - 20 = 10
    base_p50 = pd.Series([30.0])
    base_p90 = pd.Series([45.0])  # delta_high = 45 - 30 = 15
    new_p50 = pd.Series([25.0])
    cap_eff = pd.Series([48.0])

    p10_new, p90_new = _derive_rotalloc_minutes_tails(
        base_p10=base_p10,
        base_p50=base_p50,
        base_p90=base_p90,
        new_p50=new_p50,
        cap_eff=cap_eff,
    )

    # Expected: preserved deltas around new p50
    # p10_new = 25 - 10 = 15, p90_new = 25 + 15 = 40
    assert float(p10_new.iloc[0]) == 15.0
    assert float(p90_new.iloc[0]) == 40.0

    # Verify NOT fixed multipliers (which would be 0.7*25=17.5, 1.3*25=32.5)
    assert float(p10_new.iloc[0]) != 17.5  # NOT 0.7 * new_p50
    assert float(p90_new.iloc[0]) != 32.5  # NOT 1.3 * new_p50

    # Verify ordering invariants
    assert float(p10_new.iloc[0]) <= float(new_p50.iloc[0])
    assert float(p90_new.iloc[0]) >= float(new_p50.iloc[0])


def test_rotalloc_tails_ordering_preserved() -> None:
    """Verify p10 <= p50 <= p90 invariant is maintained."""
    # Edge case: delta would push p10 negative or p90 above cap
    base_p10 = pd.Series([0.0])
    base_p50 = pd.Series([20.0])  # delta_low = 20
    base_p90 = pd.Series([40.0])  # delta_high = 20
    new_p50 = pd.Series([10.0])  # small new_p50
    cap_eff = pd.Series([25.0])  # constrained cap

    p10_new, p90_new = _derive_rotalloc_minutes_tails(
        base_p10=base_p10,
        base_p50=base_p50,
        base_p90=base_p90,
        new_p50=new_p50,
        cap_eff=cap_eff,
    )

    # p10_new = max(0, 10 - 20) = 0 (clipped to non-negative)
    # p90_new = min(cap, 10 + 20) = min(25, 30) = 25 (clipped to cap)
    assert float(p10_new.iloc[0]) == 0.0
    assert float(p90_new.iloc[0]) == 25.0

    # Ordering invariants
    assert float(p10_new.iloc[0]) <= float(new_p50.iloc[0])
    assert float(p90_new.iloc[0]) >= float(new_p50.iloc[0])


