"""v1.1 Spec 7.1: Feature leakage guardrail test.

This test asserts that the RMH training schema explicitly excludes known leaky columns.
If future refactors re-introduce these columns, this test will fail loudly.
"""
from __future__ import annotations

import pytest

from projections.models.rotation_minutes_hurdle_v1.train import _EXTRA_EXCLUDED_FEATURES


# Spec 7.1: Known leaky columns that must NEVER be used as features.
LEAKY_COLUMNS = {
    # Current-game stint-derived values (leaky in a pre-tip model).
    "minutes_from_stints",
    "team_total_minutes_from_stints",
    "first_in_time_real",
    "last_out_time_real",
    "max_stint_len_real",
    # Additional stint columns that should be excluded.
    "num_stints",
}

# Columns with _real suffix are post-game/realized values (not priors).
LEAKY_SUFFIX = "_real"


def test_known_leaky_columns_are_excluded():
    """v1.1 Spec 7.1: Assert schema excludes known leaky stint-derived columns."""
    missing_from_exclusion = LEAKY_COLUMNS - _EXTRA_EXCLUDED_FEATURES
    assert not missing_from_exclusion, (
        f"LEAKAGE GUARDRAIL FAILED: The following known leaky columns are NOT in "
        f"_EXTRA_EXCLUDED_FEATURES: {sorted(missing_from_exclusion)}. "
        f"These columns contain post-game/realized values that leak future information."
    )


def test_real_suffix_pattern_documented():
    """v1.1 Spec 7.1: Verify that at least some _real columns are explicitly excluded.

    This test ensures that the exclusion list contains columns with the _real suffix,
    which indicates post-game realized values (as opposed to prior estimates).
    """
    real_suffix_columns = {col for col in _EXTRA_EXCLUDED_FEATURES if col.endswith(LEAKY_SUFFIX)}
    assert len(real_suffix_columns) >= 2, (
        f"LEAKAGE GUARDRAIL FAILED: Expected at least 2 columns with '{LEAKY_SUFFIX}' suffix "
        f"in _EXTRA_EXCLUDED_FEATURES (to guard against realized/post-game values). "
        f"Found only: {sorted(real_suffix_columns)}"
    )


def test_exclusion_list_is_non_empty():
    """Basic sanity check that the exclusion list exists and is populated."""
    assert len(_EXTRA_EXCLUDED_FEATURES) >= 5, (
        f"Expected at least 5 excluded feature columns for leakage prevention, "
        f"but found only {len(_EXTRA_EXCLUDED_FEATURES)}: {sorted(_EXTRA_EXCLUDED_FEATURES)}"
    )
