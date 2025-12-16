"""Usage shares v1 module for within-team opportunity allocation modeling."""

from projections.usage_shares_v1.features import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    GROUP_COLS,
    KEY_COLS,
    LABEL_COLS,
    NUMERIC_COLS,
    VALIDITY_COLS,
    add_derived_features,
    build_category_maps,
    encode_categoricals,
    prepare_features,
)
from projections.usage_shares_v1.metrics import (
    TargetMetrics,
    check_odds_leakage,
    compute_baseline_log_weights,
    compute_metrics,
)

__all__ = [
    "NUMERIC_COLS",
    "CATEGORICAL_COLS",
    "FEATURE_COLS",
    "LABEL_COLS",
    "VALIDITY_COLS",
    "GROUP_COLS",
    "KEY_COLS",
    "prepare_features",
    "add_derived_features",
    "build_category_maps",
    "encode_categoricals",
    "TargetMetrics",
    "compute_metrics",
    "compute_baseline_log_weights",
    "check_odds_leakage",
]
