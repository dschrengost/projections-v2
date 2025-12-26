"""Minute Share Mixture Model package.

Two-stage model for minutes prediction:
1. State classifier: predicts discrete minutes state (0-4)
2. Conditional regressors: predict expected minutes within each state

The mixture expected minutes is:
    E[minutes] = Σ_k p_state[k] * μ_k

where p_state[k] is the classifier probability and μ_k is the
conditional regressor prediction for state k.
"""

from projections.models.minute_share_mixture.labels import (
    NUM_STATES,
    STATE_NAMES,
    get_state_counts,
    minutes_to_state,
    state_to_bucket_name,
    state_to_minute_range,
)
from projections.models.minute_share_mixture.model import (
    MixtureBundle,
    predict_expected_minutes,
    predict_with_diagnostics,
    train_classifier,
    train_mixture_model,
    train_regressors_by_state,
)

__all__ = [
    # Labels
    "NUM_STATES",
    "STATE_NAMES",
    "get_state_counts",
    "minutes_to_state",
    "state_to_bucket_name",
    "state_to_minute_range",
    # Model
    "MixtureBundle",
    "predict_expected_minutes",
    "predict_with_diagnostics",
    "train_classifier",
    "train_mixture_model",
    "train_regressors_by_state",
]

