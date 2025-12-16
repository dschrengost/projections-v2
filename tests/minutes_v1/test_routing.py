from __future__ import annotations

import numpy as np
import pandas as pd

from projections.minutes_v1.routing import late_model_weight, minutes_model_used_label


def test_late_model_weight_step_function() -> None:
    t = pd.Series([120.0, 61.0, 60.0, 0.0])
    w = late_model_weight(t, late_threshold_min=60, blend_band_min=0)
    assert w.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_late_model_weight_blend_band_edges() -> None:
    t = pd.Series([120.0, 90.0, 75.0, 60.0, 30.0, 0.0])
    w = late_model_weight(t, late_threshold_min=60, blend_band_min=30)
    assert np.allclose(w.to_numpy(), np.array([0.0, 0.0, 0.5, 1.0, 1.0, 1.0]))


def test_late_model_weight_handles_missing_and_negative() -> None:
    t = pd.Series([np.nan, -5.0, 10.0])
    w = late_model_weight(t, late_threshold_min=60, blend_band_min=30)
    assert w.tolist() == [1.0, 1.0, 1.0]


def test_minutes_model_used_label() -> None:
    weights = pd.Series([0.0, 0.25, 1.0, np.nan])
    labels = minutes_model_used_label(weights)
    assert labels.tolist() == ["early", "blend", "late", "late"]

