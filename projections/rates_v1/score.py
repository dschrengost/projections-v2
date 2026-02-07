from __future__ import annotations

from typing import Dict

import pandas as pd

from projections.rates_v1.loader import RatesBundle
from projections.rates_v1.preprocess import apply_tracking_fill_values, resolve_tracking_fill_values


def predict_rates(features: pd.DataFrame, bundle: RatesBundle) -> pd.DataFrame:
    """
    Score per-minute rates using a loaded rates_v1 bundle.

    Returns a DataFrame with one column per target (same index as input).
    """
    features_prepped = features.copy()
    tracking_fill_values = resolve_tracking_fill_values(bundle.meta, bundle.feature_cols)
    if tracking_fill_values:
        features_prepped = apply_tracking_fill_values(features_prepped, tracking_fill_values)

    missing = [c for c in bundle.feature_cols if c not in features_prepped.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    X = features_prepped[bundle.feature_cols]
    preds: Dict[str, pd.Series] = {}
    for target, model in bundle.models.items():
        preds[target] = pd.Series(model.predict(X, num_iteration=model.best_iteration), index=features_prepped.index)
    return pd.DataFrame(preds, index=features_prepped.index)
