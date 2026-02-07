from __future__ import annotations

import pandas as pd

from projections.rates_v1.loader import RatesBundle
from projections.rates_v1.score import predict_rates


class _DummyModel:
    best_iteration = 1

    def predict(self, X: pd.DataFrame, num_iteration: int | None = None) -> list[float]:
        return [float(X["track_touches_per_min_szn"].iloc[0])]


def test_predict_rates_applies_bundle_tracking_fill_values() -> None:
    bundle = RatesBundle(
        models={"fga2_per_min": _DummyModel()},
        feature_cols=["track_touches_per_min_szn"],
        meta={"preprocess": {"tracking_fill_values": {"track_touches_per_min_szn": 0.42}}},
    )
    features = pd.DataFrame({"player_id": [1]})

    preds = predict_rates(features, bundle)

    assert preds.loc[0, "fga2_per_min"] == 0.42

