from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.ownership_v1.loader import OwnershipBundle
from projections.ownership_v1.score import compute_ownership_features, predict_ownership


class _DummyModel:
    def __init__(self, preds: list[float]):
        self._preds = np.asarray(preds, dtype=float)
        self.best_iteration = 1

    def predict(self, X, num_iteration=None):  # noqa: ANN001
        return self._preds


def test_predict_ownership_inverts_logit_target_transform():
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    bundle = OwnershipBundle(
        model=_DummyModel([-10.0, 0.0, 10.0]),
        feature_cols=["f1"],
        meta={"params": {"target_transform": "logit"}},
    )

    preds = predict_ownership(df, bundle)
    assert preds.iloc[0] == pytest.approx(0.0, abs=0.05)
    assert preds.iloc[1] == pytest.approx(50.0, abs=1e-9)
    assert preds.iloc[2] == pytest.approx(100.0, abs=0.05)


def test_predict_ownership_clips_percent_space_for_none_transform():
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    bundle = OwnershipBundle(
        model=_DummyModel([-10.0, 0.0, 10.0]),
        feature_cols=["f1"],
        meta={"params": {"target_transform": "none"}},
    )

    preds = predict_ownership(df, bundle)
    assert preds.tolist() == [0.0, 0.0, 10.0]


def test_compute_ownership_features_includes_v6_computed_columns():
    df = pd.DataFrame(
        {
            "salary": [3000, 4000, 8000, 9000],
            "proj_fpts": [10.0, 12.0, 40.0, 45.0],
            "pos": ["PG", "SG", "SF/PF", "C"],
            "team": ["AAA", "BBB", "CCC", "DDD"],
            "team_outs_count": [2, 0, 1, 3],
        }
    )

    feats = compute_ownership_features(df, slate_id_col=None)

    expected_cols = {
        "slate_size",
        "salary_pct_of_max",
        "is_min_salary",
        "slate_near_min_count",
        "value_vs_slate_avg",
        "salary_vs_median",
        "is_min_priced_by_pos",
        "game_count_on_slate",
        "value_x_value_tier",
        "outs_x_salary_rank",
    }
    assert expected_cols.issubset(set(feats.columns))

    assert feats["slate_size"].nunique() == 1
    assert int(feats["slate_size"].iloc[0]) == 4
    assert int(feats["game_count_on_slate"].iloc[0]) == 2
    assert feats["outs_x_salary_rank"].notna().all()
