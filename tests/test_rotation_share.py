"""Tests for the rotation_share (two-stage) minutes model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from projections.minutes_v1.rotation_share import predict_minutes, train_rotation_share_model


def _toy_team_frame(*, game_id: int, team_id: int) -> pd.DataFrame:
    rng = np.random.default_rng(42 + game_id + team_id)
    n_players = 12
    minutes = np.array([36, 34, 32, 28, 26, 22, 20, 18, 0, 0, 0, 0], dtype=float)
    assert minutes.sum() == 216.0
    # Scale played players to sum to 240; keep DNPs at 0.
    played = minutes > 0
    minutes[played] = minutes[played] / minutes[played].sum() * 240.0
    assert minutes.sum() == pytest.approx(240.0)

    return pd.DataFrame(
        {
            "game_id": [game_id] * n_players,
            "team_id": [team_id] * n_players,
            "player_id": list(range(team_id * 1000 + 1, team_id * 1000 + 1 + n_players)),
            "feature_1": rng.normal(size=n_players),
            "feature_2": rng.normal(size=n_players),
            "minutes": minutes,
            "is_out": [0] * n_players,
        }
    )


def test_predict_minutes_sums_to_240_per_team():
    df = pd.concat([_toy_team_frame(game_id=1, team_id=10), _toy_team_frame(game_id=1, team_id=20)], ignore_index=True)
    X = df[["feature_1", "feature_2", "is_out"]]
    y = df["minutes"]

    artifacts = train_rotation_share_model(
        X,
        y,
        random_state=7,
        play_params={"n_estimators": 50},
        share_params={"n_estimators": 50},
    )
    preds = predict_minutes(
        artifacts,
        X,
        game_ids=df["game_id"],
        team_ids=df["team_id"],
        is_out=df["is_out"],
        min_players=8,
        play_prob_exponent=1.25,
    )
    sums = preds.groupby(["game_id", "team_id"])["predicted_minutes"].sum()
    assert np.allclose(sums.to_numpy(dtype=float), 240.0, atol=1e-6)


def test_out_players_receive_zero_minutes():
    df = _toy_team_frame(game_id=2, team_id=30)
    # Mark two players out.
    df.loc[df.index[:2], "is_out"] = 1

    X = df[["feature_1", "feature_2", "is_out"]]
    y = df["minutes"]
    artifacts = train_rotation_share_model(
        X,
        y,
        random_state=11,
        play_params={"n_estimators": 30},
        share_params={"n_estimators": 30},
    )
    preds = predict_minutes(
        artifacts,
        X,
        game_ids=df["game_id"],
        team_ids=df["team_id"],
        is_out=df["is_out"],
        min_players=8,
        play_prob_exponent=1.5,
    )
    assert (preds.loc[df["is_out"].astype(bool), "predicted_minutes"] == 0.0).all()
    assert preds.groupby(["game_id", "team_id"])["predicted_minutes"].sum().iloc[0] == pytest.approx(240.0)


def test_min_players_gates_rotation_size():
    class _DummyPlayModel:
        def predict_proba(self, X):  # noqa: N803 - sklearn-style signature
            n = len(X)
            # Always predict "plays" with high probability.
            p = np.full((n,), 0.95, dtype=float)
            return np.column_stack([1.0 - p, p])

    class _DummyShareModel:
        def predict(self, X):  # noqa: N803 - sklearn-style signature
            # Positive raw shares for all rows (no truncation should occur).
            return np.full((len(X),), 0.05, dtype=float)

    df = _toy_team_frame(game_id=3, team_id=40)
    X = df[["feature_1", "feature_2", "is_out"]]

    # Construct artifacts without training so we can control model outputs.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
    artifacts = train_rotation_share_model(
        X,
        df["minutes"],
        random_state=13,
        play_params={"n_estimators": 5},
        share_params={"n_estimators": 5},
    )
    artifacts.play_model = _DummyPlayModel()  # type: ignore[assignment]
    artifacts.share_model = _DummyShareModel()  # type: ignore[assignment]
    artifacts.play_imputer = imputer
    artifacts.share_imputer = imputer
    artifacts.feature_columns = list(X.columns)

    preds = predict_minutes(
        artifacts,
        X,
        game_ids=df["game_id"],
        team_ids=df["team_id"],
        is_out=df["is_out"],
        min_players=6,
        play_prob_exponent=1.0,
    )
    # min_players is a floor for all-zero teams; it should not cap rotation size.
    assert int((preds["predicted_minutes"] > 1e-9).sum()) == len(df)
    assert preds.groupby(["game_id", "team_id"])["predicted_minutes"].sum().iloc[0] == pytest.approx(240.0)


def test_min_players_only_applies_on_all_zero_fallback():
    class _ZeroPlayModel:
        def predict_proba(self, X):  # noqa: N803 - sklearn-style signature
            n = len(X)
            p = np.zeros((n,), dtype=float)
            return np.column_stack([1.0 - p, p])

    class _ZeroShareModel:
        def predict(self, X):  # noqa: N803 - sklearn-style signature
            return np.zeros((len(X),), dtype=float)

    df = _toy_team_frame(game_id=5, team_id=60)
    X = df[["feature_1", "feature_2", "is_out"]]

    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
    artifacts = train_rotation_share_model(
        X,
        df["minutes"],
        random_state=19,
        play_params={"n_estimators": 5},
        share_params={"n_estimators": 5},
    )
    artifacts.play_model = _ZeroPlayModel()  # type: ignore[assignment]
    artifacts.share_model = _ZeroShareModel()  # type: ignore[assignment]
    artifacts.play_imputer = imputer
    artifacts.share_imputer = imputer
    artifacts.feature_columns = list(X.columns)

    preds = predict_minutes(
        artifacts,
        X,
        game_ids=df["game_id"],
        team_ids=df["team_id"],
        is_out=df["is_out"],
        min_players=6,
        play_prob_exponent=1.0,
    )
    assert int((preds["predicted_minutes"] > 1e-9).sum()) == 6
    assert preds.groupby(["game_id", "team_id"])["predicted_minutes"].sum().iloc[0] == pytest.approx(240.0)


class _ConstantTauModel:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def predict(self, X):  # noqa: N803 - sklearn-style signature
        return np.full((len(X),), self._value, dtype=float)


def test_learned_tau_is_bounded_and_deterministic():
    df = _toy_team_frame(game_id=4, team_id=50)
    X = df[["feature_1", "feature_2", "is_out"]]
    y = df["minutes"]

    artifacts = train_rotation_share_model(
        X,
        y,
        random_state=17,
        play_params={"n_estimators": 25},
        share_params={"n_estimators": 25},
    )
    # Attach a dummy tau head that saturates to tau_max.
    artifacts.tau_min = 0.5
    artifacts.tau_max = 2.0
    artifacts.tau_feature_columns = ["team_out_count"]
    artifacts.tau_imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(
        pd.DataFrame({"team_out_count": [0.0, 1.0]})
    )
    artifacts.tau_model = _ConstantTauModel(1e6)

    preds1 = predict_minutes(
        artifacts,
        X,
        game_ids=df["game_id"],
        team_ids=df["team_id"],
        is_out=df["is_out"],
        min_players=8,
        play_prob_exponent=1.0,
        use_learned_tau=True,
    )
    preds2 = predict_minutes(
        artifacts,
        X,
        game_ids=df["game_id"],
        team_ids=df["team_id"],
        is_out=df["is_out"],
        min_players=8,
        play_prob_exponent=1.0,
        use_learned_tau=True,
    )

    assert preds1.groupby(["game_id", "team_id"])["normalized_share"].sum().iloc[0] == pytest.approx(1.0)
    assert preds1.groupby(["game_id", "team_id"])["predicted_minutes"].sum().iloc[0] == pytest.approx(240.0)
    assert (preds1["tau"] >= artifacts.tau_min - 1e-9).all()
    assert (preds1["tau"] <= artifacts.tau_max + 1e-9).all()
    assert np.allclose(
        preds1["predicted_minutes"].to_numpy(dtype=float),
        preds2["predicted_minutes"].to_numpy(dtype=float),
        atol=0.0,
        rtol=0.0,
    )
