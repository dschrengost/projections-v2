"""Rotation-aware minutes model (two-stage): P(play) + minute share.

Goal: Improve "next man up" realism by explicitly modeling who plays, then
distributing 240 team minutes over the likely rotation.

Architecture:
1) Binary classifier: `played_flag = minutes > 0`
2) Regressor: minute share (minutes / 240) conditional on played_flag == 1

Inference:
- Weight = max(0, share_pred) * (play_prob ** play_prob_exponent)
- Optional: `min_players` is a safety floor used only when a team-game would
  otherwise end up with all-zero weights (e.g., extreme missingness). It does
  not cap the rotation size.
- Normalize weights per team-game and scale to 240 minutes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


TEAM_TOTAL_MINUTES = 240.0
TAU_MIN_DEFAULT = 0.5
TAU_MAX_DEFAULT = 2.0


@dataclass
class RotationShareArtifacts:
    """Artifacts for the rotation-aware two-stage model."""

    play_model: lgb.LGBMClassifier
    play_imputer: SimpleImputer
    share_model: lgb.LGBMRegressor
    share_imputer: SimpleImputer
    feature_columns: list[str]
    # Optional learned team-level temperature (rotation tightness).
    tau_model: lgb.LGBMRegressor | None = None
    tau_imputer: SimpleImputer | None = None
    tau_feature_columns: list[str] = field(default_factory=list)
    tau_min: float = TAU_MIN_DEFAULT
    tau_max: float = TAU_MAX_DEFAULT
    params: dict[str, Any] = field(default_factory=dict)


def _ensure_features(artifacts: RotationShareArtifacts, X: pd.DataFrame) -> pd.DataFrame:
    missing = set(artifacts.feature_columns) - set(X.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(sorted(missing))}")
    return X[artifacts.feature_columns]


def _default_lgbm_classifier_params(*, random_state: int) -> dict[str, float | int]:
    return {
        "n_estimators": 600,
        "learning_rate": 0.04,
        "num_leaves": 63,
        "min_data_in_leaf": 64,
        "max_depth": -1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 0.1,
        "random_state": random_state,
    }


def _default_lgbm_regressor_params(*, random_state: int) -> dict[str, float | int]:
    return {
        "n_estimators": 600,
        "learning_rate": 0.04,
        "num_leaves": 63,
        "min_data_in_leaf": 64,
        "max_depth": -1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 0.1,
        "random_state": random_state,
    }


def train_rotation_share_model(
    X_train: pd.DataFrame,
    y_minutes: pd.Series,
    *,
    random_state: int = 42,
    in_rotation_minutes_threshold: float = 0.0,
    play_params: Mapping[str, float | int] | None = None,
    share_params: Mapping[str, float | int] | None = None,
) -> RotationShareArtifacts:
    """Train rotation classifier + conditional minute-share regressor.

    The classifier target is a *rotation inclusion* label, not just "played a
    second". Use `in_rotation_minutes_threshold` > 0 (e.g., 8–10) to make the
    model learn a realistic rotation set and avoid relying on hard caps.
    """

    if X_train.empty:
        raise ValueError("Cannot train on an empty feature frame.")
    y = pd.to_numeric(y_minutes, errors="coerce").fillna(0.0).astype(float)
    threshold = float(in_rotation_minutes_threshold)
    if threshold < 0.0:
        raise ValueError("in_rotation_minutes_threshold must be >= 0.")
    # Backward compatible semantics:
    # - threshold == 0 => treat as "played" (minutes > 0)
    # - threshold > 0  => treat as "in rotation" (minutes >= threshold)
    if threshold <= 0.0:
        y_play = (y > 0.0).astype(int)
    else:
        y_play = (y >= threshold).astype(int)

    play_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_play = play_imputer.fit_transform(X_train)
    play_model = lgb.LGBMClassifier(**(_default_lgbm_classifier_params(random_state=random_state) | dict(play_params or {})))
    play_model.fit(X_play, y_play.to_numpy(dtype=int))

    played_mask = y_play.astype(bool)
    if not played_mask.any():
        raise ValueError(
            "All rows are below the rotation minutes threshold; cannot fit share model. "
            "Lower in_rotation_minutes_threshold or widen the training window."
        )

    share_target = (y[played_mask] / TEAM_TOTAL_MINUTES).clip(lower=0.0).to_numpy(dtype=float)
    share_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_share = share_imputer.fit_transform(X_train.loc[played_mask])
    share_model = lgb.LGBMRegressor(**(_default_lgbm_regressor_params(random_state=random_state) | dict(share_params or {})))
    share_model.fit(X_share, share_target)

    return RotationShareArtifacts(
        play_model=play_model,
        play_imputer=play_imputer,
        share_model=share_model,
        share_imputer=share_imputer,
        feature_columns=list(X_train.columns),
        params={
            "random_state": random_state,
            "in_rotation_minutes_threshold": threshold,
            "play_params": dict(play_params or {}),
            "share_params": dict(share_params or {}),
        },
    )


def predict_play_prob(artifacts: RotationShareArtifacts, X: pd.DataFrame) -> np.ndarray:
    """Predict P(play) for each row."""

    X_feat = _ensure_features(artifacts, X)
    X_imp = artifacts.play_imputer.transform(X_feat)
    if hasattr(artifacts.play_model, "predict_proba"):
        proba = artifacts.play_model.predict_proba(X_imp)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(float)
    # Fallback: treat predict output as probability-ish.
    return artifacts.play_model.predict(X_imp).astype(float)


def predict_raw_share(artifacts: RotationShareArtifacts, X: pd.DataFrame) -> np.ndarray:
    """Predict raw (un-normalized) minute shares for played players."""

    X_feat = _ensure_features(artifacts, X)
    X_imp = artifacts.share_imputer.transform(X_feat)
    raw = artifacts.share_model.predict(X_imp).astype(float)
    return np.maximum(0.0, raw)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    # Clip to avoid overflow; exp(-60) is effectively 0.
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def _tau_from_logit(logit: np.ndarray, *, tau_min: float, tau_max: float) -> np.ndarray:
    lo = float(tau_min)
    hi = float(tau_max)
    if hi <= lo:
        raise ValueError("tau_max must be greater than tau_min.")
    s = _sigmoid(logit)
    return lo + (hi - lo) * s


def build_tau_team_features(
    df: pd.DataFrame,
    *,
    game_ids: pd.Series | np.ndarray,
    team_ids: pd.Series | np.ndarray,
    is_out: pd.Series | np.ndarray | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build a team-game feature frame for tau (rotation tightness).

    This intentionally operates on the *full* input frame (not the contract
    feature matrix) so it can use simple pre-lock context signals like
    out counts and vegas/rest flags.
    """

    games = np.asarray(game_ids)
    teams = np.asarray(team_ids)
    if games.shape[0] != len(df) or teams.shape[0] != len(df):
        raise ValueError("game_ids and team_ids must align with df rows.")

    requested = list(feature_columns or [])
    base_cols = [c for c in requested if c not in {"team_out_count", "team_starter_out_count"}]

    base = pd.DataFrame(
        {
            "game_id": games.astype(int),
            "team_id": teams.astype(int),
        },
        index=df.index,
    )

    # OUT + starter OUT counts.
    out_flag = np.zeros(len(df), dtype=int)
    if is_out is not None:
        out_flag = np.asarray(is_out).astype(int)
    base["_out"] = out_flag

    starter_flag = np.zeros(len(df), dtype=int)
    if "starter_flag_label" in df.columns:
        starter_flag = (
            pd.to_numeric(df["starter_flag_label"], errors="coerce").fillna(0).astype(int) > 0
        ).astype(int).to_numpy()
    elif "starter_prev_game_asof" in df.columns:
        starter_flag = (
            pd.to_numeric(df["starter_prev_game_asof"], errors="coerce").fillna(0).astype(int) > 0
        ).astype(int).to_numpy()
    else:
        for col in ("is_confirmed_starter", "is_projected_starter", "starter_flag", "is_starter"):
            if col in df.columns:
                starter_flag = starter_flag | (
                    pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int) > 0
                ).astype(int).to_numpy()
    base["_starter"] = starter_flag
    base["_starter_out"] = (base["_out"] & base["_starter"]).astype(int)

    for col in base_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required tau feature column '{col}'.")
        base[col] = pd.to_numeric(df[col], errors="coerce")

    agg: dict[str, str] = {"_out": "sum", "_starter_out": "sum"}
    for col in base_cols:
        agg[col] = "mean"

    grouped = base.groupby(["game_id", "team_id"], sort=False).agg(agg).reset_index()
    grouped = grouped.rename(
        columns={
            "_out": "team_out_count",
            "_starter_out": "team_starter_out_count",
        }
    )
    return grouped


def _predict_tau_per_row(
    artifacts: RotationShareArtifacts,
    X: pd.DataFrame,
    game_ids: np.ndarray,
    team_ids: np.ndarray,
    *,
    is_out: pd.Series | np.ndarray | None,
) -> np.ndarray:
    tau_model = getattr(artifacts, "tau_model", None)
    tau_imputer = getattr(artifacts, "tau_imputer", None)
    tau_cols = list(getattr(artifacts, "tau_feature_columns", []) or [])
    if tau_model is None or tau_imputer is None or not tau_cols:
        return np.ones(len(X), dtype=float)

    tau_min = float(getattr(artifacts, "tau_min", TAU_MIN_DEFAULT))
    tau_max = float(getattr(artifacts, "tau_max", TAU_MAX_DEFAULT))

    team_features = build_tau_team_features(
        X,
        game_ids=game_ids,
        team_ids=team_ids,
        is_out=is_out,
        feature_columns=tau_cols,
    )
    missing = set(tau_cols) - set(team_features.columns)
    if missing:
        raise ValueError(f"Missing required tau feature columns: {', '.join(sorted(missing))}")

    X_tau = team_features[tau_cols]
    X_tau_imp = tau_imputer.transform(X_tau)
    tau_logit = tau_model.predict(X_tau_imp).astype(float)
    tau_team = _tau_from_logit(tau_logit, tau_min=tau_min, tau_max=tau_max)

    mapping = team_features[["game_id", "team_id"]].copy()
    mapping["tau"] = tau_team
    key_df = pd.DataFrame({"game_id": game_ids.astype(int), "team_id": team_ids.astype(int)}, index=X.index)
    merged = key_df.merge(mapping, on=["game_id", "team_id"], how="left")
    return pd.to_numeric(merged["tau"], errors="coerce").fillna(1.0).to_numpy(dtype=float)


def _normalize_weights_per_team(
    weights: np.ndarray,
    game_ids: np.ndarray,
    team_ids: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    df = pd.DataFrame(
        {
            "w": weights,
            "game_id": game_ids,
            "team_id": team_ids,
        }
    )
    sums = df.groupby(["game_id", "team_id"], sort=False)["w"].transform("sum").to_numpy(dtype=float)
    denom = np.where(sums > 0.0, sums, 1.0)
    return weights / (denom + epsilon)


def predict_minutes(
    artifacts: RotationShareArtifacts,
    X: pd.DataFrame,
    *,
    game_ids: pd.Series | np.ndarray,
    team_ids: pd.Series | np.ndarray,
    is_out: pd.Series | np.ndarray | None = None,
    min_players: int = 8,
    play_prob_exponent: float = 1.0,
    use_learned_tau: bool = True,
    select_expected_rotation: bool | None = None,
    team_total_minutes: float = TEAM_TOTAL_MINUTES,
) -> pd.DataFrame:
    """Predict 240-feasible minutes per team via two-stage rotation weighting."""

    if len(X) == 0:
        return pd.DataFrame(
            columns=[
                "game_id",
                "team_id",
                "play_prob",
                "raw_share",
                "weight",
                "normalized_share",
                "predicted_minutes",
            ]
        )

    games = np.asarray(game_ids)
    teams = np.asarray(team_ids)
    if games.shape[0] != len(X) or teams.shape[0] != len(X):
        raise ValueError("game_ids and team_ids must align with X rows.")

    play_prob = predict_play_prob(artifacts, X)
    raw_share = predict_raw_share(artifacts, X)
    share_weight = raw_share.copy()

    out_mask = None
    if is_out is not None:
        out_mask = np.asarray(is_out).astype(bool)
        share_weight = np.where(out_mask, 0.0, share_weight)
        play_prob = np.where(out_mask, 0.0, play_prob)

    tau = np.ones(len(X), dtype=float)
    if use_learned_tau:
        try:
            tau = _predict_tau_per_row(artifacts, X, games, teams, is_out=is_out)
        except ValueError:
            # If tau features are missing in this run, fall back to tau=1.
            tau = np.ones(len(X), dtype=float)
    tau = np.where(np.isfinite(tau) & (tau > 0.0), tau, 1.0)
    # Apply temperature on conditional share weights only (not play-prob).
    # This is closer to "softmax(logits/tau)" where logits come from the share head.
    share_weight = np.power(np.maximum(0.0, share_weight), 1.0 / tau)
    weight = share_weight * np.power(np.clip(play_prob, 0.0, 1.0), float(play_prob_exponent))

    # Optional: turn rotation probabilities into a deterministic expected-rotation set by
    # selecting top-K players per team-game, where K ~= sum(play_prob).
    #
    # This avoids allocating meaningful minutes to deep-bench players when the model
    # believes they're unlikely to be in the rotation.
    #
    # Default: enable only when the play head is trained as an "in rotation" classifier
    # (i.e., threshold > 0).
    if select_expected_rotation is None:
        threshold = float((getattr(artifacts, "params", {}) or {}).get("in_rotation_minutes_threshold", 0.0) or 0.0)
        select_expected_rotation = threshold > 0.0

    if select_expected_rotation:
        df_sel = pd.DataFrame(
            {
                "idx": np.arange(len(X), dtype=int),
                "game_id": games.astype(int),
                "team_id": teams.astype(int),
            }
        )
        for (_, _), g in df_sel.groupby(["game_id", "team_id"], sort=False):
            idx = g["idx"].to_numpy(dtype=int)
            if out_mask is not None:
                active_idx = idx[~out_mask[idx]]
            else:
                active_idx = idx
            if active_idx.size == 0:
                continue
            expected_k = float(np.sum(np.clip(play_prob[active_idx], 0.0, 1.0)))
            k = int(np.rint(expected_k))
            k = max(1, min(k, int(active_idx.size)))
            if k >= int(active_idx.size):
                continue
            order = np.argsort(-weight[active_idx], kind="mergesort")
            keep = set(active_idx[order[:k]].tolist())
            drop = [i for i in active_idx.tolist() if i not in keep]
            if drop:
                weight[np.asarray(drop, dtype=int)] = 0.0

    # Final safeguard: if a team-game still has all-zero weights (e.g., all players
    # predicted DNP), fall back to play_prob (or uniform) for the top-N players.
    #
    # NOTE: `min_players` only impacts this safety fallback; it does not truncate
    # rotations when weights are already positive.
    df_groups = pd.DataFrame(
        {
            "idx": np.arange(len(X)),
            "game_id": games,
            "team_id": teams,
            "weight": weight,
        }
    )
    for (_, _), g in df_groups.groupby(["game_id", "team_id"], sort=False):
        idx = g["idx"].to_numpy(dtype=int)
        if float(weight[idx].sum()) > 0.0:
            continue
        probs = play_prob[idx]
        if out_mask is not None:
            active = ~out_mask[idx]
        else:
            active = np.ones_like(probs, dtype=bool)
        candidate_idx = idx[active]
        candidate_probs = probs[active]
        if candidate_idx.size == 0:
            candidate_idx = idx
            candidate_probs = probs
        k = int(min_players) if min_players and min_players > 0 else min(8, len(candidate_idx))
        k = max(1, min(k, len(candidate_idx)))
        order = np.argsort(-candidate_probs, kind="mergesort")[:k]
        chosen = candidate_idx[order]
        fallback = np.maximum(candidate_probs[order], 0.0)
        if float(fallback.sum()) <= 0.0:
            fallback = np.ones_like(fallback, dtype=float)
        weight[chosen] = fallback

    normalized = _normalize_weights_per_team(weight, games, teams)
    predicted_minutes = normalized * float(team_total_minutes)

    return pd.DataFrame(
        {
            "game_id": games,
            "team_id": teams,
            "tau": tau,
            "play_prob": play_prob,
            "raw_share": raw_share,
            "weight": weight,
            "normalized_share": normalized,
            "predicted_minutes": predicted_minutes,
        },
        index=X.index,
    )


__all__ = [
    "RotationShareArtifacts",
    "TAU_MAX_DEFAULT",
    "TAU_MIN_DEFAULT",
    "TEAM_TOTAL_MINUTES",
    "build_tau_team_features",
    "predict_minutes",
    "predict_play_prob",
    "predict_raw_share",
    "train_rotation_share_model",
]
