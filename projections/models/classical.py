"""Classical ML model utilities (e.g. XGBoost)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from xgboost import XGBRegressor


@dataclass
class TrainingResult:
    """Container for a fitted model plus optional metrics."""

    model: XGBRegressor
    metrics: Mapping[str, float]


def train_xgboost_model(
    X_train,
    y_train,
    *,
    X_valid=None,
    y_valid=None,
    params: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Train a default XGBoost regressor suitable for minutes prediction."""

    default_params: dict[str, Any] = {
        "objective": "reg:squarederror",
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    model = XGBRegressor(**(default_params | (params or {})))
    eval_set = None
    if X_valid is not None and y_valid is not None:
        eval_set = [(X_valid, y_valid)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    metrics: dict[str, float] = {}
    if eval_set:
        preds = model.predict(X_valid)
        mae = float(np.mean(np.abs(preds - y_valid)))
        metrics["mae"] = mae
    return TrainingResult(model=model, metrics=metrics)


def train_lightgbm_model(
    X_train,
    y_train,
    *,
    X_valid=None,
    y_valid=None,
    params: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Train a LightGBM regressor."""
    from lightgbm import LGBMRegressor

    default_params: dict[str, Any] = {
        "objective": "regression",
        "n_estimators": 500,
        "max_depth": 7,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }
    model = LGBMRegressor(**(default_params | (params or {})))
    
    eval_set = None
    callbacks = None
    if X_valid is not None and y_valid is not None:
        eval_set = [(X_valid, y_valid)]
        
    model.fit(X_train, y_train, eval_set=eval_set)

    metrics: dict[str, float] = {}
    if eval_set:
        preds = model.predict(X_valid)
        mae = float(np.mean(np.abs(preds - y_valid)))
        metrics["mae"] = mae
    return TrainingResult(model=model, metrics=metrics)
