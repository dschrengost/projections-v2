"""Modeling helpers for the Minutes V1 quick start."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from projections.models.minutes_features import infer_feature_columns


@dataclass
class BaselineResult:
    """Artifacts for the ridge baseline."""

    model: Pipeline
    metrics: Mapping[str, float]


@dataclass
class QuantileArtifacts:
    """Container for LightGBM quantile models."""

    models: dict[float, lgb.LGBMRegressor]
    imputer: SimpleImputer


@dataclass
class QuickstartModelArtifacts:
    """Full training outputs for the quick-start pipeline."""

    baseline: BaselineResult
    quantiles: QuantileArtifacts
    conformal: "ConformalIntervalCalibrator"
    feature_columns: list[str]


class ConformalIntervalCalibrator:
    """Simple two-sided conformal interval calibrator."""

    def __init__(self, alpha_low: float = 0.1, alpha_high: float = 0.1) -> None:
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self._low_adjustment: float = 0.0
        self._high_adjustment: float = 0.0
        self._fitted: bool = False

    def fit(self, y_true: np.ndarray, low_preds: np.ndarray, high_preds: np.ndarray) -> None:
        if y_true.size == 0:
            raise ValueError("Cannot conformalize with an empty calibration set.")
        low_errors = low_preds - y_true
        high_errors = y_true - high_preds
        self._low_adjustment = float(np.quantile(low_errors, 1 - self.alpha_low))
        self._high_adjustment = float(np.quantile(high_errors, 1 - self.alpha_high))
        self._fitted = True

    def calibrate(self, low_preds: np.ndarray, high_preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_fitted():
            raise ValueError("ConformalIntervalCalibrator.calibrate() called before fit().")
        return low_preds - self._low_adjustment, high_preds + self._high_adjustment

    def export_offsets(self) -> dict[str, float]:
        """Return the learned offsets for persistence/monitoring."""

        if not self._is_fitted():
            raise ValueError("ConformalIntervalCalibrator.export_offsets() called before fit().")
        return {
            "alpha_low": self.alpha_low,
            "alpha_high": self.alpha_high,
            "low_adjustment": self._low_adjustment,
            "high_adjustment": self._high_adjustment,
        }

    def _is_fitted(self) -> bool:
        """Best-effort check that supports legacy artifacts missing _fitted."""

        fitted_flag = getattr(self, "_fitted", None)
        has_offsets = ("_low_adjustment" in self.__dict__) and ("_high_adjustment" in self.__dict__)
        if fitted_flag is None:
            return has_offsets
        return bool(fitted_flag)
def train_ridge_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> BaselineResult:
    """Train a ridge regression baseline with standardization."""

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=(0.1, 1.0, 10.0))),
        ]
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_valid)
    metrics = {"mae": float(mean_absolute_error(y_valid, preds))}
    return BaselineResult(model=pipeline, metrics=metrics)


def train_lightgbm_quantiles(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    quantiles: Iterable[float] = (0.1, 0.5, 0.9),
    random_state: int = 42,
    params: Mapping[str, float] | None = None,
) -> QuantileArtifacts:
    """Train independent LightGBM models for each requested quantile."""

    base_params: dict[str, float | int] = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 64,
        "max_depth": -1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 0.1,
        "random_state": random_state,
    }
    merged_params = base_params | (params or {})

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_imputed_df = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index)
    models: dict[float, lgb.LGBMRegressor] = {}
    for quantile in quantiles:
        if not 0 < quantile < 1:
            raise ValueError("Quantiles must be in (0, 1).")
        model = lgb.LGBMRegressor(objective="quantile", alpha=quantile, **merged_params)
        model.fit(X_train_imputed_df, y_train)
        models[float(quantile)] = model
    return QuantileArtifacts(models=models, imputer=imputer)


def predict_quantiles(artifacts: QuantileArtifacts, X: pd.DataFrame) -> dict[float, np.ndarray]:
    """Generate quantile predictions using the fitted models."""

    X_array = artifacts.imputer.transform(X)
    X_imputed = pd.DataFrame(X_array, columns=X.columns, index=X.index)
    return {quantile: model.predict(X_imputed) for quantile, model in artifacts.models.items()}


def train_minutes_quickstart_models(
    df: pd.DataFrame,
    *,
    target_col: str = "minutes",
    feature_cols: Iterable[str] | None = None,
    calibration_size: float = 0.2,
    random_state: int = 42,
) -> QuickstartModelArtifacts:
    """End-to-end training for the Quick Start spec."""

    if feature_cols is None:
        feature_columns = infer_feature_columns(
            df,
            target_col=target_col,
            excluded={"prior_play_prob", "play_prob", "play_probability", "p_play"},
        )
    else:
        feature_columns = list(feature_cols)
    if not 0 < calibration_size < 0.5:
        raise ValueError("Calibration size should be in (0, 0.5).")
    X = df[feature_columns]
    y = df[target_col]
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=calibration_size, random_state=random_state, shuffle=True
    )

    baseline = train_ridge_baseline(X_train, y_train, X_valid=X_cal, y_valid=y_cal)
    quantiles = train_lightgbm_quantiles(X_train, y_train, random_state=random_state)
    cal_quantiles = predict_quantiles(quantiles, X_cal)

    calibrator = ConformalIntervalCalibrator(alpha_low=0.1, alpha_high=0.1)
    calibrator.fit(y_cal.to_numpy(), cal_quantiles[0.1], cal_quantiles[0.9])

    return QuickstartModelArtifacts(
        baseline=baseline,
        quantiles=quantiles,
        conformal=calibrator,
        feature_columns=feature_columns,
    )


def predict_minutes(
    artifacts: QuickstartModelArtifacts,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score a dataframe and return raw + conformalized quantiles."""

    missing = set(artifacts.feature_columns) - set(feature_df.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(sorted(missing))}")
    X = feature_df[artifacts.feature_columns]
    baseline_preds = artifacts.baseline.model.predict(X)
    quantile_preds = predict_quantiles(artifacts.quantiles, X)
    low_adj, high_adj = artifacts.conformal.calibrate(
        quantile_preds[0.1], quantile_preds[0.9]
    )
    return pd.DataFrame(
        {
            "baseline_pred": baseline_preds,
            "p10": quantile_preds[0.1],
            "p50": quantile_preds[0.5],
            "p90": quantile_preds[0.9],
            "p10_calibrated": low_adj,
            "p90_calibrated": high_adj,
        },
        index=feature_df.index,
    )
