"""Minute Share Mixture Model - training and prediction.

Two-stage model:
1. State classifier (multiclass LightGBM) predicts p(state | features)
2. Per-state regressors predict E[minutes | state, features]

Combined expected minutes:
    E[minutes] = Σ_k p_state[k] * μ_k
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from projections.models.minute_share_mixture.labels import (
    NUM_STATES,
    STATE_NAMES,
    minutes_to_state,
)

LOGGER = logging.getLogger(__name__)

# LightGBM defaults for classifier
CLASSIFIER_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": NUM_STATES,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 0.1,
    "verbose": -1,
}

# LightGBM defaults for regressors
REGRESSOR_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "n_estimators": 150,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 30,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 0.1,
    "verbose": -1,
}


@dataclass
class MixtureBundle:
    """Bundle of trained mixture model artifacts."""
    
    classifier: lgb.LGBMClassifier
    regressors: dict[int, lgb.LGBMRegressor]  # state -> regressor (S1-S4)
    imputer: SimpleImputer
    feature_columns: list[str]
    state_means: dict[int, float] = field(default_factory=dict)  # fallback means
    train_meta: dict[str, Any] = field(default_factory=dict)
    
    def save(self, bundle_dir: Path) -> None:
        """Save bundle to directory."""
        bundle_dir = Path(bundle_dir)
        models_dir = bundle_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save classifier
        joblib.dump(self.classifier, models_dir / "state_classifier.joblib")
        
        # Save regressors
        for state, reg in self.regressors.items():
            joblib.dump(reg, models_dir / f"regressor_state_{state}.joblib")
        
        # Save imputer
        joblib.dump(self.imputer, bundle_dir / "imputer.joblib")
        
        # Save feature columns
        (bundle_dir / "feature_columns.json").write_text(
            json.dumps(self.feature_columns, indent=2)
        )
        
        # Save training metadata
        meta = {
            **self.train_meta,
            "state_means": self.state_means,
            "num_states": NUM_STATES,
            "state_names": STATE_NAMES,
        }
        (bundle_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
        
        LOGGER.info(f"Saved mixture bundle to {bundle_dir}")
    
    @classmethod
    def load(cls, bundle_dir: Path) -> "MixtureBundle":
        """Load bundle from directory."""
        bundle_dir = Path(bundle_dir)
        models_dir = bundle_dir / "models"
        
        # Load classifier
        classifier = joblib.load(models_dir / "state_classifier.joblib")
        
        # Load regressors (S1-S4)
        regressors = {}
        for state in range(1, NUM_STATES):
            path = models_dir / f"regressor_state_{state}.joblib"
            if path.exists():
                regressors[state] = joblib.load(path)
        
        # Load imputer
        imputer = joblib.load(bundle_dir / "imputer.joblib")
        
        # Load feature columns
        feature_columns = json.loads(
            (bundle_dir / "feature_columns.json").read_text()
        )
        
        # Load training metadata
        train_meta = json.loads(
            (bundle_dir / "train_meta.json").read_text()
        )
        state_means = train_meta.pop("state_means", {})
        # Convert keys back to int
        state_means = {int(k): v for k, v in state_means.items()}
        
        return cls(
            classifier=classifier,
            regressors=regressors,
            imputer=imputer,
            feature_columns=feature_columns,
            state_means=state_means,
            train_meta=train_meta,
        )


def train_classifier(
    X: pd.DataFrame,
    y_minutes: np.ndarray | pd.Series,
    *,
    random_state: int = 42,
    params: dict[str, Any] | None = None,
) -> lgb.LGBMClassifier:
    """Train multiclass state classifier.
    
    Args:
        X: Feature dataframe (already imputed)
        y_minutes: Raw minutes labels
        random_state: Random seed
        params: Optional parameter overrides
        
    Returns:
        Trained LGBMClassifier
    """
    # Convert minutes to states
    y_states = minutes_to_state(np.asarray(y_minutes))
    
    # Merge params
    merged = dict(CLASSIFIER_PARAMS)
    merged["random_state"] = random_state
    if params:
        merged.update(params)
    
    # Train
    clf = lgb.LGBMClassifier(**merged)
    clf.fit(X, y_states)
    
    # Log class distribution
    unique, counts = np.unique(y_states, return_counts=True)
    LOGGER.info(f"Classifier trained on {len(y_states)} samples")
    for s, c in zip(unique, counts):
        LOGGER.info(f"  State {s} ({STATE_NAMES.get(s, '?')}): {c} ({100*c/len(y_states):.1f}%)")
    
    return clf


def train_regressors_by_state(
    X: pd.DataFrame,
    y_minutes: np.ndarray | pd.Series,
    *,
    random_state: int = 42,
    params: dict[str, Any] | None = None,
    min_samples_per_state: int = 100,
) -> tuple[dict[int, lgb.LGBMRegressor], dict[int, float]]:
    """Train per-state conditional regressors.
    
    Trains a regressor for each non-zero state (S1-S4) predicting
    E[minutes | state].
    
    Args:
        X: Feature dataframe (already imputed)
        y_minutes: Raw minutes labels
        random_state: Random seed
        params: Optional parameter overrides
        min_samples_per_state: Minimum samples to train a regressor
        
    Returns:
        Tuple of (regressors dict, state_means dict for fallback)
    """
    y_min = np.asarray(y_minutes, dtype=np.float64)
    y_states = minutes_to_state(y_min)
    
    # Merge params
    merged = dict(REGRESSOR_PARAMS)
    merged["random_state"] = random_state
    if params:
        merged.update(params)
    
    regressors: dict[int, lgb.LGBMRegressor] = {}
    state_means: dict[int, float] = {0: 0.0}  # S0 always predicts 0
    
    for state in range(1, NUM_STATES):
        mask = y_states == state
        n_samples = mask.sum()
        
        if n_samples < min_samples_per_state:
            LOGGER.warning(
                f"State {state} has only {n_samples} samples "
                f"(< {min_samples_per_state}), using mean fallback"
            )
            state_means[state] = float(y_min[mask].mean()) if n_samples > 0 else 0.0
            continue
        
        X_state = X.loc[mask]
        y_state = y_min[mask]
        
        reg = lgb.LGBMRegressor(**merged)
        reg.fit(X_state, y_state)
        regressors[state] = reg
        
        # Also compute mean for fallback
        state_means[state] = float(y_state.mean())
        
        LOGGER.info(
            f"State {state} regressor: n={n_samples}, "
            f"mean={state_means[state]:.1f} min"
        )
    
    return regressors, state_means


def train_mixture_model(
    X: pd.DataFrame,
    y_minutes: np.ndarray | pd.Series,
    *,
    random_state: int = 42,
    classifier_params: dict[str, Any] | None = None,
    regressor_params: dict[str, Any] | None = None,
) -> MixtureBundle:
    """Train complete mixture model (classifier + regressors).
    
    Args:
        X: Feature dataframe
        y_minutes: Raw minutes labels
        random_state: Random seed
        classifier_params: Optional classifier parameter overrides
        regressor_params: Optional regressor parameter overrides
        
    Returns:
        MixtureBundle with all trained components
    """
    feature_columns = list(X.columns)
    
    # Impute missing values
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=feature_columns,
        index=X.index,
    )
    
    # Train classifier
    LOGGER.info("Training state classifier...")
    classifier = train_classifier(
        X_imputed, y_minutes,
        random_state=random_state,
        params=classifier_params,
    )
    
    # Train regressors
    LOGGER.info("Training per-state regressors...")
    regressors, state_means = train_regressors_by_state(
        X_imputed, y_minutes,
        random_state=random_state,
        params=regressor_params,
    )
    
    return MixtureBundle(
        classifier=classifier,
        regressors=regressors,
        imputer=imputer,
        feature_columns=feature_columns,
        state_means=state_means,
        train_meta={
            "random_state": random_state,
            "n_samples": len(y_minutes),
            "n_features": len(feature_columns),
        },
    )


def predict_expected_minutes(
    X: pd.DataFrame,
    bundle: MixtureBundle,
    *,
    cap_max: float = 48.0,
) -> np.ndarray:
    """Predict expected minutes using mixture model.
    
    Computes:
        E[minutes] = Σ_k p_state[k] * μ_k
        
    where p_state[k] is the classifier probability and μ_k is the
    conditional regressor prediction (or mean fallback) for state k.
    
    Args:
        X: Feature dataframe
        bundle: Trained mixture bundle
        cap_max: Maximum minutes cap
        
    Returns:
        Array of expected minutes predictions
    """
    # Align and impute features
    missing = set(bundle.feature_columns) - set(X.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    
    X_aligned = X[bundle.feature_columns]
    X_imputed = bundle.imputer.transform(X_aligned)
    X_df = pd.DataFrame(X_imputed, columns=bundle.feature_columns, index=X.index)
    
    n = len(X)
    
    # Get state probabilities from classifier
    p_states = bundle.classifier.predict_proba(X_df)  # (n, NUM_STATES)
    
    # Get conditional minutes for each state
    mu = np.zeros((n, NUM_STATES), dtype=np.float64)
    mu[:, 0] = 0.0  # S0 always predicts 0
    
    for state in range(1, NUM_STATES):
        if state in bundle.regressors:
            mu[:, state] = bundle.regressors[state].predict(X_df)
        else:
            # Fallback to training mean
            mu[:, state] = bundle.state_means.get(state, 0.0)
    
    # Clip conditional predictions to reasonable range
    mu = np.clip(mu, 0, cap_max)
    
    # Compute expected minutes: E[min] = Σ p[k] * μ[k]
    expected_minutes = (p_states * mu).sum(axis=1)
    
    # Final clip
    expected_minutes = np.clip(expected_minutes, 0, cap_max)
    
    return expected_minutes


def predict_with_diagnostics(
    X: pd.DataFrame,
    bundle: MixtureBundle,
    *,
    cap_max: float = 48.0,
) -> pd.DataFrame:
    """Predict with full diagnostic information.
    
    Args:
        X: Feature dataframe
        bundle: Trained mixture bundle
        cap_max: Maximum minutes cap
        
    Returns:
        DataFrame with columns:
        - expected_minutes: E[minutes]
        - predicted_state: argmax(p_states)
        - p_state_0, p_state_1, ...: state probabilities
        - mu_state_1, mu_state_2, ...: conditional means
    """
    # Align and impute features
    X_aligned = X[bundle.feature_columns]
    X_imputed = bundle.imputer.transform(X_aligned)
    X_df = pd.DataFrame(X_imputed, columns=bundle.feature_columns, index=X.index)
    
    n = len(X)
    
    # Get state probabilities
    p_states = bundle.classifier.predict_proba(X_df)
    
    # Get conditional minutes
    mu = np.zeros((n, NUM_STATES), dtype=np.float64)
    mu[:, 0] = 0.0
    
    for state in range(1, NUM_STATES):
        if state in bundle.regressors:
            mu[:, state] = bundle.regressors[state].predict(X_df)
        else:
            mu[:, state] = bundle.state_means.get(state, 0.0)
    
    mu = np.clip(mu, 0, cap_max)
    
    # Compute expected minutes
    expected_minutes = (p_states * mu).sum(axis=1)
    expected_minutes = np.clip(expected_minutes, 0, cap_max)
    
    # Build output dataframe
    result = pd.DataFrame(index=X.index)
    result["expected_minutes"] = expected_minutes
    result["predicted_state"] = p_states.argmax(axis=1)
    
    for s in range(NUM_STATES):
        result[f"p_state_{s}"] = p_states[:, s]
        if s > 0:
            result[f"mu_state_{s}"] = mu[:, s]
    
    return result
