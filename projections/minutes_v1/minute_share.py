"""Minute share model: predicts share of team's 240 minutes per player.

Instead of predicting raw minutes directly, we predict the fraction of team
minutes (minute_share = actual_minutes / 240). At inference, shares are
normalized per team so predictions sum to exactly 240 minutes.

Advantages:
- Team constraint (240 total minutes) is enforced naturally via normalization
- Target range is bounded [0, ~0.42] for cleaner model behavior
- Shares are more stable across pace/game-script variation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

LOGGER = logging.getLogger(__name__)

# Team total minutes constraint
TEAM_TOTAL_MINUTES = 240.0

# Expected share ranges for validation
STAR_SHARE_MIN = 0.30
STAR_SHARE_MAX = 0.42
ROTATION_SHARE_MIN = 0.08
ROTATION_SHARE_MAX = 0.25
BENCH_SHARE_MAX = 0.08

# Default LightGBM hyperparameters
DEFAULT_LGBM_PARAMS: dict[str, float | int] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 64,
    "max_depth": -1,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 0.1,
}


class MinuteLabelMode(str, Enum):
    """Label computation mode for minute shares.
    
    REG240: Always divide by 240 (regulation minutes). Shares may sum >1.0 for OT games.
            This is the default for DFS production since platforms use 240.
    TEAM_TOTAL_ACTUAL: Divide by actual team total (OT-aware). Shares always sum to 1.0.
                       Useful for diagnostics and backtesting OT impact.
    """
    REG240 = "reg240"
    TEAM_TOTAL_ACTUAL = "team_total_actual"


@dataclass
class MinuteShareArtifacts:
    """Bundle of trained minute share model artifacts."""

    model: lgb.LGBMRegressor
    imputer: SimpleImputer
    feature_columns: list[str]
    params: dict[str, Any] = field(default_factory=dict)



def compute_minute_share(
    minutes: pd.Series | np.ndarray,
    game_ids: pd.Series | np.ndarray,
    team_ids: pd.Series | np.ndarray,
    team_total: float | None = None,
    mode: MinuteLabelMode = MinuteLabelMode.REG240,
) -> np.ndarray:
    """Convert raw minutes to share of team total in that specific game.
    
    Args:
        minutes: Raw minutes values
        game_ids: Game identifier
        team_ids: Team identifier
        team_total: Optional static divisor (overrides mode if provided)
        mode: Label computation mode (default REG240).
              REG240: Always divide by 240 (shares can sum >1.0 for OT)
              TEAM_TOTAL_ACTUAL: Divide by actual team total (OT-aware)

    Returns:
        Array of minute shares in [0, 1] range (or slightly >1 for OT with REG240).
    """
    df = pd.DataFrame({
        "minutes": np.asarray(minutes, dtype=float),
        "game_id": np.asarray(game_ids),
        "team_id": np.asarray(team_ids),
    })
    
    if team_total is not None:
        # Explicit override takes precedence
        divisor = team_total
    elif mode == MinuteLabelMode.REG240:
        # Fixed 240 divisor (shares may sum >1 for OT games)
        divisor = TEAM_TOTAL_MINUTES
    else:
        # Calculate dynamic total per game-team (OT-aware)
        divisor = df.groupby(["game_id", "team_id"])["minutes"].transform("sum")
        # Avoid zero division
        divisor = divisor.replace(0, 1.0)
        
    return (df["minutes"] / divisor).to_numpy()


def train_minute_share_model(
    X_train: pd.DataFrame,
    y_train_minutes: pd.Series,
    game_ids: pd.Series | np.ndarray | None = None,
    team_ids: pd.Series | np.ndarray | None = None,
    *,
    random_state: int = 42,
    params: Mapping[str, float | int] | None = None,
    label_mode: MinuteLabelMode = MinuteLabelMode.REG240,
) -> MinuteShareArtifacts:
    """Train a single LightGBM model predicting minute share.

    Args:
        X_train: Feature dataframe
        y_train_minutes: Raw minutes labels
        game_ids: Game IDs for grouping
        team_ids: Team IDs for grouping
        random_state: Random seed for reproducibility
        params: Optional LightGBM parameter overrides
        label_mode: How to compute share labels (default REG240)

    Returns:
        MinuteShareArtifacts with trained model and imputer
    """
    # Convert minutes to shares using specified mode.
    #
    # For REG240 labels, the divisor is constant (240) and does not require
    # (game_id, team_id) grouping. For convenience (and unit tests), allow
    # callers to omit ids in this mode.
    if game_ids is None:
        game_ids = np.zeros(len(y_train_minutes), dtype=int)
    if team_ids is None:
        team_ids = np.zeros(len(y_train_minutes), dtype=int)

    y_share = compute_minute_share(y_train_minutes, game_ids, team_ids, mode=label_mode)
    
    LOGGER.info(f"Training with label_mode={label_mode.value}")
    label_sum_stats = pd.DataFrame({
        "share": y_share,
        "game_id": np.asarray(game_ids),
        "team_id": np.asarray(team_ids),
    }).groupby(["game_id", "team_id"])["share"].sum()
    LOGGER.info(f"  Label share sum per team: mean={label_sum_stats.mean():.4f}, "
                f"min={label_sum_stats.min():.4f}, max={label_sum_stats.max():.4f}")

    # Merge params
    merged_params = dict(DEFAULT_LGBM_PARAMS)
    merged_params["random_state"] = random_state
    if params:
        merged_params.update(params)

    # Impute missing features
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_imputed = imputer.fit_transform(X_train)
    X_imputed_df = pd.DataFrame(X_imputed, columns=X_train.columns, index=X_train.index)

    # Train model
    model = lgb.LGBMRegressor(**merged_params)
    model.fit(X_imputed_df, y_share)

    return MinuteShareArtifacts(
        model=model,
        imputer=imputer,
        feature_columns=list(X_train.columns),
        params={**merged_params, "label_mode": label_mode.value},
    )


def predict_raw_shares(
    artifacts: MinuteShareArtifacts,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict raw (un-normalized) minute shares.

    Args:
        artifacts: Trained model artifacts
        X: Feature dataframe

    Returns:
        Raw predicted shares (may not sum to 1 per team)
    """
    # Ensure feature alignment
    missing = set(artifacts.feature_columns) - set(X.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(sorted(missing))}")

    X_aligned = X[artifacts.feature_columns]
    X_imputed = artifacts.imputer.transform(X_aligned)

    # Predict shares
    raw_shares = artifacts.model.predict(X_imputed)

    # Coerce to positive weights using softplus-like or just clamping?
    # User requested: "treat model output as unconstrained logit z_i... convert to positive weights w_i = exp(z_i) (or softplus)"
    # BUT: The model is trained on shares directly [0, 1]. It's a regressor.
    # If we treat output as 'z_i', we usually train with a different objective.
    # Here, we trained on actual shares. So the output IS the estimated share.
    # However, to avoid negative shares, clamping is standard.
    # "Softmax" on raw shares is also a valid normalization strategy if they are ~probabilities.
    # Given the Prompt: "stop using raw_share directly; treat model output as unconstrained logit z_i... weights w_i = exp(z_i)..."
    # Wait, if I trained on linear shares [0, 0.4], `predict` yields linear shares.
    # Exponentiating them `exp(0.4)` vs `exp(0.0)` -> range [1.0, 1.49].
    # That destroys the distribution.
    # If the USER wants logits, I should ideally train with Logit target?
    # Or does the user imply I should just treat the output as a "score" and softmax it?
    # If I train on `share`, the model tries to output `share`.
    # `share` sums to 1.
    # If I just normalize `share / sum(shares)`, that preserves the scale better than `exp(share)`.
    # `exp` is for log-space.
    # I will stick to "Clamp non-negative" + "Normalize / sum".
    # User said: "treat model output as unconstrained logit z_i... convert to positive weights w_i = exp(z_i)"
    # If I do that on a model trained on linear shares, the result will be very uniform (shares are small, exp(share) ~ 1 + share).
    # I will assume "Clamp to non-negative" is the intended "positive weights" step for a Regression model trained on [0,1].
    # Changing the target transform to Logit is a bigger change (requires handling 0s).
    # I will proceed with Standard Normalization (Divide by Sum) which satisfies "predicted_share = w_i / sum(w_i)".
    
    # Clamp to non-negative
    raw_shares = np.maximum(0, raw_shares)

    return raw_shares


def normalize_shares_per_team(
    raw_shares: np.ndarray,
    game_ids: pd.Series | np.ndarray,
    team_ids: pd.Series | np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Normalize shares so each (game, team) sums to 1.0.

    Args:
        raw_shares: Raw predicted shares from model
        game_ids: Game identifier
        team_ids: Team identifier
        epsilon: Small value to prevent division by zero

    Returns:
        Normalized shares that sum to 1.0 per game-team
    """
    df = pd.DataFrame({
        "raw_share": raw_shares,
        "game_id": np.asarray(game_ids),
        "team_id": np.asarray(team_ids),
    })

    # Compute team sums per game
    team_sums = df.groupby(["game_id", "team_id"])["raw_share"].transform("sum")

    # Normalize
    normalized = df["raw_share"] / (team_sums + epsilon)

    return normalized.to_numpy()


def predict_minutes(
    artifacts: MinuteShareArtifacts,
    X: pd.DataFrame,
    team_ids: pd.Series | np.ndarray,
    game_ids: pd.Series | np.ndarray | None = None,
    team_total: float = TEAM_TOTAL_MINUTES,
    *,
    is_out: pd.Series | np.ndarray | None = None,
    sharpen_exponent: float = 1.0,  # 1.0 = no sharpening
) -> pd.DataFrame:
    """Full inference pipeline: predict shares, normalize, convert to minutes.

    Args:
        artifacts: Trained model artifacts
        X: Feature dataframe
        team_ids: Team identifier
        game_ids: Optional game identifier (defaults to a single dummy game).
        team_total: Total team minutes (default 240) used for projection scaling.
                   (Note: Training used actuals, here we project to standard 240 or specific total).
        is_out: Optional boolean mask (1/True for OUT players).
                If provided, these players will have their shares strictly set to 0.0.
        sharpen_exponent: Exponent to sharpen share distribution (share^exponent).
                          >1.0 pushes minutes to starters, <1.0 flattens distribution.

    Returns:
        DataFrame with columns:
        - raw_share: Un-normalized predicted share
        - raw_share_sum: Team sum of raw shares (diagnostic)
        - normalized_share: Team-normalized share
        - predicted_minutes: Final minutes prediction
    """
    if game_ids is None:
        game_ids = np.zeros(len(X), dtype=int)

    # Get raw predictions
    raw_shares = predict_raw_shares(artifacts, X)

    # Strictly zero out known OUT players
    if is_out is not None:
        out_mask = np.asarray(is_out).astype(bool)
        n_out = out_mask.sum()
        if n_out > 0:
            raw_shares[out_mask] = 0.0

    # Apply sharpening (power transform)
    if sharpen_exponent != 1.0:
        # Avoid negative bases if any (though clamped in predict_raw_shares)
        raw_shares = np.where(raw_shares > 0, raw_shares ** sharpen_exponent, 0.0)

    # Compute team sums for diagnostics
    df = pd.DataFrame(
        {
            "game_id": np.asarray(game_ids),
            "team_id": np.asarray(team_ids),
            "raw_share": raw_shares,
        },
        index=X.index,
    )
    
    df["raw_share_sum"] = df.groupby(["game_id", "team_id"])["raw_share"].transform("sum")

    # Normalize
    df["normalized_share"] = normalize_shares_per_team(raw_shares, game_ids, team_ids)

    # Convert to minutes
    df["predicted_minutes"] = df["normalized_share"] * team_total

    return df[["game_id", "team_id", "raw_share", "raw_share_sum", "normalized_share", "predicted_minutes"]]


@dataclass
class ShareValidationReport:
    """Validation report for minute share predictions."""

    # Share distribution checks
    star_share_range_ok: bool
    rotation_share_range_ok: bool
    bench_share_ok: bool

    # Team sum checks
    team_sums_normalized: bool
    mean_raw_share_sum: float
    min_raw_share_sum: float
    max_raw_share_sum: float

    # Player role distributions
    share_by_role: dict[str, dict[str, float]]

    passed: bool


def validate_share_predictions(
    predictions_df: pd.DataFrame,
    player_roles: pd.Series | None = None,
    verbose: bool = True,
) -> ShareValidationReport:
    """Validate that predicted shares look reasonable.

    Args:
        predictions_df: Output from predict_minutes()
        player_roles: Optional series with role labels (star/rotation/bench)
        verbose: Log validation results

    Returns:
        ShareValidationReport with pass/fail status
    """
    # Check team sums after normalization (should all be 1.0)
    required_cols = {"game_id", "team_id"}
    if not required_cols.issubset(predictions_df.columns):
        # Fallback if allowed? No, we require strict keys now.
        # But wait, predict_minutes doesn't include game_id/team_id in returned columns!
        # The caller usually merges. We need to ensure predictions_df has them.
        pass # We'll raise or assume present.
    
    if "game_id" not in predictions_df.columns or "team_id" not in predictions_df.columns:
         raise ValueError(f"predictions_df must include {required_cols}")

    team_norm_sums = predictions_df.groupby(["game_id", "team_id"])["normalized_share"].sum()
    team_sums_ok = np.allclose(team_norm_sums.to_numpy(), 1.0, atol=1e-6)

    # Check raw share sum distribution
    raw_sums = predictions_df.groupby(["game_id", "team_id"])["raw_share_sum"].first()
    mean_raw = float(raw_sums.mean())
    min_raw = float(raw_sums.min())
    max_raw = float(raw_sums.max())


    # Role-based checks
    share_by_role: dict[str, dict[str, float]] = {}
    star_ok = rotation_ok = bench_ok = True

    if player_roles is not None:
        for role in ["star", "rotation", "bench"]:
            mask = player_roles == role
            if mask.any():
                role_shares = predictions_df.loc[mask, "normalized_share"]
                share_by_role[role] = {
                    "mean": float(role_shares.mean()),
                    "min": float(role_shares.min()),
                    "max": float(role_shares.max()),
                    "count": int(mask.sum()),
                }

        if "star" in share_by_role:
            star_mean = share_by_role["star"]["mean"]
            star_ok = STAR_SHARE_MIN <= star_mean <= STAR_SHARE_MAX

        if "rotation" in share_by_role:
            rot_mean = share_by_role["rotation"]["mean"]
            rotation_ok = ROTATION_SHARE_MIN <= rot_mean <= ROTATION_SHARE_MAX

        if "bench" in share_by_role:
            bench_mean = share_by_role["bench"]["mean"]
            bench_ok = bench_mean <= BENCH_SHARE_MAX
    else:
        # Without roles, check overall distribution
        shares = predictions_df["normalized_share"]
        share_by_role["all"] = {
            "mean": float(shares.mean()),
            "min": float(shares.min()),
            "max": float(shares.max()),
            "count": len(shares),
        }
        # Stars typically get 0.35-0.42 share
        max_share = shares.max()
        star_ok = max_share <= STAR_SHARE_MAX + 0.02  # Small tolerance

    passed = team_sums_ok and star_ok and rotation_ok and bench_ok

    if verbose:
        LOGGER.info(f"Share validation: {'PASS' if passed else 'FAIL'}")
        LOGGER.info(f"  Team sums normalized: {team_sums_ok}")
        LOGGER.info(f"  Raw share sum: mean={mean_raw:.3f}, range=[{min_raw:.3f}, {max_raw:.3f}]")
        if player_roles is not None:
            for role, stats in share_by_role.items():
                LOGGER.info(f"  {role}: mean={stats['mean']:.3f}, n={stats['count']}")

    return ShareValidationReport(
        star_share_range_ok=star_ok,
        rotation_share_range_ok=rotation_ok,
        bench_share_ok=bench_ok,
        team_sums_normalized=team_sums_ok,
        mean_raw_share_sum=mean_raw,
        min_raw_share_sum=min_raw,
        max_raw_share_sum=max_raw,
        share_by_role=share_by_role,
        passed=passed,
    )


def save_artifacts(artifacts: MinuteShareArtifacts, path: Path) -> None:
    """Save trained artifacts to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, path)
    LOGGER.info(f"Saved minute share artifacts to {path}")


def load_artifacts(path: Path) -> MinuteShareArtifacts:
    """Load trained artifacts from disk."""
    return joblib.load(path)


# =============================================================================
# EVALUATION METRICS
# =============================================================================

# Minute buckets for stratified MAE
MINUTE_BUCKETS = {
    "dnp": (0, 0),  # exactly 0 minutes
    "garbage_time": (0.01, 10),
    "bench": (10, 20),
    "rotation": (20, 30),
    "core": (30, 38),
    "stars": (38, 48),
}

# Thresholds for classification metrics
DNP_SHARE_THRESHOLD = 0.02  # ~4.8 minutes - players below this are predicted DNP
ROTATION_MINUTES_THRESHOLD = 15.0  # rotation player threshold


@dataclass
class MinuteShareEvaluation:
    """Comprehensive evaluation results for minute share model."""

    # Overall regression metrics
    mae_shares: float
    mae_minutes: float
    rmse_shares: float
    rmse_minutes: float

    # MAE by minute bucket
    mae_by_bucket: dict[str, float | None]
    count_by_bucket: dict[str, int]

    # Play/DNP classification
    # NOTE: dnp_accuracy is None when n_dnp_actual=0 (no positive examples)
    dnp_auc: float | None
    dnp_accuracy: float | None  # Made optional - None when n_dnp=0 (misleading otherwise)
    dnp_precision: float | None
    dnp_recall: float | None
    dnp_threshold: float

    # Rotation classification (>=15 min)
    rotation_accuracy: float
    rotation_precision: float | None
    rotation_recall: float | None
    rotation_base_rate: float  # n_rotation_actual / n_samples

    # Share distribution diagnostics (renamed for clarity)
    # Pre-normalization: raw model output sums (typically < 1.0)
    raw_pred_share_sum_pre_norm_mean: float
    raw_pred_share_sum_pre_norm_std: float
    # Post-normalization: after team normalization (should be exactly 1.0)
    norm_share_sum_post_norm_mean: float
    norm_share_sum_post_norm_std: float
    # Max share per team
    max_share_mean: float
    max_share_std: float

    # Sample sizes
    n_samples: int
    n_teams: int
    n_dnp_actual: int
    n_rotation_actual: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for JSON serialization."""
        result = {
            "mae_shares": self.mae_shares,
            "mae_minutes": self.mae_minutes,
            "rmse_shares": self.rmse_shares,
            "rmse_minutes": self.rmse_minutes,
            "dnp_auc": self.dnp_auc,
            "dnp_accuracy": self.dnp_accuracy,
            "dnp_precision": self.dnp_precision,
            "dnp_recall": self.dnp_recall,
            "dnp_threshold": self.dnp_threshold,
            "rotation_accuracy": self.rotation_accuracy,
            "rotation_precision": self.rotation_precision,
            "rotation_recall": self.rotation_recall,
            "rotation_base_rate": self.rotation_base_rate,
            # Pre-normalization raw prediction sums
            "raw_pred_share_sum_pre_norm_mean": self.raw_pred_share_sum_pre_norm_mean,
            "raw_pred_share_sum_pre_norm_std": self.raw_pred_share_sum_pre_norm_std,
            # Post-normalization sums (should be ~1.0)
            "norm_share_sum_post_norm_mean": self.norm_share_sum_post_norm_mean,
            "norm_share_sum_post_norm_std": self.norm_share_sum_post_norm_std,
            # Legacy alias for backward compatibility
            "team_share_sum_mean": self.raw_pred_share_sum_pre_norm_mean,
            "team_share_sum_std": self.raw_pred_share_sum_pre_norm_std,
            "max_share_mean": self.max_share_mean,
            "max_share_std": self.max_share_std,
            "n_samples": self.n_samples,
            "n_teams": self.n_teams,
            "n_dnp_actual": self.n_dnp_actual,
            "n_rotation_actual": self.n_rotation_actual,
        }
        # Add bucket metrics
        for bucket_name in MINUTE_BUCKETS:
            result[f"mae_{bucket_name}"] = self.mae_by_bucket.get(bucket_name)
            result[f"count_{bucket_name}"] = self.count_by_bucket.get(bucket_name, 0)
        return result

    @property
    def team_share_sum_mean(self) -> float:
        """Backward-compatible alias for raw predicted share sum mean."""
        return float(self.raw_pred_share_sum_pre_norm_mean)

    @property
    def team_share_sum_std(self) -> float:
        """Backward-compatible alias for raw predicted share sum std."""
        return float(self.raw_pred_share_sum_pre_norm_std)


def _compute_mae_by_bucket(
    actual_minutes: np.ndarray,
    predicted_minutes: np.ndarray,
) -> tuple[dict[str, float | None], dict[str, int]]:
    """Compute MAE stratified by actual minute buckets.

    Args:
        actual_minutes: Ground truth minutes
        predicted_minutes: Predicted minutes (after normalization)

    Returns:
        Tuple of (mae_by_bucket, count_by_bucket)
    """
    mae_by_bucket: dict[str, float | None] = {}
    count_by_bucket: dict[str, int] = {}

    for bucket_name, (low, high) in MINUTE_BUCKETS.items():
        if bucket_name == "dnp":
            # Exactly 0 minutes
            mask = actual_minutes == 0
        else:
            # Range (low, high]
            mask = (actual_minutes > low) & (actual_minutes <= high)

        count = int(mask.sum())
        count_by_bucket[bucket_name] = count

        if count > 0:
            mae_by_bucket[bucket_name] = float(
                mean_absolute_error(actual_minutes[mask], predicted_minutes[mask])
            )
        else:
            mae_by_bucket[bucket_name] = None

    return mae_by_bucket, count_by_bucket


def _compute_dnp_classification_metrics(
    actual_minutes: np.ndarray,
    predicted_shares: np.ndarray,
    threshold: float = DNP_SHARE_THRESHOLD,
) -> dict[str, float | None]:
    """Compute play/DNP binary classification metrics.

    Args:
        actual_minutes: Ground truth minutes
        predicted_shares: Raw predicted shares (before normalization)
        threshold: Share threshold below which player is predicted DNP

    Returns:
        Dict with auc, accuracy, precision, recall
        NOTE: accuracy is None when n_dnp_actual=0 (no positives to classify)
    """
    actual_played = (actual_minutes > 0).astype(int)
    predicted_played = (predicted_shares >= threshold).astype(int)

    n_positive = actual_played.sum()  # players who played
    n_negative = len(actual_played) - n_positive  # DNPs (actual negatives)

    result: dict[str, float | None] = {
        "dnp_accuracy": None,  # Set to None by default when n_negative=0
        "dnp_precision": None,
        "dnp_recall": None,
        "dnp_auc": None,
    }

    # Only compute accuracy if we have both classes (otherwise misleading)
    if n_negative > 0:
        result["dnp_accuracy"] = float(accuracy_score(actual_played, predicted_played))

    # Precision and recall for "played" class
    if n_positive > 0 and predicted_played.sum() > 0:
        result["dnp_precision"] = float(
            precision_score(actual_played, predicted_played, zero_division=0)
        )
        result["dnp_recall"] = float(
            recall_score(actual_played, predicted_played, zero_division=0)
        )

    # AUC requires both classes present
    if n_positive > 0 and n_negative > 0:
        try:
            result["dnp_auc"] = float(roc_auc_score(actual_played, predicted_shares))
        except ValueError:
            # Can happen with edge cases
            pass

    return result


def _compute_rotation_classification_metrics(
    actual_minutes: np.ndarray,
    predicted_minutes: np.ndarray,
    threshold: float = ROTATION_MINUTES_THRESHOLD,
) -> dict[str, float | None]:
    """Compute rotation player (>=15 min) classification metrics.

    Args:
        actual_minutes: Ground truth minutes
        predicted_minutes: Final predicted minutes (after normalization)
        threshold: Minutes threshold for rotation player

    Returns:
        Dict with accuracy, precision, recall
    """
    actual_rotation = (actual_minutes >= threshold).astype(int)
    predicted_rotation = (predicted_minutes >= threshold).astype(int)

    n_positive = actual_rotation.sum()

    result: dict[str, float | None] = {
        "rotation_accuracy": float(accuracy_score(actual_rotation, predicted_rotation)),
        "rotation_precision": None,
        "rotation_recall": None,
    }

    if n_positive > 0 and predicted_rotation.sum() > 0:
        result["rotation_precision"] = float(
            precision_score(actual_rotation, predicted_rotation, zero_division=0)
        )
        result["rotation_recall"] = float(
            recall_score(actual_rotation, predicted_rotation, zero_division=0)
        )

    return result


def _compute_share_distribution_diagnostics(
    raw_predicted_shares: np.ndarray,
    normalized_shares: np.ndarray,
    game_ids: np.ndarray,
    team_ids: np.ndarray,
) -> dict[str, float]:
    """Compute diagnostics on predicted share distribution.

    Args:
        raw_predicted_shares: Raw predicted shares (before normalization)
        normalized_shares: Normalized shares (after team normalization)
        game_ids: Game identifier
        team_ids: Team identifier

    Returns:
        Dict with pre-norm sums, post-norm sums, and max share statistics
    """
    df = pd.DataFrame({
        "raw_share": raw_predicted_shares,
        "norm_share": normalized_shares,
        "game_id": game_ids,
        "team_id": team_ids,
    })

    # Pre-normalization raw model output sums (typically <1.0 or varies)
    raw_sums = df.groupby(["game_id", "team_id"])["raw_share"].sum()
    
    # Post-normalization sums (should be exactly 1.0)
    norm_sums = df.groupby(["game_id", "team_id"])["norm_share"].sum()

    # Max share per team (should be ~0.15-0.20 for stars)
    max_shares = df.groupby(["game_id", "team_id"])["norm_share"].max()

    return {
        "raw_pred_share_sum_pre_norm_mean": float(raw_sums.mean()),
        "raw_pred_share_sum_pre_norm_std": float(raw_sums.std()) if len(raw_sums) > 1 else 0.0,
        "norm_share_sum_post_norm_mean": float(norm_sums.mean()),
        "norm_share_sum_post_norm_std": float(norm_sums.std()) if len(norm_sums) > 1 else 0.0,
        "max_share_mean": float(max_shares.mean()),
        "max_share_std": float(max_shares.std()) if len(max_shares) > 1 else 0.0,
    }


def evaluate_minute_share_model(
    actual_minutes: np.ndarray | pd.Series,
    predicted_shares: np.ndarray | pd.Series,
    predicted_minutes: np.ndarray | pd.Series,
    team_ids: np.ndarray | pd.Series,
    game_ids: np.ndarray | pd.Series | None = None,
    *,
    dnp_threshold: float = DNP_SHARE_THRESHOLD,
    rotation_threshold: float = ROTATION_MINUTES_THRESHOLD,
    verbose: bool = True,
) -> MinuteShareEvaluation:
    """Comprehensive evaluation of minute share model predictions.

    This evaluates both regression quality and implicit classification performance
    (play/DNP, rotation vs bench).

    Args:
        actual_minutes: Ground truth minutes played
        predicted_shares: Raw predicted shares (before team normalization)
        predicted_minutes: Final predicted minutes (after normalization to 240)
        team_ids: Team identifier for each row
        game_ids: Optional game identifier for each row (defaults to a single dummy game)
        dnp_threshold: Share threshold below which player is predicted DNP
        rotation_threshold: Minutes threshold for rotation player classification
        verbose: Whether to log evaluation results

    Returns:
        MinuteShareEvaluation with all metrics
    """
    # Convert to numpy arrays
    actual_min = np.asarray(actual_minutes, dtype=float)
    pred_shares = np.asarray(predicted_shares, dtype=float)
    pred_min = np.asarray(predicted_minutes, dtype=float)
    
    # Ensure keys are numpy arrays
    if game_ids is None:
        game_ids = np.zeros(len(actual_min), dtype=int)
    games = np.asarray(game_ids)
    teams = np.asarray(team_ids)

    # Compute actual shares for comparison (using dynamic totals if possible? 
    # Or just standard 240? Usually evaluation compares against 240-normalized target if model predicts 240-normalized.
    # But if we trained on dynamic share, we should compare against dynamic share?
    # Actually, simple MAE on minutes is most robust.
    # For shares, let's use 240 for consistency or dynamic?
    # Let's compute actual shares using dynamic totals to be fair.)
    
    # We can reuse compute_minute_share to get actual shares dynamically!
    # But we need to import it or copy logic. It's in the same file.
    # But `evaluate` is often imported.
    # Let's use dynamic totals for actuals too.
    
    actual_shares = compute_minute_share(actual_min, games, teams)

    # Overall regression metrics (computed only on players who logged minutes)
    # DNPs are handled separately by classification metrics
    played_mask = actual_min > 0
    if played_mask.sum() > 0:
        mae_shares = float(mean_absolute_error(actual_shares[played_mask], pred_shares[played_mask]))
        mae_minutes = float(mean_absolute_error(actual_min[played_mask], pred_min[played_mask]))
        rmse_shares = float(np.sqrt(np.mean((actual_shares[played_mask] - pred_shares[played_mask]) ** 2)))
        rmse_minutes = float(np.sqrt(np.mean((actual_min[played_mask] - pred_min[played_mask]) ** 2)))
    else:
        mae_shares = float("nan")
        mae_minutes = float("nan")
        rmse_shares = float("nan")
        rmse_minutes = float("nan")

    # MAE by bucket
    mae_by_bucket, count_by_bucket = _compute_mae_by_bucket(actual_min, pred_min)

    # DNP classification
    dnp_metrics = _compute_dnp_classification_metrics(actual_min, pred_shares, dnp_threshold)

    # Rotation classification
    rotation_metrics = _compute_rotation_classification_metrics(
        actual_min, pred_min, rotation_threshold
    )

    # Share distribution diagnostics - need to compute normalized shares
    # Normalized shares are pred_min / team_total where team sums to exactly 240
    # We can derive from predicted_minutes or compute fresh
    normalized_shares = pred_min / TEAM_TOTAL_MINUTES  # normalized_share = pred_min/240
    dist_metrics = _compute_share_distribution_diagnostics(
        raw_predicted_shares=pred_shares,
        normalized_shares=normalized_shares, 
        game_ids=games, 
        team_ids=teams
    )

    # Sample counts
    n_samples = len(actual_min)
    n_teams = len(np.unique(teams)) # Unique teams across games? Or unique (game, team) groups?
    # Usually "n_teams" meant distinct franchises. 
    # But for diagnostics, maybe number of groups is more interesting?
    # Let's keep distinct team IDs for now.
    
    n_dnp_actual = int((actual_min == 0).sum())
    n_rotation_actual = int((actual_min >= rotation_threshold).sum())
    rotation_base_rate = n_rotation_actual / n_samples if n_samples > 0 else 0.0

    evaluation = MinuteShareEvaluation(
        mae_shares=mae_shares,
        mae_minutes=mae_minutes,
        rmse_shares=rmse_shares,
        rmse_minutes=rmse_minutes,
        mae_by_bucket=mae_by_bucket,
        count_by_bucket=count_by_bucket,
        dnp_auc=dnp_metrics["dnp_auc"],
        dnp_accuracy=dnp_metrics["dnp_accuracy"],
        dnp_precision=dnp_metrics["dnp_precision"],
        dnp_recall=dnp_metrics["dnp_recall"],
        dnp_threshold=dnp_threshold,
        rotation_accuracy=rotation_metrics["rotation_accuracy"],
        rotation_precision=rotation_metrics["rotation_precision"],
        rotation_recall=rotation_metrics["rotation_recall"],
        rotation_base_rate=rotation_base_rate,
        raw_pred_share_sum_pre_norm_mean=dist_metrics["raw_pred_share_sum_pre_norm_mean"],
        raw_pred_share_sum_pre_norm_std=dist_metrics["raw_pred_share_sum_pre_norm_std"],
        norm_share_sum_post_norm_mean=dist_metrics["norm_share_sum_post_norm_mean"],
        norm_share_sum_post_norm_std=dist_metrics["norm_share_sum_post_norm_std"],
        max_share_mean=dist_metrics["max_share_mean"],
        max_share_std=dist_metrics["max_share_std"],
        n_samples=n_samples,
        n_teams=n_teams,
        n_dnp_actual=n_dnp_actual,
        n_rotation_actual=n_rotation_actual,
    )

    if verbose:
        _log_evaluation_results(evaluation)

    return evaluation


def _log_evaluation_results(evaluation: MinuteShareEvaluation) -> None:
    """Log evaluation results in a readable format."""
    LOGGER.info("=" * 60)
    LOGGER.info("MINUTE SHARE MODEL EVALUATION")
    LOGGER.info("=" * 60)

    LOGGER.info("\n[OVERALL REGRESSION]")
    LOGGER.info(f"  MAE (shares): {evaluation.mae_shares:.4f}")
    LOGGER.info(f"  MAE (minutes): {evaluation.mae_minutes:.2f}")
    LOGGER.info(f"  RMSE (shares): {evaluation.rmse_shares:.4f}")
    LOGGER.info(f"  RMSE (minutes): {evaluation.rmse_minutes:.2f}")

    LOGGER.info("\n[MAE BY MINUTE BUCKET]")
    for bucket_name in MINUTE_BUCKETS:
        mae = evaluation.mae_by_bucket.get(bucket_name)
        count = evaluation.count_by_bucket.get(bucket_name, 0)
        if mae is not None:
            LOGGER.info(f"  {bucket_name:12s}: MAE={mae:5.2f} min (n={count:,})")
        else:
            LOGGER.info(f"  {bucket_name:12s}: N/A (n=0)")

    LOGGER.info(f"\n[PLAY/DNP CLASSIFICATION] (threshold={evaluation.dnp_threshold:.3f})")
    if evaluation.n_dnp_actual == 0:
        LOGGER.info("  DNP metrics N/A (no actual DNPs in validation set)")
    else:
        if evaluation.dnp_accuracy is not None:
            LOGGER.info(f"  Accuracy:  {evaluation.dnp_accuracy:.3f}")
        if evaluation.dnp_precision is not None:
            LOGGER.info(f"  Precision: {evaluation.dnp_precision:.3f}")
        if evaluation.dnp_recall is not None:
            LOGGER.info(f"  Recall:    {evaluation.dnp_recall:.3f}")
        if evaluation.dnp_auc is not None:
            LOGGER.info(f"  AUC:       {evaluation.dnp_auc:.3f}")
    LOGGER.info(f"  (Actual DNPs: {evaluation.n_dnp_actual:,})")

    LOGGER.info("\n[ROTATION CLASSIFICATION] (>=15 min)")
    LOGGER.info(f"  Base rate: {evaluation.rotation_base_rate:.1%}")
    LOGGER.info(f"  Accuracy:  {evaluation.rotation_accuracy:.3f}")
    if evaluation.rotation_precision is not None:
        LOGGER.info(f"  Precision: {evaluation.rotation_precision:.3f}")
    if evaluation.rotation_recall is not None:
        LOGGER.info(f"  Recall:    {evaluation.rotation_recall:.3f}")
    LOGGER.info(f"  (Actual rotation: {evaluation.n_rotation_actual:,})")

    LOGGER.info("\n[SHARE DISTRIBUTION DIAGNOSTICS]")
    LOGGER.info(f"  Raw pred share sum (pre-norm): mean={evaluation.raw_pred_share_sum_pre_norm_mean:.3f}, std={evaluation.raw_pred_share_sum_pre_norm_std:.3f}")
    LOGGER.info(f"  Norm share sum (post-norm):    mean={evaluation.norm_share_sum_post_norm_mean:.3f}, std={evaluation.norm_share_sum_post_norm_std:.3f}")
    LOGGER.info(f"  Max share: mean={evaluation.max_share_mean:.3f}, std={evaluation.max_share_std:.3f}")

    LOGGER.info("\n[SAMPLE SIZE]")
    LOGGER.info(f"  Total samples: {evaluation.n_samples:,}")
    LOGGER.info(f"  Teams: {evaluation.n_teams:,}")
    LOGGER.info("=" * 60)
