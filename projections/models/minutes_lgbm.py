"""LightGBM quantile trainer for Minutes V1 smoke slice."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import typer
from click.core import ParameterSource
from sklearn.metrics import mean_absolute_error
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from projections import paths
from projections.metrics.minutes import compute_mae_by_actual_minutes_bucket
from projections.minutes_v1 import modeling
from projections.minutes_v1.artifacts import compute_feature_hash, ensure_run_directory, write_json
from projections.minutes_v1.constants import AvailabilityStatus
from projections.minutes_v1.config import load_training_config
from projections.minutes_v1.datasets import (
    KEY_COLUMNS,
    deduplicate_latest,
    ensure_columns,
    load_feature_frame,
    write_ids_csv,
)
from projections.minutes_v1.validation import sample_anti_leak_check
from projections.validation.data_quality import validate_feature_ranges
from projections.registry.manifest import (
    load_manifest,
    save_manifest,
    register_model,
)


UTC = timezone.utc
DEFAULT_TRAIN_START = datetime(2024, 12, 1, tzinfo=UTC)
DEFAULT_TRAIN_END = datetime(2024, 12, 18, tzinfo=UTC)
DEFAULT_CAL_END = datetime(2024, 12, 24, tzinfo=UTC)
DEFAULT_VAL_END = datetime(2024, 12, 31, tzinfo=UTC)
ALPHA_TARGET = 0.1
STRICT_VAL_TOLERANCE = 0.02
RELAXED_VAL_TOLERANCE = 0.05
RAW_NON_MONO_ALERT = 0.35
FINAL_NON_MONO_TOL = 1e-6
BUCKET_MIN_ENFORCEMENT = 200
WINKLER_ALPHA = 0.2
CENTER_WIDTH_CENTER_GRID = np.arange(-4.0, 4.0 + 1e-9, 0.5)
CENTER_WIDTH_SCALE_GRID = np.arange(0.1, 1.8 + 1e-9, 0.1)
CENTER_WIDTH_TARGET_TOL = 0.02
CENTER_WIDTH_MIN_HALF_WIDTH = 0.25
MIN_MINUTES = 0.0
MAX_MINUTES = 48.0
TWO_SIDED_CENTER_MIN = -3.0
TWO_SIDED_CENTER_MAX = 3.0
TWO_SIDED_CENTER_STEP = 0.5
TWO_SIDED_LEFT_SCALE_MIN = 0.05
TWO_SIDED_LEFT_SCALE_MAX = 1.6
TWO_SIDED_LEFT_SCALE_STEP = 0.05
TWO_SIDED_RIGHT_SCALE_MIN = 0.4
TWO_SIDED_RIGHT_SCALE_MAX = 1.2
TWO_SIDED_RIGHT_SCALE_STEP = 0.05
TWO_SIDED_PENALTY = 200.0
TWO_SIDED_LOW_UNDER_WEIGHT = 3.0
TWO_SIDED_LOW_OVER_WEIGHT = 1.0
TWO_SIDED_HIGH_OVER_WEIGHT = 3.0
TWO_SIDED_HIGH_UNDER_WEIGHT = 1.0
PLAYABLE_MIN_P50_DEFAULT = 10.0
PLAYABLE_P10_ACCEPTANCE = (0.08, 0.12)
PLAYABLE_P90_ACCEPTANCE = (0.87, 0.94)
PLAYABLE_WINKLER_TOLERANCE = 1.0


@dataclass
class TwoSidedParams:
    """Per-bucket two-sided calibration parameters."""

    a: float
    s_lo: float
    s_hi: float
    n: int


@dataclass
class PlayProbabilityArtifacts:
    """Artifacts for the play-probability model + calibrator."""

    model: lgb.LGBMClassifier
    imputer: SimpleImputer
    calibrator: IsotonicRegression | None = None
    feature_names: list[str] | None = None


TwoSidedParams.__module__ = "projections.models.minutes_lgbm"
PlayProbabilityArtifacts.__module__ = "projections.models.minutes_lgbm"

module_alias = sys.modules.setdefault("projections.models.minutes_lgbm", sys.modules[__name__])
setattr(module_alias, "TwoSidedParams", TwoSidedParams)
setattr(module_alias, "PlayProbabilityArtifacts", PlayProbabilityArtifacts)


app = typer.Typer(help=__doc__)


_DEFAULT_PARAMETER_SOURCES = {ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP}

# MLFlow tracking URI - uses same SQLite backend as the systemd service
MLFLOW_TRACKING_URI = "sqlite:////home/daniel/projections-data/mlflow/mlflow.db"
MLFLOW_EXPERIMENT_NAME = "minutes_v1_training"


def _log_to_mlflow(
    *,
    run_id: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    quantiles: Any,  # QuantileArtifacts or dict
    feature_columns: list[str],
    run_dir: Path,
    windows_meta: dict[str, Any],
) -> None:
    """Log training run to MLFlow with params, metrics, feature importance, and artifacts."""
    if not MLFLOW_AVAILABLE:
        typer.echo("[mlflow] MLFlow not available, skipping logging", err=True)
        return

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=run_id):
            # Log parameters
            flat_params = {
                "run_id": run_id,
                "train_start": windows_meta["train"]["start"],
                "train_end": windows_meta["train"]["end"],
                "cal_start": windows_meta["cal"]["start"],
                "cal_end": windows_meta["cal"]["end"],
                "val_start": windows_meta["val"]["start"],
                "val_end": windows_meta["val"]["end"],
                "n_features": len(feature_columns),
            }
            for key, value in params.items():
                if value is not None and not isinstance(value, (dict, list)):
                    flat_params[key] = value
            mlflow.log_params(flat_params)

            # Log metrics - flatten nested dicts
            def sanitize_name(name: str) -> str:
                """Sanitize metric name for MLFlow (alphanumerics, underscores, dashes, periods, spaces, colons, slashes)."""
                return name.replace("|", "_").replace("<", "lt").replace(">", "gt").replace("(", "").replace(")", "")

            flat_metrics: dict[str, float] = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    # Handle per_bucket and other nested dicts
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            for sub_sub_key, sub_sub_value in sub_value.items():
                                if isinstance(sub_sub_value, (int, float)) and sub_sub_value is not None:
                                    metric_name = sanitize_name(f"{key}_{sub_key}_{sub_sub_key}")
                                    flat_metrics[metric_name] = float(sub_sub_value)
                        elif isinstance(sub_value, (int, float)) and sub_value is not None:
                            metric_name = sanitize_name(f"{key}_{sub_key}")
                            flat_metrics[metric_name] = float(sub_value)
                elif isinstance(value, (int, float)) and value is not None:
                    flat_metrics[key] = float(value)
            mlflow.log_metrics(flat_metrics)

            # Log feature importance from the p50 quantile model
            # Handle both QuantileArtifacts and dict types
            models_dict = getattr(quantiles, "models", quantiles) if hasattr(quantiles, "models") else quantiles
            if isinstance(models_dict, dict) and 0.5 in models_dict:
                model_p50 = models_dict[0.5]
                importance_gain = model_p50.feature_importances_
                importance_df = pd.DataFrame({
                    "feature": feature_columns,
                    "importance_gain": importance_gain,
                }).sort_values("importance_gain", ascending=False)

                # Log as artifact
                importance_path = run_dir / "feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(str(importance_path))

                # Log top features as metrics for quick comparison
                for i, row in importance_df.head(20).iterrows():
                    safe_name = str(row["feature"]).replace(".", "_")[:50]
                    mlflow.log_metric(f"importance_{safe_name}", float(row["importance_gain"]))

            # Log bucket-level MAE as metrics for easy comparison
            for key in metrics:
                if key.startswith("val_mae_"):
                    mlflow.log_metric(key, float(metrics[key]))

            # Log key artifacts
            for artifact_name in ["metrics.json", "meta.json", "conformal_offsets.json", "feature_columns.json"]:
                artifact_path = run_dir / artifact_name
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))

            typer.echo(f"[mlflow] Logged run '{run_id}' to experiment '{MLFLOW_EXPERIMENT_NAME}'")

    except Exception as e:
        typer.echo(f"[mlflow] Warning: Failed to log to MLFlow: {e}", err=True)


def _apply_training_overrides(
    ctx: typer.Context,
    cli_params: dict[str, Any],
    config_path: Path | None,
) -> dict[str, Any]:
    """Load YAML overrides and merge them into CLI defaults."""

    if config_path is None:
        return cli_params

    config = load_training_config(config_path)
    overrides = config.model_dump(exclude_unset=True)
    for name, value in overrides.items():
        if name not in cli_params:
            continue
        source = ctx.get_parameter_source(name)
        if source in _DEFAULT_PARAMETER_SOURCES:
            cli_params[name] = value
    return cli_params


def _normalize_utc(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


@dataclass(frozen=True)
class DateWindow:
    """Simple inclusive date window tracked in UTC."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def from_bounds(cls, name: str, start: datetime, end: datetime) -> "DateWindow":
        start_ts = _normalize_utc(start)
        end_ts = _normalize_utc(end)
        if end_ts < start_ts:
            raise ValueError(f"{name.title()} window end precedes start ({start_ts} → {end_ts})")
        return cls(name=name, start=start_ts, end=end_ts)

    def slice(self, df: pd.DataFrame, *, date_column: str = "game_date") -> pd.DataFrame:
        ensure_columns(df, [date_column])
        working = df.copy()
        working[date_column] = pd.to_datetime(working[date_column]).dt.normalize()
        start_date = self.start.tz_convert(None).normalize()
        end_date = self.end.tz_convert(None).normalize()
        mask = (working[date_column] >= start_date) & (working[date_column] <= end_date)
        sliced = working.loc[mask].copy()
        if sliced.empty:
            raise ValueError(
                f"{self.name.title()} window produced zero rows "
                f"[{start_date.date()} → {end_date.date()}]"
            )
        return sliced

    def to_metadata(self) -> dict[str, str]:
        return {
            "start": self.start.isoformat().replace("+00:00", "Z"),
            "end": self.end.isoformat().replace("+00:00", "Z"),
        }


def _load_feature_frame_with_schema(
    *,
    features_path: Path | None,
    data_root: Path,
    season: int | None,
    month: int | None,
    target_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    feature_df = load_feature_frame(
        features_path=features_path,
        data_root=data_root,
        season=season,
        month=month,
    )
    ensure_columns(
        feature_df,
        [target_col, "game_date", "feature_as_of_ts", *KEY_COLUMNS],
    )

    # Anti-leak validation: ensure no feature timestamps exceed tip time
    if "tip_ts" in feature_df.columns:
        sample_anti_leak_check(
            feature_df,
            tip_col="tip_ts",
            sample_size=min(2000, len(feature_df)),
        )
        typer.echo("[validation] Anti-leak check passed")

    # Data quality validation: check feature value ranges
    violations = validate_feature_ranges(feature_df, strict=False)
    if violations:
        typer.echo(f"[validation] Data quality warnings: {len(violations)} issues detected", err=True)

    # NOTE: minutes_v1 is modeling minutes | plays; exclude any probability-to-play features.
    feature_columns = modeling.infer_feature_columns(
        feature_df,
        target_col=target_col,
        excluded={"prior_play_prob", "play_prob", "play_probability", "p_play"},
    )
    return feature_df, feature_columns


def _warn_overlap(train: DateWindow, cal: DateWindow) -> None:
    if cal.start <= train.end:
        typer.echo(
            "Calibration window overlaps the training window — rows within the overlap "
            "will influence both the LightGBM fit and the conformal offsets.",
            err=True,
        )


def _coverage_rates(y_true: pd.Series, low: pd.Series, high: pd.Series) -> tuple[float, float]:
    """Return coverage rates for lower/upper quantiles."""

    if len(y_true) == 0:
        raise ValueError("Cannot compute coverage on empty series.")
    lower = float((y_true <= low).mean())
    upper = float((y_true <= high).mean())
    return lower, upper


def _enforce_coverage(
    label: str,
    value: float,
    target: float,
    tolerance: float,
    *,
    allow_failure: bool = False,
) -> None:
    delta = abs(value - target)
    if delta > tolerance:
        typer.echo(
            f"[coverage] {label}={value:.3f} outside target {target:.3f} ± {tolerance:.3f}",
            err=True,
        )
        if not allow_failure:
            raise typer.Exit(code=1)


def _winkler_score(
    y_true: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    high: pd.Series | np.ndarray,
    *,
    alpha: float = WINKLER_ALPHA,
) -> np.ndarray:
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(low, dtype=float)
    hi = np.asarray(high, dtype=float)
    width = hi - lo
    scores = width.astype(float).copy()
    below = y < lo
    above = y > hi
    scores[below] = width[below] + (2.0 / alpha) * (lo[below] - y[below])
    scores[above] = width[above] + (2.0 / alpha) * (y[above] - hi[above])
    return scores


def _minutes_bin(p50: pd.Series) -> pd.Series:
    return pd.cut(p50, bins=[-1.0, 8.0, 18.0, 1e6], labels=["<8", "8-18", ">18"])


def _bucket_key(df: pd.DataFrame, bucket_mode: str = "starter,p50bins") -> pd.Series:
    normalized = bucket_mode.strip().lower()
    if normalized == "none":
        return pd.Series("__global__", index=df.index, dtype="string")

    ensure_columns(df, {"p50_pred"})
    tokens = tuple(token.strip().lower() for token in bucket_mode.split(",") if token.strip())
    if not tokens:
        raise ValueError("Bucket mode must specify at least one component.")

    components: list[pd.Series] = []
    for token in tokens:
        if token == "starter":
            starter_series = pd.Series(df.get("starter_flag", 0), index=df.index)
            starter_labels = (
                starter_series.fillna(0)
                .astype(int)
                .map({0: "bench", 1: "starter"})
                .fillna("bench")
                .astype(str)
            )
            components.append(starter_labels)
        elif token == "p50bins":
            components.append(_minutes_bin(df["p50_pred"]).astype(str))
        elif token == "injury_snapshot":
            ensure_columns(df, {"injury_snapshot_missing"})
            injury_labels = (
                df["injury_snapshot_missing"]
                .fillna(1)
                .astype(int)
                .map({0: "injury_present", 1: "injury_missing"})
                .fillna("injury_missing")
                .astype(str)
            )
            components.append(injury_labels)
        else:
            raise ValueError(f"Unsupported bucket component '{token}'.")

    bucket_strings = components[0].astype(str)
    for series in components[1:]:
        bucket_strings = bucket_strings + "|" + series.astype(str)
    return bucket_strings


def _filter_out_players(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"status", "injury_snapshot_missing"}
    if not required_cols.issubset(df.columns):
        return df
    status_col = df["status"].astype(str).str.upper()
    injuries_missing = df["injury_snapshot_missing"].fillna(1).astype(int)
    keep_mask = (status_col != AvailabilityStatus.OUT.value) | (injuries_missing == 1)
    filtered = df.loc[keep_mask].copy()
    if filtered.empty:
        raise ValueError("Filtering OUT players removed all rows; cannot proceed.")
    return filtered


def _quantile_shift(y: np.ndarray, qhat: np.ndarray, target: float) -> float:
    lo, hi = -30.0, 30.0
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        coverage = float(np.mean(y <= (qhat + mid)))
        if coverage < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _halfwidths(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    left = np.maximum(p50 - p10, eps)
    right = np.maximum(p90 - p50, eps)
    return left, right


def _project_two_sided(
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    center_shift: np.ndarray,
    scale_lo: np.ndarray,
    scale_hi: np.ndarray,
    *,
    eps_minw: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right = _halfwidths(p10, p50, p90)
    p50_adj = p50 + center_shift
    p10_adj = p50_adj - scale_lo * left
    p90_adj = p50_adj + scale_hi * right
    if eps_minw > 0:
        p10_adj = np.minimum(p10_adj, p50_adj - eps_minw)
        p90_adj = np.maximum(p90_adj, p50_adj + eps_minw)
    return p10_adj, p50_adj, p90_adj


def _clip_quantiles(
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    *,
    lower: float = MIN_MINUTES,
    upper: float = MAX_MINUTES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p50_clipped = np.clip(p50, lower, upper)
    p10_clipped = np.clip(p10, lower, upper)
    p90_clipped = np.clip(p90, lower, upper)
    p10_final = np.minimum(p10_clipped, p50_clipped)
    p90_final = np.maximum(p90_clipped, p50_clipped)
    return p10_final, p50_clipped, p90_final


def _train_play_probability_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int,
) -> PlayProbabilityArtifacts:
    if X_train.empty:
        raise ValueError("Cannot train play probability model on an empty frame.")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train_imp = imputer.fit_transform(X_train)
    y_array = y_train.to_numpy(dtype=int)
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=64,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=0.1,
        random_state=random_state,
    )
    model.fit(X_train_imp, y_array)
    return PlayProbabilityArtifacts(
        model=model,
        imputer=imputer,
        calibrator=None,
        feature_names=list(X_train.columns),
    )


def _fit_play_probability_calibrator(
    artifacts: PlayProbabilityArtifacts,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
) -> None:
    if X_cal.empty:
        raise ValueError("Cannot calibrate play probability with an empty calibration set.")
    probs = predict_play_probability(artifacts, X_cal, calibrated=False)
    labels = y_cal.to_numpy(dtype=float)
    order = np.argsort(probs)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs[order], labels[order])
    artifacts.calibrator = iso


def predict_play_probability(
    artifacts: PlayProbabilityArtifacts,
    X: pd.DataFrame,
    *,
    calibrated: bool = True,
) -> np.ndarray:
    X_imp = artifacts.imputer.transform(X)
    if artifacts.feature_names:
        X_imp = pd.DataFrame(X_imp, columns=artifacts.feature_names, index=X.index)
    raw = artifacts.model.predict_proba(X_imp)[:, 1]
    if calibrated and artifacts.calibrator is not None:
        raw = artifacts.calibrator.predict(raw)
    return np.clip(raw, 0.0, 1.0)


def apply_play_probability_mixture(
    df: pd.DataFrame,
    play_prob: np.ndarray,
) -> pd.DataFrame:
    """Attach p_play-weighted minutes without mutating conditional quantiles.

    The minutes stack strictly models conditional minutes—i.e., distributions given the player
    checks into the game. Downstream consumers (and DFS tooling) sometimes need an expectation
    that folds in availability. When this helper is invoked, it writes additional columns rather
    than overwriting the conditional quantiles so callers can choose explicitly.
    """

    probs = np.clip(play_prob, 0.0, 1.0)
    if "play_prob" not in df.columns:
        df["play_prob"] = probs
    else:
        df["play_prob"] = np.clip(df["play_prob"].to_numpy(dtype=float), 0.0, 1.0)
    conditional_cols = ["minutes_p10", "minutes_p50", "minutes_p90"]
    missing = [col for col in conditional_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "apply_play_probability_mixture requires conditional minutes columns; "
            f"missing: {', '.join(missing)}"
        )
    play_values = df["play_prob"].to_numpy(dtype=float)
    for column in conditional_cols:
        cond = df[column].to_numpy(dtype=float)
        df[f"{column}_uncond"] = cond * play_values
    return df


def _brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((y_true - probs) ** 2))


def _expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    bins: int = 10,
) -> float:
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(probs)
    if total == 0:
        return 0.0
    ece = 0.0
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= start) & (probs < end)
        if not np.any(mask):
            continue
        mean_pred = float(np.mean(probs[mask]))
        mean_true = float(np.mean(y_true[mask]))
        ece += (mask.sum() / total) * abs(mean_pred - mean_true)
    return float(ece)


def _coverage_fraction(y: np.ndarray, quantile: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.mean(y <= quantile))


def _compute_playable_subset_metrics(
    val_df: pd.DataFrame,
    *,
    target_col: str,
    minutes_threshold: float,
) -> dict[str, Any]:
    """Compute calibration metrics for rows that meet the playable filter."""

    ensure_columns(val_df, {target_col, "p10_cond", "p90_cond", "p50"})
    playable_mask = val_df["p50"] >= minutes_threshold
    playable = val_df.loc[playable_mask].copy()
    playable_count = int(len(playable))
    play_targets = playable.get("plays_target")
    if play_targets is not None:
        play_targets = play_targets.astype(int)
    else:
        play_targets = (playable[target_col] > 0).astype(int)
    playable_cond = playable[play_targets == 1]
    playable_cond_count = int(len(playable_cond))

    play_prob_series = playable.get("play_prob")
    play_probs = (
        play_prob_series.to_numpy(dtype=float)
        if play_prob_series is not None
        else np.ones(playable_count, dtype=float)
    )
    play_labels = play_targets.to_numpy(dtype=float) if playable_count else np.array([])

    metrics: dict[str, Any] = {
        "playable_count": playable_count,
        "playable_cond_count": playable_cond_count,
        "val_p10_cond_playable": None,
        "val_p90_cond_playable": None,
        "val_inside_cond_playable": None,
        "val_winkler_cond_playable": None,
        "val_play_prob_brier_playable": None,
        "val_play_prob_ece_playable": None,
        "playable_minutes_threshold": minutes_threshold,
    }

    if playable_count == 0:
        return metrics

    metrics["val_play_prob_brier_playable"] = _brier_score(play_labels, play_probs)
    metrics["val_play_prob_ece_playable"] = _expected_calibration_error(play_labels, play_probs)

    if playable_cond_count == 0:
        return metrics

    cond_p10, cond_p90 = _coverage_rates(
        playable_cond[target_col],
        playable_cond["p10_cond"],
        playable_cond["p90_cond"],
    )
    winkler = float(
        np.mean(
            _winkler_score(
                playable_cond[target_col],
                playable_cond["p10_cond"],
                playable_cond["p90_cond"],
                alpha=WINKLER_ALPHA,
            )
        )
    )

    metrics["val_p10_cond_playable"] = cond_p10
    metrics["val_p90_cond_playable"] = cond_p90
    metrics["val_inside_cond_playable"] = cond_p90 - cond_p10
    metrics["val_winkler_cond_playable"] = winkler
    return metrics


def _evaluate_playable_acceptance(
    metrics: dict[str, Any],
    *,
    winkler_baseline: float | None,
    winkler_tolerance: float,
    p10_bounds: tuple[float, float] = PLAYABLE_P10_ACCEPTANCE,
    p90_bounds: tuple[float, float] = PLAYABLE_P90_ACCEPTANCE,
) -> dict[str, Any]:
    """Determine whether playable subset metrics pass acceptance thresholds."""

    reasons: list[str] = []
    p10 = metrics.get("val_p10_cond_playable")
    p90 = metrics.get("val_p90_cond_playable")
    winkler = metrics.get("val_winkler_cond_playable")
    playable_count = metrics.get("playable_count", 0)
    playable_cond = metrics.get("playable_cond_count", 0)

    if not playable_count:
        reasons.append("no playable rows to evaluate")
    elif not playable_cond:
        reasons.append("no positive-minute playable rows for conditional metrics")

    if p10 is None or not (p10_bounds[0] <= p10 <= p10_bounds[1]):
        reasons.append(f"p10_cond_playable={p10!r} outside {p10_bounds}")
    if p90 is None or not (p90_bounds[0] <= p90 <= p90_bounds[1]):
        reasons.append(f"p90_cond_playable={p90!r} outside {p90_bounds}")

    winkler_baseline_val = winkler_baseline
    if winkler_baseline_val is not None:
        limit = winkler_baseline_val + winkler_tolerance
        if winkler is None or winkler > limit:
            reasons.append(
                f"winkler_cond_playable={winkler!r} exceeds baseline {winkler_baseline_val} + {winkler_tolerance}"
            )

    acceptance = {
        "passed": not reasons,
        "reasons": reasons,
        "p10_bounds": {"lo": p10_bounds[0], "hi": p10_bounds[1]},
        "p90_bounds": {"lo": p90_bounds[0], "hi": p90_bounds[1]},
        "winkler_baseline": winkler_baseline_val,
        "winkler_tolerance": winkler_tolerance,
        "playable_count": playable_count,
        "playable_cond_count": playable_cond,
    }
    return acceptance


def _grid_values(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 10))
        current += step
    return values


def _fit_two_sided_params_for_group(
    y: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    *,
    tol: float,
    alpha: float,
    a_grid: list[float] | None = None,
    slo_grid: list[float] | None = None,
    shi_grid: list[float] | None = None,
) -> TwoSidedParams:
    if y.size == 0:
        raise ValueError("Cannot fit two-sided parameters on empty arrays.")
    if a_grid is None:
        a_grid = _grid_values(TWO_SIDED_CENTER_MIN, TWO_SIDED_CENTER_MAX, TWO_SIDED_CENTER_STEP)
    if slo_grid is None:
        slo_grid = _grid_values(TWO_SIDED_LEFT_SCALE_MIN, TWO_SIDED_LEFT_SCALE_MAX, TWO_SIDED_LEFT_SCALE_STEP)
    if shi_grid is None:
        shi_grid = _grid_values(TWO_SIDED_RIGHT_SCALE_MIN, TWO_SIDED_RIGHT_SCALE_MAX, TWO_SIDED_RIGHT_SCALE_STEP)

    left, right = _halfwidths(p10, p50, p90)
    best_feasible: tuple[float, float, float, float] | None = None
    best_penalized: tuple[float, float, float, float] | None = None
    for shift in a_grid:
        p50_shifted = p50 + shift
        for scale_lo in slo_grid:
            candidate_p10 = p50_shifted - scale_lo * left
            for scale_hi in shi_grid:
                candidate_p90 = p50_shifted + scale_hi * right
                eval_p10, eval_p50, eval_p90 = _clip_quantiles(candidate_p10, p50_shifted, candidate_p90)
                cov10 = _coverage_fraction(y, eval_p10)
                cov90 = _coverage_fraction(y, eval_p90)
                winkler = float(np.mean(_winkler_score(y, eval_p10, eval_p90, alpha=alpha)))
                if abs(cov10 - ALPHA_TARGET) <= tol and abs(cov90 - (1.0 - ALPHA_TARGET)) <= tol:
                    if best_feasible is None or winkler < best_feasible[0]:
                        best_feasible = (winkler, shift, scale_lo, scale_hi)
                miss_low = cov10 - ALPHA_TARGET
                miss_high = cov90 - (1.0 - ALPHA_TARGET)
                low_weight = TWO_SIDED_LOW_UNDER_WEIGHT if miss_low < 0 else TWO_SIDED_LOW_OVER_WEIGHT
                high_weight = TWO_SIDED_HIGH_OVER_WEIGHT if miss_high > 0 else TWO_SIDED_HIGH_UNDER_WEIGHT
                penalty = winkler + TWO_SIDED_PENALTY * (low_weight * miss_low**2 + high_weight * miss_high**2)
                if best_penalized is None or penalty < best_penalized[0]:
                    best_penalized = (penalty, shift, scale_lo, scale_hi)
    pick = best_feasible or best_penalized
    if pick is None:
        raise RuntimeError("Failed to fit two-sided parameters for the provided group.")
    _, a, s_lo, s_hi = pick
    return TwoSidedParams(a=float(a), s_lo=float(s_lo), s_hi=float(s_hi), n=int(len(y)))


def _fit_two_sided_offsets(
    working: pd.DataFrame,
    *,
    label_col: str,
    k: int,
    tol: float,
    alpha: float,
) -> dict[str, dict[str, float]]:
    y = working[label_col].to_numpy(dtype=float)
    p10 = working["p10_pred"].to_numpy(dtype=float)
    p50 = working["p50_pred"].to_numpy(dtype=float)
    p90 = working["p90_pred"].to_numpy(dtype=float)
    global_params = _fit_two_sided_params_for_group(y, p10, p50, p90, tol=tol, alpha=alpha)
    offsets: dict[str, dict[str, float]] = {
        "__global__": {
            "a": global_params.a,
            "s_lo": global_params.s_lo,
            "s_hi": global_params.s_hi,
            "n": global_params.n,
        }
    }
    grouped = working.groupby("bucket")
    for bucket, group in grouped:
        bucket_size = len(group)
        if bucket_size == 0:
            continue
        if bucket_size < 50:
            offsets[str(bucket)] = {
                "a": global_params.a,
                "s_lo": global_params.s_lo,
                "s_hi": global_params.s_hi,
                "n": int(bucket_size),
            }
            continue
        params = _fit_two_sided_params_for_group(
            group[label_col].to_numpy(dtype=float),
            group["p10_pred"].to_numpy(dtype=float),
            group["p50_pred"].to_numpy(dtype=float),
            group["p90_pred"].to_numpy(dtype=float),
            tol=tol,
            alpha=alpha,
        )
        weight = bucket_size / (bucket_size + k) if k > 0 else 1.0
        offsets[str(bucket)] = {
            "a": float(weight * params.a + (1 - weight) * global_params.a),
            "s_lo": float(weight * params.s_lo + (1 - weight) * global_params.s_lo),
            "s_hi": float(weight * params.s_hi + (1 - weight) * global_params.s_hi),
            "n": int(bucket_size),
        }
    return offsets


def fit_conformal_offsets(
    cal_df: pd.DataFrame,
    *,
    label_col: str = "minutes",
    k: int = 200,
    mode: str = "tail-deltas",
    bucket_mode: str = "starter,p50bins",
) -> dict[str, dict[str, float]]:
    ensure_columns(cal_df, {label_col, "p10_pred", "p50_pred", "p90_pred"})
    if k < 0:
        raise ValueError("Shrinkage strength k must be non-negative")
    working = cal_df.copy()
    working["bucket"] = _bucket_key(working, bucket_mode=bucket_mode)
    mode_key = mode.strip().lower()
    if mode_key == "tail-deltas":
        return _fit_tail_delta_offsets(working, label_col=label_col, k=k)
    if mode_key == "center-width":
        return _fit_center_width_offsets(working, label_col=label_col, k=k)
    if mode_key == "two-sided":
        return _fit_two_sided_offsets(
            working,
            label_col=label_col,
            k=k,
            tol=CENTER_WIDTH_TARGET_TOL,
            alpha=WINKLER_ALPHA,
        )
    raise ValueError(f"Unsupported conformal mode '{mode}'.")


def _fit_tail_delta_offsets(
    working: pd.DataFrame,
    *,
    label_col: str,
    k: int,
) -> dict[str, dict[str, float]]:
    y = working[label_col].to_numpy(dtype=float)
    p10 = working["p10_pred"].to_numpy(dtype=float)
    p90 = working["p90_pred"].to_numpy(dtype=float)

    global_low = _quantile_shift(y, p10, ALPHA_TARGET)
    global_high = _quantile_shift(y, p90, 1.0 - ALPHA_TARGET)

    offsets: dict[str, dict[str, float]] = {
        "__global__": {"d10": float(global_low), "d90": float(global_high), "n": int(len(working))}
    }
    grouped = working.groupby("bucket")
    for bucket, group in grouped:
        bucket_size = len(group)
        if bucket_size == 0:
            continue
        yb = group[label_col].to_numpy(dtype=float)
        p10b = group["p10_pred"].to_numpy(dtype=float)
        p90b = group["p90_pred"].to_numpy(dtype=float)
        if bucket_size < 50:
            offsets[str(bucket)] = {
                "d10": float(global_low),
                "d90": float(global_high),
                "n": int(bucket_size),
            }
            continue
        bucket_low = _quantile_shift(yb, p10b, ALPHA_TARGET)
        bucket_high = _quantile_shift(yb, p90b, 1.0 - ALPHA_TARGET)
        weight = bucket_size / (bucket_size + k)
        offsets[str(bucket)] = {
            "d10": float(weight * bucket_low + (1 - weight) * global_low),
            "d90": float(weight * bucket_high + (1 - weight) * global_high),
            "n": int(bucket_size),
        }
    return offsets


def _grid_search_center_width_params(
    y: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
) -> dict[str, float]:
    if y.size == 0:
        raise ValueError("Cannot fit center-width offsets on an empty frame.")
    half_width = np.maximum(p90 - p50, CENTER_WIDTH_MIN_HALF_WIDTH)
    best: dict[str, float] | None = None
    best_relaxed: dict[str, float] | None = None
    best_relaxed_metric: tuple[float, float] | None = None
    target_high = 1.0 - ALPHA_TARGET
    for center_shift in CENTER_WIDTH_CENTER_GRID:
        center = p50 + center_shift
        for width_scale in CENTER_WIDTH_SCALE_GRID:
            lower = center - width_scale * half_width
            upper = center + width_scale * half_width
            cov10 = float(np.mean(y <= lower))
            cov90 = float(np.mean(y <= upper))
            winkler = float(np.mean(_winkler_score(y, lower, upper, alpha=WINKLER_ALPHA)))
            delta10 = abs(cov10 - ALPHA_TARGET)
            delta90 = abs(cov90 - target_high)
            candidate = {
                "center": float(center_shift),
                "scale": float(width_scale),
                "cov10": cov10,
                "cov90": cov90,
                "delta10": delta10,
                "delta90": delta90,
                "winkler": winkler,
            }
            if delta10 <= CENTER_WIDTH_TARGET_TOL and delta90 <= CENTER_WIDTH_TARGET_TOL:
                if best is None or winkler < best["winkler"]:
                    best = candidate
            relaxed_metric = (delta10 + delta90, winkler)
            if best_relaxed is None or relaxed_metric < best_relaxed_metric:
                best_relaxed = candidate
                best_relaxed_metric = relaxed_metric
    if best is not None:
        return best
    if best_relaxed is not None:
        return best_relaxed
    raise RuntimeError("Failed to discover viable center-width parameters.")


def _fit_center_width_offsets(
    working: pd.DataFrame,
    *,
    label_col: str,
    k: int,
) -> dict[str, dict[str, float]]:
    y = working[label_col].to_numpy(dtype=float)
    p50 = working["p50_pred"].to_numpy(dtype=float)
    p90 = working["p90_pred"].to_numpy(dtype=float)
    global_params = _grid_search_center_width_params(y, p50, p90)
    offsets: dict[str, dict[str, float]] = {
        "__global__": {
            "center": float(global_params["center"]),
            "scale": float(global_params["scale"]),
            "n": int(len(working)),
        }
    }
    grouped = working.groupby("bucket")
    for bucket, group in grouped:
        bucket_size = len(group)
        if bucket_size == 0:
            continue
        if bucket_size < 50:
            offsets[str(bucket)] = {
                "center": float(global_params["center"]),
                "scale": float(global_params["scale"]),
                "n": int(bucket_size),
            }
            continue
        bucket_params = _grid_search_center_width_params(
            group[label_col].to_numpy(dtype=float),
            group["p50_pred"].to_numpy(dtype=float),
            group["p90_pred"].to_numpy(dtype=float),
        )
        weight = bucket_size / (bucket_size + k) if k > 0 else 1.0
        blended_center = float(weight * bucket_params["center"] + (1 - weight) * global_params["center"])
        blended_scale = float(weight * bucket_params["scale"] + (1 - weight) * global_params["scale"])
        offsets[str(bucket)] = {
            "center": blended_center,
            "scale": blended_scale,
            "n": int(bucket_size),
        }
    return offsets


def apply_conformal(
    df: pd.DataFrame,
    offsets: dict[str, dict[str, float]],
    *,
    mode: str = "tail-deltas",
    bucket_mode: str = "starter,p50bins",
) -> pd.DataFrame:
    ensure_columns(df, {"p10_pred", "p50_pred", "p90_pred"})
    if "__global__" not in offsets:
        raise ValueError("Offsets payload missing '__global__' entry")
    working = df.copy()
    working["bucket"] = _bucket_key(working, bucket_mode=bucket_mode)

    def _lookup(bucket: str, key: str) -> float:
        entry = offsets.get(bucket)
        if entry is None:
            entry = offsets["__global__"]
        return float(entry[key])

    mode_key = mode.strip().lower()
    if mode_key == "two-sided":
        return _apply_two_sided(working, offsets)
    if mode_key == "center-width":
        return _apply_center_width(working, offsets)

    if mode_key != "tail-deltas":
        raise ValueError(f"Unsupported conformal mode '{mode}'.")

    delta10 = working["bucket"].map(lambda bucket: _lookup(bucket, "d10")).to_numpy(dtype=float)
    delta90 = working["bucket"].map(lambda bucket: _lookup(bucket, "d90")).to_numpy(dtype=float)

    p10_base = working["p10_pred"].to_numpy(dtype=float)
    p50_base = working["p50_pred"].to_numpy(dtype=float)
    p90_base = working["p90_pred"].to_numpy(dtype=float)

    center_delta = 0.5 * (delta10 + delta90)
    diff = delta10 - delta90
    upper_limit = 2.0 * (p50_base - p10_base)
    lower_limit = -2.0 * (p90_base - p50_base)
    clipped_diff = np.clip(diff, lower_limit, upper_limit)
    working["offset_clip_flag"] = (clipped_diff != diff)
    delta10 = center_delta + 0.5 * clipped_diff
    delta90 = center_delta - 0.5 * clipped_diff
    center_delta = 0.5 * (delta10 + delta90)
    working["p50_adj"] = p50_base + center_delta

    working["p10_unc"] = p10_base + delta10
    working["p90_unc"] = p90_base + delta90
    working["p10_adj"] = np.minimum(working["p10_unc"], working["p50_adj"])
    working["p90_adj"] = np.maximum(working["p90_unc"], working["p50_adj"])
    working["raw_cross_flag"] = (working["p10_unc"] > working["p50_adj"]) | (
        working["p50_adj"] > working["p90_unc"]
    )
    working["final_cross_flag"] = (working["p10_adj"] > working["p50_adj"]) | (
        working["p50_adj"] > working["p90_adj"]
    )
    p10_final, p50_final, p90_final = _clip_quantiles(
        working["p10_adj"].to_numpy(dtype=float),
        working["p50_adj"].to_numpy(dtype=float),
        working["p90_adj"].to_numpy(dtype=float),
    )
    working["p10_adj"] = p10_final
    working["p50_adj"] = p50_final
    working["p90_adj"] = p90_final
    return working


def _apply_center_width(working: pd.DataFrame, offsets: dict[str, dict[str, float]]) -> pd.DataFrame:
    def _lookup(bucket: str, key: str) -> float:
        entry = offsets.get(bucket)
        if entry is None:
            entry = offsets["__global__"]
        value = entry.get(key)
        if value is None:
            raise KeyError(f"Offsets missing '{key}' for bucket '{bucket}'.")
        return float(value)

    center_shift = working["bucket"].map(lambda bucket: _lookup(bucket, "center")).to_numpy(dtype=float)
    width_scale = working["bucket"].map(lambda bucket: _lookup(bucket, "scale")).to_numpy(dtype=float)
    p50_base = working["p50_pred"].to_numpy(dtype=float)
    p90_base = working["p90_pred"].to_numpy(dtype=float)
    raw_half_width = np.maximum(p90_base - p50_base, 0.0)
    center_unc = p50_base + center_shift
    working["p10_unc"] = center_unc - width_scale * raw_half_width
    working["p90_unc"] = center_unc + width_scale * raw_half_width

    clipped_half_width = np.maximum(p90_base - p50_base, CENTER_WIDTH_MIN_HALF_WIDTH)
    scaled_half_width = width_scale * clipped_half_width
    working["offset_clip_flag"] = clipped_half_width != raw_half_width
    working["p50_adj"] = center_unc
    working["p10_adj"] = center_unc - scaled_half_width
    working["p90_adj"] = center_unc + scaled_half_width
    working["raw_cross_flag"] = False
    working["final_cross_flag"] = False
    p10_final, p50_final, p90_final = _clip_quantiles(
        working["p10_adj"].to_numpy(dtype=float),
        working["p50_adj"].to_numpy(dtype=float),
        working["p90_adj"].to_numpy(dtype=float),
    )
    working["p10_adj"] = p10_final
    working["p50_adj"] = p50_final
    working["p90_adj"] = p90_final
    return working


def _apply_two_sided(working: pd.DataFrame, offsets: dict[str, dict[str, float]]) -> pd.DataFrame:
    def _lookup(bucket: str, key: str) -> float:
        entry = offsets.get(bucket)
        if entry is None:
            entry = offsets["__global__"]
        value = entry.get(key)
        if value is None:
            raise KeyError(f"Offsets missing '{key}' for bucket '{bucket}'.")
        return float(value)

    bucket_series = working["bucket"]
    center_shift = bucket_series.map(lambda bucket: _lookup(bucket, "a")).to_numpy(dtype=float)
    scale_lo = bucket_series.map(lambda bucket: _lookup(bucket, "s_lo")).to_numpy(dtype=float)
    scale_hi = bucket_series.map(lambda bucket: _lookup(bucket, "s_hi")).to_numpy(dtype=float)

    p10_base = working["p10_pred"].to_numpy(dtype=float)
    p50_base = working["p50_pred"].to_numpy(dtype=float)
    p90_base = working["p90_pred"].to_numpy(dtype=float)

    p10_adj, p50_adj, p90_adj = _project_two_sided(
        p10_base,
        p50_base,
        p90_base,
        center_shift,
        scale_lo,
        scale_hi,
        eps_minw=CENTER_WIDTH_MIN_HALF_WIDTH,
    )
    working["p10_adj"] = p10_adj
    working["p50_adj"] = p50_adj
    working["p90_adj"] = p90_adj
    working["p10_unc"] = p10_adj
    working["p90_unc"] = p90_adj
    working["offset_clip_flag"] = False
    working["raw_cross_flag"] = False
    working["final_cross_flag"] = False
    p10_final, p50_final, p90_final = _clip_quantiles(p10_adj, p50_adj, p90_adj)
    working["p10_adj"] = p10_final
    working["p50_adj"] = p50_final
    working["p90_adj"] = p90_final
    return working


def _window_defaults(
    train_start: datetime,
    train_end: datetime,
    cal_start: datetime | None,
    cal_end: datetime,
    val_start: datetime | None,
    val_end: datetime,
) -> tuple[DateWindow, DateWindow, DateWindow]:
    cal_start_value = cal_start or (train_end + timedelta(days=1))
    val_start_value = val_start or (cal_end + timedelta(days=1))
    train_window = DateWindow.from_bounds("train", train_start, train_end)
    cal_window = DateWindow.from_bounds("cal", cal_start_value, cal_end)
    val_window = DateWindow.from_bounds("val", val_start_value, val_end)
    _warn_overlap(train_window, cal_window)
    if cal_window.end >= val_window.start:
        raise ValueError(
            "Validation window must start after the calibration window ends "
            f"({cal_window.to_metadata()} vs {val_window.to_metadata()})"
        )
    return train_window, cal_window, val_window


@app.command()
def main(
    ctx: typer.Context,
    config_path: Path | None = typer.Option(
        None,
        "--config",
        help="Optional YAML config file describing the training run.",
    ),
    train_start: datetime = typer.Option(
        DEFAULT_TRAIN_START,
        "--train-start",
        "--start",
        help="Training window start (UTC, inclusive).",
        show_default=True,
    ),
    train_end: datetime = typer.Option(
        DEFAULT_TRAIN_END,
        "--train-end",
        help="Training window end (UTC, inclusive).",
        show_default=True,
    ),
    cal_start: datetime | None = typer.Option(
        None,
        "--cal-start",
        help="Calibration window start (UTC, inclusive). Defaults to the day after --train-end.",
    ),
    cal_end: datetime = typer.Option(
        DEFAULT_CAL_END,
        "--cal-end",
        "--end",
        help="Calibration window end (UTC, inclusive).",
        show_default=True,
    ),
    val_start: datetime | None = typer.Option(
        None,
        "--val-start",
        help="Validation window start (UTC, inclusive). Defaults to the day after --cal-end.",
    ),
    val_end: datetime = typer.Option(
        DEFAULT_VAL_END,
        "--val-end",
        help="Validation window end (UTC, inclusive).",
        show_default=True,
    ),
    run_id: str | None = typer.Option(None, help="Unique identifier for the training run."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        help="Root directory containing data/* (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    season: int | None = typer.Option(None, help="Season year for default features path (e.g., 2024)."),
    month: int | None = typer.Option(None, help="Month partition (1-12) for default path."),
    features: Path | None = typer.Option(None, help="Explicit feature parquet path."),
    artifact_root: Path = typer.Option(Path("artifacts/minutes_lgbm"), help="Where to store run artifacts."),
    target_col: str = typer.Option("minutes", help="Target column to predict."),
    random_state: int = typer.Option(42, help="Random seed for LightGBM."),
    conformal_buckets: str = typer.Option(
        "starter,p50bins",
        help="Strategy for conformal offsets (starter,p50bins | starter,p50bins,injury_snapshot | none).",
    ),
    conformal_k: int = typer.Option(200, help="Shrinkage strength toward global offsets."),
    conformal_mode: str = typer.Option(
        "two-sided",
        help="Conformal adjustment mode ('tail-deltas', 'center-width', or 'two-sided').",
    ),
    cal_days: int | None = typer.Option(
        None,
        help="If set, derive calibration window as the N days ending before --val-start.",
    ),
    tolerance_relaxed: bool = typer.Option(
        False,
        "--tolerance-relaxed",
        help="Loosen coverage tolerance to ±0.05 for experimentation.",
        is_flag=True,
    ),
    winkler_baseline: float | None = typer.Option(
        None,
        help="Optional ceiling for Winkler score to guard against regressions.",
    ),
    allow_guard_failure: bool = typer.Option(
        False,
        "--allow-guard-failure",
        help="Emit guardrail violations as warnings instead of aborting (for analysis only).",
        is_flag=True,
    ),
    fold_id: str | None = typer.Option(
        None,
        "--fold-id",
        help="Optional identifier for the evaluation fold (recorded in metrics metadata).",
    ),
    playable_min_p50: float = typer.Option(
        PLAYABLE_MIN_P50_DEFAULT,
        "--playable-min-p50",
        help="Minimum predicted p50 minutes to treat a row as playable when computing guardrail metrics.",
    ),
    playable_winkler_baseline: float | None = typer.Option(
        None,
        "--playable-winkler-baseline",
        help="Baseline conditional Winkler value for playable subset acceptance.",
    ),
    playable_winkler_tolerance: float = typer.Option(
        PLAYABLE_WINKLER_TOLERANCE,
        "--playable-winkler-tolerance",
        help="Allowed absolute increase relative to the playable Winkler baseline.",
    ),
    enable_play_prob_head: bool = typer.Option(
        True,
        "--enable-play-prob-head/--disable-play-prob-head",
        help="Train and persist a dedicated play probability classifier head.",
    ),
    enable_play_prob_mixing: bool = typer.Option(
        False,
        "--enable-play-prob-mixing",
        help="(lab only) allow play_prob mixing for unconditional outputs; kept false to preserve conditional semantics.",
        is_flag=True,
    ),
    disable_play_prob: bool = typer.Option(
        False,
        "--disable-play-prob",
        help="Deprecated; overrides enable_play_prob_head and skips play probability training/usage.",
        is_flag=True,
    ),
    lgbm_n_estimators: int | None = typer.Option(
        None,
        "--lgbm-n-estimators",
        help="Override LightGBM n_estimators (quantile trees).",
    ),
    lgbm_max_depth: int | None = typer.Option(
        None,
        "--lgbm-max-depth",
        help="Override LightGBM max_depth (quantile trees).",
    ),
    lgbm_learning_rate: float | None = typer.Option(
        None,
        "--lgbm-learning-rate",
        help="Override LightGBM learning_rate (quantile trees).",
    ),
    split_col: str | None = typer.Option(
        None,
        "--split-col",
        help="Optional explicit split column (values: train|cal|val). When set, date windows are ignored.",
    ),
) -> None:
    """Train LightGBM quantile models with a dedicated calibration window."""

    cli_params: dict[str, Any] = {
        "train_start": train_start,
        "train_end": train_end,
        "cal_start": cal_start,
        "cal_end": cal_end,
        "val_start": val_start,
        "val_end": val_end,
        "run_id": run_id,
        "data_root": data_root,
        "season": season,
        "month": month,
        "features": features,
        "artifact_root": artifact_root,
        "target_col": target_col,
        "random_state": random_state,
        "conformal_buckets": conformal_buckets,
        "conformal_k": conformal_k,
        "conformal_mode": conformal_mode,
        "cal_days": cal_days,
        "tolerance_relaxed": tolerance_relaxed,
        "winkler_baseline": winkler_baseline,
        "allow_guard_failure": allow_guard_failure,
        "fold_id": fold_id,
        "playable_min_p50": playable_min_p50,
        "playable_winkler_baseline": playable_winkler_baseline,
        "playable_winkler_tolerance": playable_winkler_tolerance,
        "enable_play_prob_head": enable_play_prob_head,
        "enable_play_prob_mixing": enable_play_prob_mixing,
        "disable_play_prob": disable_play_prob,
        "lgbm_n_estimators": lgbm_n_estimators,
        "lgbm_max_depth": lgbm_max_depth,
        "lgbm_learning_rate": lgbm_learning_rate,
        "split_col": split_col,
    }
    resolved_params = _apply_training_overrides(ctx, cli_params, config_path)
    train_start = resolved_params["train_start"]
    train_end = resolved_params["train_end"]
    cal_start = resolved_params["cal_start"]
    cal_end = resolved_params["cal_end"]
    val_start = resolved_params["val_start"]
    val_end = resolved_params["val_end"]
    run_id = resolved_params["run_id"]
    data_root = resolved_params["data_root"]
    season = resolved_params["season"]
    month = resolved_params["month"]
    features = resolved_params["features"]
    artifact_root = resolved_params["artifact_root"]
    target_col = resolved_params["target_col"]
    random_state = resolved_params["random_state"]
    conformal_buckets = resolved_params["conformal_buckets"]
    conformal_k = resolved_params["conformal_k"]
    conformal_mode = resolved_params["conformal_mode"]
    cal_days = resolved_params["cal_days"]
    tolerance_relaxed = resolved_params["tolerance_relaxed"]
    winkler_baseline = resolved_params["winkler_baseline"]
    allow_guard_failure = resolved_params["allow_guard_failure"]
    fold_id = resolved_params["fold_id"]
    playable_min_p50 = resolved_params["playable_min_p50"]
    playable_winkler_baseline = resolved_params["playable_winkler_baseline"]
    playable_winkler_tolerance = resolved_params["playable_winkler_tolerance"]
    enable_play_prob_head = resolved_params.get("enable_play_prob_head", True)
    enable_play_prob_mixing = resolved_params.get("enable_play_prob_mixing", False)
    disable_play_prob = resolved_params.get("disable_play_prob", False)
    if disable_play_prob:
        enable_play_prob_head = False
    lgbm_n_estimators = resolved_params.get("lgbm_n_estimators")
    lgbm_max_depth = resolved_params.get("lgbm_max_depth")
    lgbm_learning_rate = resolved_params.get("lgbm_learning_rate")
    split_col = resolved_params.get("split_col")

    if run_id is None:
        raise typer.BadParameter("--run-id is required (set via CLI flag or config file).")

    bucket_mode = conformal_buckets.strip().lower()
    supported_buckets = {"starter,p50bins", "starter,p50bins,injury_snapshot", "none"}
    if bucket_mode not in supported_buckets:
        raise typer.BadParameter(
            f"Unsupported --conformal-buckets '{conformal_buckets}'. Choose from: {', '.join(sorted(supported_buckets))}."
        )
    if conformal_k < 0:
        raise typer.BadParameter("--conformal-k must be non-negative")
    mode_key = conformal_mode.strip().lower()
    supported_modes = {"tail-deltas", "center-width", "two-sided"}
    if mode_key not in supported_modes:
        raise typer.BadParameter(
            f"Unsupported --conformal-mode '{conformal_mode}'. Choose from: {', '.join(sorted(supported_modes))}."
        )
    tolerance = RELAXED_VAL_TOLERANCE if tolerance_relaxed else STRICT_VAL_TOLERANCE

    if cal_days is not None:
        if cal_days <= 0:
            raise typer.BadParameter("--cal-days must be positive")
        if val_start is None:
            raise typer.BadParameter("--cal-days requires --val-start so the calibration window can be derived")
        cal_end = val_start - timedelta(days=1)
        cal_start_candidate = val_start - timedelta(days=cal_days)
        min_cal_start = train_end + timedelta(days=1)
        if cal_start_candidate < min_cal_start:
            raise typer.BadParameter(
                "Calibration window would overlap the training window; reduce --cal-days or shorten training."
            )
        cal_start = cal_start_candidate

    train_window, cal_window, val_window = _window_defaults(
        train_start=train_start,
        train_end=train_end,
        cal_start=cal_start,
        cal_end=cal_end,
        val_start=val_start,
        val_end=val_end,
    )

    feature_df, feature_columns = _load_feature_frame_with_schema(
        features_path=features,
        data_root=data_root,
        season=season,
        month=month,
        target_col=target_col,
    )

    # NOTE: starter_flag must be inference-available.
    # Never derive it from box score minutes (e.g. top-5 minute getters), or we will leak label
    # information into both the model fit and the bucketed conformal calibration.
    starter_flag_source = "default_zero"
    if "is_confirmed_starter" in feature_df.columns or "is_projected_starter" in feature_df.columns:
        confirmed = feature_df.get("is_confirmed_starter")
        projected = feature_df.get("is_projected_starter")
        if confirmed is not None:
            confirmed_bool = confirmed.fillna(False).astype(bool)
        else:
            confirmed_bool = pd.Series(False, index=feature_df.index)
        if projected is not None:
            projected_bool = projected.fillna(False).astype(bool)
        else:
            projected_bool = pd.Series(False, index=feature_df.index)
        feature_df["starter_flag"] = (confirmed_bool | projected_bool).astype(int)
        starter_flag_source = "is_confirmed_starter|is_projected_starter"
    else:
        feature_df["starter_flag"] = 0
    typer.echo(f"[train] starter_flag derived from {starter_flag_source}")

    if split_col:
        ensure_columns(feature_df, [split_col])
        split_series = feature_df[split_col].astype(str).str.strip().str.lower()
        mapped = split_series.map({"train": "train", "cal": "cal", "calibration": "cal", "val": "val", "valid": "val", "validation": "val"})
        if mapped.isna().any():
            bad = split_series[mapped.isna()].drop_duplicates().tolist()
            raise ValueError(f"Unsupported values found in --split-col '{split_col}': {bad[:10]}")
        feature_df = feature_df.copy()
        feature_df[split_col] = mapped.astype("string")
        if "game_id" in feature_df.columns:
            game_split_counts = feature_df.groupby("game_id")[split_col].nunique(dropna=False)
            if (game_split_counts > 1).any():
                bad_games = game_split_counts[game_split_counts > 1].index.astype(str).tolist()
                raise ValueError(
                    "Explicit split column assigns the same game_id to multiple splits (sample): "
                    + ", ".join(bad_games[:10])
                )
        raw_train = feature_df[feature_df[split_col] == "train"].copy()
        raw_cal = feature_df[feature_df[split_col] == "cal"].copy()
        raw_val = feature_df[feature_df[split_col] == "val"].copy()
        missing_splits = [name for name, frame in (("train", raw_train), ("cal", raw_cal), ("val", raw_val)) if frame.empty]
        if missing_splits:
            raise ValueError(
                f"Explicit split column '{split_col}' produced empty split(s): {', '.join(missing_splits)}. "
                "Adjust split cutoffs and ensure snapshot coverage."
            )
        train_df = _filter_out_players(raw_train)
        cal_df = _filter_out_players(raw_cal)
        val_df = _filter_out_players(raw_val)
    else:
        train_df = _filter_out_players(train_window.slice(feature_df))
        cal_df = _filter_out_players(cal_window.slice(feature_df))
        val_df = _filter_out_players(val_window.slice(feature_df))
    for frame in (train_df, cal_df, val_df):
        frame["plays_target"] = (frame[target_col] > 0).astype(int)

    typer.echo(
        f"Training LightGBM quantiles on {len(train_df):,} rows "
        f"(cal={len(cal_df):,}, val={len(val_df):,}) with {len(feature_columns)} features"
    )

    play_prob_artifacts: PlayProbabilityArtifacts | None = None
    if enable_play_prob_mixing:
        typer.echo(
            "Play probability mixing flag enabled (lab); conditional minutes remain the default output.",
            err=True,
        )
    if not enable_play_prob_head:
        typer.echo("Play probability head disabled; emitting conditional minutes only.", err=True)
    else:
        X_train_play_prob = train_df[feature_columns]
        y_train_play_prob = train_df["plays_target"]
        if y_train_play_prob.nunique() >= 2:
            play_prob_artifacts = _train_play_probability_model(
                X_train_play_prob,
                y_train_play_prob,
                random_state=random_state,
            )
        else:
            typer.echo("Play probability training skipped (single class).", err=True)

    train_cond_df = train_df[train_df["plays_target"] == 1]
    cal_cond_df = cal_df[cal_df["plays_target"] == 1]
    if train_cond_df.empty or cal_cond_df.empty:
        raise ValueError("Positive-minute rows required in both train and calibration windows.")

    X_train = train_cond_df[feature_columns]
    y_train = train_cond_df[target_col]
    X_cal = cal_cond_df[feature_columns]
    y_cal = cal_cond_df[target_col]
    X_val = val_df[feature_columns]

    lgbm_params: dict[str, float | int] = {}
    if lgbm_n_estimators is not None:
        lgbm_params["n_estimators"] = int(lgbm_n_estimators)
    if lgbm_max_depth is not None:
        lgbm_params["max_depth"] = int(lgbm_max_depth)
    if lgbm_learning_rate is not None:
        lgbm_params["learning_rate"] = float(lgbm_learning_rate)
    params_arg = lgbm_params or None

    quantiles = modeling.train_lightgbm_quantiles(
        X_train,
        y_train,
        random_state=random_state,
        params=params_arg,
    )
    cal_preds = modeling.predict_quantiles(quantiles, X_cal)
    cal_p10_raw = cal_preds[0.1]
    cal_p50_raw = cal_preds[0.5]
    cal_p90_raw = cal_preds[0.9]
    cal_p10_base = np.minimum(cal_p10_raw, cal_p50_raw)
    cal_p90_base = np.maximum(cal_p90_raw, cal_p50_raw)

    calibrator = modeling.ConformalIntervalCalibrator(alpha_low=ALPHA_TARGET, alpha_high=ALPHA_TARGET)
    calibrator.fit(y_cal.to_numpy(), cal_p10_base, cal_p90_base)
    calibrator_offsets = calibrator.export_offsets()
    cal_p10_cal, cal_p90_cal = calibrator.calibrate(cal_p10_base, cal_p90_base)

    if play_prob_artifacts is not None:
        y_cal_play = cal_df["plays_target"]
        if y_cal_play.nunique() >= 2:
            _fit_play_probability_calibrator(play_prob_artifacts, cal_df[feature_columns], y_cal_play)
        else:
            typer.echo("Play probability calibration skipped (single class).", err=True)

    val_quantiles = modeling.predict_quantiles(quantiles, X_val)
    val_p10_raw = val_quantiles[0.1]
    val_p50_raw = val_quantiles[0.5]
    val_p90_raw = val_quantiles[0.9]
    val_p10_base = np.minimum(val_p10_raw, val_p50_raw)
    val_p90_base = np.maximum(val_p90_raw, val_p50_raw)
    val_p10_cal, val_p90_cal = calibrator.calibrate(val_p10_base, val_p90_base)

    cal_eval = cal_cond_df.copy()
    cal_eval["p10_pred"] = cal_p10_cal
    cal_eval["p50_pred"] = cal_p50_raw
    cal_eval["p90_pred"] = cal_p90_cal

    val_eval = val_df.copy()
    val_eval["p10_raw"] = val_p10_raw
    val_eval["p50_raw"] = val_p50_raw
    val_eval["p90_raw"] = val_p90_raw
    val_eval["p10_pred"] = val_p10_cal
    val_eval["p50_pred"] = val_p50_raw
    val_eval["p90_pred"] = val_p90_cal

    if bucket_mode in {"starter,p50bins", "starter,p50bins,injury_snapshot"}:
        offsets = fit_conformal_offsets(
            cal_eval,
            label_col=target_col,
            k=conformal_k,
            mode=mode_key,
            bucket_mode=bucket_mode,
        )
    else:
        if mode_key == "tail-deltas":
            offsets = {
                "__global__": {
                    "d10": float(-calibrator_offsets["low_adjustment"]),
                    "d90": float(calibrator_offsets["high_adjustment"]),
                    "n": len(cal_eval),
                }
            }
        elif mode_key == "center-width":
            global_params = _grid_search_center_width_params(
                cal_eval[target_col].to_numpy(dtype=float),
                cal_eval["p50_pred"].to_numpy(dtype=float),
                cal_eval["p90_pred"].to_numpy(dtype=float),
            )
            offsets = {
                "__global__": {
                    "center": float(global_params["center"]),
                    "scale": float(global_params["scale"]),
                    "n": len(cal_eval),
                }
            }
        else:  # two-sided
            params = _fit_two_sided_params_for_group(
                cal_eval[target_col].to_numpy(dtype=float),
                cal_eval["p10_pred"].to_numpy(dtype=float),
                cal_eval["p50_pred"].to_numpy(dtype=float),
                cal_eval["p90_pred"].to_numpy(dtype=float),
                tol=tolerance,
                alpha=WINKLER_ALPHA,
            )
            offsets = {
                "__global__": {
                    "a": params.a,
                    "s_lo": params.s_lo,
                    "s_hi": params.s_hi,
                    "n": len(cal_eval),
                }
            }

    val_eval = apply_conformal(val_eval, offsets, mode=mode_key, bucket_mode=bucket_mode)
    val_eval["minutes_p10"] = val_eval["p10_adj"]
    val_eval["minutes_p50"] = val_eval["p50_adj"]
    val_eval["minutes_p90"] = val_eval["p90_adj"]
    val_eval["p10_cond"] = val_eval["minutes_p10"]
    val_eval["p90_cond"] = val_eval["minutes_p90"]
    if play_prob_artifacts is not None:
        val_play_prob = predict_play_probability(play_prob_artifacts, val_df[feature_columns])
    else:
        val_play_prob = np.ones(len(val_df), dtype=float)
    val_eval["play_prob"] = val_play_prob
    val_eval["will_play_flag"] = (val_play_prob >= 0.5).astype(int)
    val_eval["p10"] = val_eval["minutes_p10"]
    val_eval["p90"] = val_eval["minutes_p90"]
    val_eval["p50"] = val_eval["minutes_p50"]

    dedup_key_cols = list(KEY_COLUMNS)
    if "horizon_min" in val_eval.columns:
        dedup_key_cols.append("horizon_min")
    val_unique = deduplicate_latest(val_eval, key_cols=dedup_key_cols, order_cols=["feature_as_of_ts"])
    join_key = ["game_id", "player_id", "tip_ts"]
    if "horizon_min" in val_unique.columns:
        join_key.append("horizon_min")
    ensure_columns(val_unique, join_key)
    duplicate_count = int(val_unique.duplicated(join_key).sum())
    if duplicate_count:
        raise AssertionError(f"Validation frame contains {duplicate_count} duplicate join keys.")
    y_val_unique = val_unique[target_col]
    raw_cross = val_unique.get("raw_cross_flag")
    final_cross = val_unique.get("final_cross_flag")
    clip_mask = val_unique.get("offset_clip_flag")
    nonmono_rate_raw = float(np.mean(raw_cross.astype(float))) if raw_cross is not None else 0.0
    nonmono_rate_final = float(np.mean(final_cross.astype(float))) if final_cross is not None else 0.0
    clip_rate = float(np.mean(clip_mask.astype(float))) if clip_mask is not None else 0.0
    if nonmono_rate_final > FINAL_NON_MONO_TOL:
        typer.echo(
            f"[coverage] final non-monotonic rate {nonmono_rate_final:.6f} exceeds tolerance {FINAL_NON_MONO_TOL:.6f}",
            err=True,
        )
        if not allow_guard_failure:
            raise typer.Exit(code=1)
    if nonmono_rate_raw > RAW_NON_MONO_ALERT:
        typer.echo(
            f"[coverage] raw non-monotonic rate {nonmono_rate_raw:.3f} exceeds alert threshold {RAW_NON_MONO_ALERT:.2f}",
            err=True,
        )

    # Coverage + MAE calculations intentionally use the conditional minutes columns.
    # Never substitute any play-probability-adjusted variant here—calibration is evaluated on the
    # "if active" distribution only.
    val_mae = float(mean_absolute_error(y_val_unique, val_unique["p50"]))
    val_mae_raw = float(mean_absolute_error(y_val_unique, val_unique["p50_raw"]))
    val_mae_by_horizon: dict[str, float] | None = None
    val_rows_by_horizon: dict[str, int] | None = None
    if "horizon_min" in val_unique.columns:
        val_mae_by_horizon = {}
        val_rows_by_horizon = {}
        for horizon, group in val_unique.groupby("horizon_min", dropna=False):
            horizon_key = "__missing__" if pd.isna(horizon) else str(int(horizon))
            horizon_y = pd.to_numeric(group[target_col], errors="coerce")
            horizon_pred = pd.to_numeric(group["p50"], errors="coerce")
            mask = horizon_y.notna() & horizon_pred.notna()
            if int(mask.sum()) == 0:
                continue
            val_mae_by_horizon[horizon_key] = float(mean_absolute_error(horizon_y[mask], horizon_pred[mask]))
            val_rows_by_horizon[horizon_key] = int(len(group))
    val_mae_buckets = compute_mae_by_actual_minutes_bucket(
        y_val_unique.to_numpy(dtype=float), val_unique["p50"].to_numpy(dtype=float)
    )
    val_p10_raw, val_p90_raw = _coverage_rates(y_val_unique, val_unique["p10_raw"], val_unique["p90_raw"])
    val_p10_cov, val_p90_cov = _coverage_rates(y_val_unique, val_unique["p10"], val_unique["p90"])
    play_prob_vals = val_unique.get("play_prob")
    play_prob_array = (
        play_prob_vals.to_numpy(dtype=float) if play_prob_vals is not None else np.ones(len(val_unique), dtype=float)
    )
    play_target_series = val_unique.get("plays_target")
    y_play_val = (
        play_target_series.to_numpy(dtype=float)
        if play_target_series is not None
        else (y_val_unique > 0).astype(int).to_numpy(dtype=float)
    )
    play_brier = _brier_score(y_play_val, play_prob_array)
    play_ece = _expected_calibration_error(y_play_val, play_prob_array)
    floor10 = float(np.mean(np.maximum(ALPHA_TARGET, 1.0 - play_prob_array)))
    floor90 = float(np.mean(np.maximum(1.0 - ALPHA_TARGET, 1.0 - play_prob_array)))
    excess10 = val_p10_cov - floor10
    excess90 = val_p90_cov - floor90
    if play_target_series is not None:
        val_positive = val_unique[val_unique["plays_target"] == 1]
    else:
        val_positive = val_unique[y_val_unique > 0]
    if val_positive.empty:
        cond_p10_cov = cond_p90_cov = 0.0
    else:
        cond_p10_cov, cond_p90_cov = _coverage_rates(val_positive[target_col], val_positive["p10_cond"], val_positive["p90_cond"])
    inside_cov = val_p90_cov - val_p10_cov
    mpiwn = float(np.mean(val_unique["p90"] - val_unique["p10"]))
    winkler_scores = _winkler_score(
        y_val_unique,
        val_unique["p10"],
        val_unique["p90"],
        alpha=WINKLER_ALPHA,
    )
    winkler_mean = float(np.mean(winkler_scores))
    if winkler_baseline is not None and winkler_mean > winkler_baseline:
        typer.echo(
            f"[coverage] Winkler {winkler_mean:.4f} exceeds baseline ceiling {winkler_baseline:.4f}",
            err=True,
        )
        if not allow_guard_failure:
            raise typer.Exit(code=1)

    playable_metrics = _compute_playable_subset_metrics(
        val_unique,
        target_col=target_col,
        minutes_threshold=playable_min_p50,
    )
    if not enable_play_prob_head or play_prob_artifacts is None:
        playable_metrics["val_play_prob_brier_playable"] = None
        playable_metrics["val_play_prob_ece_playable"] = None
    acceptance = _evaluate_playable_acceptance(
        playable_metrics,
        winkler_baseline=playable_winkler_baseline,
        winkler_tolerance=playable_winkler_tolerance,
    )
    status = "PASS" if acceptance["passed"] else "FAIL"
    message = (
        f"[playable] {fold_id or run_id}: {status} "
        f"p10_cond={playable_metrics['val_p10_cond_playable']}, "
        f"p90_cond={playable_metrics['val_p90_cond_playable']}, "
        f"winkler={playable_metrics['val_winkler_cond_playable']}"
    )
    typer.echo(message, err=not acceptance["passed"])

    per_bucket: dict[str, dict[str, float]] = {}
    for bucket, group in val_unique.groupby("bucket", dropna=False):
        bucket_key = "__missing__" if pd.isna(bucket) else str(bucket)
        bucket_y = group[target_col]
        bucket_p10_cov, bucket_p90_cov = _coverage_rates(bucket_y, group["p10"], group["p90"])
        bucket_inside = bucket_p90_cov - bucket_p10_cov
        play_targets_group = group.get("plays_target")
        if play_targets_group is not None:
            group_cond = group[play_targets_group == 1]
        else:
            group_cond = group[group[target_col] > 0]
        if group_cond.empty:
            cond_p10 = cond_p90 = cond_inside = 0.0
        else:
            cond_p10, cond_p90 = _coverage_rates(group_cond[target_col], group_cond["p10_cond"], group_cond["p90_cond"])
            cond_inside = cond_p90 - cond_p10
        per_bucket[bucket_key] = {
            "n": int(len(group)),
            "p10_cov": bucket_p10_cov,
            "p90_cov": bucket_p90_cov,
            "inside_cov": bucket_inside,
            "cond_n": int(len(group_cond)),
            "cond_p10_cov": cond_p10,
            "cond_p90_cov": cond_p90,
            "cond_inside_cov": cond_inside,
        }
        if len(group_cond) >= BUCKET_MIN_ENFORCEMENT:
            _enforce_coverage(
                f"{bucket_key} conditional P10", cond_p10, ALPHA_TARGET, tolerance, allow_failure=allow_guard_failure
            )
            _enforce_coverage(
                f"{bucket_key} conditional P90", cond_p90, 1.0 - ALPHA_TARGET, tolerance, allow_failure=allow_guard_failure
            )

    if not per_bucket:
        raise RuntimeError("Per-bucket coverage payload is empty; cannot reconcile metrics.")
    total_bucket = sum(entry["n"] for entry in per_bucket.values())
    weighted_p10 = sum(entry["n"] * entry["p10_cov"] for entry in per_bucket.values()) / total_bucket
    weighted_p90 = sum(entry["n"] * entry["p90_cov"] for entry in per_bucket.values()) / total_bucket
    if abs(weighted_p10 - val_p10_cov) > 1e-3 or abs(weighted_p90 - val_p90_cov) > 1e-3:
        raise AssertionError(
            "Weighted per-bucket coverage does not reconcile with overall validation coverage "
            f"(p10={weighted_p10:.6f} vs {val_p10_cov:.6f}, "
            f"p90={weighted_p90:.6f} vs {val_p90_cov:.6f})."
        )

    _enforce_coverage(
        "validation conditional P10", cond_p10_cov, ALPHA_TARGET, tolerance, allow_failure=allow_guard_failure
    )
    _enforce_coverage(
        "validation conditional P90", cond_p90_cov, 1.0 - ALPHA_TARGET, tolerance, allow_failure=allow_guard_failure
    )

    feature_hash = compute_feature_hash(feature_columns)
    run_dir = ensure_run_directory(run_id, root=artifact_root)
    bundle = {
        "feature_columns": feature_columns,
        "quantiles": quantiles,
        "calibrator": calibrator,
        "bucket_offsets": offsets,
        "conformal_mode": mode_key,
        "bucket_mode": bucket_mode,
        "play_prob_enabled": enable_play_prob_head,
        "play_prob_mixing_enabled": enable_play_prob_mixing,
        "play_probability": play_prob_artifacts,
    }
    joblib.dump(bundle, run_dir / "lgbm_quantiles.joblib")
    write_json(run_dir / "feature_columns.json", {"columns": feature_columns})

    write_ids_csv(val_unique, run_dir / "val_ids.csv")

    if not enable_play_prob_head or play_prob_artifacts is None:
        play_brier = None
        play_ece = None
    metrics = {
        "run_id": run_id,
        "fold_id": fold_id,
        "play_prob_head_enabled": enable_play_prob_head,
        "train_rows": len(train_df),
        "cal_rows": len(cal_df),
        "val_rows": len(val_df),
        "val_rows_unique": len(val_unique),
        "val_mae_p50": val_mae,
        "val_mae_p50_raw": val_mae_raw,
        "val_mae_by_horizon_min": val_mae_by_horizon,
        "val_rows_by_horizon_min": val_rows_by_horizon,
        "val_p10_raw_coverage": val_p10_raw,
        "val_p90_raw_coverage": val_p90_raw,
        "val_p10_coverage": val_p10_cov,
        "val_p90_coverage": val_p90_cov,
        "val_p10_conditional": cond_p10_cov,
        "val_p90_conditional": cond_p90_cov,
        "val_floor_p10": floor10,
        "val_floor_p90": floor90,
        "val_excess_p10": excess10,
        "val_excess_p90": excess90,
        "val_play_prob_brier": play_brier,
        "val_play_prob_ece": play_ece,
        "val_play_prob_mean": float(np.mean(play_prob_array)),
        "val_play_prob_threshold": 0.5,
        "val_will_play_rate": float(np.mean(y_play_val)),
        "inside_cov": inside_cov,
        "mpiwn": mpiwn,
        "winkler_alpha_0_2": winkler_mean,
        "nonmono_rate_raw": nonmono_rate_raw,
        "nonmono_rate_final": nonmono_rate_final,
        "nonmono_clip_rate": clip_rate,
        "per_bucket": per_bucket,
        "target_col": target_col,
        "conformal_buckets": bucket_mode,
        "conformal_k": conformal_k,
        "conformal_mode": mode_key,
    }
    metrics.update({f"val_{name}": value for name, value in val_mae_buckets.items()})
    metrics.update(playable_metrics)
    metrics["playable_acceptance"] = acceptance

    params = {
        "random_state": random_state,
        "quantiles": [0.1, 0.5, 0.9],
        "conformal_buckets": bucket_mode,
        "conformal_k": conformal_k,
        "conformal_mode": mode_key,
        "cal_days": cal_days,
        "coverage_tolerance": tolerance,
        "winkler_baseline": winkler_baseline,
        "playable_min_p50": playable_min_p50,
        "playable_winkler_baseline": playable_winkler_baseline,
        "playable_winkler_tolerance": playable_winkler_tolerance,
        "enable_play_prob_head": enable_play_prob_head,
        "enable_play_prob_mixing": enable_play_prob_mixing,
    }

    windows_meta = {
        "train": train_window.to_metadata(),
        "cal": cal_window.to_metadata(),
        "val": val_window.to_metadata(),
    }
    if split_col and "tip_ts" in feature_df.columns:
        def _tip_window(frame: pd.DataFrame) -> dict[str, str | None]:
            tips = pd.to_datetime(frame.get("tip_ts"), utc=True, errors="coerce").dropna()
            if tips.empty:
                return {"start": None, "end": None}
            return {
                "start": tips.min().isoformat().replace("+00:00", "Z"),
                "end": tips.max().isoformat().replace("+00:00", "Z"),
            }

        windows_meta = {"train": _tip_window(train_df), "cal": _tip_window(cal_df), "val": _tip_window(val_df)}

    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "meta.json",
        {
            "model": "lightgbm_quantiles",
            "feature_hash": feature_hash,
            "params": params,
            "target_col": target_col,
            "windows": windows_meta,
            "fold_id": fold_id,
        },
    )
    write_json(
        run_dir / "conformal_offsets.json",
        {
            "buckets": bucket_mode,
            "k": conformal_k,
            "mode": mode_key,
            "offsets": offsets,
            "cal_rows": len(cal_df),
            "cal_window": windows_meta["cal"],
            "calibrator": calibrator_offsets,
        },
    )

    typer.echo(
        "LightGBM training complete. "
        f"Validation MAE={val_mae:.3f}, "
        f"P10={val_p10_cov:.3f}, "
        f"P90={val_p90_cov:.3f}. Artifacts: {run_dir}"
    )

    # Auto-register model in registry
    try:
        manifest = load_manifest()
        register_model(
            manifest,
            model_name="minutes_v1_lgbm",
            version=run_id,
            run_id=run_id,
            artifact_path=str(run_dir),
            training_start=windows_meta["train"]["start"],
            training_end=windows_meta["train"]["end"],
            feature_schema_version="v1",
            feature_hash=feature_hash,
            metrics={
                "val_mae": val_mae,
                "val_p10_coverage": val_p10_cov,
                "val_p90_coverage": val_p90_cov,
                "val_winkler": winkler_mean,
            },
            description=f"Train {windows_meta['train']['start'][:10]} to {windows_meta['train']['end'][:10]}",
        )
        save_manifest(manifest)
        typer.echo(f"[registry] Registered minutes_v1_lgbm v{run_id} (stage=dev)")
    except Exception as e:
        typer.echo(f"[registry] Warning: Failed to register model: {e}", err=True)

    # Log to MLFlow
    _log_to_mlflow(
        run_id=run_id,
        params=params,
        metrics=metrics,
        quantiles=quantiles,
        feature_columns=feature_columns,
        run_dir=run_dir,
        windows_meta=windows_meta,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
