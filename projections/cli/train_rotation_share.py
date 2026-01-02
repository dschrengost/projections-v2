"""Train a rotation-aware two-stage minutes model (P(play) + share).

This model is designed to tighten rotations in injury regimes by explicitly
predicting who plays, then distributing 240 team minutes over the likely
rotation (team-feasible by construction).

Example:
  uv run python -m projections.cli.train_rotation_share \\
    --run-id rotshare_v0 \\
    --train-start 2022-10-01 --train-end 2025-04-30 \\
    --val-start 2025-10-01 --val-end 2025-11-30
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import typer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from projections import paths
from projections.minutes_v1.artifacts import write_json
from projections.models.feature_contract import (
    assert_no_leakage,
    filter_to_contract_features,
    save_feature_contract,
)
from projections.minutes_v1.rotation_share import (
    TAU_MAX_DEFAULT,
    TAU_MIN_DEFAULT,
    TEAM_TOTAL_MINUTES,
    build_tau_team_features,
    predict_minutes,
    predict_play_prob,
    predict_raw_share,
    train_rotation_share_model,
)

app = typer.Typer(help=__doc__)

UTC = timezone.utc


def _compute_out_mask(df: pd.DataFrame) -> np.ndarray:
    is_out = np.zeros(len(df), dtype=int)
    if "is_out" in df.columns:
        is_out = is_out | pd.to_numeric(df["is_out"], errors="coerce").fillna(0).astype(int).to_numpy()
    if "status" in df.columns:
        status_out = df["status"].astype(str).str.upper() == "OUT"
        is_out = is_out | status_out.astype(int).to_numpy()
    if "lineup_role" in df.columns:
        role_out = df["lineup_role"].astype(str).str.lower() == "out"
        is_out = is_out | role_out.astype(int).to_numpy()
    return is_out.astype(int)


def _tau_to_logit(tau: np.ndarray, *, tau_min: float, tau_max: float, eps: float = 1e-6) -> np.ndarray:
    lo = float(tau_min)
    hi = float(tau_max)
    if hi <= lo:
        raise ValueError("tau_max must be greater than tau_min.")
    scaled = (np.asarray(tau, dtype=float) - lo) / (hi - lo)
    scaled = np.clip(scaled, eps, 1.0 - eps)
    return np.log(scaled / (1.0 - scaled))


def _default_lgbm_tau_params(*, random_state: int) -> dict[str, float | int]:
    return {
        "n_estimators": 250,
        "learning_rate": 0.06,
        "num_leaves": 31,
        "min_data_in_leaf": 64,
        "max_depth": -1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 0.1,
        "random_state": random_state,
    }


def _fit_tau_model(
    *,
    artifacts,
    train_frame: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    game_ids: np.ndarray,
    team_ids: np.ndarray,
    is_out: np.ndarray,
    min_players: int,
    play_prob_exponent: float,
    random_state: int,
    tau_min: float,
    tau_max: float,
    tau_grid_size: int,
) -> dict[str, float | int]:
    """Fit a team-level tau regressor and attach it to artifacts (in-place)."""

    tau_min = float(tau_min)
    tau_max = float(tau_max)
    if tau_grid_size < 2:
        raise ValueError("tau_grid_size must be >= 2.")
    tau_grid = np.linspace(tau_min, tau_max, int(tau_grid_size), dtype=float)

    play_prob = predict_play_prob(artifacts, X_train)
    raw_share = predict_raw_share(artifacts, X_train)
    play_prob = np.clip(play_prob, 0.0, 1.0)
    play_adj = np.power(play_prob, float(play_prob_exponent))

    out_mask = np.asarray(is_out).astype(bool)
    raw_share = np.where(out_mask, 0.0, raw_share)
    play_prob = np.where(out_mask, 0.0, play_prob)
    play_adj = np.where(out_mask, 0.0, play_adj)

    train_work = pd.DataFrame(
        {
            "game_id": game_ids.astype(int),
            "team_id": team_ids.astype(int),
            "raw_share": raw_share,
            "play_prob": play_prob,
            "play_adj": play_adj,
            "minutes": y_train.astype(float),
        }
    )

    tau_rows: list[dict[str, float | int]] = []
    for (game_id, team_id), g in train_work.groupby(["game_id", "team_id"], sort=False):
        actual = g["minutes"].to_numpy(dtype=float)
        if float(np.sum(actual)) < 100.0:
            # Skip corrupted team-games (e.g., missing roster rows).
            continue
        top7_actual = float(np.sort(actual)[::-1][:7].sum())
        raw = g["raw_share"].to_numpy(dtype=float)
        probs = g["play_prob"].to_numpy(dtype=float)
        play = g["play_adj"].to_numpy(dtype=float)
        if not np.isfinite(raw).all() or float(np.sum(raw)) <= 0.0:
            raw = np.ones_like(actual, dtype=float)

        best_tau = float(tau_grid[0])
        best_top7 = float("inf")
        best_mae = float("inf")
        for tau in tau_grid:
            alpha = 1.0 / float(tau)
            w_tau = np.power(np.maximum(0.0, raw), alpha) * play

            denom = float(np.sum(w_tau))
            if denom <= 0.0:
                active = probs > 0.0
                candidate_idx = np.arange(len(w_tau), dtype=int)[active]
                candidate_probs = probs[active]
                if candidate_idx.size == 0:
                    candidate_idx = np.arange(len(w_tau), dtype=int)
                    candidate_probs = probs
                k = int(min_players) if min_players and min_players > 0 else min(8, len(candidate_idx))
                k = max(1, min(k, len(candidate_idx)))
                order = np.argsort(-candidate_probs, kind="mergesort")[:k]
                chosen = candidate_idx[order]
                fallback = np.maximum(candidate_probs[order], 0.0)
                if float(np.sum(fallback)) <= 0.0:
                    fallback = np.ones_like(fallback, dtype=float)
                w_tau = np.zeros_like(w_tau, dtype=float)
                w_tau[chosen] = fallback
                denom = float(np.sum(w_tau))
                if denom <= 0.0:
                    continue
            pred = TEAM_TOTAL_MINUTES * (w_tau / denom)
            top7_pred = float(np.sort(pred)[::-1][:7].sum())
            loss_top7 = float(abs(top7_pred - top7_actual))
            loss_mae = float(np.mean(np.abs(pred - actual)))
            if (loss_top7 < best_top7 - 1e-9) or (abs(loss_top7 - best_top7) <= 1e-9 and loss_mae < best_mae):
                best_top7 = loss_top7
                best_mae = loss_mae
                best_tau = float(tau)
        tau_rows.append(
            {
                "game_id": int(game_id),
                "team_id": int(team_id),
                "tau_target": best_tau,
                "tau_loss_top7": best_top7,
                "tau_loss_mae": best_mae,
            }
        )

    tau_targets = pd.DataFrame(tau_rows)
    if tau_targets.empty:
        raise ValueError("Failed to compute any tau targets; aborting tau model fit.")

    # Team-level tau feature frame (aggregated from pre-lock signals).
    tau_feature_columns = [
        "team_out_count",
        "team_starter_out_count",
        "spread_home",
        "total",
        "home_flag",
        "is_b2b",
        "is_3in4",
        "is_4in6",
        "team_minutes_dispersion_prior",
        "close_game_score",
        "blowout_index",
    ]
    tau_feature_columns = [
        c for c in tau_feature_columns if c in {"team_out_count", "team_starter_out_count"} or c in train_frame.columns
    ]

    tau_features = build_tau_team_features(
        train_frame,
        game_ids=game_ids,
        team_ids=team_ids,
        is_out=is_out,
        feature_columns=tau_feature_columns,
    )
    tau_train = tau_targets.merge(tau_features, on=["game_id", "team_id"], how="inner")
    if tau_train.empty:
        raise ValueError("Tau targets did not match tau feature frame; cannot fit tau model.")

    X_tau = tau_train[tau_feature_columns].select_dtypes(include=["number", "bool", "boolean"]).copy()
    y_tau = _tau_to_logit(tau_train["tau_target"].to_numpy(dtype=float), tau_min=tau_min, tau_max=tau_max)

    tau_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_tau_imp = tau_imputer.fit_transform(X_tau)
    tau_model = lgb.LGBMRegressor(**_default_lgbm_tau_params(random_state=random_state))
    tau_model.fit(X_tau_imp, y_tau)

    artifacts.tau_model = tau_model
    artifacts.tau_imputer = tau_imputer
    artifacts.tau_feature_columns = list(X_tau.columns)
    artifacts.tau_min = float(tau_min)
    artifacts.tau_max = float(tau_max)

    return {
        "tau_train_team_games": int(len(tau_train)),
        "tau_target_mean": float(np.mean(tau_train["tau_target"].to_numpy(dtype=float))),
        "tau_target_p10": float(np.quantile(tau_train["tau_target"].to_numpy(dtype=float), 0.10)),
        "tau_target_p90": float(np.quantile(tau_train["tau_target"].to_numpy(dtype=float), 0.90)),
        "tau_loss_top7_mean": float(np.mean(tau_train["tau_loss_top7"].to_numpy(dtype=float))),
        "tau_loss_mae_mean": float(np.mean(tau_train["tau_loss_mae"].to_numpy(dtype=float))),
    }


def _load_features_root(features_root: Path) -> pd.DataFrame:
    files = sorted(features_root.rglob("features.parquet"))
    if not files:
        raise FileNotFoundError(f"No features.parquet files found under {features_root}")
    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def _time_slice(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    start_day = pd.Timestamp(start).tz_convert("UTC").tz_localize(None).normalize()
    end_day = pd.Timestamp(end).tz_convert("UTC").tz_localize(None).normalize()
    mask = (df["game_date"] >= start_day) & (df["game_date"] <= end_day)
    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError(f"Empty slice for {start_day.date()} → {end_day.date()}")
    return out


@app.command()
def main(
    run_id: str = typer.Option(..., help="Unique run identifier for artifacts."),
    train_start: datetime = typer.Option(..., help="Training window start (UTC, inclusive)."),
    train_end: datetime = typer.Option(..., help="Training window end (UTC, inclusive)."),
    val_start: datetime = typer.Option(..., help="Validation window start (UTC, inclusive)."),
    val_end: datetime = typer.Option(..., help="Validation window end (UTC, inclusive)."),
    data_root: Path = typer.Option(paths.get_data_root(), help="Data root (defaults to PROJECTIONS_DATA_ROOT)."),
    features_root: Path | None = typer.Option(
        None,
        help="Optional features root (defaults to <data_root>/gold/features_minutes_v1).",
    ),
    artifact_root: Path = typer.Option(
        Path("artifacts/minutes_rotation_share"),
        help="Directory to write run artifacts.",
    ),
    random_state: int = typer.Option(42, help="Deterministic seed for LightGBM."),
    in_rotation_minutes_threshold: float = typer.Option(
        10.0,
        help="Classifier target: in-rotation if minutes >= threshold (set 0 to reproduce played>0 behavior).",
    ),
    min_players: int = typer.Option(8, help="Inference min_players used for validation metrics."),
    play_prob_exponent: float = typer.Option(1.0, help="Inference play_prob exponent used for validation metrics."),
    train_tau: bool = typer.Option(True, help="Train a team-level tau (rotation tightness) regressor."),
    tau_min: float = typer.Option(TAU_MIN_DEFAULT, help="Lower bound for tau (more tight when smaller)."),
    tau_max: float = typer.Option(TAU_MAX_DEFAULT, help="Upper bound for tau (more flat when larger)."),
    tau_grid_size: int = typer.Option(31, help="Grid size for per-team-game tau target search."),
) -> None:
    data_root = data_root.expanduser().resolve()
    if features_root is None:
        features_root = data_root / "gold" / "features_minutes_v1"
    features_root = features_root.expanduser().resolve()

    train_start = train_start.replace(tzinfo=UTC)
    train_end = train_end.replace(tzinfo=UTC)
    val_start = val_start.replace(tzinfo=UTC)
    val_end = val_end.replace(tzinfo=UTC)

    typer.echo(f"[load] {features_root}")
    features = _load_features_root(features_root)
    required = {"game_id", "team_id", "player_id", "minutes", "game_date"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Features missing required columns: {', '.join(sorted(missing))}")

    train_df = _time_slice(features, train_start, train_end)
    val_df = _time_slice(features, val_start, val_end)

    # Feature contract (leakage-safe, train/live parity).
    feature_columns = filter_to_contract_features(train_df, target_col="minutes")
    assert_no_leakage(feature_columns)
    typer.echo(f"[features] {len(feature_columns)} contract features")

    X_train = train_df[feature_columns].select_dtypes(include=["number", "bool", "boolean"])
    y_train = pd.to_numeric(train_df["minutes"], errors="coerce").fillna(0.0)
    is_out_train = _compute_out_mask(train_df)

    threshold = float(in_rotation_minutes_threshold)
    typer.echo(
        f"[train] rows={len(train_df):,}  (in_rotation>= {threshold:g}m: {int((y_train>=threshold).sum()):,})"
    )
    artifacts = train_rotation_share_model(
        X_train,
        y_train,
        random_state=random_state,
        in_rotation_minutes_threshold=threshold,
    )

    X_val = val_df[feature_columns].select_dtypes(include=["number", "bool", "boolean"])
    y_val = pd.to_numeric(val_df["minutes"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    game_ids = pd.to_numeric(val_df["game_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
    team_ids = pd.to_numeric(val_df["team_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
    is_out_val = _compute_out_mask(val_df)

    tau_metrics: dict[str, float | int] | None = None
    if train_tau:
        typer.echo("[tau] fitting team-level rotation tightness model")
        tau_metrics = _fit_tau_model(
            artifacts=artifacts,
            train_frame=train_df,
            X_train=X_train,
            y_train=y_train.to_numpy(dtype=float),
            game_ids=pd.to_numeric(train_df["game_id"], errors="coerce").fillna(-1).astype(int).to_numpy(),
            team_ids=pd.to_numeric(train_df["team_id"], errors="coerce").fillna(-1).astype(int).to_numpy(),
            is_out=is_out_train,
            min_players=min_players,
            play_prob_exponent=play_prob_exponent,
            random_state=random_state,
            tau_min=tau_min,
            tau_max=tau_max,
            tau_grid_size=tau_grid_size,
        )

    preds = predict_minutes(
        artifacts,
        val_df,
        game_ids=game_ids,
        team_ids=team_ids,
        is_out=is_out_val,
        min_players=min_players,
        play_prob_exponent=play_prob_exponent,
    )
    mae = float(np.mean(np.abs(preds["predicted_minutes"].to_numpy(dtype=float) - y_val)))
    team_sums = preds.groupby(["game_id", "team_id"])["predicted_minutes"].sum()
    team_sum_dev_max = float(np.max(np.abs(team_sums.to_numpy(dtype=float) - TEAM_TOTAL_MINUTES)))

    # Play model AUC on full val (includes DNP).
    play_prob = predict_play_prob(artifacts, X_val)
    y_play = (y_val >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y_play, play_prob))
    except ValueError:
        auc = float("nan")

    metrics = {
        "val_player_mae_minutes": mae,
        "val_team_sum_dev_max": team_sum_dev_max,
        "val_play_auc": auc,
        "val_rows": int(len(val_df)),
        "val_play_rate": float(y_play.mean()) if len(y_play) else float("nan"),
        "in_rotation_minutes_threshold": threshold,
        "inference": {
            "min_players": int(min_players),
            "play_prob_exponent": float(play_prob_exponent),
        },
    }
    if tau_metrics is not None:
        metrics["tau"] = tau_metrics

    run_dir = (artifact_root / run_id).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, run_dir / "rotation_share_model.joblib")

    meta = {
        "model": "rotation_share_v0",
        "run_id": run_id,
        "windows": {
            "train": {"start": train_start.isoformat(), "end": train_end.isoformat()},
            "val": {"start": val_start.isoformat(), "end": val_end.isoformat()},
        },
        "random_state": int(random_state),
        "feature_columns": feature_columns,
        "team_total_minutes": TEAM_TOTAL_MINUTES,
        "in_rotation_minutes_threshold": threshold,
    }

    write_json(run_dir / "meta.json", meta)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "feature_columns.json", {"columns": feature_columns})
    save_feature_contract(feature_columns, run_dir / "feature_contract.json", metadata={"run_id": run_id})
    (run_dir / "TRAINING_DONE").write_text("", encoding="utf-8")

    typer.echo(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
