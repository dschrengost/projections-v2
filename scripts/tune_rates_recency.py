#!/usr/bin/env python
"""
Tune recency-weighting hyperparameters for rates_v1 using walk-forward folds.

Example:
    uv run python scripts/tune_rates_recency.py --season 2024 --n-trials 80
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import typer

from projections.eval.walk_forward import (
    assert_fold_integrity,
    generate_expanding_date_folds,
    iter_time_folds,
)
from projections.paths import data_path
from scripts.rates.train_rates_v1 import (
    FEATURES_STAGE0,
    FEATURES_STAGE1,
    FEATURES_STAGE2_TRACKING,
    FEATURES_STAGE3_CONTEXT,
    TARGET_LABEL_MAP,
    TARGETS,
    _clean_frame,
    _impute_odds,
    _load_training_base,
    _prepare_features,
    train_rates,
)

app = typer.Typer(add_completion=False)


def _utc_timestamp_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _season_start_year_for_date(dt: pd.Timestamp) -> int:
    # NBA season is labeled by the start year (Oct -> Jun).
    return int(dt.year) if int(dt.month) >= 10 else int(dt.year) - 1


def _resolve_weight_time_col(
    df: pd.DataFrame,
    *,
    requested: str,
    allow_game_date_weighting: bool,
) -> tuple[str, str]:
    """Return (weight_time_col_used, resolution)."""

    if requested in df.columns:
        if requested == "game_date" and not allow_game_date_weighting:
            raise typer.BadParameter(
                "--weight-time-col=game_date requires --allow-game-date-weighting (day-level weighting is risky)."
            )
        resolution = "day-level" if requested == "game_date" else "timestamp-level"
        return requested, resolution

    if requested == "feature_as_of_ts":
        if not allow_game_date_weighting:
            raise typer.BadParameter(
                "feature_as_of_ts is missing from rates_training_base; "
                "pass --allow-game-date-weighting to fall back to game_date (day-level), "
                "or rebuild rates_training_base to include feature_as_of_ts."
            )
        if "game_date" not in df.columns:
            raise typer.BadParameter(
                "feature_as_of_ts missing and cannot fall back: game_date also missing."
            )
        typer.echo(
            "[tune][WARNING] Using game_date for recency weights (day-level). "
            "This is less meaningful than feature_as_of_ts and can be leaky depending on label semantics."
        )
        return "game_date", "day-level"

    raise typer.BadParameter(f"Missing --weight-time-col={requested} in dataset columns.")


@dataclass(frozen=True)
class RecencyParams:
    half_life_days: float
    w_min: float
    season_boost: float


def _compute_recency_weights(
    train_df: pd.DataFrame,
    *,
    train_end_ts: pd.Timestamp,
    weight_time_col: str,
    params: RecencyParams,
) -> tuple[np.ndarray, dict[str, float | int]]:
    if weight_time_col not in train_df.columns:
        raise KeyError(f"Missing weight_time_col={weight_time_col} in training frame; cannot compute recency weights.")

    asof = pd.to_datetime(train_df[weight_time_col], errors="coerce")
    if asof.isna().any():
        raise ValueError(f"{weight_time_col} contains NaT; cannot compute recency weights.")

    # Align train_end_ts tz-awareness with asof if needed.
    end = pd.Timestamp(train_end_ts)
    if getattr(asof.dt, "tz", None) is not None and end.tzinfo is None:
        end = end.tz_localize(asof.dt.tz)

    age_days = (end - asof).dt.total_seconds() / 86400.0
    if (age_days < -1e-6).any():
        raise ValueError(
            f"Detected {weight_time_col} > train_end_ts rows (potential leakage / tz mismatch)."
        )
    age_days = age_days.clip(lower=0.0)

    w = 0.5 ** (age_days / float(params.half_life_days))
    w = np.maximum(float(params.w_min), w.to_numpy(dtype=float, copy=False))

    if "season" in train_df.columns and params.season_boost > 0:
        current_season = _season_start_year_for_date(pd.Timestamp(end).tz_localize(None))
        same_season = train_df["season"].astype(int) == int(current_season)
        if same_season.any():
            w = w * np.where(same_season.to_numpy(), 1.0 + float(params.season_boost), 1.0)

    sum_w = float(w.sum())
    sum_w2 = float(np.square(w).sum())
    neff = float((sum_w * sum_w) / sum_w2) if sum_w2 > 0 else 0.0

    age_arr = age_days.to_numpy(dtype=float, copy=False)
    w_arr = np.asarray(w, dtype=float)
    corr = float("nan")
    if w_arr.size >= 2 and float(np.std(w_arr)) > 0 and float(np.std(age_arr)) > 0:
        corr = float(np.corrcoef(w_arr, age_arr)[0, 1])
    n_at_w_min = int(np.sum(np.isclose(w_arr, float(params.w_min))))

    return w_arr, {
        "neff": neff,
        "mean_w": float(np.mean(w_arr)),
        "min_w": float(np.min(w_arr)),
        "max_w": float(np.max(w_arr)),
        "mean_age_days": float(np.mean(age_arr)),
        "corr_w_age_days": corr,
        "n_at_w_min": n_at_w_min,
        "frac_at_w_min": float(n_at_w_min / w_arr.size) if w_arr.size else 0.0,
    }


def _fold_overall_val_mae(val_df: pd.DataFrame, preds: dict[str, np.ndarray]) -> tuple[float, dict[str, Any]]:
    maes: dict[str, float] = {}
    ns: dict[str, int] = {}
    for target in TARGETS:
        label_col = TARGET_LABEL_MAP.get(target, target)
        if label_col not in val_df.columns:
            continue
        mask = val_df[label_col].notna().to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        pred = preds.get(target)
        if pred is None:
            continue
        y = val_df.loc[mask, label_col].to_numpy(dtype=float, copy=False)
        yhat = np.asarray(pred)[mask]
        maes[target] = float(np.mean(np.abs(yhat - y)))
        ns[target] = n
    if not maes:
        return float("nan"), {"per_target_mae": maes, "per_target_n": ns}
    overall = float(np.mean(list(maes.values())))
    return overall, {"per_target_mae": maes, "per_target_n": ns}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


@app.command()
def main(
    season: int = typer.Option(..., "--season", help="Season start year (folder season=YYYY)."),
    start: str | None = typer.Option(None, "--start", help="Optional start date YYYY-MM-DD (overrides season min)."),
    end: str | None = typer.Option(None, "--end", help="Optional end date YYYY-MM-DD (overrides season max)."),
    time_col: str = typer.Option("game_date", "--time-col", help="Timestamp column used to slice folds."),
    weight_time_col: str = typer.Option(
        "feature_as_of_ts", "--weight-time-col", help="Timestamp column used to compute recency weights."
    ),
    allow_game_date_weighting: bool = typer.Option(
        False,
        "--allow-game-date-weighting",
        is_flag=True,
        help="Allow using game_date as the weighting timestamp (day-level; risky).",
    ),
    feature_set: str = typer.Option("stage3_context", "--feature-set", help="rates_v1 feature set key."),
    n_trials: int = typer.Option(80, "--n-trials", min=1),
    seed: int = typer.Option(1337, "--seed"),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional run id (defaults to UTC timestamp)."),
    data_root: Path | None = typer.Option(None, "--data-root", help="Root containing gold/rates_training_base."),
    output_root: Path = typer.Option(
        Path("artifacts") / "tuning" / "rates_recency",
        "--output-root",
        help="Base output dir (default: artifacts/tuning/rates_recency).",
    ),
    train_months: int = typer.Option(4, "--train-months", min=1, help="Minimum training window size in months (~30d)."),
    cal_weeks: int = typer.Option(2, "--cal-weeks", min=0, help="Calibration window size in weeks."),
    val_weeks: int = typer.Option(2, "--val-weeks", min=1, help="Validation window size in weeks."),
    step_weeks: int = typer.Option(2, "--step-weeks", min=1, help="Weeks to advance the cutoff between folds."),
    season_aware: bool = typer.Option(True, "--season-aware/--no-season-aware", help="Skip folds with offseason validation."),
    max_folds: int | None = typer.Option(None, "--max-folds", min=1, help="Optional cap on folds (useful for smoke runs)."),
    min_train_rows: int = typer.Option(1000, "--min-train-rows", min=1),
    min_val_rows: int = typer.Option(200, "--min-val-rows", min=1),
    allow_minutes_actual_fallback: bool = typer.Option(
        False,
        "--allow-minutes-actual-fallback/--no-minutes-actual-fallback",
        help="Allow fallback to minutes_actual when minutes_pred_* are missing (leaky). Default: disabled.",
    ),
) -> None:
    np.random.seed(seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    root = data_root or data_path()
    season_root = root / "gold" / "rates_training_base" / f"season={season}"
    if not season_root.exists():
        raise typer.BadParameter(f"Missing season folder: {season_root}")

    # Infer date bounds from available partitions when not explicitly provided.
    available_days = sorted(
        pd.Timestamp(p.name.split("=", 1)[1]).normalize()
        for p in season_root.glob("game_date=*")
        if (p / "rates_training_base.parquet").exists()
    )
    if not available_days:
        raise typer.BadParameter(f"No rates_training_base partitions found under {season_root}")
    inferred_start = available_days[0]
    inferred_end = available_days[-1]

    start_ts = pd.Timestamp(start).normalize() if start else inferred_start
    end_ts = pd.Timestamp(end).normalize() if end else inferred_end
    if end_ts <= start_ts:
        raise typer.BadParameter("--end must be after --start")

    resolved_run_id = run_id or _utc_timestamp_now()
    out_dir = output_root / resolved_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_set_key = feature_set.lower()
    feature_map = {
        "stage0": FEATURES_STAGE0,
        "stage1": FEATURES_STAGE1,
        "stage2_tracking": FEATURES_STAGE2_TRACKING,
        "stage3_context": FEATURES_STAGE3_CONTEXT,
    }
    if feature_set_key not in feature_map:
        raise typer.BadParameter(f"--feature-set must be one of {sorted(feature_map)}")
    feature_cols = feature_map[feature_set_key]
    use_predicted_minutes = feature_set_key != "stage0"
    use_tracking_features = feature_set_key in {"stage2_tracking", "stage3_context"}

    # Load and prepare the full modeling dataframe once; folds slice by `game_date`.
    df = _load_training_base(root, start_ts, end_ts)
    df = df[df["season"].astype(int) == int(season)].copy()
    if df.empty:
        raise typer.BadParameter(f"No rows found for season={season} in [{start_ts.date()}..{end_ts.date()}].")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    df = _prepare_features(
        df,
        use_predicted_minutes=use_predicted_minutes,
        fallback_minutes_with_actual=allow_minutes_actual_fallback,
        use_tracking_features=use_tracking_features,
    )
    if time_col not in df.columns:
        raise typer.BadParameter(f"Missing --time-col={time_col} in dataset columns.")
    weight_time_col_used, weight_time_resolution = _resolve_weight_time_col(
        df, requested=weight_time_col, allow_game_date_weighting=allow_game_date_weighting
    )
    if weight_time_col_used == "game_date" and allow_game_date_weighting:
        typer.echo(
            f"[tune][WARNING] weight_time_col={weight_time_col_used} (day-level). "
            "This can create misleading tuning wins vs true feature snapshot timestamps."
        )

    folds = generate_expanding_date_folds(
        data_start=start_ts.to_pydatetime(),
        data_end=end_ts.to_pydatetime(),
        min_train_months=train_months,
        cal_weeks=cal_weeks,
        val_weeks=val_weeks,
        step_weeks=step_weeks,
        season_aware=season_aware,
        uses_calibration=True,
        fold_id_format="fold_{fold_num:03d}",
    )
    if not folds:
        raise typer.BadParameter("No folds generated; widen the date window or reduce train/cal/val sizes.")

    # Build fold iterator (attach cal_df so we can early-stop without peeking at val).
    fold_iter = list(
        iter_time_folds(
            df,
            time_col=time_col,
            folds=folds,
            min_train_rows=min_train_rows,
            min_val_rows=min_val_rows,
            attach_cal_df=True,
        )
    )
    if max_folds is not None:
        fold_iter = fold_iter[: int(max_folds)]
    if not fold_iter:
        raise typer.BadParameter("No folds satisfied min row constraints; widen window or lower thresholds.")

    def eval_params(
        recency: RecencyParams | None,
        *,
        trial: optuna.Trial | None = None,
        label: str,
    ) -> tuple[float, dict[str, Any]]:
        fold_maes: list[float] = []
        fold_neff: list[float] = []
        fold_details: list[dict[str, Any]] = []

        for fold_idx, (train_raw, val_raw, train_end_ts, fold_meta) in enumerate(fold_iter):
            cal_raw = fold_meta.cal_df if fold_meta.cal_df is not None else train_raw.iloc[0:0].copy()

            train_df, cal_df, val_df = _impute_odds(train_raw, cal_raw, val_raw)
            train_df = _clean_frame(train_df, TARGET_LABEL_MAP, feature_cols)
            cal_df = _clean_frame(cal_df, TARGET_LABEL_MAP, feature_cols)
            val_df = _clean_frame(val_df, TARGET_LABEL_MAP, feature_cols)

            assert_fold_integrity(
                train_df,
                val_df,
                time_col=time_col,
                train_end_ts=train_end_ts,
                fold_id=fold_meta.fold.fold_id,
                key_cols=None,
            )

            weights = None
            weight_stats: dict[str, float | int] | None = None
            if recency is not None:
                # Use a timestamp reference consistent with weight_time_col (UTC timestamps),
                # not the fold's day-boundary train_end_ts (which is typically a naive date).
                if "tip_ts" in train_df.columns:
                    tip_ref = pd.to_datetime(train_df["tip_ts"], utc=True, errors="coerce")
                    if tip_ref.isna().any():
                        raise ValueError(
                            f"tip_ts contains NaT in training fold_id={fold_meta.fold.fold_id}; cannot weight safely."
                        )
                    weight_ref_ts = pd.Timestamp(tip_ref.max())
                else:
                    weight_ref = pd.to_datetime(train_df[weight_time_col_used], utc=True, errors="coerce")
                    if weight_ref.isna().any():
                        raise ValueError(
                            f"{weight_time_col_used} contains NaT in training fold_id={fold_meta.fold.fold_id}; cannot weight safely."
                        )
                    weight_ref_ts = pd.Timestamp(weight_ref.max())

                weights, weight_stats = _compute_recency_weights(
                    train_df,
                    train_end_ts=weight_ref_ts,
                    weight_time_col=weight_time_col_used,
                    params=recency,
                )
                if not np.isfinite(weights).all() or (weights <= 0).any():
                    raise ValueError(
                        f"Invalid sample_weight computed for fold_id={fold_meta.fold.fold_id} "
                        f"(finite={bool(np.isfinite(weights).all())}, min={float(np.min(weights))})."
                    )
            model, _ = train_rates(
                train_df,
                cal_df=cal_df,
                feature_cols=feature_cols,
                sample_weight=weights,
                seed=seed + fold_idx,
                params=None,  # use defaults; determinism comes from seed plumbing
            )
            preds = {k: v for k, v in model.boosters.items()}
            # predict_rates() predicts all rows; we avoid importing it to keep evaluation local.
            X_val = val_df[model.feature_cols]
            val_preds: dict[str, np.ndarray] = {
                target: booster.predict(X_val, num_iteration=booster.best_iteration)
                for target, booster in preds.items()
            }

            fold_mae, extra = _fold_overall_val_mae(val_df, val_preds)
            fold_maes.append(fold_mae)
            if weight_stats is not None:
                fold_neff.append(float(weight_stats["neff"]))
            fold_details.append(
                {
                    "fold_id": fold_meta.fold.fold_id,
                    "train_end_ts": str(train_end_ts),
                    "train_rows": int(len(train_df)),
                    "cal_rows": int(len(cal_df)),
                    "val_rows": int(len(val_df)),
                    "val_mae_mean": fold_mae,
                    "weights": weight_stats,
                    "targets": extra,
                }
            )

            if trial is not None:
                running = float(np.nanmean(fold_maes))
                trial.report(running, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mean_mae = float(np.nanmean(fold_maes))
        mean_neff = float(np.mean(fold_neff)) if fold_neff else float("nan")
        mean_train_rows = float(np.mean([d["train_rows"] for d in fold_details])) if fold_details else 0.0
        neff_ratio = (mean_neff / mean_train_rows) if (mean_train_rows > 0 and np.isfinite(mean_neff)) else float("nan")

        # Conservative penalty: only activate when effective sample size collapses.
        neff_ratio_floor = 0.20
        neff_penalty_scale = 0.02
        neff_penalty = 0.0
        if np.isfinite(neff_ratio):
            neff_penalty = float(max(0.0, neff_ratio_floor - neff_ratio) * neff_penalty_scale)

        score = float(mean_mae + neff_penalty)
        return score, {
            "label": label,
            "score": score,
            "mean_mae": mean_mae,
            "neff_penalty": neff_penalty,
            "mean_neff": mean_neff,
            "neff_ratio": neff_ratio,
            "mean_age_days": float(
                np.mean(
                    [
                        float(d["weights"]["mean_age_days"])
                        for d in fold_details
                        if d.get("weights") and d["weights"].get("mean_age_days") is not None
                    ]
                )
            )
            if any(d.get("weights") for d in fold_details)
            else None,
            "folds": fold_details,
        }

    def objective(trial: optuna.Trial) -> float:
        params = RecencyParams(
            half_life_days=float(trial.suggest_float("half_life_days", 30.0, 365.0, log=True)),
            w_min=float(trial.suggest_float("w_min", 0.05, 0.30)),
            season_boost=float(trial.suggest_float("season_boost", 0.0, 1.0)),
        )
        value, _ = eval_params(params, trial=trial, label="trial")
        return float(value)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(10, n_trials // 8), n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_params = RecencyParams(
        half_life_days=float(best_trial.params["half_life_days"]),
        w_min=float(best_trial.params["w_min"]),
        season_boost=float(best_trial.params["season_boost"]),
    )
    best_value, best_detail = eval_params(best_params, trial=None, label="best")

    baseline_default_params = RecencyParams(half_life_days=180.0, w_min=0.2, season_boost=0.0)
    baseline_default_score, baseline_default_detail = eval_params(
        baseline_default_params, trial=None, label="baseline_default"
    )
    baseline_unweighted_score, baseline_unweighted_detail = eval_params(None, trial=None, label="baseline_unweighted")

    typer.echo("[tune] best trial fold diagnostics:")
    for fold in best_detail.get("folds", []):
        wstats = fold.get("weights") or {}
        corr = wstats.get("corr_w_age_days")
        corr_str = f"{float(corr):+.3f}" if corr is not None and np.isfinite(float(corr)) else "nan"
        frac_wmin = wstats.get("frac_at_w_min")
        frac_wmin_str = f"{float(frac_wmin):.1%}" if frac_wmin is not None else "n/a"
        typer.echo(
            f"  {fold['fold_id']} mae={fold['val_mae_mean']:.6f} "
            f"neff={wstats.get('neff')} age_days={wstats.get('mean_age_days')} "
            f"corr(w,age)={corr_str} w==w_min%={frac_wmin_str}"
        )
        if corr is not None and np.isfinite(float(corr)) and float(corr) >= 0:
            typer.echo(f"  [tune][WARNING] corr(weight, age_days) is non-negative in {fold['fold_id']}: {corr_str}")

    payload = {
        "run_id": resolved_run_id,
        "seed": seed,
        "data_root": str(root),
        "season": int(season),
        "date_window": {"start": str(start_ts.date()), "end": str(end_ts.date())},
        "baseline_unweighted_score": float(baseline_unweighted_score),
        "baseline_default_score": float(baseline_default_score),
        "best_score": float(best_value),
        "folds": {
            "n_folds": len(fold_iter),
            "train_months": train_months,
            "cal_weeks": cal_weeks,
            "val_weeks": val_weeks,
            "step_weeks": step_weeks,
            "season_aware": season_aware,
        },
        "feature_set": feature_set_key,
        "time_col": time_col,
        "weight_time_col_requested": weight_time_col,
        "weight_time_col_used": weight_time_col_used,
        "weight_time_resolution": weight_time_resolution,
        "baselines": {
            "unweighted": baseline_unweighted_detail,
            "default": {
                "params": {
                    "half_life_days": baseline_default_params.half_life_days,
                    "w_min": baseline_default_params.w_min,
                    "season_boost": baseline_default_params.season_boost,
                },
                **baseline_default_detail,
            },
        },
        "best": {
            "params": best_trial.params,
            "value": float(best_value),
            "detail": best_detail,
        },
        "trials": [
            {
                "number": int(t.number),
                "state": str(t.state),
                "value": float(t.value) if t.value is not None else None,
                "params": dict(t.params),
            }
            for t in study.trials
        ],
    }

    _write_json(out_dir / "study.json", payload)
    _write_json(out_dir / "best_params.json", {"best_params": best_trial.params, "best_value": float(best_value)})

    typer.echo(
        f"[tune] baseline_unweighted={baseline_unweighted_score:.6f} "
        f"baseline_default={baseline_default_score:.6f} best={best_value:.6f}"
    )
    typer.echo(
        f"[tune] delta(best-baseline_unweighted)={best_value - baseline_unweighted_score:+.6f} "
        f"delta(best-baseline_default)={best_value - baseline_default_score:+.6f}"
    )
    typer.echo(f"[tune] wrote {out_dir / 'study.json'}")
    typer.echo(f"[tune] best_value={best_value:.6f} best_params={best_trial.params}")


if __name__ == "__main__":
    app()
