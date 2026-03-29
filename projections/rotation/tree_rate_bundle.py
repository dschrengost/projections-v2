from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb


KEY_COLS = ["game_id", "team_id", "player_id", "game_date"]
AST_REB_TARGETS = ["ast_per_min", "oreb_per_min", "dreb_per_min"]
FULL_RATE_TARGETS = [
    "fga2_per_min",
    "fga3_per_min",
    "fta_per_min",
    "ast_per_min",
    "tov_per_min",
    "oreb_per_min",
    "dreb_per_min",
    "stl_per_min",
    "blk_per_min",
]
EFF_TARGETS = ["fg2_pct_label", "fg3_pct_label", "ft_pct_label"]
FULL_TARGETS = FULL_RATE_TARGETS + EFF_TARGETS
EXCLUDE_COLS = {
    *KEY_COLS,
    "minutes_actual",
    "rates_non_null_count",
    "efficiency_non_null_count",
    "rates_label_available_any",
    "rates_label_available_all_rate_targets",
    "rates_loss_eligible",
    "fg2_pct_label",
    "fg3_pct_label",
    "ft_pct_label",
    "fga2_per_min",
    "fga3_per_min",
    "fta_per_min",
    "tov_per_min",
    "stl_per_min",
    "blk_per_min",
    *FULL_TARGETS,
}

BASE_LGBM_PARAMS: dict[str, object] = {
    "objective": "regression",
    "metric": "l2",
    "boosting_type": "gbdt",
    "num_leaves": 64,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "max_depth": -1,
    "lambda_l2": 1.0,
    "verbosity": -1,
}


@dataclass(slots=True)
class TargetModelResult:
    target: str
    model: Any
    best_iteration: int | None
    val_mae: float
    val_rmse: float
    train_rows: int
    cal_rows: int


@dataclass(slots=True)
class LoadedTreeRateBundle:
    bundle_dir: Path
    model_type: str
    feature_columns: list[str]
    targets: list[str]
    models: dict[str, Any]
    best_iterations: dict[str, int | None]
    metadata: dict[str, Any]


def _coerce_keys(df: pd.DataFrame, *, require_game_date: bool = True) -> pd.DataFrame:
    out = df.copy()
    for col in ("game_id", "team_id", "player_id"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    elif require_game_date:
        raise KeyError("game_date")
    return out


def _numeric_feature_cols(frame: pd.DataFrame, *, allowed_live_cols: set[str] | None = None) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col in EXCLUDE_COLS:
            continue
        if allowed_live_cols is not None and col not in allowed_live_cols:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric feature columns found for training.")
    return cols


def _clean_features(df: pd.DataFrame, *, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _split_train_cal(train_df: pd.DataFrame, *, cal_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_days = sorted(train_df["game_date"].dropna().unique().tolist())
    if not unique_days:
        raise ValueError("No training dates available.")
    cal_days = max(1, int(cal_days))
    if len(unique_days) <= cal_days:
        return train_df.iloc[0:0].copy(), train_df.copy()
    cal_dates = set(unique_days[-cal_days:])
    cal_df = train_df.loc[train_df["game_date"].isin(cal_dates)].copy()
    fit_df = train_df.loc[~train_df["game_date"].isin(cal_dates)].copy()
    return fit_df, cal_df


def _sample_weights(
    df: pd.DataFrame,
    *,
    target: str,
    ast_line_threshold: float,
    ast_weight_mult: float,
    reb_line_threshold: float,
    reb_weight_mult: float,
) -> np.ndarray | None:
    weights = np.ones(len(df), dtype=float)
    if target == "ast_per_min" and ast_weight_mult > 1.0 and "an_ast_line" in df.columns:
        line = pd.to_numeric(df["an_ast_line"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        weights[line >= float(ast_line_threshold)] *= float(ast_weight_mult)
    elif target in {"oreb_per_min", "dreb_per_min"} and reb_weight_mult > 1.0 and "an_reb_line" in df.columns:
        line = pd.to_numeric(df["an_reb_line"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        weights[line >= float(reb_line_threshold)] *= float(reb_weight_mult)
    if np.allclose(weights, 1.0):
        return None
    return weights


def _predict_model(model: Any, X: pd.DataFrame, *, model_type: str, best_iteration: int | None) -> np.ndarray:
    if model_type == "lgbm":
        return model.predict(X, num_iteration=best_iteration)
    if model_type == "xgb":
        dmat = xgb.DMatrix(X)
        iteration_range = (0, int(best_iteration) + 1) if best_iteration is not None else None
        return model.predict(dmat, iteration_range=iteration_range)
    raise ValueError(f"Unsupported model_type={model_type}")


def _train_target(
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    *,
    model_type: str,
    target: str,
    feature_cols: list[str],
    num_boost_round: int,
    ast_line_threshold: float,
    ast_weight_mult: float,
    reb_line_threshold: float,
    reb_weight_mult: float,
) -> TargetModelResult:
    fit_mask = train_df[target].notna()
    cal_mask = cal_df[target].notna()
    if not fit_mask.any():
        raise ValueError(f"No non-null training labels for target={target}")

    X_train = train_df.loc[fit_mask, feature_cols]
    y_train = train_df.loc[fit_mask, target]
    X_cal = cal_df.loc[cal_mask, feature_cols] if cal_mask.any() else None
    y_cal = cal_df.loc[cal_mask, target] if cal_mask.any() else None
    train_weights = _sample_weights(
        train_df.loc[fit_mask].copy(),
        target=target,
        ast_line_threshold=ast_line_threshold,
        ast_weight_mult=ast_weight_mult,
        reb_line_threshold=reb_line_threshold,
        reb_weight_mult=reb_weight_mult,
    )
    cal_weights = _sample_weights(
        cal_df.loc[cal_mask].copy(),
        target=target,
        ast_line_threshold=ast_line_threshold,
        ast_weight_mult=ast_weight_mult,
        reb_line_threshold=reb_line_threshold,
        reb_weight_mult=reb_weight_mult,
    )

    if model_type == "lgbm":
        train_set = lgb.Dataset(
            X_train,
            label=y_train,
            weight=train_weights,
            free_raw_data=False,
        )
        valid_sets = []
        callbacks = []
        if cal_mask.any():
            cal_set = lgb.Dataset(
                X_cal,
                label=y_cal,
                weight=cal_weights,
                reference=train_set,
                free_raw_data=False,
            )
            valid_sets = [cal_set]
            callbacks.append(lgb.early_stopping(stopping_rounds=200, verbose=False))
        model = lgb.train(
            params=BASE_LGBM_PARAMS,
            train_set=train_set,
            valid_sets=valid_sets,
            num_boost_round=int(num_boost_round),
            callbacks=callbacks,
        )
        best_iteration = model.best_iteration
    elif model_type == "xgb":
        train_weights_arr = train_weights if train_weights is not None else np.ones(len(y_train), dtype=float)
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights_arr)
        evals = []
        if cal_mask.any():
            cal_weights_arr = cal_weights if cal_weights is not None else np.ones(len(y_cal), dtype=float)
            dcal = xgb.DMatrix(X_cal, label=y_cal, weight=cal_weights_arr)
            evals = [(dcal, "eval")]
        model = xgb.train(
            params={
                "objective": "reg:squarederror",
                "eta": 0.05,
                "max_depth": 8,
                "min_child_weight": 5.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "lambda": 1.0,
                "tree_method": "hist",
                "seed": 42,
            },
            dtrain=dtrain,
            num_boost_round=int(num_boost_round),
            evals=evals,
            verbose_eval=False,
            early_stopping_rounds=200 if cal_mask.any() else None,
        )
        best_iteration = getattr(model, "best_iteration", None)
    else:
        raise ValueError(f"Unsupported model_type={model_type}")

    preds = _predict_model(model, X_cal, model_type=model_type, best_iteration=best_iteration) if cal_mask.any() else np.array([])
    truth = cal_df.loc[cal_mask, target].to_numpy(dtype=float) if cal_mask.any() else np.array([])
    mae = float(np.mean(np.abs(preds - truth))) if len(truth) else float("nan")
    rmse = float(np.sqrt(np.mean((preds - truth) ** 2))) if len(truth) else float("nan")
    return TargetModelResult(
        target=target,
        model=model,
        best_iteration=int(best_iteration) if best_iteration is not None else None,
        val_mae=mae,
        val_rmse=rmse,
        train_rows=int(fit_mask.sum()),
        cal_rows=int(cal_mask.sum()),
    )


def _target_model_filename(target: str, model_type: str) -> str:
    ext = "txt" if model_type == "lgbm" else "json"
    return f"{target}.{ext}"


def train_tree_rate_bundle(
    *,
    dataset_dir: Path,
    bundle_dir: Path,
    model_type: str = "lgbm",
    target_set: str = "astreb",
    cal_days: int = 30,
    num_boost_round: int = 5000,
    ast_line_threshold: float = 7.0,
    ast_weight_mult: float = 1.0,
    reb_line_threshold: float = 10.0,
    reb_weight_mult: float = 1.0,
    live_feature_columns: set[str] | None = None,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    features = _coerce_keys(pd.read_parquet(dataset_dir / "features.parquet"))
    labels = _coerce_keys(pd.read_parquet(dataset_dir / "labels_rates.parquet"))
    base = features.merge(labels, on=KEY_COLS, how="inner")
    target_list = AST_REB_TARGETS if str(target_set) == "astreb" else FULL_TARGETS
    train_df, cal_df = _split_train_cal(base, cal_days=int(cal_days))
    train_pool = train_df if not train_df.empty else base
    feature_cols = _numeric_feature_cols(train_pool, allowed_live_cols=live_feature_columns)
    train_pool = _clean_features(train_pool, feature_cols=feature_cols)
    cal_df = _clean_features(cal_df, feature_cols=feature_cols)

    model_results: dict[str, TargetModelResult] = {}
    models_dir = bundle_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for target in target_list:
        res = _train_target(
            train_pool,
            cal_df,
            model_type=model_type,
            target=target,
            feature_cols=feature_cols,
            num_boost_round=int(num_boost_round),
            ast_line_threshold=float(ast_line_threshold),
            ast_weight_mult=float(ast_weight_mult),
            reb_line_threshold=float(reb_line_threshold),
            reb_weight_mult=float(reb_weight_mult),
        )
        model_path = models_dir / _target_model_filename(target, model_type)
        if model_type == "lgbm":
            res.model.save_model(str(model_path), num_iteration=res.best_iteration)
        elif model_type == "xgb":
            res.model.save_model(str(model_path))
        model_results[target] = res

    metadata = {
        "bundle_type": "tree_rate_v1",
        "model_type": model_type,
        "target_set": target_set,
        "targets": target_list,
        "feature_columns": feature_cols,
        "training_args": {
            "dataset_dir": str(dataset_dir),
            "cal_days": int(cal_days),
            "num_boost_round": int(num_boost_round),
            "ast_line_threshold": float(ast_line_threshold),
            "ast_weight_mult": float(ast_weight_mult),
            "reb_line_threshold": float(reb_line_threshold),
            "reb_weight_mult": float(reb_weight_mult),
        },
        "targets_meta": {
            target: {
                "model_path": str((models_dir / _target_model_filename(target, model_type)).resolve()),
                "best_iteration": int(res.best_iteration) if res.best_iteration is not None else None,
                "val_mae": res.val_mae,
                "val_rmse": res.val_rmse,
                "train_rows": res.train_rows,
                "cal_rows": res.cal_rows,
            }
            for target, res in model_results.items()
        },
    }
    (bundle_dir / "bundle_meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def load_tree_rate_bundle(bundle_dir: Path) -> LoadedTreeRateBundle:
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    meta_path = bundle_dir / "bundle_meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    model_type = str(metadata["model_type"])
    feature_columns = [str(col) for col in metadata["feature_columns"]]
    targets = [str(t) for t in metadata["targets"]]
    models: dict[str, Any] = {}
    best_iterations: dict[str, int | None] = {}
    for target in targets:
        target_meta = metadata["targets_meta"][target]
        model_path = Path(str(target_meta["model_path"])).expanduser().resolve()
        if model_type == "lgbm":
            model = lgb.Booster(model_file=str(model_path))
        elif model_type == "xgb":
            model = xgb.Booster()
            model.load_model(str(model_path))
        else:
            raise ValueError(f"Unsupported model_type={model_type}")
        models[target] = model
        best_iterations[target] = (
            int(target_meta["best_iteration"])
            if target_meta.get("best_iteration") is not None
            else None
        )
    return LoadedTreeRateBundle(
        bundle_dir=bundle_dir,
        model_type=model_type,
        feature_columns=feature_columns,
        targets=targets,
        models=models,
        best_iterations=best_iterations,
        metadata=metadata,
    )


def score_tree_rate_bundle_features(
    *,
    features_df: pd.DataFrame,
    bundle: LoadedTreeRateBundle,
    include_extra_cols: list[str] | None = None,
    game_date: str | None = None,
) -> pd.DataFrame:
    frame_in = features_df.copy()
    if "game_date" not in frame_in.columns and game_date is not None:
        frame_in["game_date"] = str(game_date)
    frame = _coerce_keys(frame_in)
    missing = [col for col in bundle.feature_columns if col not in frame.columns]
    if missing:
        raise KeyError(f"Features missing required tree bundle columns: {missing[:20]}")
    frame = _clean_features(frame, feature_cols=bundle.feature_columns)
    extra_cols = [col for col in (include_extra_cols or []) if col in frame.columns and col not in KEY_COLS]
    out = frame.loc[:, KEY_COLS + extra_cols].copy()
    for target in bundle.targets:
        preds = _predict_model(
            bundle.models[target],
            frame[bundle.feature_columns],
            model_type=bundle.model_type,
            best_iteration=bundle.best_iterations.get(target),
        )
        pred_col = f"pred_{target}"
        preds = np.asarray(preds, dtype=float)
        if target.endswith("_pct_label"):
            preds = np.clip(preds, 0.0, 1.0)
        else:
            preds = np.clip(preds, 0.0, None)
        out[pred_col] = preds
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date.astype(str)
    return out


def score_tree_rate_bundle_features_to_csv(
    *,
    features_df: pd.DataFrame,
    bundle_dir: Path,
    output_csv: Path,
    include_extra_cols: list[str] | None = None,
    game_date: str | None = None,
) -> dict[str, Any]:
    bundle = load_tree_rate_bundle(bundle_dir)
    scored = score_tree_rate_bundle_features(
        features_df=features_df,
        bundle=bundle,
        include_extra_cols=include_extra_cols,
        game_date=game_date,
    )
    output_csv = Path(output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_csv, index=False)
    return {
        "bundle_dir": str(bundle_dir),
        "output_csv": str(output_csv),
        "row_count": int(len(scored)),
        "targets": list(bundle.targets),
        "feature_count": int(len(bundle.feature_columns)),
        "model_type": bundle.model_type,
    }
