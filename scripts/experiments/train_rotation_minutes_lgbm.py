#!/usr/bin/env python3
"""Train a rotation+minutes LGBM baseline for 240-minute team allocation.

Pipeline:
  1) rot8 classifier: y = 1 if minutes_actual >= rot_threshold else 0
  2) minutes regressor: train only on y==1 rows

Example:
  uv run python -m scripts.experiments.train_rotation_minutes_lgbm \\
    --train-parquet artifacts/experiments/team_alloc_dataset/minutes_teamalloc_train_no_ot.parquet \\
    --out-dir artifacts/experiments/lgbm_rotalloc_v1 \\
    --split-mode time --val-frac 0.2 --rot-threshold 8 --seed 7 \\
    --lgbm-artifact minutes_v1_safe_starter_20251214
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from projections.models.calibration import SigmoidCalibrator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


KEY_COLUMNS = ("game_id", "team_id", "player_id")


@dataclass(frozen=True)
class RotAllocTrainConfig:
    train_parquet: str
    out_dir: str
    split_mode: str
    val_frac: float
    label_mode: str
    rot_threshold: float
    rot_topk: int
    exclude_out: bool
    seed: int
    calibration: str
    features: list[str]
    used_lgbm_artifact_features: bool


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_feature_list_from_artifact(run_id_or_path: str) -> list[str]:
    bundle_dir = Path(run_id_or_path)
    if not bundle_dir.exists():
        bundle_dir = Path("artifacts") / "minutes_lgbm" / run_id_or_path
    feature_path = bundle_dir / "feature_columns.json"
    payload = json.loads(feature_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(x) for x in payload]
    if isinstance(payload, dict) and isinstance(payload.get("columns"), list):
        return [str(x) for x in payload["columns"]]
    raise ValueError(f"Unrecognized feature_columns.json format at {feature_path}")


def _infer_feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    blocked = set(KEY_COLUMNS) | {label_col, "game_date", "tip_ts", "feature_as_of_ts"}
    cols: list[str] = []
    for col in df.columns:
        if col in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric feature columns detected; provide --features or --lgbm-artifact")
    return sorted(cols)


def add_missingness_indicators(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    suffix: str = "_is_nan",
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    new_cols: list[str] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if not df[col].isna().any():
            continue
        ind_col = f"{col}{suffix}"
        if ind_col in df.columns:
            continue
        df[ind_col] = df[col].isna().astype(np.float32)
        new_cols.append(ind_col)
    return df, new_cols


def ensure_infer_feature_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    indicator_suffix: str = "_is_nan",
) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col.endswith(indicator_suffix):
            base = col[: -len(indicator_suffix)]
            if base not in df.columns:
                df[base] = 0.0
        elif col not in df.columns:
            df[col] = 0.0
    for col in feature_cols:
        if not col.endswith(indicator_suffix):
            continue
        base = col[: -len(indicator_suffix)]
        if base in df.columns:
            df[col] = df[base].isna().astype(np.float32)
        else:
            df[col] = 1.0
    return df


def fill_missing_values(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _split_train_val_team_games(
    df: pd.DataFrame,
    *,
    val_frac: float,
    seed: int,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = df.loc[:, ["game_id", "team_id", "game_date"]].drop_duplicates().copy()
    if mode not in {"random", "time"}:
        raise ValueError("split-mode must be random or time")

    if mode == "random":
        rng = np.random.RandomState(seed)
        order = np.arange(len(keys))
        rng.shuffle(order)
        keys = keys.iloc[order].reset_index(drop=True)
    else:
        if "game_date" not in keys.columns:
            raise ValueError("Time split requested but game_date column missing")
        dt = pd.to_datetime(keys["game_date"], errors="raise")
        keys = keys.assign(_dt=dt)
        keys = keys.sort_values(["_dt", "game_id", "team_id"], kind="mergesort").drop(columns="_dt")

    n_val = int(len(keys) * float(val_frac))
    n_val = max(1, n_val)
    val_keys = keys.tail(n_val) if mode == "time" else keys.head(n_val)
    train_keys = keys.iloc[: len(keys) - n_val] if mode == "time" else keys.iloc[n_val:]

    train_df = df.merge(train_keys.loc[:, ["game_id", "team_id"]], on=["game_id", "team_id"], how="inner")
    val_df = df.merge(val_keys.loc[:, ["game_id", "team_id"]], on=["game_id", "team_id"], how="inner")

    if mode == "time" and len(val_keys):
        train_dt = pd.to_datetime(train_keys["game_date"], errors="coerce")
        val_dt = pd.to_datetime(val_keys["game_date"], errors="coerce")
        logger.info(
            "Time split: "
            f"train={len(train_keys)} ({train_dt.min().date()}..{train_dt.max().date()}) | "
            f"val={len(val_keys)} ({val_dt.min().date()}..{val_dt.max().date()})"
        )
    else:
        logger.info(f"Split: {len(train_keys)} train, {len(val_keys)} val team-games")

    return train_df, val_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-mode", type=str, choices=["random", "time"], default="time")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--label-mode", type=str, choices=["threshold", "topk"], default="threshold")
    parser.add_argument("--rot-threshold", type=float, default=8.0)
    parser.add_argument("--rot-topk", type=int, default=9)
    parser.add_argument(
        "--exclude-out",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude status=='OUT' rows before splitting/training (default: True)",
    )
    parser.add_argument("--calibration", type=str, choices=["isotonic", "sigmoid", "none"], default="isotonic")
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument(
        "--lgbm-artifact",
        type=str,
        default=None,
        help="Optional artifact id/path to load feature_columns.json only",
    )
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()
    _set_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.train_parquet)
    required = {"game_id", "team_id", "player_id", "game_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training parquet missing required columns: {sorted(missing)}")

    label_col = None
    for cand in ("minutes_actual", "minutes"):
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise ValueError("Training parquet missing minutes label column (expected minutes_actual or minutes)")

    df = df.rename(columns={label_col: "minutes_actual"}).copy()

    # Optionally exclude OUT rows from candidate set.
    if args.exclude_out and "status" in df.columns:
        status = df["status"].astype(str).str.upper()
        before = len(df)
        df = df.loc[status != "OUT"].copy()
        logger.info(f"Excluded OUT rows: {before} -> {len(df)}")

    dup = int(df.duplicated(subset=list(KEY_COLUMNS)).sum())
    if dup:
        raise ValueError(f"Found duplicate key rows (game_id, team_id, player_id): dup_rows={dup}")

    # Features
    used_artifact = False
    if args.features:
        feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    elif args.lgbm_artifact:
        feature_cols = _load_feature_list_from_artifact(args.lgbm_artifact)
        used_artifact = True
    else:
        feature_cols = _infer_feature_columns(df, label_col="minutes_actual")

    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        raise ValueError(
            f"Requested feature columns missing from training parquet: {missing_feats[:20]}"
            f"{'...' if len(missing_feats) > 20 else ''}"
        )

    # Missingness indicators + fill.
    df, indicator_cols = add_missingness_indicators(df, feature_cols)
    feature_cols = feature_cols + indicator_cols
    df = ensure_infer_feature_columns(df, feature_cols)
    df = fill_missing_values(df, feature_cols)

    # Labels
    label_mode = str(args.label_mode).strip().lower()
    if label_mode == "threshold":
        df["y_rot"] = (df["minutes_actual"] >= float(args.rot_threshold)).astype(np.int8)
    elif label_mode == "topk":
        rot_topk = int(args.rot_topk)
        if rot_topk <= 0:
            raise ValueError("--rot-topk must be >= 1 when --label-mode topk")
        df = df.sort_values(
            ["game_id", "team_id", "minutes_actual", "player_id"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        played = df["minutes_actual"] > 0
        rank = (
            df.loc[played]
            .groupby(["game_id", "team_id"], sort=False)["minutes_actual"]
            .rank(method="first", ascending=False)
        )
        df["y_rot"] = 0
        df.loc[played, "y_rot"] = (rank <= float(rot_topk)).astype(np.int8).to_numpy()
        df["y_rot"] = df["y_rot"].astype(np.int8)
        logger.info(f"Using topk label mode: rot_topk={rot_topk}")
    else:
        raise ValueError(f"Unknown label mode: {label_mode}")

    # Split by team-game.
    train_df, val_df = _split_train_val_team_games(
        df, val_frac=float(args.val_frac), seed=int(args.seed), mode=str(args.split_mode)
    )

    y_train = train_df["y_rot"]
    X_val = val_df[feature_cols]
    y_val = val_df["y_rot"]

    # Train classifier.
    # Hold out a calibration split from the TRAIN set (not val) to avoid leakage.
    train_team_games = train_df.loc[:, ["game_id", "team_id"]].drop_duplicates().copy()
    rng = np.random.RandomState(int(args.seed))
    order = np.arange(len(train_team_games))
    rng.shuffle(order)
    train_team_games = train_team_games.iloc[order].reset_index(drop=True)
    n_cal_tg = max(1, int(0.2 * len(train_team_games)))
    cal_keys = train_team_games.head(n_cal_tg)
    fit_df = train_df.merge(cal_keys, on=["game_id", "team_id"], how="left", indicator=True)
    cal_df = fit_df.loc[fit_df["_merge"] == "both"].drop(columns="_merge")
    fit_df = fit_df.loc[fit_df["_merge"] == "left_only"].drop(columns="_merge")

    if fit_df.empty:
        # Degenerate: all went to cal due to small data; just use full train.
        fit_df = train_df
        cal_df = train_df.iloc[:0].copy()

    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=64,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=0.1,
        random_state=int(args.seed),
        n_jobs=1,
    )
    clf.fit(fit_df[feature_cols], fit_df["y_rot"])

    calibration = str(args.calibration).strip().lower()
    calibrator: IsotonicRegression | SigmoidCalibrator | None = None
    if calibration == "none":
        logger.info("Skipping calibration (--calibration none).")
    elif cal_df.empty or cal_df["y_rot"].nunique() != 2:
        logger.info("Skipping calibration (cal split missing or has only one class).")
    else:
        p_cal_raw = clf.predict_proba(cal_df[feature_cols])[:, 1]
        y_cal = cal_df["y_rot"].to_numpy()
        if calibration == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(p_cal_raw, y_cal)
            logger.info(f"Fit isotonic calibrator on {len(cal_df)} calibration rows")
        elif calibration == "sigmoid":
            eps = 1e-6
            p_clip = np.clip(np.asarray(p_cal_raw, dtype=np.float64), eps, 1.0 - eps)
            x = np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)
            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(x, y_cal)
            calibrator = SigmoidCalibrator(
                coef=float(lr.coef_[0, 0]),
                intercept=float(lr.intercept_[0]),
                eps=float(eps),
            )
            logger.info(f"Fit sigmoid calibrator on {len(cal_df)} calibration rows")
        else:
            raise ValueError(f"Unknown calibration method: {calibration}")

    p_val_raw = clf.predict_proba(X_val)[:, 1]
    p_val = calibrator.transform(p_val_raw) if calibrator is not None else p_val_raw

    # Train regressor on y_rot==1 rows.
    train_pos = train_df["y_rot"] == 1
    if int(train_pos.sum()) == 0:
        raise ValueError("No positive rot8 rows in training set (y_rot==1); cannot train regressor")
    reg = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=64,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=0.1,
        random_state=int(args.seed),
        n_jobs=1,
    )
    reg.fit(train_df.loc[train_pos, feature_cols], train_df.loc[train_pos, "minutes_actual"])

    # Save artifacts.
    config = RotAllocTrainConfig(
        train_parquet=str(args.train_parquet),
        out_dir=str(args.out_dir),
        split_mode=str(args.split_mode),
        val_frac=float(args.val_frac),
        label_mode=label_mode,
        rot_threshold=float(args.rot_threshold),
        rot_topk=int(args.rot_topk),
        exclude_out=bool(args.exclude_out),
        seed=int(args.seed),
        calibration=calibration,
        features=list(feature_cols),
        used_lgbm_artifact_features=bool(used_artifact),
    )

    joblib.dump(clf, args.out_dir / "rot8_classifier.joblib")
    joblib.dump(reg, args.out_dir / "minutes_regressor.joblib")
    if calibrator is not None:
        if isinstance(calibrator, IsotonicRegression):
            joblib.dump(calibrator, args.out_dir / "rot8_calibrator_isotonic.joblib")
        else:
            joblib.dump(calibrator, args.out_dir / "rot8_calibrator_sigmoid.joblib")

    (args.out_dir / "feature_columns.json").write_text(
        json.dumps({"columns": list(feature_cols)}, indent=2), encoding="utf-8"
    )
    (args.out_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    # Minimal training diagnostics.
    metrics: dict[str, Any] = {
        "val_p_rot_mean": float(np.mean(p_val)),
        "val_p_rot_p90": float(np.quantile(p_val, 0.9)),
        "val_pos_rate": float(y_val.mean()),
        "train_pos_rate": float(y_train.mean()),
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_train_team_games": int(train_df.groupby(["game_id", "team_id"]).ngroups),
        "n_val_team_games": int(val_df.groupby(["game_id", "team_id"]).ngroups),
        "n_features": int(len(feature_cols)),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(f"Wrote artifacts to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
