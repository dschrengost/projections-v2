#!/usr/bin/env python3
"""Train a lightweight sparse-emergency apply/no-apply gate from full-history labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score

from projections import paths


DEFAULT_FEATURES = (
    "prior_play_prob",
    "minutes_from_stints_prior_20",
    "recent_start_pct_10",
    "started_proxy_rate_prior_10",
    "started_proxy_rate_prior_20",
    "an_implied_minutes",
    "lineup_starter_announced",
)


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_history_dir(value: str | None) -> Path:
    root = paths.get_data_root() / "training" / "runs"
    if value:
        p = Path(value).expanduser()
        if p.exists():
            return p.resolve()
        p2 = root / value
        if p2.exists():
            return p2.resolve()
        raise FileNotFoundError(f"History directory not found: {value}")
    candidates = sorted(root.glob("next_man_up_history_*"))
    if not candidates:
        raise FileNotFoundError(f"No next_man_up_history_* directories found under {root}")
    return candidates[-1].resolve()


def _compute_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_score = float("-inf")
    best_payload: dict[str, float] = {}
    for threshold in np.linspace(0.05, 0.95, 37):
        y_pred = y_prob >= threshold
        score = float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_payload = {
                "f0_5": score,
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "pred_positive_rate": float(np.mean(y_pred)),
            }
    return best_threshold, best_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--val-days", type=int, default=60)
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--require-propless", action="store_true")
    parser.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--c", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_dir = _resolve_history_dir(args.history_dir)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (paths.get_data_root() / "training" / "runs" / f"sparse_emergency_gate_{_utc_now_compact()}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(history_dir / "labeled_rows.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    feature_names = [c.strip() for c in str(args.features).split(",") if c.strip()]
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing gate features in labeled_rows.parquet: {missing}")

    mask = pd.Series(np.ones(len(df), dtype=bool), index=df.index)
    if bool(args.candidate_only):
        if "sparse_prior_signal" not in df.columns:
            raise ValueError("candidate-only training requires sparse_prior_signal in labeled_rows.parquet")
        mask &= pd.to_numeric(df["sparse_prior_signal"], errors="coerce").fillna(0.0).ge(0.5)
    if bool(args.require_propless):
        if "propless" not in df.columns:
            raise ValueError("require-propless training requires propless in labeled_rows.parquet")
        mask &= pd.to_numeric(df["propless"], errors="coerce").fillna(0.0).ge(0.5)
    work = df.loc[mask].copy()
    if work.empty:
        raise RuntimeError("No rows available after gate training filters")

    target = work["primary_archetype"].astype(str).ne("none").to_numpy(dtype=np.int64)
    cutoff = work["game_date"].max() - pd.Timedelta(days=int(args.val_days))
    train_mask = work["game_date"] < cutoff
    val_mask = ~train_mask
    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
        raise RuntimeError("Gate split produced empty train or val partition")

    x_all = work.loc[:, feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    x_train = x_all[train_mask.to_numpy()]
    x_val = x_all[val_mask.to_numpy()]
    y_train = target[train_mask.to_numpy()]
    y_val = target[val_mask.to_numpy()]

    feat_mean = x_train.mean(axis=0)
    feat_std = x_train.std(axis=0)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
    x_train_z = (x_train - feat_mean) / feat_std
    x_val_z = (x_val - feat_mean) / feat_std

    model = LogisticRegression(
        C=float(args.c),
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    model.fit(x_train_z, y_train)

    val_prob = model.predict_proba(x_val_z)[:, 1]
    threshold, threshold_payload = _compute_threshold(y_val, val_prob)
    y_val_pred = val_prob >= threshold

    artifact = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "history_dir": str(history_dir),
        "feature_names": feature_names,
        "feature_means": feat_mean.tolist(),
        "feature_stds": feat_std.tolist(),
        "coefficients": model.coef_[0].astype(float).tolist(),
        "intercept": float(model.intercept_[0]),
        "prob_threshold": float(threshold),
        "training": {
            "candidate_only": bool(args.candidate_only),
            "require_propless": bool(args.require_propless),
            "val_days": int(args.val_days),
            "c": float(args.c),
            "train_rows": int(len(y_train)),
            "val_rows": int(len(y_val)),
            "train_positive_rate": float(y_train.mean()),
            "val_positive_rate": float(y_val.mean()),
        },
        "validation": {
            "roc_auc": float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else float("nan"),
            "average_precision": float(average_precision_score(y_val, val_prob)),
            **threshold_payload,
        },
    }
    (out_dir / "gate_artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "threshold": threshold, "validation": artifact["validation"]}, indent=2))


if __name__ == "__main__":
    main()
