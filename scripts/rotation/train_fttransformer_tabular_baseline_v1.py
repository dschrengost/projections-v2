#!/usr/bin/env python3
"""Train a pytorch-tabular FT-Transformer baseline on joint rotation/rates data.

This baseline is row-wise (tabular), not set-based. It is intended for quick
comparison against current production models and the joint set model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from projections import paths

KEY_COLS = ["game_id", "team_id", "player_id", "game_date"]
RATE_TARGETS = [
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paths.get_project_root())  # noqa: S603,S607
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _coerce_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("game_id", "team_id", "player_id"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    return out


def _load_feature_columns(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        cols = payload.get("columns")
    else:
        cols = payload
    if not isinstance(cols, list) or not cols:
        raise ValueError(f"Invalid feature column payload in {path}")
    return [str(c) for c in cols]


def _resolve_split_window(
    frame: pd.DataFrame,
    *,
    split_manifest_path: Path | None,
    val_start_date: str | None,
    val_end_date: str | None,
    val_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if split_manifest_path is not None:
        payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        split = payload.get("split") or {}
        start = split.get("val_min_date")
        end = split.get("val_max_date")
        if start and end:
            return pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()

    if val_start_date is not None or val_end_date is not None:
        start = pd.Timestamp(val_start_date).normalize() if val_start_date else pd.Timestamp(frame["game_date"].min()).normalize()
        end = pd.Timestamp(val_end_date).normalize() if val_end_date else pd.Timestamp(frame["game_date"].max()).normalize()
        return start, end

    unique_days = sorted(pd.to_datetime(frame["game_date"]).dropna().dt.normalize().unique().tolist())
    if not unique_days:
        raise ValueError("No game_date values available for split.")
    window = max(1, int(val_days))
    val_dates = unique_days[-window:]
    return pd.Timestamp(val_dates[0]).normalize(), pd.Timestamp(val_dates[-1]).normalize()


def _prepare_task_frame(base: pd.DataFrame, *, task: str) -> tuple[pd.DataFrame, list[str]]:
    work = base.copy()
    if task == "minutes":
        targets = ["minutes_label"]
        work = work[work["minutes_label"].notna()].copy()
    elif task == "rates":
        targets = RATE_TARGETS
        work = work[pd.to_numeric(work.get("rates_loss_eligible", 0), errors="coerce").fillna(0).astype(int) > 0].copy()
        work = work.dropna(subset=targets)
    elif task == "eff":
        targets = EFF_TARGETS
        work = work[pd.to_numeric(work.get("rates_loss_eligible", 0), errors="coerce").fillna(0).astype(int) > 0].copy()
        work = work.dropna(subset=targets)
    else:
        raise ValueError(f"Unsupported task={task}")

    if work.empty:
        raise ValueError(f"No rows left after filtering for task={task}")
    return work, targets


def _infer_default_feature_columns(df: pd.DataFrame, *, targets: list[str]) -> list[str]:
    excluded = set(KEY_COLS + targets + ["rates_loss_eligible", "rates_label_available_any", "rates_label_available_all_rate_targets"])
    cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric/bool feature columns found.")
    return cols


def _resolve_accelerator(accelerator_arg: str) -> str:
    value = str(accelerator_arg).strip().lower()
    if value in {"", "auto"}:
        if torch.cuda.is_available():
            return "gpu"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and bool(mps_backend.is_available()):
            return "mps"
        return "cpu"
    if value in {"cpu", "gpu", "cuda", "mps"}:
        if value in {"gpu", "cuda"} and not torch.cuda.is_available():
            raise ValueError(f"--accelerator={accelerator_arg!r} requested GPU, but CUDA is not available")
        if value == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is None or not bool(mps_backend.is_available()):
                raise ValueError("--accelerator='mps' requested, but MPS is not available")
        return "gpu" if value == "cuda" else value
    raise ValueError(f"Unsupported --accelerator={accelerator_arg!r}; expected auto/cpu/gpu/cuda/mps")


def _prepare_features(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = df.copy()
    categorical = [c for c in categorical_cols if c in feature_cols and c in out.columns]
    continuous = [c for c in feature_cols if c not in set(categorical)]

    for col in continuous:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")

    for col in categorical:
        if col not in out.columns:
            out[col] = "__MISSING__"
        out[col] = out[col].astype("string").fillna("__MISSING__")

    return out, continuous, categorical


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[valid] - y_pred[valid])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", type=str, required=True, help="Joint dataset directory containing features/labels parquet files.")
    parser.add_argument("--task", choices=["minutes", "rates", "eff"], default="minutes")
    parser.add_argument("--feature-columns-json", type=str, default=None, help="Optional feature_columns.json payload to enforce.")
    parser.add_argument("--split-manifest", type=str, default=None, help="Optional manifest with split.val_min_date/val_max_date.")
    parser.add_argument("--val-start-date", type=str, default=None, help="Optional explicit validation start date (YYYY-MM-DD).")
    parser.add_argument("--val-end-date", type=str, default=None, help="Optional explicit validation end date (YYYY-MM-DD).")
    parser.add_argument("--val-days", type=int, default=14, help="Validation window size when explicit dates are not provided.")
    parser.add_argument("--categorical-cols", type=str, default="", help="Comma-separated categorical feature columns.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap after task filtering (for rapid iteration).")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes for pytorch-tabular.")
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--input-embed-dim", type=int, default=32)
    parser.add_argument("--num-attn-blocks", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", type=str, default="auto", help="pytorch-lightning accelerator (auto/cpu/gpu/cuda/mps).")
    parser.add_argument("--devices", type=int, default=1, help="Device count for trainer.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output root. Defaults to $PROJECTIONS_DATA_ROOT/artifacts/joint_rotation_rates_v1/pytorch_tabular_runs",
    )
    parser.add_argument("--run-tag", type=str, default="fttransformer_tabular_baseline")
    args = parser.parse_args()
    try:
        from pytorch_tabular import TabularModel
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
        from pytorch_tabular.models import FTTransformerConfig
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "pytorch-tabular is not installed. Install with: uv add pytorch-tabular"
        ) from exc

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    out_root = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (paths.get_data_root() / "artifacts" / "joint_rotation_rates_v1" / "pytorch_tabular_runs").resolve()
    )
    run_id = f"{args.run_tag}_{args.task}_{_utc_now_compact()}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[tabular_train] dataset_dir={dataset_dir}")
    print(f"[tabular_train] run_dir={run_dir}")
    resolved_accelerator = _resolve_accelerator(str(args.accelerator))
    using_cuda = resolved_accelerator == "gpu" and torch.cuda.is_available()
    if using_cuda:
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"[tabular_train] accelerator={resolved_accelerator} ({device_name})")
    else:
        print(f"[tabular_train] accelerator={resolved_accelerator}")

    features = _coerce_keys(pd.read_parquet(dataset_dir / "features.parquet"))
    labels_minutes = _coerce_keys(pd.read_parquet(dataset_dir / "labels_minutes.parquet"))
    labels_rates = _coerce_keys(pd.read_parquet(dataset_dir / "labels_rates.parquet"))

    base = pd.concat(
        [
            features.reset_index(drop=True),
            labels_minutes.drop(columns=[c for c in KEY_COLS if c in labels_minutes.columns], errors="ignore").reset_index(drop=True),
            labels_rates.drop(columns=[c for c in KEY_COLS if c in labels_rates.columns], errors="ignore").reset_index(drop=True),
        ],
        axis=1,
    )
    base = _coerce_keys(base)

    frame, targets = _prepare_task_frame(base, task=str(args.task))
    if args.max_rows is not None and args.max_rows > 0 and len(frame) > args.max_rows:
        frame = frame.iloc[: int(args.max_rows)].copy()
        print(f"[tabular_train] max_rows applied -> {len(frame)}")

    feature_cols = _load_feature_columns(Path(args.feature_columns_json).expanduser().resolve() if args.feature_columns_json else None)
    if feature_cols is None:
        feature_cols = _infer_default_feature_columns(frame, targets=targets)
    else:
        feature_cols = [c for c in feature_cols if c in frame.columns]
    if not feature_cols:
        raise ValueError("No feature columns available after filtering.")

    categorical_cols = [c.strip() for c in str(args.categorical_cols).split(",") if c.strip()]
    frame, continuous_cols, categorical_cols = _prepare_features(
        frame,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
    )
    model_cols = list(dict.fromkeys(KEY_COLS + feature_cols + targets))
    frame = frame.loc[:, [c for c in model_cols if c in frame.columns]].copy()

    split_manifest = Path(args.split_manifest).expanduser().resolve() if args.split_manifest else None
    val_start, val_end = _resolve_split_window(
        frame,
        split_manifest_path=split_manifest,
        val_start_date=args.val_start_date,
        val_end_date=args.val_end_date,
        val_days=int(args.val_days),
    )
    is_val = (frame["game_date"] >= val_start) & (frame["game_date"] <= val_end)
    train_df = frame.loc[~is_val].copy()
    val_df = frame.loc[is_val].copy()
    if train_df.empty or val_df.empty:
        raise ValueError(f"Invalid split: train={len(train_df)} val={len(val_df)}")

    # pytorch-tabular can still emit NaN predictions when continuous features contain
    # unresolved NaNs. Impute from train medians to keep train/val consistent.
    if continuous_cols:
        train_medians = train_df[continuous_cols].median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for col in continuous_cols:
            fill = float(train_medians.get(col, 0.0))
            train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan).fillna(fill).astype("float32")
            val_df[col] = val_df[col].replace([np.inf, -np.inf], np.nan).fillna(fill).astype("float32")

    print(
        f"[tabular_train] task={args.task} targets={targets} "
        f"rows(train={len(train_df)}, val={len(val_df)}) "
        f"features(continuous={len(continuous_cols)}, categorical={len(categorical_cols)})"
    )

    data_config = DataConfig(
        target=targets,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=using_cuda,
        normalize_continuous_features=True,
        handle_missing_values=True,
    )
    trainer_config = TrainerConfig(
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        accelerator=resolved_accelerator,
        devices=int(args.devices),
        progress_bar="none",
        checkpoints="valid_loss",
        checkpoints_path=str(run_dir / "checkpoints"),
        checkpoints_name="best",
        checkpoints_save_top_k=1,
        load_best=True,
        seed=int(args.seed),
    )
    optimizer_config = OptimizerConfig(optimizer="AdamW")
    model_config = FTTransformerConfig(
        task="regression",
        learning_rate=float(args.learning_rate),
        input_embed_dim=int(args.input_embed_dim),
        num_attn_blocks=int(args.num_attn_blocks),
        num_heads=int(args.num_heads),
        attn_dropout=float(args.dropout),
        add_norm_dropout=float(args.dropout),
        ff_dropout=float(args.dropout),
        seed=int(args.seed),
    )

    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    model.fit(train=train_df, validation=val_df)

    pred_df = model.predict(val_df, include_input_features=False)
    metrics_by_target: dict[str, dict[str, Any]] = {}
    weighted_abs = 0.0
    weighted_n = 0

    output = val_df.loc[:, KEY_COLS + targets].copy()
    for target in targets:
        pred_col = f"{target}_prediction"
        if pred_col not in pred_df.columns:
            raise KeyError(f"Missing prediction column: {pred_col}")
        y_true = pd.to_numeric(val_df[target], errors="coerce").to_numpy(dtype=np.float32)
        y_pred = pd.to_numeric(pred_df[pred_col], errors="coerce").to_numpy(dtype=np.float32)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        mae = _mae(y_true, y_pred)
        n = int(valid.sum())
        metrics_by_target[target] = {"mae": mae, "count": n}
        if n > 0:
            weighted_abs += float(np.abs(y_true[valid] - y_pred[valid]).sum())
            weighted_n += n
        output[pred_col] = y_pred

    overall_mae = float(weighted_abs / weighted_n) if weighted_n > 0 else float("nan")

    model.save_model(str(run_dir / "model"))
    output.to_parquet(run_dir / "val_predictions.parquet", index=False)

    manifest = {
        "version": "fttransformer_tabular_baseline_v1",
        "created_at": _utc_now_iso(),
        "git_sha": _git_sha(),
        "run_id": run_id,
        "task": args.task,
        "targets": targets,
        "dataset_dir": str(dataset_dir),
        "split": {
            "val_start_date": str(val_start.date()),
            "val_end_date": str(val_end.date()),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "train_game_dates": int(pd.to_datetime(train_df["game_date"]).nunique()),
            "val_game_dates": int(pd.to_datetime(val_df["game_date"]).nunique()),
        },
        "features": {
            "total": int(len(feature_cols)),
            "continuous": int(len(continuous_cols)),
            "categorical": int(len(categorical_cols)),
            "columns": feature_cols,
            "categorical_columns": categorical_cols,
        },
        "metrics": {
            "val_mae_overall": overall_mae,
            "val_mae_by_target": metrics_by_target,
            "val_metric_count": int(weighted_n),
        },
        "args": vars(args),
        "outputs": {
            "model_dir": str(run_dir / "model"),
            "val_predictions": str(run_dir / "val_predictions.parquet"),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[tabular_train] finished run_id={run_id} val_mae={overall_mae:.6f} metrics_n={weighted_n}")
    print(f"[tabular_train] manifest={run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
