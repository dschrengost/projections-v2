from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import typer

from projections.fpts_v2.current import load_current_fpts_bundle
from projections.fpts_v2.features import CATEGORICAL_FEATURES_DEFAULT, build_fpts_design_matrix
from projections.sim_v1.residuals import assign_bucket, default_buckets

app = typer.Typer(add_completion=False)


def _parse_date(value: str) -> date:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid date format: {value} (expected YYYY-MM-DD)") from exc


def _season_from_date(day: date) -> int:
    return day.year if day.month >= 8 else day.year - 1


def _iter_days(start: date, end: date):
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.abs(y_true[mask] - y_pred[mask]).mean())


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _assign_buckets(df: pd.DataFrame) -> pd.Series:
    bucket_defs = default_buckets()
    working = df.copy()
    if "minutes_pred_p50" not in working.columns and "minutes_p50" in working.columns:
        working["minutes_pred_p50"] = working["minutes_p50"]
    if "minutes_pred_p50" not in working.columns and "minutes_actual" in working.columns:
        working["minutes_pred_p50"] = working["minutes_actual"]
    if "is_starter" not in working.columns:
        working["is_starter"] = 0
    return working.apply(lambda row: assign_bucket(row, bucket_defs), axis=1)


def _ensure_predictions(merged: pd.DataFrame, data_root: Path, join_keys: list[str]) -> pd.DataFrame:
    has_pred_col = "dk_fpts_pred" in merged.columns
    has_non_nan = has_pred_col and merged["dk_fpts_pred"].notna().any()
    if has_non_nan:
        return merged

    bundle = load_current_fpts_bundle(data_root=data_root)
    feature_cols = bundle.feature_cols
    missing = [c for c in feature_cols if c not in merged.columns]
    if missing:
        raise RuntimeError(f"debug_worlds_vs_labels: missing FPTS feature columns: {missing}")

    base_slice = merged.drop_duplicates(subset=join_keys)
    X = build_fpts_design_matrix(
        base_slice,
        feature_cols,
        categorical_cols=CATEGORICAL_FEATURES_DEFAULT,
        fill_missing_with_zero=False,
    )
    num_iter = getattr(bundle.model, "best_iteration", None) or getattr(bundle.model, "best_iteration_", None)
    preds = bundle.model.predict(
        X.values, num_iteration=int(num_iter) if num_iter and num_iter > 0 else None
    )
    pred_frame = base_slice[join_keys].copy()
    pred_frame["dk_fpts_pred"] = preds

    merged = merged.drop(columns=["dk_fpts_pred"], errors="ignore")
    merged = merged.merge(pred_frame, on=join_keys, how="left")
    return merged


def _log_intra_game_correlation(worlds: pd.DataFrame) -> None:
    if worlds.empty or "game_id" not in worlds.columns or "world_id" not in worlds.columns:
        return
    grouped = worlds.groupby("game_id")
    for _, grp in grouped:
        if grp["player_id"].nunique() < 2 or grp["world_id"].nunique() < 2:
            continue
        pivot = grp.pivot(index="world_id", columns="player_id", values="dk_fpts_world")
        if pivot.shape[1] < 2:
            continue
        cols = pivot.columns[:2]
        corr = pivot[cols[0]].corr(pivot[cols[1]])
        if pd.notna(corr):
            typer.echo(f"[debug_worlds] sample intra-game corr across worlds: {corr:.3f}")
            return


@app.command()
def main(
    data_root: Path = typer.Option(..., "--data-root"),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    n_worlds_limit: Optional[int] = typer.Option(
        None,
        "--n-worlds-limit",
        help="Optional cap on worlds per player for speed",
    ),
) -> None:
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    per_player_frames: list[pd.DataFrame] = []

    for day in _iter_days(start_dt, end_dt):
        season = _season_from_date(day)
        day_token = day.isoformat()

        worlds_path = (
            data_root / "artifacts" / "sim_v1" / "worlds" / f"season={season}" / f"game_date={day_token}" / "worlds.parquet"
        )
        base_path = (
            data_root / "gold" / "fpts_training_base" / f"season={season}" / f"game_date={day_token}" / "fpts_training_base.parquet"
        )

        if not worlds_path.exists() or not base_path.exists():
            typer.echo(f"[debug_worlds] {day_token} missing {'worlds' if not worlds_path.exists() else ''} {'base' if not base_path.exists() else ''}; skipping.")
            continue

        worlds = pd.read_parquet(worlds_path)
        base = pd.read_parquet(base_path)

        if n_worlds_limit is not None and "world_id" in worlds.columns:
            world_ids = sorted(pd.unique(worlds["world_id"]))
            keep_ids = set(world_ids[:n_worlds_limit])
            worlds = worlds[worlds["world_id"].isin(keep_ids)]

        join_keys = ["season", "game_date", "game_id", "team_id", "player_id"]
        merged = worlds.merge(base, on=join_keys, how="inner", suffixes=("", "_base"))
        if merged.empty:
            typer.echo(f"[debug_worlds] {day_token} join produced 0 rows; skipping.")
            continue

        merged = _ensure_predictions(merged, data_root, join_keys)
        _log_intra_game_correlation(worlds)

        label_candidates = [c for c in ("dk_fpts_actual", "dk_fpts", "fpts_dk_label") if c in merged.columns]
        if not label_candidates:
            raise RuntimeError("debug_worlds_vs_labels: missing dk_fpts label column.")
        label_col = label_candidates[0]
        pred_col = "dk_fpts_pred" if "dk_fpts_pred" in merged.columns else None

        grouped = merged.groupby(join_keys, dropna=False)
        agg_rows = []
        for _, grp in grouped:
            label = grp[label_col].iloc[0] if label_col in grp else np.nan
            pred = grp[pred_col].iloc[0] if pred_col else np.nan
            minutes = grp["minutes_p50"].iloc[0] if "minutes_p50" in grp else grp.get("minutes_actual", pd.Series([np.nan])).iloc[0]
            player_name = grp["player_name"].iloc[0] if "player_name" in grp else None
            sim_mean = float(grp["dk_fpts_world"].mean())
            sim_std = float(grp["dk_fpts_world"].std(ddof=0))
            agg_rows.append(
                {
                    "season": grp["season"].iloc[0],
                    "game_date": grp["game_date"].iloc[0],
                    "game_id": grp["game_id"].iloc[0],
                    "team_id": grp["team_id"].iloc[0],
                    "player_id": grp["player_id"].iloc[0],
                    "player_name": player_name,
                    "dk_fpts_label": label,
                    "dk_fpts_pred": pred,
                    "minutes_p50": minutes,
                    "sim_mean": sim_mean,
                    "sim_std": sim_std,
                }
            )

        per_player_frames.append(pd.DataFrame(agg_rows))

    if not per_player_frames:
        typer.echo("[debug_worlds] no data found in the requested window.")
        raise typer.Exit(code=0)

    df_all = pd.concat(per_player_frames, ignore_index=True)
    df_all["bucket"] = _assign_buckets(df_all)

    labels_raw = pd.to_numeric(df_all["dk_fpts_label"], errors="coerce").to_numpy()
    preds_raw = pd.to_numeric(df_all["dk_fpts_pred"], errors="coerce").to_numpy()
    sim_means_raw = pd.to_numeric(df_all["sim_mean"], errors="coerce").to_numpy()

    mask = ~np.isnan(labels_raw)
    labels = labels_raw[mask]
    preds = preds_raw[mask]
    sim_means = sim_means_raw[mask]

    global_metrics: dict[str, Any] = {
        "n": int(len(labels)),
        "mae_pred": _mae(labels, preds),
        "rmse_pred": _rmse(labels, preds),
        "mae_sim": _mae(labels, sim_means),
        "rmse_sim": _rmse(labels, sim_means),
        "corr_pred_label": _corr(preds, labels),
        "corr_sim_label": _corr(sim_means, labels),
    }

    typer.echo(
        "[debug_worlds] "
        f"n={global_metrics['n']} "
        f"corr_pred_label={global_metrics['corr_pred_label']:.3f} "
        f"corr_sim_label={global_metrics['corr_sim_label']:.3f} "
        f"mae_pred={global_metrics['mae_pred']:.3f} mae_sim={global_metrics['mae_sim']:.3f} "
        f"rmse_pred={global_metrics['rmse_pred']:.3f} rmse_sim={global_metrics['rmse_sim']:.3f}"
    )

    bucket_metrics: list[dict[str, Any]] = []
    for bucket, grp in df_all.groupby("bucket", dropna=False):
        labels_b_raw = pd.to_numeric(grp["dk_fpts_label"], errors="coerce").to_numpy()
        preds_b_raw = pd.to_numeric(grp["dk_fpts_pred"], errors="coerce").to_numpy()
        sim_means_b_raw = pd.to_numeric(grp["sim_mean"], errors="coerce").to_numpy()
        sim_stds_b = pd.to_numeric(grp["sim_std"], errors="coerce").to_numpy()
        mask_b = ~np.isnan(labels_b_raw)
        labels_b = labels_b_raw[mask_b]
        preds_b = preds_b_raw[mask_b]
        sim_means_b = sim_means_b_raw[mask_b]
        resid_std = float(np.nanstd(labels_b - preds_b)) if labels_b.size > 0 else float("nan")
        bucket_metrics.append(
            {
                "bucket": bucket,
                "n": int(len(grp)),
                "mae_pred": _mae(labels_b, preds_b),
                "mae_sim": _mae(labels_b, sim_means_b),
                "rmse_pred": _rmse(labels_b, preds_b),
                "rmse_sim": _rmse(labels_b, sim_means_b),
                "mean_sim_std": float(np.nanmean(sim_stds_b)) if len(sim_stds_b) else float("nan"),
                "resid_std": resid_std,
            }
        )

    bucket_df = pd.DataFrame(bucket_metrics)
    if not bucket_df.empty:
        display_cols = ["bucket", "n", "mae_pred", "mae_sim", "mean_sim_std"]
        typer.echo(bucket_df[display_cols].to_string(index=False))

    sample_cols = [c for c in ("player_name", "player_id", "dk_fpts_label", "dk_fpts_pred", "sim_mean", "sim_std", "minutes_p50") if c in df_all.columns]
    sample = df_all[sample_cols].head(10)
    typer.echo("[debug_worlds] sample rows:")
    typer.echo(sample.to_string(index=False))

    out_path = (
        data_root
        / "artifacts"
        / "sim_v1"
        / f"debug_worlds_vs_labels_{start_dt}_{end_dt}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "global_metrics": global_metrics,
        "bucket_metrics": bucket_metrics,
    }
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    typer.echo(f"[debug_worlds] wrote summary to {out_path}")


if __name__ == "__main__":
    app()
