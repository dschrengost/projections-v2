"""
Compare rates_v1 GBM vs simple baselines on a shared val split.

Baselines:
- baseline_season: season-long per-minute averages (e.g., fga2 = season_fga_per_min - season_3pa_per_min).
  NaNs filled by (position_primary, is_starter) mean, else global mean.
- baseline_pos_role: group mean by (position_primary, is_starter) from the training split.

Example:
    uv run python -m scripts.rates.compare_baselines \
        --data-root /home/daniel/projections-data \
        --rates-run-id rates_v1_stage0_20251127_172108 \
        --start-date 2023-10-01 \
        --end-date   2025-11-26 \
        --train-end-date 2024-06-30 \
        --cal-end-date   2025-03-01
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.paths import data_path
from projections.rates_v1.loader import load_rates_bundle
from projections.rates_v1.score import predict_rates

TARGETS = [
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

app = typer.Typer(add_completion=False)


def _iter_partitions(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    base = root / "gold" / "rates_training_base"
    parts: list[Path] = []
    for season_dir in base.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if start <= day <= end:
                candidate = day_dir / "rates_training_base.parquet"
                if candidate.exists():
                    parts.append(candidate)
    if not parts:
        raise FileNotFoundError("No rates_training_base partitions found for window.")
    return parts


def _load_base(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    paths = _iter_partitions(root, start, end)
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def _split(df: pd.DataFrame, train_end: pd.Timestamp, cal_end: pd.Timestamp):
    train_df = df[df["game_date"] < train_end].copy()
    cal_df = df[(df["game_date"] >= train_end) & (df["game_date"] < cal_end)].copy()
    val_df = df[df["game_date"] >= cal_end].copy()
    return train_df, cal_df, val_df


def _impute_features(train_df: pd.DataFrame, *others: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    odds_cols = ["spread_close", "total_close", "team_itt", "opp_itt"]
    medians = {c: train_df[c].median(skipna=True) for c in odds_cols}
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in ("is_starter", "home_flag"):
            if col in out.columns:
                out[col] = out[col].fillna(0).astype(int)
        for col in odds_cols:
            out[col] = out[col].fillna(medians[col])
        fill_zero_cols = [
            "season_fga_per_min",
            "season_3pa_per_min",
            "season_fta_per_min",
            "season_ast_per_min",
            "season_tov_per_min",
            "season_reb_per_min",
            "season_stl_per_min",
            "season_blk_per_min",
            "days_rest",
        ]
        for col in fill_zero_cols:
            if col in out.columns:
                out[col] = out[col].fillna(0)
        return out
    return tuple([_prep(train_df)] + [_prep(df) for df in others])


def _baseline_season(val_df: pd.DataFrame) -> pd.DataFrame:
    """
    Season-long per-minute averages; columns are suffixed _baseline_season.
    """
    out = pd.DataFrame(index=val_df.index)
    # FGA split
    if {"season_fga_per_min", "season_3pa_per_min"}.issubset(val_df.columns):
        season_fga = val_df["season_fga_per_min"]
        season_3pa = val_df["season_3pa_per_min"]
        out["fga2_per_min_baseline_season"] = season_fga - season_3pa
        out["fga3_per_min_baseline_season"] = season_3pa
    else:
        mean_fga2 = val_df["fga2_per_min"].mean()
        mean_fga3 = val_df["fga3_per_min"].mean()
        out["fga2_per_min_baseline_season"] = mean_fga2
        out["fga3_per_min_baseline_season"] = mean_fga3

    mapping = {
        "fta_per_min": "season_fta_per_min",
        "ast_per_min": "season_ast_per_min",
        "tov_per_min": "season_tov_per_min",
        "oreb_per_min": "season_oreb_per_min",
        "dreb_per_min": "season_dreb_per_min",
        "stl_per_min": "season_stl_per_min",
        "blk_per_min": "season_blk_per_min",
    }
    for tgt, src in mapping.items():
        col = f"{tgt}_baseline_season"
        if src in val_df.columns:
            out[col] = val_df[src]
        else:
            out[col] = val_df[tgt].mean()

    for tgt in TARGETS:
        col = f"{tgt}_baseline_season"
        if col in out.columns:
            mean_val = out[col].mean()
            out[col] = out[col].fillna(mean_val)
    return out


def _baseline_pos_role(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean per-minute by (position_primary, is_starter) from training split; suffixed _baseline_posrole.
    """
    group_cols = ["position_primary", "is_starter"]
    means = train_df.groupby(group_cols)[TARGETS].mean().reset_index()
    merged = val_df[group_cols].merge(means, on=group_cols, how="left")
    out = pd.DataFrame(index=val_df.index)
    for tgt in TARGETS:
        col = f"{tgt}_baseline_posrole"
        if tgt in merged.columns:
            out[col] = merged[tgt]
        else:
            out[col] = val_df[tgt].mean()
        mean_val = out[col].mean()
        out[col] = out[col].fillna(mean_val)
    return out


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    err = y_pred - y_true
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    return {"mae": mae, "rmse": rmse, "n": int(len(y_true))}


@app.command()
def main(
    data_root: Optional[Path] = typer.Option(None, help="Data root (defaults PROJECTIONS_DATA_ROOT or ./data)"),
    rates_run_id: str = typer.Option("rates_v1_stage0_20251127_172108", help="Trained rates_v1 run id"),
    start_date: str = typer.Option(..., help="Start date (inclusive) for loading base"),
    end_date: str = typer.Option(..., help="End date (inclusive) for loading base"),
    train_end_date: str = typer.Option(..., help="Cutoff: game_date < train_end_date goes to train"),
    cal_end_date: str = typer.Option(..., help="Cutoff: train_end_date <= date < cal_end_date goes to cal; rest to val"),
) -> None:
    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    train_cut = pd.Timestamp(train_end_date).normalize()
    cal_cut = pd.Timestamp(cal_end_date).normalize()

    df = _load_base(root, start, end)
    train_df, cal_df, val_df = _split(df, train_cut, cal_cut)
    if val_df.empty:
        typer.echo("[compare] val set is empty; adjust date window.")
        raise SystemExit(1)

    train_df, cal_df, val_df = _impute_features(train_df, cal_df, val_df)

    bundle = load_rates_bundle(rates_run_id, base_artifacts_root=root)
    val_features = val_df[bundle.feature_cols]
    df_gbm = predict_rates(val_features, bundle)

    base_season = _baseline_season(val_df)
    base_posrole = _baseline_pos_role(train_df, val_df)

    results: dict[str, dict] = {}
    for tgt in TARGETS:
        y_true = val_df[tgt].to_numpy()
        y_pred_gbm = df_gbm[f"{tgt}_mean"].to_numpy() if f"{tgt}_mean" in df_gbm.columns else df_gbm[tgt].to_numpy()
        y_pred_season = base_season[f"{tgt}_baseline_season"].to_numpy()
        y_pred_posrole = base_posrole[f"{tgt}_baseline_posrole"].to_numpy()
        res = {
            "gbm": _metrics(y_true, y_pred_gbm),
            "baseline_season": _metrics(y_true, y_pred_season),
            "baseline_posrole": _metrics(y_true, y_pred_posrole),
        }
        results[tgt] = res

    # Print table
    typer.echo("Target\tModel\tMAE\tRMSE\tn")
    for tgt, res in results.items():
        for model, vals in res.items():
            typer.echo(f"{tgt}\t{model}\t{vals['mae']:.4f}\t{vals['rmse']:.4f}\t{vals['n']}")

    # Write JSON
    out_path = root / "artifacts" / "rates_v1" / "runs" / rates_run_id / "baseline_compare.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"val": results}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"[compare] wrote baseline comparison to {out_path}")


if __name__ == "__main__":
    app()
