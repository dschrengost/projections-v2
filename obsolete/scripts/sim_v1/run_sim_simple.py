from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.fpts_v2.current import _get_data_root, _load_current_run_id, load_current_fpts_bundle
from projections.sim_v1.sampler import FptsResidualSampler

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


def _prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    feature_frame = df[feature_cols].copy()
    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")

    categorical_cols = {"track_role_cluster", "track_role_is_low_minutes"}
    for col in categorical_cols & set(feature_frame.columns):
        feature_frame[col] = feature_frame[col].fillna(-1).astype(int)

    feature_frame = feature_frame.fillna(0.0)
    return feature_frame


def _score_predictions(df: pd.DataFrame, feature_cols: list[str], model) -> np.ndarray:
    feature_frame = _prepare_features(df, feature_cols)
    mask = feature_frame.notna().all(axis=1)
    preds = np.full(len(df), np.nan)
    if mask.any():
        num_iter = getattr(model, "best_iteration", None) or getattr(model, "best_iteration_", None)
        preds[mask] = model.predict(
            feature_frame.loc[mask].values, num_iteration=int(num_iter) if num_iter and num_iter > 0 else None
        )
    return preds


def _load_minutes_for_day(data_root: Path, day: date) -> pd.DataFrame:
    season = _season_from_date(day)
    path = (
        data_root
        / "gold"
        / "minutes_for_rates"
        / f"season={season}"
        / f"game_date={day.isoformat()}"
        / "minutes_for_rates.parquet"
    )
    if not path.exists():
        return pd.DataFrame()
    minutes = pd.read_parquet(path)
    minutes["game_date"] = pd.to_datetime(minutes["game_date"]).dt.normalize()
    return minutes


def _ensure_minutes(df: pd.DataFrame, minutes_preds: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "minutes_pred_p50" in working.columns:
        return working

    if not minutes_preds.empty:
        key_cols = ["season", "game_date", "team_id", "player_id"]
        join_cols = [
            c
            for c in ("minutes_pred_p50", "minutes_pred_p10", "minutes_pred_p90", "minutes_pred_play_prob")
            if c in minutes_preds.columns
        ]
        minutes_slice = minutes_preds[key_cols + join_cols].drop_duplicates(subset=key_cols)
        merged = working.merge(minutes_slice, on=key_cols, how="left", suffixes=("", "_minutes"))
        if "minutes_pred_p50" in merged.columns and merged["minutes_pred_p50"].notna().any():
            return merged

    if "minutes_p50" in working.columns:
        typer.echo("[sim_worlds] minutes_pred_p50 missing; falling back to minutes_p50")
        working["minutes_pred_p50"] = working["minutes_p50"]
        return working

    if "minutes_actual" in working.columns:
        typer.echo("[sim_worlds] minutes_pred_p50 missing; falling back to minutes_actual")
        working["minutes_pred_p50"] = working["minutes_actual"]
    else:
        typer.echo("[sim_worlds] missing minutes_pred_p50 and minutes_actual; minutes left empty")
        working["minutes_pred_p50"] = np.nan
    return working


@app.command()
def main(
    data_root: Path = typer.Option(..., "--data-root"),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    n_worlds: int = typer.Option(1000, "--n-worlds"),
    residual_model_path: Optional[Path] = typer.Option(
        None,
        "--residual-model-path",
        help="Defaults to PROJECTIONS_DATA_ROOT/artifacts/sim_v1/fpts_residual_model.json",
    ),
    game_factor_std: float = typer.Option(0.0, "--game-factor-std"),
    output_root: Optional[Path] = typer.Option(
        None,
        "--output-root",
        help="Defaults to PROJECTIONS_DATA_ROOT/artifacts/sim_v1/worlds",
    ),
) -> None:
    """
    Sample simple worlds for historical slates using the residual model.
    """

    root = data_root or _get_data_root()
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    residual_path = residual_model_path or (root / "artifacts" / "sim_v1" / "fpts_residual_model.json")
    sampler = FptsResidualSampler.from_json_file(residual_path)

    bundle = load_current_fpts_bundle(data_root=root)
    run_id = _load_current_run_id()
    typer.echo(f"[sim_worlds] using fpts_v2 run_id={run_id} feature_cols={len(bundle.feature_cols)}")

    out_root = output_root or (root / "artifacts" / "sim_v1" / "worlds")
    out_root.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_worlds = 0

    for day in _iter_days(start_dt, end_dt):
        season = _season_from_date(day)
        day_token = day.isoformat()
        base_path = (
            root / "gold" / "fpts_training_base" / f"season={season}" / f"game_date={day_token}" / "fpts_training_base.parquet"
        )
        if not base_path.exists():
            typer.echo(f"[sim_worlds] {day_token}: missing {base_path}, skipping.")
            continue

        slate = pd.read_parquet(base_path)
        slate["game_date"] = pd.to_datetime(slate["game_date"]).dt.normalize()

        minutes_preds = _load_minutes_for_day(root, day)
        slate = _ensure_minutes(slate, minutes_preds)
        if "is_starter" not in slate.columns:
            typer.echo(f"[sim_worlds] {day_token}: is_starter missing; defaulting to bench=0")
            slate["is_starter"] = 0

        if "dk_fpts_pred" not in slate.columns or slate["dk_fpts_pred"].isna().all():
            preds = _score_predictions(slate, bundle.feature_cols, bundle.model)
            slate["dk_fpts_pred"] = preds

        required_cols = ["season", "game_date", "game_id", "team_id", "player_id", "dk_fpts_pred", "minutes_pred_p50", "is_starter"]
        missing = [c for c in required_cols if c not in slate.columns]
        if missing:
            raise KeyError(f"[sim_worlds] missing required columns for sampling: {missing}")

        worlds = sampler.sample_worlds(
            slate[required_cols],
            fpts_mean_col="dk_fpts_pred",
            minutes_col="minutes_pred_p50",
            is_starter_col="is_starter",
            n_worlds=n_worlds,
            game_factor_std=game_factor_std,
            game_id_col="game_id",
        )

        out_dir = out_root / f"season={season}" / f"game_date={day_token}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "worlds.parquet"
        worlds.to_parquet(out_path, index=False)

        total_rows += len(slate)
        total_worlds += len(worlds)

        summary = (
            worlds.groupby(["game_id"])["dk_fpts_world"]
            .agg(["mean", "std"])
            .reset_index()
        )
        typer.echo(
            f"[sim_worlds] {day_token}: rows={len(slate)} worlds={len(worlds)} "
            f"game_factor_std={game_factor_std:.3f} written={out_path}"
        )
        typer.echo(summary.to_string(index=False))

    typer.echo(f"[sim_worlds] total_rows={total_rows} total_worlds={total_worlds}")


if __name__ == "__main__":
    app()
