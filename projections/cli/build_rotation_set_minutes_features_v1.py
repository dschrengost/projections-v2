"""Build live `rotation_set_minutes_v1` feature slices for inference.

Output:
  live/features_rotation_set_minutes_v1/{date}/run={id}/features.parquet
"""

from __future__ import annotations

import json
import os
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from projections import paths
from projections.rotation.live_features_v1 import (
    KEY_COLS,
    RotationLiveFeaturesError,
    RotationSetMinutesV1FeatureSpec,
    build_rotation_set_minutes_v1_features,
    load_rotation_priors_for_live_inference,
)

UTC = timezone.utc

app = typer.Typer(help=__doc__)

DEFAULT_DATA_ROOT = paths.get_data_root()
DEFAULT_MINUTES_FEATURES_ROOT = paths.data_path("live", "features_minutes_v1")
DEFAULT_OUTPUT_ROOT = paths.data_path("live", "features_rotation_set_minutes_v1")
DEFAULT_ROTATION_CONFIG = Path("config/rotation_set_minutes_live.json")

FEATURE_FILENAME = "features.parquet"
SUMMARY_FILENAME = "summary.json"
LATEST_POINTER = "latest_run.json"


def _normalize_day(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _season_for_day(day: pd.Timestamp) -> int:
    return int(day.year) if int(day.month) >= 8 else int(day.year) - 1


def _read_latest_run_id(day_dir: Path) -> str | None:
    pointer = day_dir / LATEST_POINTER
    if not pointer.exists():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    run_id = payload.get("run_id")
    return str(run_id) if run_id else None


def _resolve_run_id(day_dir: Path, run_id: str | None) -> str:
    if run_id:
        return str(run_id)
    latest = _read_latest_run_id(day_dir)
    if latest:
        return latest
    raise typer.BadParameter(f"--run-id is required (no {LATEST_POINTER} under {day_dir})", param_hint="run_id")


def _write_latest_pointer(day_dir: Path, *, run_id: str, run_as_of_ts: pd.Timestamp) -> None:
    if os.environ.get("PROJECTIONS_SKIP_POINTER_WRITES", "").strip().lower() in {"1", "true", "yes"}:
        return
    from projections.pipeline import writer_guard

    writer_guard.assert_can_write_pointers(purpose=f"build_rotation_set_minutes_features_v1 promote {day_dir}")
    pointer = day_dir / LATEST_POINTER
    payload = {"run_id": run_id, "run_as_of_ts": run_as_of_ts.isoformat()}
    pointer.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).mean())


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise RotationLiveFeaturesError(f"Invalid JSON at {path}: {exc}") from exc


@app.command()
def main(
    *,
    date: str = typer.Option(..., "--date", help="Slate date (YYYY-MM-DD)."),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id (defaults to latest minutes features run)."),
    model_dir: Path | None = typer.Option(
        None,
        "--model-dir",
        exists=False,
        file_okay=False,
        dir_okay=True,
        help="Model directory containing feature_columns.json.",
    ),
    rotation_config: Path = typer.Option(
        DEFAULT_ROTATION_CONFIG,
        "--rotation-config",
        help="Rotation overlay config (default model_dir + priors fallback toggle).",
    ),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, "--data-root", help="PROJECTIONS_DATA_ROOT override."),
    minutes_features_root: Path = typer.Option(
        DEFAULT_MINUTES_FEATURES_ROOT, "--minutes-features-root", help="Root for live minutes features."
    ),
    out_root: Path = typer.Option(DEFAULT_OUTPUT_ROOT, "--out-root", help="Output root for rotation features."),
) -> None:
    day = _normalize_day(date)
    season = _season_for_day(day)

    minutes_day_dir = Path(minutes_features_root) / day.strftime("%Y-%m-%d")
    resolved_run_id = _resolve_run_id(minutes_day_dir, run_id)
    minutes_path = minutes_day_dir / f"run={resolved_run_id}" / FEATURE_FILENAME
    if not minutes_path.exists():
        raise typer.BadParameter(f"Minutes features not found at {minutes_path}", param_hint="minutes_features_root")

    cfg = _load_json(Path(rotation_config))
    allow_priors_fallback = bool(cfg.get("allow_priors_fallback", False))

    resolved_model_dir = model_dir
    if resolved_model_dir is None and os.environ.get("ROTATION_SET_MODEL_DIR"):
        resolved_model_dir = Path(os.environ.get("ROTATION_SET_MODEL_DIR", "")).expanduser()
    if resolved_model_dir is None and cfg.get("model_dir"):
        resolved_model_dir = Path(str(cfg["model_dir"])).expanduser()
    if not resolved_model_dir or str(resolved_model_dir).strip() in {"", "."}:
        raise typer.BadParameter(
            "--model-dir is required (or set ROTATION_SET_MODEL_DIR / config.model_dir)", param_hint="model_dir"
        )
    spec = RotationSetMinutesV1FeatureSpec.load(resolved_model_dir)

    typer.echo(
        f"[rotation_set_minutes_v1] build features date={day.date()} run_id={resolved_run_id} "
        f"season={season} model_dir={spec.model_dir}",
        err=True,
    )

    minutes_df = pd.read_parquet(minutes_path)
    if minutes_df.empty:
        raise RotationLiveFeaturesError(f"Minutes features slice is empty at {minutes_path}")

    game_ids = (
        pd.to_numeric(minutes_df["game_id"], errors="coerce")
        .dropna()
        .astype(int)
        .astype(str)
        .str.zfill(10)
        .unique()
        .tolist()
    )
    team_ids = (
        pd.to_numeric(minutes_df["team_id"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    player_ids = (
        pd.to_numeric(minutes_df["player_id"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    priors_result = load_rotation_priors_for_live_inference(
        Path(data_root),
        season=season,
        game_date=day.strftime("%Y-%m-%d"),
        game_ids=game_ids,
        team_ids=team_ids,
        player_ids=player_ids,
        allow_priors_fallback=allow_priors_fallback,
    )
    team_priors = priors_result.team_priors
    player_priors = priors_result.player_priors

    if priors_result.used_latest_fallback:
        typer.echo(
            f"[rotation_set_minutes_v1] WARNING: {priors_result.warning_message}",
            err=True,
        )
        typer.echo(
            f"[rotation_set_minutes_v1] Priors coverage: teams={priors_result.teams_found}/{priors_result.teams_found + priors_result.teams_missing} "
            f"players={priors_result.players_found}/{priors_result.players_found + priors_result.players_missing}",
            err=True,
        )

    if team_priors.empty or player_priors.empty:
        raise RotationLiveFeaturesError(
            "rotation_priors_v1 missing for slate; build priors first via "
            "`uv run python scripts/rotation/build_rotation_priors_v1.py` "
            f"(season={season}, game_ids={len(game_ids)})"
        )

    try:
        result = build_rotation_set_minutes_v1_features(
            minutes_df,
            team_priors=team_priors,
            player_priors=player_priors,
            feature_columns=spec.feature_columns,
        )
    except RotationLiveFeaturesError:
        raise
    except Exception as exc:
        raise RotationLiveFeaturesError(f"Failed to build rotation features: {exc}") from exc

    out_df = result.features

    dup_mask = out_df.duplicated(subset=list(KEY_COLS), keep=False)
    if dup_mask.any():
        sample = out_df.loc[dup_mask, list(KEY_COLS)].head(20).to_dict(orient="records")
        raise RotationLiveFeaturesError(f"Duplicate keys found in output: sample={sample}")

    if result.dropped_extra_columns:
        preview = result.dropped_extra_columns[:25]
        suffix = "..." if len(result.dropped_extra_columns) > len(preview) else ""
        typer.echo(
            f"[rotation_set_minutes_v1] WARNING: dropped {len(result.dropped_extra_columns)} extra columns: "
            f"{preview}{suffix}",
            err=True,
        )

    out_day_dir = Path(out_root) / day.strftime("%Y-%m-%d")
    out_run_dir = out_day_dir / f"run={resolved_run_id}"
    out_run_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_run_dir / FEATURE_FILENAME
    out_df.to_parquet(out_path, index=False)

    summary: dict[str, Any] = {
        "date": day.strftime("%Y-%m-%d"),
        "generated_at": pd.Timestamp.now(tz=UTC).isoformat(),
        "run_id": resolved_run_id,
        "model_dir": str(spec.model_dir),
        "counts": {
            "rows": int(len(out_df)),
            "players": int(out_df["player_id"].nunique()),
            "teams": int(out_df["team_id"].nunique()),
            "games": int(out_df["game_id"].nunique()),
        },
        "rates": {
            "minutes_features_row_missing": _rate(out_df.get("minutes_features_row_missing", pd.Series(dtype=float))),
            "injury_snapshot_missing": _rate(out_df.get("injury_snapshot_missing", pd.Series(dtype=float))),
        },
        "priors": {
            "used_latest_fallback": priors_result.used_latest_fallback,
            "teams_found": priors_result.teams_found,
            "teams_missing": priors_result.teams_missing,
            "players_found": priors_result.players_found,
            "players_missing": priors_result.players_missing,
        },
        "dropped_extra_columns": result.dropped_extra_columns,
        "feature_columns": {"n": int(len(spec.feature_columns))},
    }
    (out_run_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_latest_pointer(out_day_dir, run_id=resolved_run_id, run_as_of_ts=pd.Timestamp.now(tz=UTC))

    typer.echo(f"[rotation_set_minutes_v1] wrote {len(out_df)} rows -> {out_path}", err=True)


if __name__ == "__main__":
    app()
