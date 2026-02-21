"""Build and train a recency-weighted Minutes V1 retrain run."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
import typer

from projections.minutes_v1.retrain_dataset import build_retrain_dataset

app = typer.Typer(help=__doc__)


def _default_run_id() -> str:
    return datetime.now(tz=UTC).strftime("minutes_v1_recency_h35_%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _attach_action_props_to_retrain_dataset(*, dataset_path: Path, data_root: Path) -> dict[str, Any]:
    """Augment a retrain dataset parquet with Action Network prop-market features (an_*)."""

    action_props_dir = data_root / "bronze" / "action_network" / "props"
    diagnostics: dict[str, Any] = {
        "enabled": bool(action_props_dir.exists()),
        "source_dir": str(action_props_dir),
        "snapshot_rows": 0,
        "matched_rows": 0,
        "coverage": 0.0,
    }
    if not action_props_dir.exists():
        return diagnostics

    from projections.features.action_props import (
        attach_action_props_features,
        load_action_props_feature_snapshots_for_date,
    )

    df = pd.read_parquet(dataset_path)
    if df.empty:
        return diagnostics

    game_dates = (
        pd.to_datetime(df.get("game_date"), errors="coerce")
        .dt.normalize()
        .dropna()
        .unique()
        .tolist()
    )
    unique_days = sorted(pd.Timestamp(day) for day in game_dates)
    days_to_load: set[pd.Timestamp] = set(unique_days)
    for day in unique_days:
        days_to_load.add(day + pd.Timedelta(days=1))

    frames: list[pd.DataFrame] = []
    for day in sorted(days_to_load):
        snap = load_action_props_feature_snapshots_for_date(props_dir=action_props_dir, game_date=day)
        if not snap.empty:
            frames.append(snap)
    snaps = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    diagnostics["snapshot_rows"] = int(len(snaps))

    out = attach_action_props_features(
        df,
        snaps,
        strict_asof=True,
        as_of_col="feature_as_of_ts",
        tip_col="tip_ts",
        game_date_offsets=(0, -1),
        clamp_late_asof_to_game_date=True,
    )
    if "an_has_any_props" in out.columns:
        matched = (
            pd.to_numeric(out["an_has_any_props"], errors="coerce")
            .fillna(0.0)
            .gt(0.0)
            .sum()
        )
        diagnostics["matched_rows"] = int(matched)
        diagnostics["coverage"] = float(matched) / float(len(out)) if len(out) else 0.0

    out.to_parquet(dataset_path, index=False)
    return diagnostics


def _attach_prop_implied_minutes_to_retrain_dataset(
    *,
    dataset_path: Path,
    data_root: Path,
    season: int,
) -> dict[str, Any]:
    """Augment a retrain dataset parquet with prop-implied minutes (an_implied_minutes)."""

    diagnostics: dict[str, Any] = {
        "enabled": True,
        "season": int(season),
        "history_start": None,
        "history_end": None,
        "lookback_days": 365,
        "history_rows": 0,
        "prior_rows": 0,
        "matched_rows": 0,
        "coverage": 0.0,
    }

    from projections.features.prop_implied_minutes import (
        attach_prop_implied_minutes,
        compute_player_game_pra_priors,
        load_fpts_training_base_history_multi_season,
    )

    df = pd.read_parquet(dataset_path)
    if df.empty:
        return diagnostics
    if "game_date" not in df.columns:
        return diagnostics

    game_dates = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    game_dates = game_dates.dropna()
    if game_dates.empty:
        return diagnostics

    lookback_days = 365
    history_end = pd.Timestamp(game_dates.max())
    history_start = (pd.Timestamp(game_dates.min()) - pd.Timedelta(days=lookback_days)).normalize()
    diagnostics["history_start"] = history_start.date().isoformat()
    diagnostics["history_end"] = history_end.date().isoformat()
    diagnostics["lookback_days"] = int(lookback_days)

    player_ids = None
    if "an_has_any_props" in df.columns and "player_id" in df.columns:
        has_props = pd.to_numeric(df["an_has_any_props"], errors="coerce").fillna(0.0).gt(0.0)
        pid_series = pd.to_numeric(df.loc[has_props, "player_id"], errors="coerce").dropna()
        if not pid_series.empty:
            player_ids = pid_series.astype(int).unique().tolist()

    history = load_fpts_training_base_history_multi_season(
        data_root=data_root,
        start=history_start,
        end=history_end,
        player_ids=player_ids,
    )
    diagnostics["history_rows"] = int(len(history))

    priors = compute_player_game_pra_priors(history)
    diagnostics["prior_rows"] = int(len(priors))

    out = attach_prop_implied_minutes(
        df,
        priors=priors,
        join_keys=("game_id", "player_id"),
    )
    if "an_implied_minutes" in out.columns:
        matched = (
            pd.to_numeric(out["an_implied_minutes"], errors="coerce")
            .fillna(0.0)
            .gt(0.0)
            .sum()
        )
        diagnostics["matched_rows"] = int(matched)
        diagnostics["coverage"] = float(matched) / float(len(out)) if len(out) else 0.0

    out.to_parquet(dataset_path, index=False)
    return diagnostics


@app.command("build-dataset")
def build_dataset(
    run_id: str = typer.Option(
        "",
        "--run-id",
        help="Dataset run id under $PROJECTIONS_DATA_ROOT/artifacts/minutes_retrain_runs/.",
    ),
    data_root: Path = typer.Option(
        Path("/home/daniel/projections-data"),
        "--data-root",
        help="Projections data root.",
    ),
    season: int = typer.Option(2025, "--season", help="Season partition in labels/features paths."),
    train_start_date: str = typer.Option("2025-02-01", "--train-start-date"),
    train_end_date: str = typer.Option("2026-01-31", "--train-end-date"),
    cal_start_date: str = typer.Option("2026-02-01", "--cal-start-date"),
    cal_end_date: str = typer.Option("2026-02-05", "--cal-end-date"),
    half_life_days: float = typer.Option(35.0, "--half-life-days"),
    attach_action_props: bool = typer.Option(
        False,
        "--attach-action-props",
        help="Attach Action Network prop-market features (an_*) to the dataset parquet.",
        is_flag=True,
    ),
    attach_prop_implied_minutes: bool = typer.Option(
        False,
        "--attach-prop-implied-minutes",
        help="Attach prop-derived implied minutes features (an_implied_minutes) to the dataset parquet.",
        is_flag=True,
    ),
) -> None:
    """Build the leakage-safe retrain parquet with recency weights."""

    effective_run_id = run_id.strip() or _default_run_id()
    result = build_retrain_dataset(
        data_root=data_root,
        run_id=effective_run_id,
        season=season,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        cal_start_date=cal_start_date,
        cal_end_date=cal_end_date,
        half_life_days=half_life_days,
    )
    effective_attach_action_props = bool(attach_action_props or attach_prop_implied_minutes)
    if effective_attach_action_props:
        diag = _attach_action_props_to_retrain_dataset(dataset_path=result.dataset_path, data_root=data_root)
        meta = _read_json(result.meta_path)
        meta["action_props"] = diag
        _write_json(result.meta_path, meta)
        typer.echo(
            "[retrain-dataset] "
            f"action_props snapshots={diag.get('snapshot_rows')} "
            f"matched_rows={diag.get('matched_rows')}/{meta.get('summary', {}).get('dataset_rows')} "
            f"coverage={float(diag.get('coverage') or 0.0):.1%}"
        )
    if attach_prop_implied_minutes:
        diag = _attach_prop_implied_minutes_to_retrain_dataset(
            dataset_path=result.dataset_path,
            data_root=data_root,
            season=season,
        )
        meta = _read_json(result.meta_path)
        meta["prop_implied_minutes"] = diag
        _write_json(result.meta_path, meta)
        typer.echo(
            "[retrain-dataset] "
            f"prop_implied_minutes matched_rows={diag.get('matched_rows')}/{meta.get('summary', {}).get('dataset_rows')} "
            f"coverage={float(diag.get('coverage') or 0.0):.1%}"
        )
    meta = _read_json(result.meta_path)
    summary = meta.get("summary", {})
    typer.echo(
        "[retrain-dataset] "
        f"run_id={result.run_id} rows={summary.get('dataset_rows')} "
        f"split_counts={summary.get('split_counts')} "
        f"path={result.dataset_path}"
    )
    typer.echo(f"[retrain-dataset] meta={result.meta_path}")


@app.command("run")
def run(
    run_id: str = typer.Option(
        "",
        "--run-id",
        help="Run id used for both dataset artifact and trained model bundle.",
    ),
    data_root: Path = typer.Option(
        Path("/home/daniel/projections-data"),
        "--data-root",
        help="Projections data root.",
    ),
    season: int = typer.Option(2025, "--season", help="Season partition in labels/features paths."),
    train_start_date: str = typer.Option("2025-02-01", "--train-start-date"),
    train_end_date: str = typer.Option("2026-01-31", "--train-end-date"),
    cal_start_date: str = typer.Option("2026-02-01", "--cal-start-date"),
    cal_end_date: str = typer.Option("2026-02-05", "--cal-end-date"),
    half_life_days: float = typer.Option(35.0, "--half-life-days"),
    train_random_state: int = typer.Option(42, "--train-random-state"),
    allow_guard_failure: bool = typer.Option(
        False,
        "--allow-guard-failure",
        help="Forwarded to minutes_lgbm to emit guardrail violations as warnings instead of aborting.",
        is_flag=True,
    ),
    attach_action_props: bool = typer.Option(
        False,
        "--attach-action-props",
        help="Attach Action Network prop-market features (an_*) to the retrain dataset before training.",
        is_flag=True,
    ),
    attach_prop_implied_minutes: bool = typer.Option(
        False,
        "--attach-prop-implied-minutes",
        help="Attach prop-derived implied minutes features (an_implied_minutes) to the retrain dataset before training.",
        is_flag=True,
    ),
) -> None:
    """Build dataset, train minutes_lgbm with recency weights, and print validation gates."""

    effective_run_id = run_id.strip() or _default_run_id()
    dataset = build_retrain_dataset(
        data_root=data_root,
        run_id=effective_run_id,
        season=season,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        cal_start_date=cal_start_date,
        cal_end_date=cal_end_date,
        half_life_days=half_life_days,
    )
    dataset_meta = _read_json(dataset.meta_path)
    action_props_diag = None
    effective_attach_action_props = bool(attach_action_props or attach_prop_implied_minutes)
    if effective_attach_action_props:
        action_props_diag = _attach_action_props_to_retrain_dataset(
            dataset_path=dataset.dataset_path,
            data_root=data_root,
        )
        dataset_meta["action_props"] = action_props_diag
        _write_json(dataset.meta_path, dataset_meta)
        typer.echo(
            "[retrain-dataset] "
            f"action_props snapshots={action_props_diag.get('snapshot_rows')} "
            f"matched_rows={action_props_diag.get('matched_rows')}/{dataset_meta.get('summary', {}).get('dataset_rows')} "
            f"coverage={float(action_props_diag.get('coverage') or 0.0):.1%}"
        )

    prop_implied_minutes_diag = None
    if attach_prop_implied_minutes:
        prop_implied_minutes_diag = _attach_prop_implied_minutes_to_retrain_dataset(
            dataset_path=dataset.dataset_path,
            data_root=data_root,
            season=season,
        )
        dataset_meta = _read_json(dataset.meta_path)
        dataset_meta["prop_implied_minutes"] = prop_implied_minutes_diag
        _write_json(dataset.meta_path, dataset_meta)
        typer.echo(
            "[retrain-dataset] "
            f"prop_implied_minutes matched_rows={prop_implied_minutes_diag.get('matched_rows')}/{dataset_meta.get('summary', {}).get('dataset_rows')} "
            f"coverage={float(prop_implied_minutes_diag.get('coverage') or 0.0):.1%}"
        )

    model_root = data_root / "artifacts" / "minutes_lgbm"
    model_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "projections.models.minutes_lgbm",
        "--features",
        str(dataset.dataset_path),
        "--run-id",
        effective_run_id,
        "--artifact-root",
        str(model_root),
        "--sample-weight-col",
        "weight_recency",
        "--target-col",
        "minutes",
        "--train-start",
        train_start_date,
        "--train-end",
        train_end_date,
        "--cal-start",
        cal_start_date,
        "--cal-end",
        cal_end_date,
        "--val-start",
        cal_start_date,
        "--val-end",
        cal_end_date,
        "--allow-val-cal-overlap",
        "--random-state",
        str(train_random_state),
    ]
    if allow_guard_failure:
        cmd.append("--allow-guard-failure")
    typer.echo("[retrain-train] running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    model_run_dir = model_root / effective_run_id
    metrics_path = model_run_dir / "metrics.json"
    model_meta_path = model_run_dir / "meta.json"
    if not metrics_path.exists() or not model_meta_path.exists():
        raise RuntimeError(f"Training completed but expected artifacts are missing under {model_run_dir}")

    metrics = _read_json(metrics_path)
    model_meta = _read_json(model_meta_path)
    model_meta["retrain_dataset"] = {
        "run_id": dataset.run_id,
        "dataset_path": str(dataset.dataset_path),
        "meta_path": str(dataset.meta_path),
    }
    if action_props_diag is not None:
        model_meta["retrain_dataset"]["action_props"] = action_props_diag
    if prop_implied_minutes_diag is not None:
        model_meta["retrain_dataset"]["prop_implied_minutes"] = prop_implied_minutes_diag
    model_meta["recency_decay"] = dataset_meta.get("recency_decay", {})
    model_meta["train_cal_windows"] = dataset_meta.get("effective_windows", {})
    _write_json(model_meta_path, model_meta)

    typer.echo(
        "[retrain-metrics] "
        f"false_active={metrics.get('val_false_active_rate_p_ge_0_5')} "
        f"false_inactive={metrics.get('val_false_inactive_rate_p_le_0_2')} "
        f"brier={metrics.get('val_play_prob_brier')} "
        f"mae_p50_cond={metrics.get('val_mae_p50_conditional')} "
        f"bench_smear={metrics.get('val_bench_smear_proxy_p50_gt_10_actual_lt_1')}"
    )
    typer.echo(f"[retrain-output] dataset={dataset.dataset_path}")
    typer.echo(f"[retrain-output] model_bundle={model_run_dir}")


if __name__ == "__main__":  # pragma: no cover
    app()
