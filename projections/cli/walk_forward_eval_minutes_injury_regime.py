"""Walk-forward evaluation of minutes models on injury-regime slices.

This runner:
1) Trains a rotshare bundle on all data up to a month start
2) Scores that month into a candidate predictions root
3) Runs `eval_minutes_injury_regime` for that month and writes a report JSON

Goal: provide a time-forward, production-like comparison of rotshare vs the
current production minutes pipeline across multiple months.
"""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import typer

from projections import paths
from projections.cli import eval_minutes_injury_regime
from projections.cli.score_minutes_v1 import score_minutes_range_to_parquet
from projections.minutes_v1.artifacts import write_json
from projections.minutes_v1.rotation_share import TEAM_TOTAL_MINUTES, train_rotation_share_model
from projections.models.feature_contract import (
    assert_no_leakage,
    filter_to_contract_features,
    save_feature_contract,
)


app = typer.Typer(help=__doc__)

UTC = timezone.utc


def extract_injury_regime_report_metrics(path: Path) -> dict[str, object] | None:
    """Extract a compact, month-level metric bundle from an eval_minutes_injury_regime report."""

    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models", {})
    if "current" not in models:
        return None

    def _f(obj: dict[str, object], key: str) -> float:
        value = obj.get(key, float("nan"))
        try:
            return float(value)  # type: ignore[arg-type]
        except Exception:
            return float("nan")

    out: dict[str, object] = {"path": str(path)}
    for slice_name in ("all_games", "injury_regime", "non_injury"):
        if slice_name not in models["current"]:
            continue
        cur = models["current"][slice_name]
        cand = models.get("candidate", {}).get(slice_name, {})
        out[slice_name] = {
            "n_player_rows": int(_f(cur, "n_player_rows") or 0),
            "n_team_games": int(_f(cur, "n_team_games") or 0),
            "player_mae_current": _f(cur, "player_mae"),
            "player_mae_candidate": _f(cand, "player_mae"),
            "bench_core_mae_current": _f(cur, "bench_core_mae"),
            "bench_core_mae_candidate": _f(cand, "bench_core_mae"),
            "top7_sum_mae_current": _f(cur, "top7_sum_mae"),
            "top7_sum_mae_candidate": _f(cand, "top7_sum_mae"),
            "top9_player_mae_current": _f(cur, "top9_player_mae"),
            "top9_player_mae_candidate": _f(cand, "top9_player_mae"),
            "top9_sum_mae_current": _f(cur, "top9_sum_mae"),
            "top9_sum_mae_candidate": _f(cand, "top9_sum_mae"),
            "top9_sum_mae_team240_current": _f(cur, "top9_sum_mae_team240"),
            "top9_sum_mae_team240_candidate": _f(cand, "top9_sum_mae_team240"),
            "top9_sum_bias_team240_current": _f(cur, "top9_sum_bias_team240"),
            "top9_sum_bias_team240_candidate": _f(cand, "top9_sum_bias_team240"),
            "tail_minutes_mae_team240_current": _f(cur, "tail_minutes_mae_team240"),
            "tail_minutes_mae_team240_candidate": _f(cand, "tail_minutes_mae_team240"),
            "tail_minutes_bias_team240_current": _f(cur, "tail_minutes_bias_team240"),
            "tail_minutes_bias_team240_candidate": _f(cand, "tail_minutes_bias_team240"),
        }
    return out


def aggregate_walk_forward_metrics(extracted_months: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    def _weighted_mean(rows: list[dict[str, object]], key: str, weight_key: str) -> float:
        numer = 0.0
        denom = 0.0
        for row in rows:
            w = float(row.get(weight_key, 0.0) or 0.0)
            v = float(row.get(key, float("nan")))
            if not np.isfinite(v) or w <= 0.0:
                continue
            numer += v * w
            denom += w
        return float(numer / denom) if denom > 0.0 else float("nan")

    aggregates: dict[str, dict[str, float]] = {}
    for slice_name in ("all_games", "injury_regime", "non_injury"):
        slice_rows = [m[slice_name] for m in extracted_months if isinstance(m.get(slice_name), dict)]
        if not slice_rows:
            continue

        def _pair(metric: str, weight: str) -> dict[str, float]:
            cur = _weighted_mean(slice_rows, f"{metric}_current", weight)
            cand = _weighted_mean(slice_rows, f"{metric}_candidate", weight)
            delta = float(cand - cur) if np.isfinite(cur) and np.isfinite(cand) else float("nan")
            return {f"{metric}_current": cur, f"{metric}_candidate": cand, f"{metric}_delta": delta}

        agg: dict[str, float] = {}
        agg.update(_pair("player_mae", "n_player_rows"))
        agg.update(_pair("bench_core_mae", "n_team_games"))
        agg.update(_pair("top7_sum_mae", "n_team_games"))
        agg.update(_pair("top9_player_mae", "n_player_rows"))
        agg.update(_pair("top9_sum_mae", "n_team_games"))
        agg.update(_pair("top9_sum_mae_team240", "n_team_games"))
        agg.update(_pair("tail_minutes_mae_team240", "n_team_games"))
        aggregates[slice_name] = agg

    return aggregates


def _month_end(day: date) -> date:
    anchor = date(day.year, day.month, 28) + timedelta(days=4)
    return anchor.replace(day=1) - timedelta(days=1)


def _iter_month_windows(start: date, end: date) -> Iterable[tuple[date, date]]:
    if end < start:
        return
    cur = date(start.year, start.month, 1)
    while cur <= end:
        win_start = max(cur, start)
        win_end = min(_month_end(cur), end)
        yield win_start, win_end
        nxt = cur.replace(day=28) + timedelta(days=4)
        cur = date(nxt.year, nxt.month, 1)


def _iter_feature_month_windows(features: pd.DataFrame, *, start: date, end: date) -> list[tuple[date, date]]:
    if features.empty:
        return []
    dates = pd.to_datetime(features["game_date"], errors="coerce").dropna()
    if dates.empty:
        return []
    dates = dates[(dates.dt.date >= start) & (dates.dt.date <= end)]
    if dates.empty:
        return []
    periods = pd.PeriodIndex(dates.dt.to_period("M")).unique().sort_values()
    windows: list[tuple[date, date]] = []
    for p in periods.tolist():
        month_start = max(start, p.start_time.date())
        month_end = min(end, p.end_time.date())
        if month_end >= month_start:
            windows.append((month_start, month_end))
    return windows


def _load_features_root(features_root: Path) -> pd.DataFrame:
    files = sorted(features_root.rglob("features.parquet"))
    if not files:
        raise FileNotFoundError(f"No features.parquet files found under {features_root}")
    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def _time_slice(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    start_day = pd.Timestamp(start).tz_convert("UTC").tz_localize(None).normalize()
    end_day = pd.Timestamp(end).tz_convert("UTC").tz_localize(None).normalize()
    mask = (df["game_date"] >= start_day) & (df["game_date"] <= end_day)
    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError(f"Empty slice for {start_day.date()} -> {end_day.date()}")
    return out


@dataclass(frozen=True)
class MonthRun:
    month_start: date
    month_end: date
    train_end: date
    run_id: str
    bundle_dir: Path
    report_path: Path


def _build_month_run(
    *,
    month_start: date,
    month_end: date,
    run_prefix: str,
    bundle_root: Path,
    report_root: Path,
) -> MonthRun:
    train_end = month_start - timedelta(days=1)
    stamp = f"{month_start:%Y-%m}"
    run_id = f"{run_prefix}_{stamp}"
    bundle_dir = bundle_root / run_prefix / stamp
    report_path = report_root / run_prefix / f"{stamp}.json"
    return MonthRun(
        month_start=month_start,
        month_end=month_end,
        train_end=train_end,
        run_id=run_id,
        bundle_dir=bundle_dir,
        report_path=report_path,
    )


def _train_rotshare_bundle(
    *,
    features: pd.DataFrame,
    run: MonthRun,
    train_start: datetime,
    train_end: datetime,
    val_start: datetime,
    val_end: datetime,
    random_state: int,
    in_rotation_minutes_threshold: float,
) -> None:
    train_df = _time_slice(features, train_start, train_end)
    feature_columns = filter_to_contract_features(train_df, target_col="minutes")
    assert_no_leakage(feature_columns)
    X_train = train_df[feature_columns].select_dtypes(include=["number", "bool", "boolean"])
    y_train = pd.to_numeric(train_df["minutes"], errors="coerce").fillna(0.0)

    artifacts = train_rotation_share_model(
        X_train,
        y_train,
        random_state=random_state,
        in_rotation_minutes_threshold=float(in_rotation_minutes_threshold),
    )

    run.bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, run.bundle_dir / "rotation_share_model.joblib")

    meta = {
        "model": "rotation_share_v0",
        "run_id": run.run_id,
        "windows": {
            "train": {"start": train_start.isoformat(), "end": train_end.isoformat()},
            "val": {"start": val_start.isoformat(), "end": val_end.isoformat()},
        },
        "random_state": int(random_state),
        "in_rotation_minutes_threshold": float(in_rotation_minutes_threshold),
        "feature_columns": feature_columns,
        "team_total_minutes": TEAM_TOTAL_MINUTES,
    }
    write_json(run.bundle_dir / "meta.json", meta)
    write_json(run.bundle_dir / "feature_columns.json", {"columns": feature_columns})
    save_feature_contract(feature_columns, run.bundle_dir / "feature_contract.json", metadata={"run_id": run.run_id})
    (run.bundle_dir / "TRAINING_DONE").write_text("", encoding="utf-8")


def _bundle_is_trained(bundle_dir: Path) -> bool:
    return (bundle_dir / "TRAINING_DONE").exists() and (bundle_dir / "rotation_share_model.joblib").exists()


@app.command()
def main(
    start_date: str = typer.Option("2023-10-01", help="Start game_date (YYYY-MM-DD, inclusive)."),
    end_date: str | None = typer.Option(None, help="Optional end game_date (YYYY-MM-DD, inclusive). Defaults to max in features."),
    train_start_date: str | None = typer.Option(
        None,
        help="Optional training start date (YYYY-MM-DD, inclusive). Defaults to start_date.",
    ),
    data_root: Path = typer.Option(paths.get_data_root(), help="Data root (defaults to PROJECTIONS_DATA_ROOT)."),
    features_root: Path | None = typer.Option(None, help="Optional features root (defaults to <data_root>/gold/features_minutes_v1)."),
    preds_root: Path = typer.Option(
        paths.data_path("gold", "projections_minutes_v1"),
        help="Root containing per-day minutes.parquet predictions for the current pipeline.",
    ),
    candidate_root: Path = typer.Option(
        Path("artifacts/rotshare_walk_forward_preds"),
        help="Root to write candidate per-day minutes.parquet predictions.",
    ),
    bundle_root: Path = typer.Option(
        Path("artifacts/minutes_rotation_share/walk_forward"),
        help="Root to write per-month rotshare bundles.",
    ),
    report_root: Path = typer.Option(
        Path("reports/minutes_injury_regime/walk_forward"),
        help="Root to write per-month eval JSON reports.",
    ),
    run_prefix: str = typer.Option("rotshare_wf_k10_e1_seed42", help="Prefix for bundle/report grouping."),
    min_starters_out: int = typer.Option(1, help="Injury regime if >= this many previous-game starters are OUT."),
    min_team_out: int = typer.Option(2, help="Injury regime if >= this many total OUT players on team."),
    baseline_top_k: int = typer.Option(8, help="Baseline compress-to-top-K heuristic."),
    lookback_days: int = typer.Option(30, help="Lookback window for previous-game starters (days)."),
    rotshare_min_players: int = typer.Option(10, help="rotshare min_players used for scoring."),
    rotshare_play_prob_exponent: float = typer.Option(1.0, help="rotshare play_prob exponent used for scoring."),
    in_rotation_minutes_threshold: float = typer.Option(
        10.0,
        help="Rotation inclusion label threshold (minutes >= threshold). Set 0 to use minutes > 0 semantics.",
    ),
    random_state: int = typer.Option(42, help="Deterministic seed for rotshare training."),
    reuse_bundles: bool = typer.Option(True, help="Reuse existing trained bundles if present."),
    reuse_scores: bool = typer.Option(True, help="Reuse existing scored month reports if present."),
) -> None:
    data_root = data_root.expanduser().resolve()
    preds_root = preds_root.expanduser().resolve()
    candidate_root = candidate_root.expanduser().resolve()
    bundle_root = bundle_root.expanduser().resolve()
    report_root = report_root.expanduser().resolve()

    if features_root is None:
        features_root = data_root / "gold" / "features_minutes_v1"
    features_root = features_root.expanduser().resolve()

    start = pd.Timestamp(start_date).date()
    train_start = pd.Timestamp(train_start_date).date() if train_start_date is not None else start
    if start < train_start:
        raise typer.BadParameter("train_start_date must be on or before start_date", param_name="train_start_date")
    typer.echo(f"[walk] loading features from {features_root}")
    features = _load_features_root(features_root)
    if end_date is None:
        max_day = pd.to_datetime(features["game_date"]).max()
        if pd.isna(max_day):
            raise ValueError("Could not determine end_date from features (empty game_date).")
        end = pd.Timestamp(max_day).date()
    else:
        end = pd.Timestamp(end_date).date()

    if end < start:
        raise typer.BadParameter("end_date must be on or after start_date", param_name="end_date")

    month_runs: list[MonthRun] = []
    for win_start, win_end in _iter_feature_month_windows(features, start=start, end=end):
        month_runs.append(
            _build_month_run(
                month_start=win_start,
                month_end=win_end,
                run_prefix=run_prefix,
                bundle_root=bundle_root,
                report_root=report_root,
            )
        )

    summary_rows: list[dict[str, object]] = []
    for run in month_runs:
        stamp = f"{run.month_start:%Y-%m}"
        if reuse_scores and run.report_path.exists():
            typer.echo(f"[walk] {stamp}: report exists, skipping")
            summary_rows.append({"month": stamp, "report": str(run.report_path), "skipped": True})
            continue

        train_start_dt = datetime(start.year, start.month, start.day, tzinfo=UTC)
        if train_start != start:
            train_start_dt = datetime(train_start.year, train_start.month, train_start.day, tzinfo=UTC)
        train_end_dt = datetime(run.train_end.year, run.train_end.month, run.train_end.day, tzinfo=UTC)
        val_start_dt = datetime(run.month_start.year, run.month_start.month, run.month_start.day, tzinfo=UTC)
        val_end_dt = datetime(run.month_end.year, run.month_end.month, run.month_end.day, tzinfo=UTC)

        if run.train_end < train_start:
            typer.echo(f"[walk] {stamp}: no prior training window available (train_end={run.train_end}), skipping")
            summary_rows.append({"month": stamp, "report": None, "skipped": True})
            continue

        # Train monthly bundle.
        if reuse_bundles and _bundle_is_trained(run.bundle_dir):
            typer.echo(f"[walk] {stamp}: reusing bundle {run.bundle_dir}")
        else:
            typer.echo(f"[walk] {stamp}: training bundle (train_end={run.train_end})")
            _train_rotshare_bundle(
                features=features,
                run=run,
                train_start=train_start_dt,
                train_end=train_end_dt,
                val_start=val_start_dt,
                val_end=val_end_dt,
                random_state=random_state,
                in_rotation_minutes_threshold=float(in_rotation_minutes_threshold),
            )

        # Score the month with the trained bundle.
        month_candidate_root = candidate_root / run_prefix
        typer.echo(f"[walk] {stamp}: scoring candidate preds into {month_candidate_root}")
        score_minutes_range_to_parquet(
            run.month_start,
            run.month_end,
            features_root=features_root,
            bundle_dir=run.bundle_dir,
            artifact_root=month_candidate_root,
            mode="historical",
            reconcile_team_minutes="none",
            enable_upside_adjustment=False,
            rotshare_min_players=rotshare_min_players,
            rotshare_play_prob_exponent=rotshare_play_prob_exponent,
            rotshare_use_learned_tau=False,
        )

        # Run eval for this month. Suppress JSON echo; report is written to file.
        run.report_path.parent.mkdir(parents=True, exist_ok=True)
        typer.echo(f"[walk] {stamp}: evaluating -> {run.report_path}")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            eval_minutes_injury_regime.main(
                start_date=run.month_start.isoformat(),
                end_date=run.month_end.isoformat(),
                data_root=data_root,
                preds_root=preds_root,
                preds_run_id=None,
                candidate_root=month_candidate_root,
                candidate_run_id=None,
                min_starters_out=min_starters_out,
                min_team_out=min_team_out,
                baseline_top_k=baseline_top_k,
                out=run.report_path,
                lookback_days=lookback_days,
            )

        summary_rows.append(
            {
                "month": stamp,
                "report": str(run.report_path),
                "bundle_dir": str(run.bundle_dir),
                "candidate_root": str(month_candidate_root),
                "skipped": False,
            }
        )

    extracted_months: list[dict[str, object]] = []
    for row in summary_rows:
        report = row.get("report")
        if not report:
            continue
        extracted = extract_injury_regime_report_metrics(Path(str(report)))
        if extracted is None:
            continue
        extracted["month"] = row.get("month")
        extracted_months.append(extracted)

    aggregates = aggregate_walk_forward_metrics(extracted_months)

    summary = {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "train_window": {"start": train_start.isoformat(), "end": end.isoformat()},
        "run_prefix": run_prefix,
        "bundle_root": str(bundle_root),
        "candidate_root": str(candidate_root),
        "report_root": str(report_root),
        "params": {
            "min_starters_out": int(min_starters_out),
            "min_team_out": int(min_team_out),
            "baseline_top_k": int(baseline_top_k),
            "lookback_days": int(lookback_days),
            "rotshare_min_players": int(rotshare_min_players),
            "rotshare_play_prob_exponent": float(rotshare_play_prob_exponent),
            "in_rotation_minutes_threshold": float(in_rotation_minutes_threshold),
            "random_state": int(random_state),
        },
        "aggregates": aggregates,
        "month_metrics": extracted_months,
        "months": summary_rows,
    }
    out_path = report_root / run_prefix / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    typer.echo(f"[walk] wrote {out_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
