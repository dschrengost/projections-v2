"""Day-by-day sequential backtest for Minutes V1 rolling calibration."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import typer

from projections import paths
from projections.minutes_v1 import calibration, modeling
from projections.minutes_v1.datasets import (
    KEY_COLUMNS,
    deduplicate_latest,
    default_features_path,
    ensure_columns,
    load_feature_frame,
)


# ---------------------------------------------------------------------------
# Helper utilities (formerly in score_minutes.py)
# ---------------------------------------------------------------------------
def _normalize_date(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _filter_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    working = df.copy()
    working["game_date"] = pd.to_datetime(working["game_date"]).dt.normalize()
    mask = (working["game_date"] >= start) & (working["game_date"] <= end)
    sliced = working.loc[mask].copy()
    if sliced.empty:
        raise ValueError(
            f"No feature rows found between {start.date()} and {end.date()} — verify feature inputs."
        )
    return sliced


def _resolve_feature_source(
    features_path: Path | None,
    *,
    data_root: Path,
    season: int | None,
    month: int | None,
) -> Path | None:
    if features_path is not None:
        return features_path
    if season is not None and month is not None:
        return default_features_path(data_root, season, month)
    return None


app = typer.Typer(help=__doc__)

ALPHA_TARGET = 0.10
P10_FLOOR = 0.06
P90_TARGET = 0.90
MONTH_TOLERANCE = 0.04


def _window_slice(
    feature_df: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    history_days: int,
) -> pd.DataFrame:
    """Return a dataframe spanning [start-history_days, end]."""

    history_start = start - pd.Timedelta(days=history_days)
    return _filter_window(feature_df, history_start, end)


def _coverage_share(group: pd.DataFrame, *, label_col: str, quantile_col: str) -> float:
    scoped = group.dropna(subset=[label_col, quantile_col])
    if scoped.empty:
        return float("nan")
    return float((scoped[label_col] <= scoped[quantile_col]).mean())


def _daily_coverage_table(
    df: pd.DataFrame,
    *,
    label_col: str,
    quantile_map: dict[str, str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for score_date, group in df.groupby("game_date"):
        scoped = group.dropna(subset=[label_col])
        if scoped.empty:
            continue
        record: dict[str, object] = {
            "game_date": score_date.strftime("%Y-%m-%d"),
            "rows": int(len(scoped)),
        }
        for column_name, quant_col in quantile_map.items():
            record[column_name] = _coverage_share(scoped, label_col=label_col, quantile_col=quant_col)
        records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame = frame.sort_values("game_date").reset_index(drop=True)
    return frame


def _summarize_daily(
    daily: pd.DataFrame,
    *,
    coverage_col: str,
    target: float,
    floor: float | None = None,
) -> dict[str, float | int]:
    if daily.empty:
        return {
            "min": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "days_below_floor": 0,
            "target": target,
        }
    stats = daily[coverage_col].agg(["min", "median", "max", "mean"])
    summary: dict[str, float | int] = {
        "min": float(stats["min"]),
        "median": float(stats["median"]),
        "max": float(stats["max"]),
        "mean": float(stats["mean"]),
        "target": target,
    }
    if floor is not None:
        summary["days_below_floor"] = int((daily[coverage_col] < floor).sum())
    return summary


def _overall_coverage(df: pd.DataFrame, *, label_col: str, quant_col: str) -> float:
    scoped = df.dropna(subset=[label_col, quant_col])
    if scoped.empty:
        return float("nan")
    return float((scoped[label_col] <= scoped[quant_col]).mean())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _ensure_prediction_inputs(df: pd.DataFrame, target_col: str) -> None:
    ensure_columns(
        df,
        [
            target_col,
            "game_date",
            "game_id",
            "player_id",
            "team_id",
            "starter_prev_game_asof",
            "ramp_flag",
            "tip_ts",
        ],
    )


def _history_partitions(season: int, month: int, history_months: int) -> list[tuple[int, int]]:
    base = pd.Period(year=season, month=month, freq="M")
    partitions: list[tuple[int, int]] = []
    for offset in range(history_months, -1, -1):
        period = base - offset
        partitions.append((period.year, period.month))
    return partitions


def _load_feature_history(
    *,
    features_path: Path | None,
    data_root: Path,
    season: int | None,
    month: int | None,
    history_months: int,
) -> pd.DataFrame:
    if features_path is not None:
        return load_feature_frame(
            features_path=features_path,
            data_root=data_root,
            season=season,
            month=month,
        )
    if season is None or month is None:
        raise ValueError("Either --features or both --season/--month must be provided.")

    partitions = _history_partitions(season, month, max(history_months, 0))
    frames: list[pd.DataFrame] = []
    for part_season, part_month in partitions:
        try:
            frame = load_feature_frame(
                features_path=None,
                data_root=data_root,
                season=part_season,
                month=part_month,
            )
        except FileNotFoundError as exc:
            if part_season == season and part_month == month:
                raise
            typer.echo(
                f"[sequential] Skipping missing history partition season={part_season}, month={part_month:02d}: {exc}",
                err=True,
            )
            continue
        frames.append(frame)
    if not frames:
        raise RuntimeError("Unable to load any feature partitions for sequential backtest.")
    return pd.concat(frames, ignore_index=True)


@app.command()
def main(
    start: datetime = typer.Option(..., help="Start date (inclusive) for the evaluation window."),
    end: datetime = typer.Option(..., help="End date (inclusive) for the evaluation window."),
    run_id: str = typer.Option(..., help="Trained LightGBM run_id to load."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        help="Root directory with data subfolders (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    season: int | None = typer.Option(None, help="Season partition for default feature path."),
    month: int | None = typer.Option(None, help="Month partition for default feature path."),
    features: Path | None = typer.Option(None, help="Explicit feature parquet path."),
    artifact_root: Path = typer.Option(Path("artifacts/minutes_v1"), help="Directory containing trained runs."),
    target_col: str = typer.Option("minutes", help="Name of the realized minutes column present in features."),
    window_days: int = typer.Option(21, "--window-days", help="Rolling calibration window size (days)."),
    min_n: int = typer.Option(400, "--min-n", help="Minimum rows required before trusting local offsets."),
    tau: float = typer.Option(150.0, "--tau", help="Shrinkage pseudo-count for rolling calibration."),
    tau_bucket: float | None = typer.Option(
        None,
        "--tau-bucket",
        help="Shrinkage pseudo-count for bucket-level rolling offsets (falls back to --tau when omitted).",
    ),
    use_buckets: bool = typer.Option(False, "--use-buckets/--no-use-buckets", help="Enable bucketed calibration."),
    min_n_bucket: int = typer.Option(250, "--min-n-bucket", help="Minimum per-bucket rows required for offsets."),
    use_spread_buckets: bool = typer.Option(
        False,
        "--use-spread-buckets/--no-use-spread-buckets",
        help="When bucketed calibration is enabled, also split by |spread| buckets.",
    ),
    cold_start_min_n: int = typer.Option(
        400,
        "--cold-start-min-n",
        help="If total calibration rows fall below this threshold, skip p10 delta shifts.",
    ),
    history_months: int = typer.Option(
        1,
        "--history-months",
        help="Number of prior months to include for calibration history (0 disables).",
    ),
    reports_root: Path = typer.Option(Path("reports/minutes_v1"), help="Directory for sequential backtest artifacts."),
    min_recent_days: int = typer.Option(
        0,
        "--min-recent-days",
        help="Require at least this many unique current-month days in a bucket window before using bucket-specific offsets.",
    ),
    p10_floor_guard: float = typer.Option(
        0.10,
        "--p10-floor-guard",
        help="Solve rolling p10 adjustments when simulated coverage would fall below this threshold (set to 0 to disable).",
    ),
    p90_floor_guard: float = typer.Option(
        0.90,
        "--p90-floor-guard",
        help="Solve rolling p90 adjustments when simulated coverage would fall below this threshold (set to 0 to disable).",
    ),
    bucket_delta_cap: float = typer.Option(
        0.30,
        "--bucket-delta-cap",
        help="Maximum allowed deviation (minutes) between bucket and overall deltas; set <=0 to disable.",
    ),
    max_abs_delta_p10: float | None = typer.Option(
        None,
        "--max-abs-delta-p10",
        help="Absolute cap (minutes) for p10 rolling deltas; leave unset to enable adaptive trust caps.",
    ),
    max_abs_delta_p90: float | None = typer.Option(
        None,
        "--max-abs-delta-p90",
        help="Absolute cap (minutes) for p90 rolling deltas; leave unset to enable adaptive trust caps.",
    ),
    recent_target_rows: int = typer.Option(
        300,
        "--recent-target-rows",
        help="Rows required before adaptive caps decay to their lower bound.",
    ),
    recency_window_days: int = typer.Option(
        14,
        "--recency-window-days",
        help="Window (days) for counting recent rows when sizing adaptive caps.",
    ),
    recency_half_life_days: float = typer.Option(
        12.0,
        "--recency-half-life-days",
        help="Half-life (days) for exponential recency weights in coverage solves.",
    ),
    bucket_floor_relief: float = typer.Option(
        0.06,
        "--bucket-floor-relief",
        help="If a bucket's simulated P10 coverage falls below this, bypass the deviation cap and solve directly.",
    ),
    warmup_days_p10: int = typer.Option(
        0,
        "--warmup-days-p10",
        help="Blend p10 rolling deltas toward raw for this many days at the month start.",
    ),
    warmup_days_p90: int = typer.Option(
        0,
        "--warmup-days-p90",
        help="Blend p90 rolling deltas toward raw for this many days at the month start.",
    ),
    use_global_p10_delta: bool = typer.Option(
        False,
        "--use-global-p10-delta/--no-use-global-p10-delta",
        help="Use the conformal global delta as the anchor for p10 adjustments.",
    ),
    use_global_p90_delta: bool = typer.Option(
        True,
        "--use-global-p90-delta/--no-use-global-p90-delta",
        help="Use the conformal global delta as the anchor for p90 adjustments.",
    ),
    global_baseline_months: int = typer.Option(
        2,
        "--global-baseline-months",
        help="Months of pre-period history for baseline global deltas.",
    ),
    global_recent_target_rows: int = typer.Option(
        300,
        "--global-recent-target-rows",
        help="Rows required in current-period data before trusting recent global deltas fully.",
    ),
    global_recent_target_rows_p10: int = typer.Option(
        180,
        "--global-recent-target-rows-p10",
        help="Override for P10 global crossfade target rows.",
    ),
    global_recent_target_rows_p90: int = typer.Option(
        300,
        "--global-recent-target-rows-p90",
        help="Override for P90 global crossfade target rows.",
    ),
    global_min_n: int = typer.Option(
        200,
        "--global-min-n",
        help="Minimum rows per stratum needed to fit a global delta.",
    ),
    global_delta_cap: float = typer.Option(
        0.30,
        "--global-delta-cap",
        help="Absolute cap for fitted global deltas.",
    ),
    bucket_min_effective_sample_size: float = typer.Option(
        360.0,
        "--bucket-min-ess",
        help="Minimum effective sample size before solving bucket adjustments.",
    ),
    bucket_inherit_if_low_n: bool = typer.Option(
        True,
        "--bucket-inherit/--no-bucket-inherit",
        help="Inherit window deltas when bucket coverage is thin.",
    ),
    cap_guardrail_rate: float | None = typer.Option(
        None,
        "--cap-guardrail-rate",
        help="Optional guardrail on the fraction of post-day-7 windows that bind the adaptive cap.",
    ),
    max_width_delta_pct: float | None = typer.Option(
        None,
        "--max-width-delta-pct",
        help="Optional guardrail on median quantile width delta (fraction).",
    ),
    monotonicity_repair_threshold: float | None = typer.Option(
        0.005,
        "--monotonicity-repair-threshold",
        help="Maximum allowed fraction of rows needing monotonicity repair.",
    ),
    apply_center_width_projection: bool = typer.Option(
        True,
        "--center-width/--no-center-width",
        help="Project p10/p90 around p50 to enforce monotone tails before repairs.",
    ),
    min_quantile_half_width: float = typer.Option(
        0.05,
        "--min-quantile-half-width",
        help="Minimum half-width (minutes) when projecting tails around p50.",
    ),
    one_sided_days: int = typer.Option(
        0,
        "--one-sided-days",
        help="For this many days from month start, force p10 deltas <=0 and p90 >=0 unless the miss exceeds the tolerance.",
    ),
    hysteresis_lower: float = typer.Option(
        0.09,
        "--p10-hysteresis-lower",
        help="Lower coverage threshold for one-sided p10 adjustments.",
    ),
    hysteresis_upper: float = typer.Option(
        0.11,
        "--p10-hysteresis-upper",
        help="Upper coverage threshold for one-sided p10 adjustments.",
    ),
    delta_smoothing_alpha_p10: float = typer.Option(
        0.0,
        "--delta-smoothing-alpha-p10",
        help="Optional EWA alpha (0-1) to smooth p10 deltas.",
    ),
    delta_smoothing_alpha_p90: float = typer.Option(
        0.3,
        "--delta-smoothing-alpha-p90",
        help="Optional EWA alpha (0-1) to smooth p90 deltas.",
    ),
) -> None:
    """Run a sequential backtest that simulates daily rolling calibration."""

    split_start = _normalize_date(start)
    split_end = _normalize_date(end)
    if split_start > split_end:
        raise typer.BadParameter("start date must be on/before end date")

    feature_source = _resolve_feature_source(features, data_root=data_root, season=season, month=month)

    feature_df = _load_feature_history(
        features_path=features,
        data_root=data_root,
        season=season,
        month=month,
        history_months=history_months,
    )
    _ensure_prediction_inputs(feature_df, target_col)

    sliced = _window_slice(
        feature_df,
        start=split_start,
        end=split_end,
        history_days=window_days,
    )
    sliced = deduplicate_latest(sliced, key_cols=KEY_COLUMNS, order_cols=["feature_as_of_ts"])
    sliced = sliced.copy()
    sliced["game_id"] = sliced["game_id"].astype(str)

    run_dir = artifact_root / run_id
    bundle_path = run_dir / "lgbm_quantiles.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Quantile artifact not found at {bundle_path}")
    bundle = joblib.load(bundle_path)
    meta_path = run_dir / "meta.json"
    meta: dict[str, object] = {}
    if meta_path.exists():
        with meta_path.open() as handle:
            meta = json.load(handle)

    feature_columns: list[str] = bundle["feature_columns"]
    missing = set(feature_columns) - set(sliced.columns)
    if missing:
        raise ValueError(f"Scoring dataset missing columns: {', '.join(sorted(missing))}")

    quantile_preds = modeling.predict_quantiles(bundle["quantiles"], sliced[feature_columns])
    scored_all = sliced.copy()
    scored_all["game_date"] = pd.to_datetime(scored_all["game_date"]).dt.normalize()
    scored_all["p10_raw"] = quantile_preds[0.1]
    scored_all["p50_raw"] = quantile_preds[0.5]
    scored_all["p90_raw"] = quantile_preds[0.9]

    eval_mask = (scored_all["game_date"] >= split_start) & (scored_all["game_date"] <= split_end)
    if not eval_mask.any():
        raise ValueError("Evaluation window produced zero rows — check date bounds or feature inputs.")

    conformal_offsets = bundle["calibrator"].export_offsets()
    global_delta_p10 = -conformal_offsets["low_adjustment"]
    global_delta_p90 = conformal_offsets["high_adjustment"]

    # Rolling calibration was previously implemented via compute_rolling_offsets/apply_rolling_offsets.
    # That logic has been removed; for sequential backtests we fall back to applying the global
    # conformal deltas on each evaluation day. This preserves the report contract (CSV + summary)
    # while avoiding stale/unsupported calibration code paths.
    typer.echo(
        "[sequential] warning: rolling calibration unavailable; using global conformal deltas for rolling outputs."
    )

    eval_dates = scored_all.loc[eval_mask, "game_date"].drop_duplicates().sort_values()
    rolling_offsets = pd.DataFrame(
        {
            "score_date": eval_dates,
            "window_start": eval_dates - pd.Timedelta(days=int(window_days)),
            "window_end": eval_dates,
            "delta_p10": float(global_delta_p10),
            "delta_p90": float(global_delta_p90),
            "shrinkage": 0.0,
            "bucket_name": pd.NA,
            "strategy": "global_fallback",
        }
    )

    eval_rows = scored_all.loc[eval_mask].copy()
    rolling_applied = eval_rows.copy()
    rolling_applied["p10_rolling"] = rolling_applied["p10_raw"] + global_delta_p10
    rolling_applied["p90_rolling"] = rolling_applied["p90_raw"] + global_delta_p90
    rolling_applied["p10_global"] = rolling_applied["p10_raw"] + global_delta_p10
    rolling_applied["p90_global"] = rolling_applied["p90_raw"] + global_delta_p90

    unique_eval = deduplicate_latest(rolling_applied, key_cols=KEY_COLUMNS, order_cols=["feature_as_of_ts"])
    unique_eval = unique_eval.dropna(subset=[target_col]).copy()
    unique_eval["game_date"] = pd.to_datetime(unique_eval["game_date"]).dt.normalize()

    if unique_eval.empty:
        raise ValueError("No labeled rows available after deduplication for coverage computation.")

    p10_daily = _daily_coverage_table(
        unique_eval,
        label_col=target_col,
        quantile_map={
            "coverage_rolling": "p10_rolling",
            "coverage_global": "p10_global",
            "coverage_raw": "p10_raw",
        },
    )
    if not p10_daily.empty:
        p10_daily["improvement_vs_global"] = p10_daily["coverage_rolling"] - p10_daily["coverage_global"]
        p10_daily["improvement_vs_raw"] = p10_daily["coverage_rolling"] - p10_daily["coverage_raw"]

    p90_daily = _daily_coverage_table(
        unique_eval,
        label_col=target_col,
        quantile_map={
            "coverage_rolling": "p90_rolling",
            "coverage_global": "p90_global",
            "coverage_raw": "p90_raw",
        },
    )
    if not p90_daily.empty:
        p90_daily["improvement_vs_global"] = p90_daily["coverage_rolling"] - p90_daily["coverage_global"]
        p90_daily["improvement_vs_raw"] = p90_daily["coverage_rolling"] - p90_daily["coverage_raw"]

    overall_p10 = _overall_coverage(unique_eval, label_col=target_col, quant_col="p10_rolling")
    overall_p90 = _overall_coverage(unique_eval, label_col=target_col, quant_col="p90_rolling")
    overall_p10_global = _overall_coverage(unique_eval, label_col=target_col, quant_col="p10_global")
    overall_p90_global = _overall_coverage(unique_eval, label_col=target_col, quant_col="p90_global")
    overall_p10_raw = _overall_coverage(unique_eval, label_col=target_col, quant_col="p10_raw")
    overall_p90_raw = _overall_coverage(unique_eval, label_col=target_col, quant_col="p90_raw")

    p10_summary = _summarize_daily(
        p10_daily,
        coverage_col="coverage_rolling",
        target=ALPHA_TARGET,
        floor=P10_FLOOR,
    )
    p90_summary = _summarize_daily(
        p90_daily,
        coverage_col="coverage_rolling",
        target=P90_TARGET,
        floor=None,
    )

    p10_guardrail = abs(overall_p10 - ALPHA_TARGET) <= MONTH_TOLERANCE
    p90_guardrail = abs(overall_p90 - P90_TARGET) <= MONTH_TOLERANCE

    summary_payload = {
        "run_id": run_id,
        "feature_source": str(feature_source) if feature_source else None,
        "date_range": {
            "start": split_start.strftime("%Y-%m-%d"),
            "end": split_end.strftime("%Y-%m-%d"),
        },
        "global_deltas": {
            "delta_p10": global_delta_p10,
            "delta_p90": global_delta_p90,
        },
        "rolling_params": {
            "window_days": window_days,
            "min_n": min_n,
            "tau": tau,
            "use_buckets": use_buckets,
            "min_n_bucket": min_n_bucket,
            "use_spread_buckets": use_spread_buckets,
        },
        "meta": meta,
        "p10": {
            "overall": overall_p10,
            "overall_global": overall_p10_global,
            "overall_raw": overall_p10_raw,
            "daily": p10_summary,
            "guardrail_pass": p10_guardrail,
        },
        "p90": {
            "overall": overall_p90,
            "overall_global": overall_p90_global,
            "overall_raw": overall_p90_raw,
            "daily": p90_summary,
            "guardrail_pass": p90_guardrail,
        },
    }

    if p10_daily.empty or p90_daily.empty:
        raise RuntimeError("Failed to compute daily coverage tables — verify evaluation dataset.")

    month_key = split_start.strftime("%Y-%m")
    if split_start.strftime("%Y-%m") != split_end.strftime("%Y-%m"):
        month_key = f"{split_start.strftime('%Y-%m')}_to_{split_end.strftime('%Y-%m')}"

    report_dir = reports_root / month_key
    report_dir.mkdir(parents=True, exist_ok=True)

    offsets_to_write = rolling_offsets.copy()
    for column in ("score_date", "window_start", "window_end"):
        if column in offsets_to_write:
            offsets_to_write[column] = pd.to_datetime(offsets_to_write[column]).dt.strftime("%Y-%m-%d")
    offsets_to_write = offsets_to_write.rename(columns={"shrinkage": "s"})
    offsets_path = report_dir / "rolling_offsets.csv"
    offsets_to_write.sort_values(["score_date", "bucket_name"], na_position="first").to_csv(offsets_path, index=False)

    p10_path = report_dir / "p10_coverage_daily.csv"
    p10_daily.to_csv(p10_path, index=False)
    p90_path = report_dir / "p90_coverage_daily.csv"
    p90_daily.to_csv(p90_path, index=False)

    summary_path = report_dir / "rolling_backtest_summary.json"
    _write_json(summary_path, summary_payload)

    typer.echo(
        f"Sequential backtest complete for {split_start.date()} → {split_end.date()}. "
        f"Artifacts: {offsets_path}, {p10_path}, {p90_path}, {summary_path}"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
