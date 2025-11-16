"""Score Minutes V1 features to produce calibrated predictions + reconciliation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import typer

from projections import paths
from projections.minutes_v1 import calibration, modeling
from projections.minutes_v1.artifacts import write_json
from projections.minutes_v1.datasets import (
    KEY_COLUMNS,
    deduplicate_latest,
    ensure_columns,
    left_anti_keys,
    load_feature_frame,
    write_ids_csv,
    default_features_path,
)
from projections.minutes_v1.reconciliation import reconcile_minutes


app = typer.Typer(help=__doc__)


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


def _default_output_path(data_root: Path, window_start: pd.Timestamp) -> Path:
    month_key = window_start.strftime("%Y-%m")
    dest_dir = data_root / "preds" / "minutes_v1" / month_key
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / "minutes_pred.parquet"


def _sidecar_path(base: Path, label: str) -> Path:
    if base.suffix:
        return base.with_name(f"{base.stem}.{label}.csv")
    return base / f"{label}.csv"


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


@app.command()
def main(
    start: datetime = typer.Option(..., help="Start date (inclusive) for scoring window."),
    end: datetime = typer.Option(..., help="End date (inclusive) for scoring window."),
    run_id: str = typer.Option(..., help="Trained LightGBM run_id to load."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        help="Root directory with data subfolders (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    season: int | None = typer.Option(None, help="Season partition for default feature path."),
    month: int | None = typer.Option(None, help="Month partition for default feature path."),
    features: Path | None = typer.Option(None, help="Explicit feature parquet path."),
    artifact_root: Path = typer.Option(Path("artifacts/minutes_v1"), help="Directory containing trained runs."),
    out: Path | None = typer.Option(None, help="Optional explicit output parquet path."),
    target_col: str = typer.Option("minutes", help="Name of the realized minutes column present in features."),
    window_days: int = typer.Option(21, "--window-days", help="Rolling calibration window size (days)."),
    min_n: int = typer.Option(400, "--min-n", help="Minimum rows required before trusting local offsets."),
    tau: float = typer.Option(150.0, "--tau", help="Shrinkage pseudo-count for rolling calibration."),
    tau_bucket: float | None = typer.Option(
        None,
        "--tau-bucket",
        help="Shrinkage pseudo-count for bucket-level offsets (falls back to --tau when omitted).",
    ),
    use_buckets: bool = typer.Option(
        False,
        "--use-buckets/--no-use-buckets",
        help="Enable bucketed rolling calibration (injury + optional spread buckets).",
    ),
    min_n_bucket: int = typer.Option(
        250,
        "--min-n-bucket",
        help="Minimum per-bucket rows required before using bucketed offsets.",
    ),
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
    min_recent_days: int = typer.Option(
        0,
        "--min-recent-days",
        help="Require at least this many unique current-month days in a bucket window before applying bucket-specific offsets.",
    ),
    p10_floor_guard: float = typer.Option(
        0.10,
        "--p10-floor-guard",
        help="Solve rolling p10 adjustments so simulated coverage meets this threshold (set to 0 to disable).",
    ),
    p90_floor_guard: float = typer.Option(
        0.90,
        "--p90-floor-guard",
        help="Solve rolling p90 adjustments so simulated coverage meets this threshold (set to 0 to disable).",
    ),
    bucket_delta_cap: float = typer.Option(
        0.30,
        "--bucket-delta-cap",
        help="Maximum allowed deviation (minutes) between bucket and overall deltas; set <=0 to disable.",
    ),
    bucket_floor_relief: float = typer.Option(
        0.06,
        "--bucket-floor-relief",
        help="If a bucket's simulated P10 coverage falls below this, bypass the deviation cap and solve directly.",
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
    warmup_days_p10: int = typer.Option(
        0,
        "--warmup-days-p10",
        help="Blend p10 rolling deltas toward raw values for this many days at the start of each month.",
    ),
    warmup_days_p90: int = typer.Option(
        0,
        "--warmup-days-p90",
        help="Blend p90 rolling deltas toward raw values for this many days at the start of each month.",
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
    """Score features, conformalize quantiles, reconcile, and write predictions parquet."""

    split_start = _normalize_date(start)
    split_end = _normalize_date(end)
    if split_start > split_end:
        raise typer.BadParameter("start date must be on/before end date")

    feature_source = _resolve_feature_source(features, data_root=data_root, season=season, month=month)

    feature_df = load_feature_frame(
        features_path=features,
        data_root=data_root,
        season=season,
        month=month,
    )
    ensure_columns(
        feature_df,
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
    sliced = _filter_window(feature_df, split_start, split_end)
    sliced = deduplicate_latest(sliced, key_cols=KEY_COLUMNS, order_cols=["feature_as_of_ts"])
    sliced = sliced.copy()
    sliced["game_id"] = sliced["game_id"].astype(str)

    run_dir = artifact_root / run_id
    bundle_path = run_dir / "lgbm_quantiles.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Quantile artifact not found at {bundle_path}")
    bundle = joblib.load(bundle_path)
    meta_path = run_dir / "meta.json"
    with meta_path.open() as handle:
        meta = json.load(handle)
    feature_columns: list[str] = bundle["feature_columns"]
    missing = set(feature_columns) - set(sliced.columns)
    if missing:
        raise ValueError(f"Scoring dataset missing columns: {', '.join(sorted(missing))}")

    quantile_preds = modeling.predict_quantiles(bundle["quantiles"], sliced[feature_columns])
    p10_raw = quantile_preds[0.1]
    p50_raw = quantile_preds[0.5]
    p90_raw = quantile_preds[0.9]

    scored = sliced.copy()
    scored["p10_raw"] = p10_raw
    scored["p50_raw"] = p50_raw
    scored["p90_raw"] = p90_raw
    scored["p50"] = p50_raw

    conformal_offsets = bundle["calibrator"].export_offsets()
    global_delta_p10 = -conformal_offsets["low_adjustment"]
    global_delta_p90 = conformal_offsets["high_adjustment"]
    rolling_cfg = calibration.RollingCalibrationConfig(
        window_days=window_days,
        min_n=min_n,
        tau=tau,
        tau_bucket=tau_bucket,
        use_buckets=use_buckets,
        min_n_bucket=min_n_bucket,
        use_spread_buckets=use_spread_buckets,
        cold_start_min_n=cold_start_min_n,
        min_recent_days=min_recent_days,
        p10_floor_guard=p10_floor_guard,
        p90_floor_guard=p90_floor_guard,
        bucket_delta_cap=bucket_delta_cap,
        bucket_floor_relief=bucket_floor_relief,
        max_abs_delta_p10=max_abs_delta_p10,
        max_abs_delta_p90=max_abs_delta_p90,
        recent_target_rows=recent_target_rows,
        recency_window_days=recency_window_days,
        recency_half_life_days=recency_half_life_days,
        warmup_days_p10=warmup_days_p10,
        warmup_days_p90=warmup_days_p90,
        use_global_p10_delta=use_global_p10_delta,
        use_global_p90_delta=use_global_p90_delta,
        global_baseline_months=global_baseline_months,
        global_recent_target_rows=global_recent_target_rows,
        global_recent_target_rows_p10=global_recent_target_rows_p10,
        global_recent_target_rows_p90=global_recent_target_rows_p90,
        global_min_n=global_min_n,
        global_delta_cap=global_delta_cap,
        bucket_min_effective_sample_size=bucket_min_effective_sample_size,
        bucket_inherit_if_low_n=bucket_inherit_if_low_n,
        cap_guardrail_rate=cap_guardrail_rate,
        max_width_delta_pct=max_width_delta_pct,
        monotonicity_repair_threshold=monotonicity_repair_threshold,
        apply_center_width_projection=apply_center_width_projection,
        min_quantile_half_width=min_quantile_half_width,
        one_sided_days=one_sided_days,
        hysteresis_lower=hysteresis_lower,
        hysteresis_upper=hysteresis_upper,
        delta_smoothing_alpha_p10=delta_smoothing_alpha_p10,
        delta_smoothing_alpha_p90=delta_smoothing_alpha_p90,
    )
    rolling_offsets = calibration.compute_rolling_offsets(
        scored,
        global_delta_p10=global_delta_p10,
        global_delta_p90=global_delta_p90,
        label_col=target_col,
        config=rolling_cfg,
    )
    scored = calibration.apply_rolling_offsets(scored, rolling_offsets, config=rolling_cfg)
    scored["quantile_width"] = scored["p90"] - scored["p10"]

    scored = reconcile_minutes(scored)
    scored = deduplicate_latest(scored, key_cols=KEY_COLUMNS, order_cols=["feature_as_of_ts"])
    scored["game_id"] = scored["game_id"].astype(str)
    scored["run_id"] = run_id
    scored["feature_hash"] = meta.get("feature_hash")

    output_path = out or _default_output_path(data_root, split_start)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    offsets_path = _sidecar_path(output_path, "rolling_offsets")
    if feature_source is not None and feature_source.is_file():
        ids_path = _sidecar_path(feature_source, "ids")
        if ids_path.exists():
            expected_ids = pd.read_csv(ids_path, dtype={"game_id": str})
        else:
            expected_ids = sliced.loc[:, list(KEY_COLUMNS)]
    else:
        expected_ids = sliced.loc[:, list(KEY_COLUMNS)]
    expected_ids = expected_ids.copy()
    expected_ids["game_id"] = expected_ids["game_id"].astype(str)

    missing = left_anti_keys(expected_ids, scored)
    if not missing.empty:
        diff_path = _sidecar_path(output_path, "left_anti_ids")
        missing.to_csv(diff_path, index=False)
        raise RuntimeError(
            f"Predictions missing {len(missing)} feature rows. See {diff_path} for details."
        )

    rolling_offsets.sort_values("score_date").to_csv(offsets_path, index=False)

    scored.to_parquet(output_path, index=False)
    write_ids_csv(scored, _sidecar_path(output_path, "ids"))

    meta_payload = {
        "run_id": run_id,
        "feature_hash": meta.get("feature_hash"),
        "start": split_start.strftime("%Y-%m-%d"),
        "end": split_end.strftime("%Y-%m-%d"),
        "rows": len(scored),
        "target_col": target_col,
        "calibration": {
            "strategy": "rolling",
            "params": rolling_cfg.to_dict(),
            "global_delta_p10": global_delta_p10,
            "global_delta_p90": global_delta_p90,
            "provenance": str(offsets_path),
        },
    }
    write_json(output_path.with_suffix(".meta.json"), meta_payload)
    typer.echo(f"Wrote {len(scored):,} predictions → {output_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
