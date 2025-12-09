"""Score fantasy points per minute using the production bundle."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from projections import paths
from projections.fpts_v1.datasets import FptsDatasetBuilder
from projections.fpts_v1.production import (
    DEFAULT_ARTIFACT_ROOT as DEFAULT_FPTS_ARTIFACT_ROOT,
)
from projections.fpts_v1.production import (
    ProductionFptsBundle,
    load_fpts_model,
    load_production_fpts_bundle,
    predict_fpts,
)
from projections.rates_v1.score import predict_rates
from projections.rates_v1.production import load_production_rates_bundle
from projections.minutes_v1.pos import canonical_pos_bucket_series
from projections.pipeline.status import JobStatus, write_status

DEFAULT_MINUTES_ROOT = Path("artifacts/minutes_v1/daily")
DEFAULT_FEATURES_ROOT = paths.data_path("live", "features_minutes_v1")
DEFAULT_OUT_ROOT = paths.data_path("gold", "projections_fpts_v1")
DEFAULT_RATES_LIVE_ROOT = paths.data_path("gold", "rates_v1_live")
DEFAULT_RATES_TRAIN_ROOT = paths.data_path("gold", "rates_training_base")
DEFAULT_PRODUCTION_CONFIG = Path("config/fpts_current_run.json")
MINUTES_FILENAME = "minutes.parquet"
MINUTES_SUMMARY = "summary.json"
FEATURE_FILENAME = "features.parquet"
OUTPUT_FILENAME = "fpts.parquet"
SUMMARY_FILENAME = "summary.json"
LATEST_POINTER = "latest_run.json"

app = typer.Typer(help=__doc__)


def load_fpts_bundle_context(
    fpts_run_id: str | None,
    artifact_root: Path,
    bundle_config: Path,
) -> ProductionFptsBundle:
    artifact_root = artifact_root.expanduser().resolve()
    if fpts_run_id:
        bundle = load_fpts_model(fpts_run_id, artifact_root=artifact_root)
        run_dir = artifact_root / fpts_run_id
        metadata = bundle.metadata or {}
        scoring = str(metadata.get("scoring_system") or "dk")
        resolved_run_id = str(metadata.get("run_id") or fpts_run_id)
        return ProductionFptsBundle(
            bundle=bundle,
            run_dir=run_dir,
            run_id=resolved_run_id,
            scoring_system=scoring,
        )
    return load_production_fpts_bundle(config_path=bundle_config)


def _normalize_day(value: datetime | None) -> date:
    if value is None:
        return datetime.now(tz=UTC).date()
    return value.date()


def _load_minutes_run(
    root: Path,
    slate_day: date,
    run_id: str | None,
) -> tuple[Path, str]:
    day_dir = root / slate_day.isoformat()
    if not day_dir.exists():
        raise FileNotFoundError(f"No minutes outputs found under {day_dir}")
    resolved_run = run_id
    if not resolved_run:
        pointer = day_dir / LATEST_POINTER
        if pointer.exists():
            try:
                payload = json.loads(pointer.read_text(encoding="utf-8"))
                resolved_run = payload.get("run_id")
            except json.JSONDecodeError:
                resolved_run = None
        if not resolved_run:
            raise FileNotFoundError(
                f"Run id missing for {slate_day.isoformat()} (expected pointer at {pointer})."
            )
    run_dir = day_dir / f"run={resolved_run}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Minutes run directory missing at {run_dir}")
    return run_dir, resolved_run


def resolve_minutes_run(
    minutes_root: Path,
    slate_day: date,
    run_id: str | None,
) -> tuple[Path, str]:
    """Public helper for resolving a minutes run directory."""

    return _load_minutes_run(minutes_root, slate_day, run_id)


def _load_features_path(
    root: Path,
    slate_day: date,
    run_id: str,
    explicit_path: Path | None,
) -> Path:
    if explicit_path is not None:
        resolved = explicit_path.expanduser().resolve()
        if resolved.is_dir():
            return resolved / FEATURE_FILENAME
        return resolved
    candidate = root / slate_day.isoformat() / f"run={run_id}" / FEATURE_FILENAME
    candidate = candidate.expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Live features parquet missing at {candidate}")
    return candidate


def resolve_features_path(
    live_features_root: Path,
    slate_day: date,
    run_id: str,
    explicit_path: Path | None,
) -> Path:
    """Public helper for resolving the features parquet path."""

    return _load_features_path(live_features_root, slate_day, run_id, explicit_path)


def _nan_rate(df: pd.DataFrame, cols: list[str]) -> float | None:
    present = [col for col in cols if col in df.columns]
    if not present or df.empty:
        return 0.0
    return float(df[present].isna().mean().mean())


def _attach_rate_predictions(
    slate_day: date,
    merged: pd.DataFrame,
    *,
    enabled: bool,
    require_rates: bool,
    precomputed_path: Path | None,
    rates_train_root: Path,
) -> pd.DataFrame:
    """Optionally attach rates_v1 predictions needed by fpts_v2 bundles.

    For fpts_v2 runs, the model expects columns like ``pred_fga2_per_min`` etc.
    We can derive these by scoring the current rates_v1 bundle against the
    rates_training_base features for the slate date when available.
    """

    if not enabled:
        return merged

    # Live path: use precomputed rates predictions if provided.
    if precomputed_path is not None:
        if not precomputed_path.exists():
            msg = f"[fpts-live] missing live rates predictions at {precomputed_path}; {'aborting' if require_rates else 'skipping'}."
            if require_rates:
                raise RuntimeError(msg)
            typer.echo(msg, err=True)
            return merged
        rates_df = pd.read_parquet(precomputed_path)
        if rates_df.empty:
            msg = f"[fpts-live] live rates predictions empty at {precomputed_path}; {'aborting' if require_rates else 'skipping'}."
            if require_rates:
                raise RuntimeError(msg)
            typer.echo(msg, err=True)
            return merged
        required_keys = {"game_id", "player_id", "team_id"}
        if not required_keys.issubset(rates_df.columns):
            msg = f"[fpts-live] live rates predictions missing keys {required_keys - set(rates_df.columns)}; {'aborting' if require_rates else 'skipping'}."
            if require_rates:
                raise RuntimeError(msg)
            typer.echo(msg, err=True)
            return merged
        pred_cols = [c for c in rates_df.columns if c.startswith("pred_")]
        if not pred_cols:
            msg = f"[fpts-live] live rates predictions missing pred_* columns at {precomputed_path}; {'aborting' if require_rates else 'skipping'}."
            if require_rates:
                raise RuntimeError(msg)
            typer.echo(msg, err=True)
            return merged
        for key in required_keys:
            rates_df[key] = pd.to_numeric(rates_df[key], errors="coerce")
        merged = merged.merge(rates_df, on=["game_id", "player_id", "team_id"], how="left")
        return merged

    # Historical/eval path: score rates features from training base for the date.
    try:
        rates_bundle = load_production_rates_bundle()
    except Exception as exc:  # noqa: BLE001
        msg = f"[fpts-live] unable to load rates bundle ({exc}); {'aborting' if require_rates else 'skipping'} rate preds."
        if require_rates:
            raise RuntimeError(msg)
        typer.echo("[fpts-live] warning: " + msg, err=True)
        return merged

    season = slate_day.year
    rates_path = (
        rates_train_root
        / f"season={season}"
        / f"game_date={slate_day.isoformat()}"
        / "rates_training_base.parquet"
    )
    if not rates_path.exists():
        msg = f"[fpts-live] missing rates features at {rates_path}; {'aborting' if require_rates else 'skipping'} rate preds."
        if require_rates:
            raise RuntimeError(msg)
        typer.echo(msg, err=True)
        return merged

    rates_df = pd.read_parquet(rates_path)
    rates_df = rates_df.copy()
    for key in ("game_id", "player_id", "team_id"):
        if key in rates_df.columns:
            rates_df[key] = pd.to_numeric(rates_df[key], errors="coerce")

    missing = [c for c in rates_bundle.feature_cols if c not in rates_df.columns]
    if missing:
        msg = f"[fpts-live] rates features missing columns {missing}; {'aborting' if require_rates else 'filling with 0.'}"
        if require_rates:
            raise RuntimeError(msg)
        typer.echo(msg, err=True)
        for col in missing:
            rates_df[col] = 0.0

    rate_features = rates_df[rates_bundle.feature_cols].copy()
    preds = predict_rates(rate_features, rates_bundle)
    preds = preds.rename(columns={col: f"pred_{col}" for col in preds.columns})
    rates_df = pd.concat([rates_df[["game_id", "player_id", "team_id"]], preds], axis=1)

    merged = merged.merge(rates_df, on=["game_id", "player_id", "team_id"], how="left")
    return merged


def _augment_minutes_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    for col, source in (
        ("minutes_p10_pred", working.get("minutes_p10")),
        ("minutes_p50_pred", working.get("minutes_p50")),
        ("minutes_p90_pred", working.get("minutes_p90")),
        ("play_prob_pred", working.get("play_prob")),
    ):
        if col in working.columns:
            continue
        if source is not None:
            working[col] = source
    for col in ("minutes_p10_pred", "minutes_p50_pred", "minutes_p90_pred", "play_prob_pred"):
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0.0)
    if {"minutes_p10_pred", "minutes_p90_pred"}.issubset(working.columns):
        working["minutes_volatility_pred"] = (
            working["minutes_p90_pred"] - working["minutes_p10_pred"]
        ).clip(lower=0.0)
    if {"minutes_p50_pred", "minutes_volatility_pred"}.issubset(working.columns):
        working["minutes_range_ratio"] = np.where(
            working["minutes_p50_pred"] > 0,
            working["minutes_volatility_pred"] / working["minutes_p50_pred"].clip(lower=1e-3),
            0.0,
        )

    if "pos_bucket" in working.columns:
        pos_series = canonical_pos_bucket_series(working["pos_bucket"])
        working["pos_is_guard"] = (pos_series == "G").astype(int)
        working["pos_is_wing"] = (pos_series == "W").astype(int)
        working["pos_is_big"] = (pos_series == "B").astype(int)
    lineup_role = working.get("lineup_role")
    if lineup_role is not None:
        working["lineup_role_tier"] = (
            lineup_role.fillna("")
            .str.lower()
            .map({"starter": 2, "bench": 1})
            .fillna(0)
            .astype(int)
        )

    if "status" in working.columns:
        status_series = working["status"].fillna("").str.lower()
    else:
        status_series = pd.Series("", index=working.index)
    out_mask = status_series.str.startswith("out")
    questionable_mask = status_series.str.startswith("q")
    out_flags = out_mask.astype(int)
    q_flags = questionable_mask.astype(int)
    if {"game_id", "team_id"}.issubset(working.columns):
        team_out = out_flags.groupby(
            [working["game_id"], working["team_id"]]
        ).transform("sum")
        team_q = q_flags.groupby(
            [working["game_id"], working["team_id"]]
        ).transform("sum")
        working["teammate_out_count"] = (team_out - out_flags).clip(lower=0)
        working["teammate_questionable_count"] = (team_q - q_flags).clip(lower=0)
        prior_cols = [
            col for col in ("fpts_per_min_prior_10", "fpts_per_min_prior_5") if col in working.columns
        ]
        if prior_cols:
            base_prior = working[prior_cols[0]].fillna(working.get("fpts_per_min_prior_5"))
            prior_contrib = base_prior * working.get("minutes_p50_pred", 0.0)
            out_usage = pd.Series(
                np.where(out_mask, prior_contrib, 0.0), index=working.index
            )
            team_out_usage = out_usage.groupby(
                [working["game_id"], working["team_id"]]
            ).transform("sum")
            working["teammate_out_usage_sum"] = (team_out_usage - out_usage).clip(lower=0.0)
        if "pos_bucket" in working.columns:
            same_pos_out = out_flags.groupby(
                [working["game_id"], working["team_id"], working["pos_bucket"]]
            ).transform("sum")
            working["same_pos_teammate_out_count"] = (same_pos_out - out_flags).clip(lower=0)

    if {"total", "spread_home", "home_flag"}.issubset(working.columns):
        total = pd.to_numeric(working["total"], errors="coerce")
        spread_home = pd.to_numeric(working["spread_home"], errors="coerce")
        home_flag = working["home_flag"].fillna(0).astype(int)
        working["team_implied_total"] = np.where(
            home_flag == 1,
            total / 2 - spread_home / 2,
            total / 2 + spread_home / 2,
        )
        working["opponent_implied_total"] = np.where(
            home_flag == 1,
            total / 2 + spread_home / 2,
            total / 2 - spread_home / 2,
        )
    return working


def _write_pointer(day_dir: Path, run_id: str) -> None:
    pointer = day_dir / LATEST_POINTER
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    pointer.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def score_fpts_for_date(
    *,
    slate_day: date,
    run_id: str | None,
    minutes_root: Path,
    live_features_root: Path,
    features_path: Path | None,
    out_root: Path,
    bundle_ctx: ProductionFptsBundle,
    data_root: Path,
    resolved_minutes_dir: Path | None = None,
    resolved_run_id: str | None = None,
    resolved_features_path: Path | None = None,
    builder: FptsDatasetBuilder,
    quiet: bool = False,
    require_rates: bool = True,
    rates_live_root: Path | None = None,
    use_live_rates: bool = True,
    rates_train_root: Path | None = None,
) -> Path:
    """Score FPTS for a single slate date."""

    if resolved_minutes_dir is None or resolved_run_id is None:
        minutes_run_dir, resolved_run = _load_minutes_run(minutes_root, slate_day, run_id)
    else:
        minutes_run_dir = resolved_minutes_dir
        resolved_run = resolved_run_id

    if resolved_features_path is None:
        features_file = _load_features_path(
            live_features_root,
            slate_day,
            resolved_run,
            features_path,
        )
    else:
        features_file = resolved_features_path

    minutes_parquet = minutes_run_dir / MINUTES_FILENAME
    if not minutes_parquet.exists():
        raise FileNotFoundError(f"Minutes parquet missing at {minutes_parquet}")
    minutes_df = pd.read_parquet(minutes_parquet)
    features_df = pd.read_parquet(features_file)
    features_df = builder.enrich_live_features(slate_day, features_df)
    features_df["_fpts_feature_present"] = 1

    for col in ("game_id", "player_id", "team_id"):
        minutes_df[col] = pd.to_numeric(minutes_df[col], errors="coerce").astype("Int64")
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce").astype("Int64")
    merged = minutes_df.merge(
        features_df,
        on=["game_id", "player_id"],
        how="left",
        suffixes=("", "_feat"),
    )
    missing_mask = merged["_fpts_feature_present"].isna()
    if missing_mask.any():
        missing_rows = int(missing_mask.sum())
        typer.echo(
            f"[fpts-live] warning: dropping {missing_rows} rows with missing features.",
            err=True,
        )
    merged = merged.loc[~missing_mask].copy()
    merged.drop(columns=["_fpts_feature_present"], inplace=True, errors="ignore")
    if merged.empty:
        raise RuntimeError(f"[fpts-live] {slate_day.isoformat()}: no rows remaining after join.")
    merged = _augment_minutes_features(merged)

    # Attach rate predictions when using fpts_v2 bundles (total-FPTS head).
    is_fpts_v2 = str(bundle_ctx.run_id).lower().startswith("fpts_v2")
    precomputed_rates = None
    if use_live_rates and rates_live_root is not None:
        # Resolve rates run ID from rates' own latest_run.json (not minutes run_id)
        rates_day_dir = rates_live_root / slate_day.isoformat()
        rates_pointer = rates_day_dir / LATEST_POINTER
        rates_run_id = None
        if rates_pointer.exists():
            try:
                payload = json.loads(rates_pointer.read_text(encoding="utf-8"))
                rates_run_id = payload.get("run_id")
            except json.JSONDecodeError:
                rates_run_id = None
        if rates_run_id:
            precomputed_rates = rates_day_dir / f"run={rates_run_id}" / "rates.parquet"
        else:
            # Fallback: try to find latest run directory
            run_dirs = sorted(rates_day_dir.glob("run=*"), reverse=True) if rates_day_dir.exists() else []
            if run_dirs:
                precomputed_rates = run_dirs[0] / "rates.parquet"
    merged = _attach_rate_predictions(
        slate_day,
        merged,
        enabled=is_fpts_v2,
        require_rates=require_rates,
        precomputed_path=precomputed_rates,
        rates_train_root=rates_train_root or DEFAULT_RATES_TRAIN_ROOT,
    )

    preds = predict_fpts(bundle_ctx.bundle, merged)
    enriched = merged.copy()
    enriched["fpts_per_min_pred"] = preds["fpts_per_min_pred"]
    enriched["proj_fpts"] = preds["proj_fpts"]
    enriched["scoring_system"] = bundle_ctx.scoring_system
    enriched["minutes_run_id"] = resolved_run
    enriched["fpts_model_run_id"] = bundle_ctx.run_id

    out_day_dir = out_root / slate_day.isoformat()
    out_day_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_day_dir / f"run={resolved_run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    export_columns = [
        "game_date",
        "tip_ts",
        "game_id",
        "player_id",
        "player_name",
        "team_id",
        "team_name",
        "team_tricode",
        "opponent_team_id",
        "opponent_team_name",
        "opponent_team_tricode",
        "starter_flag",
        "pos_bucket",
        "spread_home",
        "total",
        "team_implied_total",
        "opponent_implied_total",
        "play_prob",
        "minutes_p10",
        "minutes_p50",
        "minutes_p90",
        "minutes_p10_cond",
        "minutes_p50_cond",
        "minutes_p90_cond",
        "minutes_p10_uncond",
        "minutes_p50_uncond",
        "minutes_p90_uncond",
        "fpts_per_min_pred",
        "proj_fpts",
        "scoring_system",
        "minutes_run_id",
        "fpts_model_run_id",
    ]
    available_cols = [col for col in export_columns if col in enriched.columns]
    enriched.loc[:, available_cols].to_parquet(run_dir / OUTPUT_FILENAME, index=False)

    minutes_summary = {}
    summary_path = minutes_run_dir / MINUTES_SUMMARY
    if summary_path.exists():
        try:
            minutes_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            minutes_summary = {}
    summary_payload = {
        "date": slate_day.isoformat(),
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "minutes_run_id": resolved_run,
        "minutes_run_as_of_ts": minutes_summary.get("run_as_of_ts"),
        "fpts_model_run_id": bundle_ctx.run_id,
        "scoring_system": bundle_ctx.scoring_system,
        "counts": {
            "rows": int(len(enriched)),
            "players": int(enriched["player_id"].nunique()),
        },
    }
    (run_dir / SUMMARY_FILENAME).write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_pointer(out_day_dir, resolved_run)
    if not quiet:
        typer.echo(
            f"[fpts-live] {slate_day.isoformat()} run={resolved_run}: "
            f"wrote {len(enriched)} rows to {run_dir / OUTPUT_FILENAME}"
        )
    return run_dir / OUTPUT_FILENAME


@app.command()
def main(
    date_value: datetime = typer.Option(..., "--date", help="Slate date (YYYY-MM-DD)."),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Minutes run identifier (defaults to latest).",
    ),
    minutes_root: Path = typer.Option(
        DEFAULT_MINUTES_ROOT,
        "--minutes-root",
        help="Root directory for minutes outputs.",
    ),
    live_features_root: Path = typer.Option(
        DEFAULT_FEATURES_ROOT,
        "--live-features-root",
        help="Root directory for live feature runs.",
    ),
    features_path: Path | None = typer.Option(
        None,
        "--features-path",
        help="Explicit features.parquet path (overrides --live-features-root).",
    ),
    out_root: Path = typer.Option(
        DEFAULT_OUT_ROOT,
        "--out-root",
        help="Directory where gold FPTS outputs will be written.",
    ),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Canonical data root containing historical boxscores/features.",
    ),
    bundle_config: Path = typer.Option(
        DEFAULT_PRODUCTION_CONFIG,
        "--bundle-config",
        help="JSON file pointing to the production FPTS bundle.",
    ),
    fpts_run_id: str | None = typer.Option(
        None,
        "--fpts-run-id",
        help="Override the production FPTS run id for scoring.",
    ),
    fpts_artifact_root: Path = typer.Option(
        DEFAULT_FPTS_ARTIFACT_ROOT,
        "--fpts-artifact-root",
        help="Root directory containing FPTS artifacts (used with --fpts-run-id).",
    ),
    allow_missing_rates: bool = typer.Option(
        False,
        "--allow-missing-rates",
        help="If set, skip rate preds when rates features are missing instead of failing.",
    ),
    rates_live_root: Path = typer.Option(
        DEFAULT_RATES_LIVE_ROOT,
        "--rates-live-root",
        help="Root containing live rates predictions (run=<id>/rates.parquet).",
    ),
    rates_train_root: Path = typer.Option(
        DEFAULT_RATES_TRAIN_ROOT,
        "--rates-train-root",
        help="Root containing rates training base partitions (for historical/eval scoring).",
    ),
    use_live_rates: bool = typer.Option(
        True,
        "--use-live-rates/--use-training-rates",
        help="Use live rates predictions (default) or fall back to training-base scoring.",
    ),
) -> None:
    slate_day = _normalize_day(date_value)
    minutes_root = minutes_root.expanduser().resolve()
    live_features_root = live_features_root.expanduser().resolve()
    out_root = out_root.expanduser().resolve()
    data_root = data_root.expanduser().resolve()

    target_date = slate_day.isoformat()
    run_ts_iso = datetime.now(tz=UTC).isoformat()
    rows_written = 0
    nan_rate = None
    try:
        try:
            bundle_ctx = load_fpts_bundle_context(fpts_run_id, fpts_artifact_root, bundle_config)
        except (RuntimeError, FileNotFoundError) as exc:
            typer.echo(f"[fpts-live] warning: {exc}; skipping FPTS scoring.", err=True)
            raise typer.Exit(code=0)
        metadata = bundle_ctx.bundle.metadata or {}
        bundle_minutes_source = str(metadata.get("minutes_source") or "actual").strip().lower()
        if bundle_minutes_source not in ("predicted", "actual"):
            bundle_minutes_source = "actual"
        builder = FptsDatasetBuilder(
            data_root=data_root,
            minutes_source=bundle_minutes_source,
        )
        try:
            output_path = score_fpts_for_date(
                slate_day=slate_day,
                run_id=run_id,
                minutes_root=minutes_root,
                live_features_root=live_features_root,
                features_path=features_path,
                out_root=out_root,
                bundle_ctx=bundle_ctx,
                data_root=data_root,
                builder=builder,
                require_rates=not allow_missing_rates,
                rates_live_root=rates_live_root,
                use_live_rates=use_live_rates,
                rates_train_root=rates_train_root,
            )
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=0)
        run_dir = output_path.parent
        summary_path = run_dir / SUMMARY_FILENAME
        if summary_path.exists():
            try:
                summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
                rows_written = int(summary_data.get("counts", {}).get("rows", 0))
            except Exception:
                rows_written = 0
        if rows_written == 0:
            try:
                df = pd.read_parquet(output_path, columns=["proj_fpts", "fpts_per_min_pred"])
                rows_written = len(df)
                nan_rate = _nan_rate(df, ["proj_fpts", "fpts_per_min_pred"])
            except Exception:
                rows_written = 0
        if nan_rate is None and output_path.exists():
            try:
                df = pd.read_parquet(output_path, columns=["proj_fpts", "fpts_per_min_pred"])
                nan_rate = _nan_rate(df, ["proj_fpts", "fpts_per_min_pred"])
            except Exception:
                nan_rate = None
        write_status(
            JobStatus(
                job_name="score_fpts_live",
                stage="projections",
                target_date=target_date,
                run_ts=run_ts_iso,
                status="success",
                rows_written=rows_written,
                expected_rows=rows_written if rows_written else None,
                nan_rate_key_cols=nan_rate,
            )
        )
    except typer.Exit as exc:
        write_status(
            JobStatus(
                job_name="score_fpts_live",
                stage="projections",
                target_date=target_date,
                run_ts=run_ts_iso,
                status="error",
                rows_written=rows_written,
                expected_rows=None,
                message=str(exc),
            )
        )
        raise
    except Exception as exc:  # noqa: BLE001
        write_status(
            JobStatus(
                job_name="score_fpts_live",
                stage="projections",
                target_date=target_date,
                run_ts=run_ts_iso,
                status="error",
                rows_written=rows_written,
                expected_rows=None,
                message=str(exc),
            )
        )
        raise


if __name__ == "__main__":
    app()
