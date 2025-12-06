"""Score rates_v1 predictions for a live slate."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections import paths
from projections.rates_v1.production import load_production_rates_bundle
from projections.rates_v1.score import predict_rates
from projections.rates_v1.schemas import FeatureSchemaMismatchError
from projections.pipeline.status import JobStatus, write_status

# Changed from features_minutes_v1 to features_rates_v1
DEFAULT_FEATURES_ROOT = paths.data_path("live", "features_rates_v1")
DEFAULT_OUT_ROOT = paths.data_path("gold", "rates_v1_live")
FEATURE_FILENAME = "features.parquet"
OUTPUT_FILENAME = "rates.parquet"
SUMMARY_FILENAME = "summary.json"
LATEST_POINTER = "latest_run.json"

app = typer.Typer(help=__doc__)


def _normalize_day(value: datetime | None) -> date:
    if value is None:
        return datetime.now(tz=UTC).date()
    return value.date()


def _resolve_run_id(day_dir: Path, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    pointer = day_dir / LATEST_POINTER
    if pointer.exists():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            latest = payload.get("run_id")
            if latest:
                return str(latest)
        except json.JSONDecodeError:
            pass
    raise FileNotFoundError(f"Run id missing for {day_dir}; expected {LATEST_POINTER} or --run-id.")


def _load_features_path(root: Path, slate_day: date, run_id: str, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        resolved = explicit.expanduser().resolve()
        if resolved.is_dir():
            return resolved / FEATURE_FILENAME
        return resolved
    candidate = root / slate_day.isoformat() / f"run={run_id}" / FEATURE_FILENAME
    candidate = candidate.expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Live features parquet missing at {candidate}")
    return candidate


def _write_pointer(day_dir: Path, run_id: str) -> None:
    payload = {"run_id": run_id, "generated_at": datetime.now(tz=UTC).isoformat()}
    (day_dir / LATEST_POINTER).write_text(json.dumps(payload, indent=2), encoding="utf-8")


@app.command()
def main(
    date_value: datetime = typer.Option(..., "--date", help="Slate date (YYYY-MM-DD)"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Run id matching minutes/features (defaults to latest)."),
    features_root: Path = typer.Option(
        DEFAULT_FEATURES_ROOT, "--features-root", help="Root containing live minutes features (run=<id>/features.parquet)."
    ),
    features_path: Optional[Path] = typer.Option(
        None, "--features-path", help="Optional explicit features parquet (overrides --features-root)."
    ),
    out_root: Path = typer.Option(
        DEFAULT_OUT_ROOT,
        "--out-root",
        help="Output root for live rates predictions (per-day subdir with run=<id>/rates.parquet).",
    ),
    bundle_config: Optional[Path] = typer.Option(
        None, "--bundle-config", help="Optional override for config/rates_current_run.json."
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="Raise error on missing features (default: True). Use --no-strict to fall back to zero-fill.",
    ),
) -> None:
    slate_day = _normalize_day(date_value)
    features_root = features_root.expanduser().resolve()
    out_root = out_root.expanduser().resolve()

    day_dir = features_root / slate_day.isoformat()
    resolved_run = _resolve_run_id(day_dir, run_id)
    feat_path = _load_features_path(features_root, slate_day, resolved_run, features_path)

    run_ts_iso = datetime.now(tz=UTC).isoformat()
    rows_written = 0
    nan_rate = None
    try:
        bundle = load_production_rates_bundle(config_path=bundle_config)
        features_df = pd.read_parquet(feat_path)
        for key in ("game_id", "player_id", "team_id"):
            if key in features_df.columns:
                features_df[key] = pd.to_numeric(features_df[key], errors="coerce")
        missing = [c for c in bundle.feature_cols if c not in features_df.columns]
        if missing:
            if strict:
                raise FeatureSchemaMismatchError(
                    f"Rates model requires columns missing from features: {sorted(missing)}. "
                    f"Run build_rates_features_live first to generate proper features, "
                    f"or use --no-strict to fall back to zero-fill (not recommended)."
                )
            else:
                typer.echo(
                    f"[rates-live] WARNING: filling missing feature columns with 0: {missing}",
                    err=True,
                )
                for col in missing:
                    features_df[col] = 0.0
        preds = predict_rates(features_df, bundle)
        preds = preds.rename(columns={col: f"pred_{col}" for col in preds.columns})
        pct_clamps = {
            "pred_fg2_pct": (0.3, 0.75),
            "pred_fg3_pct": (0.2, 0.55),
            "pred_ft_pct": (0.5, 0.95),
        }
        for col, (lo, hi) in pct_clamps.items():
            if col in preds.columns:
                preds[col] = preds[col].clip(lo, hi)
        payload = pd.concat(
            [
                features_df[["game_id", "team_id", "player_id"]].reset_index(drop=True),
                preds.reset_index(drop=True),
            ],
            axis=1,
        )
        payload["game_date"] = pd.to_datetime(slate_day).normalize()

        out_day_dir = out_root / slate_day.isoformat()
        out_run_dir = out_day_dir / f"run={resolved_run}"
        out_run_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_run_dir / OUTPUT_FILENAME
        payload.to_parquet(out_path, index=False)
        rows_written = len(payload)
        nan_rate = float(payload.isna().mean().mean())

        summary = {
            "date": slate_day.isoformat(),
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "run_id": resolved_run,
            "rates_model_run_id": bundle.meta.get("run_id") if hasattr(bundle, "meta") else None,
            "counts": {"rows": rows_written, "players": int(payload["player_id"].nunique())},
        }
        (out_run_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_pointer(out_day_dir, resolved_run)
        typer.echo(f"[rates-live] {slate_day} run={resolved_run}: wrote {rows_written} rows -> {out_path}")
        write_status(
            JobStatus(
                job_name="score_rates_live",
                stage="projections",
                target_date=slate_day.isoformat(),
                run_ts=run_ts_iso,
                status="success",
                rows_written=rows_written,
                expected_rows=rows_written,
                nan_rate_key_cols=nan_rate,
            )
        )
    except Exception as exc:  # noqa: BLE001
        write_status(
            JobStatus(
                job_name="score_rates_live",
                stage="projections",
                target_date=slate_day.isoformat(),
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
