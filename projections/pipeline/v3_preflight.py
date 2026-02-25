"""Fail-fast preflight gates for the v3 live pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from projections.pipeline import parity_checks
from projections.pipeline.parity_manifest import load_parity_manifest


class V3PreflightError(RuntimeError):
    """Raised when v3 preflight contract checks fail."""


def _coerce_ts(value: str | datetime) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise V3PreflightError(f"invalid timestamp value: {value!r}")
    return ts


def _check_required_inputs(
    *,
    required_inputs: Mapping[str, Path],
    as_of_ts: pd.Timestamp,
    input_max_age_minutes: float,
) -> dict[str, float]:
    freshness: dict[str, float] = {}
    for name, raw_path in required_inputs.items():
        path = Path(raw_path)
        if not path.exists():
            raise V3PreflightError(f"required input missing: {name} -> {path}")
        mtime = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
        age_minutes = float((as_of_ts - mtime).total_seconds() / 60.0)
        if age_minutes < -5.0:
            raise V3PreflightError(
                f"required input mtime is after as_of_ts: {name} age_minutes={age_minutes:.2f}"
            )
        if age_minutes > float(input_max_age_minutes):
            raise V3PreflightError(
                f"required input too stale: {name} age_minutes={age_minutes:.2f} "
                f"> max_age_minutes={float(input_max_age_minutes):.2f}"
            )
        freshness[str(name)] = age_minutes
    return freshness


def _check_run_dirs_clean_writable(run_dirs: Sequence[Path]) -> list[str]:
    checked: list[str] = []
    for raw_dir in run_dirs:
        run_dir = Path(raw_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if any(run_dir.iterdir()):
            raise V3PreflightError(f"run output dir is not clean: {run_dir}")
        probe = run_dir / ".v3_write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
        finally:
            if probe.exists():
                probe.unlink()
        checked.append(str(run_dir))
    return checked


def run_preflight_gate(
    *,
    as_of_ts: str | datetime,
    required_inputs: Mapping[str, Path],
    run_dirs: Sequence[Path],
    features_path: Path,
    parity_manifest_path: Path,
    observed_transform_manifest: Mapping[str, Any],
    observed_integrity: Mapping[str, Any],
    input_max_age_minutes: float = 360.0,
    bundle_config_path: Path | None = None,
) -> dict[str, Any]:
    """Execute strict v3 preflight checks before model scoring."""
    ts = _coerce_ts(as_of_ts)

    freshness = _check_required_inputs(
        required_inputs=required_inputs,
        as_of_ts=ts,
        input_max_age_minutes=float(input_max_age_minutes),
    )
    run_dirs_checked = _check_run_dirs_clean_writable(run_dirs)

    if not Path(features_path).exists():
        raise V3PreflightError(f"features file missing: {features_path}")

    manifest = load_parity_manifest(Path(parity_manifest_path))
    features_df = pd.read_parquet(features_path)

    feature_report = parity_checks.validate_feature_frame_against_manifest(
        features_df,
        manifest,
        require_exact_order=True,
    )
    transform_report = parity_checks.validate_transform_manifest(
        manifest,
        observed_transform_manifest,
    )
    integrity_report = parity_checks.validate_integrity_manifest(
        manifest,
        observed_integrity,
    )
    distribution_report = parity_checks.validate_feature_distribution_contract(
        features_df,
        manifest,
        bundle_config_path=bundle_config_path,
    )

    return {
        "as_of_ts": ts.isoformat(),
        "required_inputs_age_minutes": freshness,
        "run_dirs_checked": run_dirs_checked,
        "feature_report": feature_report,
        "transform_report": transform_report,
        "integrity_report": integrity_report,
        "distribution_report": distribution_report,
        "parity_manifest_path": str(parity_manifest_path),
    }
