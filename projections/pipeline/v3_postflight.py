"""Strict postflight gates for the v3 live pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from projections.pipeline import parity_checks, writer_guard
from projections.pipeline.parity_manifest import load_parity_manifest


class V3PostflightError(RuntimeError):
    """Raised when v3 postflight contract checks fail."""


_REQUIRED_WORLD_ZERO_KEYS = (
    "team_minutes_not_240",
    "minutes_negative",
    "minutes_over_48",
    "negative_stats",
    "fg2m_gt_fga2",
    "fg3m_gt_fga3",
    "ftm_gt_fta",
    "inactive_nonzero_stats",
    "inactive_nonzero_fpts_proxy",
)


def _load_world_checks(
    *,
    world_contract_summary_path: Path | None,
    world_contract_checks: Mapping[str, Any] | None,
) -> dict[str, int]:
    if world_contract_checks is not None:
        payload = dict(world_contract_checks)
    elif world_contract_summary_path is not None:
        try:
            raw = json.loads(Path(world_contract_summary_path).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise V3PostflightError(f"world contract summary missing: {world_contract_summary_path}") from exc
        except json.JSONDecodeError as exc:
            raise V3PostflightError(
                f"world contract summary is invalid JSON: {world_contract_summary_path}"
            ) from exc
        if isinstance(raw, dict) and isinstance(raw.get("contract_checks"), dict):
            payload = dict(raw["contract_checks"])
        elif isinstance(raw, dict):
            payload = dict(raw)
        else:
            raise V3PostflightError("world contract summary must be an object")
    else:
        raise V3PostflightError("world contract checks are required")

    out: dict[str, int] = {}
    for k, v in payload.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            out[str(k)] = 0
    return out


def _validate_world_contracts(world_checks: Mapping[str, int]) -> dict[str, int]:
    bad: dict[str, int] = {}
    for key in _REQUIRED_WORLD_ZERO_KEYS:
        val = int(world_checks.get(key, 0))
        if val != 0:
            bad[key] = val
    if bad:
        raise V3PostflightError(f"world contract check failed: {bad}")
    return {k: int(world_checks.get(k, 0)) for k in _REQUIRED_WORLD_ZERO_KEYS}


def _validate_projection_rows(
    projections_df: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    min_rows: int,
) -> dict[str, Any]:
    if int(len(projections_df)) < int(min_rows):
        raise V3PostflightError(
            f"projection row count too low: {len(projections_df)} < {int(min_rows)}"
        )

    missing = [c for c in key_columns if c not in projections_df.columns]
    if missing:
        raise V3PostflightError(f"projection key columns missing: {missing}")

    null_counts = {
        c: int(projections_df[c].isna().sum()) for c in key_columns if c in projections_df.columns
    }
    bad_nulls = {k: v for k, v in null_counts.items() if v > 0}
    if bad_nulls:
        raise V3PostflightError(f"projection key columns contain nulls: {bad_nulls}")

    dupes = int(projections_df.duplicated(subset=list(key_columns)).sum())
    if dupes > 0:
        raise V3PostflightError(f"projection key duplication detected: duplicates={dupes}")

    return {
        "row_count": int(len(projections_df)),
        "duplicate_keys": int(dupes),
        "key_columns": list(key_columns),
    }


def run_postflight_gate(
    *,
    projections_path: Path,
    parity_manifest_path: Path,
    world_contract_summary_path: Path | None = None,
    world_contract_checks: Mapping[str, Any] | None = None,
    key_columns: Sequence[str] = ("game_id", "team_id", "player_id"),
    min_rows: int = 20,
) -> dict[str, Any]:
    """Execute strict postflight checks before atomic pointer publish."""
    writer_guard.assert_can_write_pointers(purpose="v3 postflight gate")

    checks = _load_world_checks(
        world_contract_summary_path=world_contract_summary_path,
        world_contract_checks=world_contract_checks,
    )
    world_report = _validate_world_contracts(checks)

    if not Path(projections_path).exists():
        raise V3PostflightError(f"projections file missing: {projections_path}")
    projections_df = pd.read_parquet(projections_path)

    manifest = load_parity_manifest(Path(parity_manifest_path))
    schema_report = parity_checks.validate_projection_output_columns(projections_df, manifest)
    row_report = _validate_projection_rows(
        projections_df,
        key_columns=tuple(key_columns),
        min_rows=int(min_rows),
    )

    return {
        "world_contract_report": world_report,
        "projection_schema_report": schema_report,
        "projection_row_report": row_report,
        "projections_path": str(projections_path),
    }
