"""Build effective inputs consumed by downstream live pipeline stages.

Effective inputs apply manual overrides from the *authorized* source and
materialize a deterministic parquet layer for downstream consumption.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from projections.minutes.depth_chart_crosswalk import (
    refresh_realgm_player_crosswalk_from_minutes,
    summarize_crosswalk_json,
)
from projections.minutes.depth_chart_prior import apply_depth_chart_prior_from_realgm
from projections.ops.overrides import load_overrides_map

EFFECTIVE_MINUTES_FILENAME = "effective_minutes.parquet"
EFFECTIVE_RATES_FILENAME = "effective_rates.parquet"
EFFECTIVE_INPUTS_SUMMARY = "effective_inputs_summary.json"

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}.json")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True, slots=True)
class EffectiveInputsResult:
    effective_minutes_path: Path
    summary_path: Path
    overrides_count: int


def _coerce_timestamp(value: object) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _load_run_as_of_ts(run_dir: Path) -> tuple[pd.Timestamp | None, Path | None]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None, None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None, manifest_path
    if not isinstance(payload, dict):
        return None, manifest_path

    as_of = _coerce_timestamp(payload.get("as_of_ts"))
    if as_of is not None:
        return as_of, manifest_path

    # Back-compat when only run_as_of_ts was written to run summaries.
    fallback = _coerce_timestamp(payload.get("run_as_of_ts"))
    return fallback, manifest_path


def _log_depth_chart_prior(diagnostics: dict[str, Any]) -> None:
    if not isinstance(diagnostics, dict):
        return
    logger.info(
        "[dc-prior] applied=%s reason=%s matched_id=%s matched_name_fallback=%s unmatched=%s snapshot=%s",
        diagnostics.get("applied"),
        diagnostics.get("reason"),
        diagnostics.get("matched_id"),
        diagnostics.get("matched_name_fallback"),
        diagnostics.get("unmatched"),
        diagnostics.get("dc_snapshot_ts"),
    )
    if diagnostics.get("role_distribution"):
        logger.info("[dc-prior] role_distribution=%s", diagnostics.get("role_distribution"))
    if diagnostics.get("top_rotation_prob_deltas"):
        logger.info("[dc-prior] top_rotation_prob_deltas=%s", diagnostics.get("top_rotation_prob_deltas"))
    if diagnostics.get("top_play_prob_deltas"):
        logger.info("[dc-prior] top_play_prob_deltas=%s", diagnostics.get("top_play_prob_deltas"))
    if diagnostics.get("cap_hits_by_role"):
        logger.info("[dc-cap] cap_hits_by_role=%s", diagnostics.get("cap_hits_by_role"))
    if diagnostics.get("largest_q_reductions"):
        logger.info("[dc-cap] largest_q_reductions=%s", diagnostics.get("largest_q_reductions"))
    if diagnostics.get("model_vs_depth_disagreements"):
        logger.info("[dc-disagree] top=%s", diagnostics.get("model_vs_depth_disagreements"))
    if diagnostics.get("dnp_guardrail"):
        logger.info("[dnp-guardrail] %s", diagnostics.get("dnp_guardrail"))
    if diagnostics.get("has_alerts"):
        logger.warning(
            "[dc-alert] flags=%s matched_rate=%s snapshot_age_minutes=%s",
            diagnostics.get("alert_flags"),
            diagnostics.get("matched_rate"),
            diagnostics.get("snapshot_age_minutes"),
        )


def _log_depth_chart_crosswalk(diag: dict[str, Any]) -> None:
    logger.info("[dc-crosswalk] %s", summarize_crosswalk_json(diag))
    if not bool(diag.get("applied", False)):
        logger.warning("[dc-alert] crosswalk_not_applied reason=%s", diag.get("reason"))
        return
    match_rate = diag.get("match_rate")
    if match_rate is None:
        return
    try:
        threshold = float(os.environ.get("PROJECTIONS_DC_CROSSWALK_WARN_MIN_MATCH_RATE", "0.30"))
    except ValueError:
        threshold = 0.30
    if float(match_rate) < threshold:
        logger.warning(
            "[dc-alert] crosswalk_low_match_rate rate=%.3f threshold=%.3f matched_rows=%s unmatched_snapshot_rows=%s",
            float(match_rate),
            threshold,
            diag.get("matched_rows"),
            diag.get("unmatched_snapshot_rows"),
        )


def build_effective_minutes(
    *,
    game_date: date,
    minutes_df: pd.DataFrame,
    data_root: Path,
    source: str = "gameview",
    run_as_of_ts: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply authorized overrides to minutes and emit a structured diff summary."""
    from projections.ops.overrides import apply_overrides_to_minutes_df

    overrides_map = load_overrides_map(game_date, data_root=data_root)
    before = minutes_df.copy()
    after = apply_overrides_to_minutes_df(
        before,
        game_date=game_date,
        data_root=data_root,
        log_diagnostics=True,
        force_reconcile=True,
    )
    crosswalk_diag = refresh_realgm_player_crosswalk_from_minutes(
        after,
        data_root=data_root,
        as_of_ts=run_as_of_ts,
    )
    _log_depth_chart_crosswalk(crosswalk_diag)
    depth_prior = apply_depth_chart_prior_from_realgm(
        after,
        data_root=data_root,
        as_of_ts=run_as_of_ts,
    )
    after = depth_prior.frame
    _log_depth_chart_prior(depth_prior.diagnostics)

    key_cols = [c for c in ("game_id", "player_id", "player_name", "team_id", "team_tricode") if c in before.columns]
    tracked_cols = [
        c
        for c in (
            "status",
            "play_prob",
            "rotation_prob",
            "is_confirmed_starter",
            "is_projected_starter",
            "minutes_p10",
            "minutes_p50",
            "minutes_p90",
            "minutes_p10_cond",
            "minutes_p50_cond",
            "minutes_p90_cond",
        )
        if c in before.columns and c in after.columns
    ]

    diff_rows: list[dict[str, Any]] = []
    if key_cols and tracked_cols and not before.empty and not after.empty:
        left = before[key_cols + tracked_cols].copy()
        right = after[key_cols + tracked_cols].copy()
        for col in key_cols:
            left[col] = left[col].astype(str)
            right[col] = right[col].astype(str)

        merged = left.merge(right, on=key_cols, how="outer", suffixes=("_before", "_after"), indicator=True)
        for col in tracked_cols:
            merged[col] = merged[f"{col}_before"].where(merged[f"{col}_before"].notna(), merged[f"{col}_after"])
            merged[f"{col}_changed"] = merged[f"{col}_before"] != merged[f"{col}_after"]

        changed_cols = [f"{c}_changed" for c in tracked_cols]
        changed = merged[changed_cols].any(axis=1)
        for _, row in merged.loc[changed].iterrows():
            record: dict[str, Any] = {"source": source}
            for col in key_cols:
                record[col] = row[col]
            changes: dict[str, Any] = {}
            for col in tracked_cols:
                if not bool(row.get(f"{col}_changed", False)):
                    continue
                before_val = row.get(f"{col}_before")
                after_val = row.get(f"{col}_after")
                changes[col] = {"before": None if pd.isna(before_val) else before_val, "after": None if pd.isna(after_val) else after_val}
            record["changes"] = changes
            # If an ops override exists for this (game_id, player_id), include metadata.
            gid = str(record.get("game_id") or "")
            pid = str(record.get("player_id") or "")
            override_key = next((k for k in overrides_map.keys() if k.game_id == gid and k.player_id == pid), None)
            if override_key is not None:
                ov = overrides_map.get(override_key, {})
                record["override_updated_at"] = ov.get("updated_at")
                record["override_note"] = ov.get("note")
                record["override_fields"] = (ov.get("fields") if isinstance(ov.get("fields"), dict) else None)
            diff_rows.append(record)

    summary = {
        "version": 1,
        "game_date": game_date.isoformat(),
        "generated_at": _utc_now_iso(),
        "source": source,
        "overrides_count": len(overrides_map),
        "changed_players": len(diff_rows),
        "diffs": diff_rows,
        "run_as_of_ts": run_as_of_ts.isoformat().replace("+00:00", "Z") if run_as_of_ts is not None else None,
        "depth_chart_crosswalk": crosswalk_diag,
        "depth_chart_prior": depth_prior.diagnostics,
        "depth_chart_alerts": list(depth_prior.diagnostics.get("alert_flags") or []),
    }
    return after, summary


def write_effective_minutes_layer(
    *,
    game_date: date,
    minutes_path: Path,
    out_dir: Path,
    data_root: Path,
    source: str = "gameview",
) -> EffectiveInputsResult:
    """Load minutes parquet, write effective_minutes.parquet + effective_inputs_summary.json."""
    df = pd.read_parquet(minutes_path)
    run_dir = minutes_path.parent
    run_as_of_ts, manifest_path = _load_run_as_of_ts(run_dir)
    if manifest_path is not None:
        logger.info(
            "[effective-inputs] manifest=%s run_as_of_ts=%s",
            manifest_path,
            run_as_of_ts.isoformat().replace("+00:00", "Z") if run_as_of_ts is not None else None,
        )
    effective, summary = build_effective_minutes(
        game_date=game_date,
        minutes_df=df,
        data_root=data_root,
        source=source,
        run_as_of_ts=run_as_of_ts,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    eff_path = out_dir / EFFECTIVE_MINUTES_FILENAME
    effective.to_parquet(eff_path, index=False)

    summary_path = out_dir / EFFECTIVE_INPUTS_SUMMARY
    _atomic_write_json(summary_path, summary)

    return EffectiveInputsResult(
        effective_minutes_path=eff_path,
        summary_path=summary_path,
        overrides_count=int(summary.get("overrides_count", 0)),
    )


def build_effective_rates(
    *,
    game_date: date,
    rates_df: pd.DataFrame,
    data_root: Path,
    source: str = "gameview",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply authorized overrides to rates and emit a structured diff summary."""
    from projections.ops.overrides import USAGE_RATE_FIELDS, apply_overrides_to_rates_df

    overrides_map = load_overrides_map(game_date, data_root=data_root)
    before = rates_df.copy()
    after = apply_overrides_to_rates_df(before, game_date=game_date, data_root=data_root)

    key_cols = [c for c in ("game_id", "player_id", "player_name", "team_id", "team_tricode") if c in before.columns]
    tracked_cols = [c for c in USAGE_RATE_FIELDS if c in before.columns and c in after.columns]

    diff_rows: list[dict[str, Any]] = []
    if key_cols and tracked_cols and not before.empty and not after.empty:
        left = before[key_cols + tracked_cols].copy()
        right = after[key_cols + tracked_cols].copy()
        for col in key_cols:
            left[col] = left[col].astype(str)
            right[col] = right[col].astype(str)

        merged = left.merge(right, on=key_cols, how="outer", suffixes=("_before", "_after"), indicator=True)
        for col in tracked_cols:
            merged[col] = merged[f"{col}_before"].where(merged[f"{col}_before"].notna(), merged[f"{col}_after"])
            merged[f"{col}_changed"] = merged[f"{col}_before"] != merged[f"{col}_after"]

        changed_cols = [f"{c}_changed" for c in tracked_cols]
        changed = merged[changed_cols].any(axis=1)
        for _, row in merged.loc[changed].iterrows():
            record: dict[str, Any] = {"source": source}
            for col in key_cols:
                record[col] = row[col]
            changes: dict[str, Any] = {}
            for col in tracked_cols:
                if not bool(row.get(f"{col}_changed", False)):
                    continue
                before_val = row.get(f"{col}_before")
                after_val = row.get(f"{col}_after")
                changes[col] = {
                    "before": None if pd.isna(before_val) else before_val,
                    "after": None if pd.isna(after_val) else after_val,
                }
            record["changes"] = changes
            diff_rows.append(record)

    summary = {
        "version": 1,
        "game_date": game_date.isoformat(),
        "generated_at": _utc_now_iso(),
        "source": source,
        "overrides_count": len(overrides_map),
        "changed_players": len(diff_rows),
        "diffs": diff_rows,
    }
    return after, summary


def write_effective_rates_layer(
    *,
    game_date: date,
    rates_path: Path,
    out_dir: Path,
    data_root: Path,
    source: str = "gameview",
) -> Path:
    """Load rates parquet, write effective_rates.parquet."""
    df = pd.read_parquet(rates_path)
    effective, _summary = build_effective_rates(game_date=game_date, rates_df=df, data_root=data_root, source=source)
    out_dir.mkdir(parents=True, exist_ok=True)
    eff_path = out_dir / EFFECTIVE_RATES_FILENAME
    effective.to_parquet(eff_path, index=False)
    return eff_path
