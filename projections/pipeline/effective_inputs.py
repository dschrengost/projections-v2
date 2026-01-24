"""Build effective inputs consumed by downstream live pipeline stages.

Effective inputs apply manual overrides from the *authorized* source and
materialize a deterministic parquet layer for downstream consumption.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from projections.ops.overrides import load_overrides_map

EFFECTIVE_MINUTES_FILENAME = "effective_minutes.parquet"
EFFECTIVE_RATES_FILENAME = "effective_rates.parquet"
EFFECTIVE_INPUTS_SUMMARY = "effective_inputs_summary.json"


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


def build_effective_minutes(
    *,
    game_date: date,
    minutes_df: pd.DataFrame,
    data_root: Path,
    source: str = "gameview",
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

    key_cols = [c for c in ("game_id", "player_id", "player_name", "team_id", "team_tricode") if c in before.columns]
    tracked_cols = [
        c
        for c in (
            "status",
            "play_prob",
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
    effective, summary = build_effective_minutes(game_date=game_date, minutes_df=df, data_root=data_root, source=source)

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
