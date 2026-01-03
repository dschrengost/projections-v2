from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from projections import paths

OPS_OVERRIDES_VERSION = 1

# Curated set: usage + efficiency predictions (rates_v1 outputs).
USAGE_RATE_FIELDS: tuple[str, ...] = (
    "pred_fga2_per_min",
    "pred_fga3_per_min",
    "pred_fta_per_min",
    "pred_ast_per_min",
    "pred_tov_per_min",
    "pred_oreb_per_min",
    "pred_dreb_per_min",
    "pred_stl_per_min",
    "pred_blk_per_min",
    "pred_fg2_pct",
    "pred_fg3_pct",
    "pred_ft_pct",
)

MINUTES_FIELDS: tuple[str, ...] = (
    "minutes_p10",
    "minutes_p50",
    "minutes_p90",
    "minutes_p10_cond",
    "minutes_p50_cond",
    "minutes_p90_cond",
    "minutes_delta",  # Additive adjustment to model quantiles (e.g., +5 or -3)
    "play_prob",
    "status",
    "is_confirmed_starter",
    "is_projected_starter",
)

OPS_OVERRIDE_FIELDS: tuple[str, ...] = tuple(sorted(set(USAGE_RATE_FIELDS + MINUTES_FIELDS)))
STICKY_FIELDS_KEY = "sticky_fields"


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_id_str_series(series: pd.Series) -> pd.Series:
    """Normalize identifier values to stable string tokens (e.g., 123.0 -> '123')."""

    if series.empty:
        return pd.Series([], index=series.index, dtype="string")

    # Start with string view (preserves non-numeric ids).
    out = series.astype("string", copy=False).fillna("")

    # For numeric ids stored as floats (common when parquet contains NA), coerce to Int64
    # when integer-like to avoid '123.0' string mismatches.
    numeric = pd.to_numeric(series, errors="coerce")
    int_like = numeric.notna() & (numeric % 1 == 0)
    if int_like.any():
        out = out.where(~int_like, numeric.where(int_like).astype("Int64").astype("string"))

    # Defensive cleanup for any remaining '.0' suffixes.
    out = out.str.replace(r"\.0$", "", regex=True)
    return out


def _ops_key_for_df(df: pd.DataFrame) -> pd.Series:
    return _normalize_id_str_series(df["game_id"]) + "|" + _normalize_id_str_series(df["player_id"])


def _overrides_dir(data_root: Path, game_date: date) -> Path:
    return data_root / "artifacts" / "ops" / "overrides_v1" / f"game_date={game_date.isoformat()}"


def overrides_path(game_date: date, *, data_root: Path | None = None) -> Path:
    root = data_root or paths.data_path()
    return _overrides_dir(root, game_date) / "overrides.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".tmp.{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}.json")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


@dataclass(frozen=True, slots=True)
class OverrideKey:
    game_id: str
    player_id: str

    @classmethod
    def from_values(cls, game_id: Any, player_id: Any) -> "OverrideKey":
        return cls(game_id=str(game_id), player_id=str(player_id))


def _coerce_override_field(name: str, value: Any) -> Any:
    if value is None:
        return None

    if name in {"is_confirmed_starter", "is_projected_starter"}:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(int(value))
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        raise ValueError(f"Invalid boolean for {name}: {value!r}")

    if name == "status":
        if not isinstance(value, str):
            raise ValueError(f"Invalid status: {value!r}")
        return value.strip()

    if name.endswith("_pct"):
        numeric = float(value)
        return float(max(0.0, min(1.0, numeric)))

    if name.startswith("pred_") or name.startswith("minutes_") or name == "play_prob":
        numeric = float(value)
        if name == "minutes_delta":
            # Delta can be negative, clip to [-48, 48]
            return float(max(-48.0, min(48.0, numeric)))
        if name.startswith("minutes_"):
            return float(max(0.0, min(48.0, numeric)))
        if name == "play_prob":
            return float(max(0.0, min(1.0, numeric)))
        return float(max(0.0, numeric))

    return value


def _normalize_override_fields(fields: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in fields.items():
        if key not in OPS_OVERRIDE_FIELDS:
            continue
        if value is None:
            continue
        normalized[key] = _coerce_override_field(key, value)
    return normalized


def load_overrides_map(game_date: date, *, data_root: Path | None = None) -> dict[OverrideKey, dict[str, Any]]:
    path = overrides_path(game_date, data_root=data_root)
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("overrides", [])
    else:
        return {}

    overrides: dict[OverrideKey, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        game_id = item.get("game_id")
        player_id = item.get("player_id")
        if game_id is None or player_id is None:
            continue
        fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
        normalized = _normalize_override_fields(fields)
        if not normalized:
            continue
        sticky_fields_raw = item.get(STICKY_FIELDS_KEY)
        sticky_fields: list[str] = []
        if isinstance(sticky_fields_raw, list):
            sticky_fields = [str(x) for x in sticky_fields_raw if x is not None]
        record = {
            "game_id": str(game_id),
            "player_id": str(player_id),
            "fields": normalized,
            "updated_at": str(item.get("updated_at") or ""),
            "note": item.get("note"),
            STICKY_FIELDS_KEY: sticky_fields,
        }
        overrides[OverrideKey.from_values(game_id, player_id)] = record

    return overrides


def list_overrides(game_date: date, *, data_root: Path | None = None) -> list[dict[str, Any]]:
    overrides = load_overrides_map(game_date, data_root=data_root)
    return sorted(overrides.values(), key=lambda r: (r.get("game_id", ""), r.get("player_id", "")))


def upsert_overrides(
    game_date: date,
    updates: Iterable[dict[str, Any]],
    *,
    note: str | None = None,
    data_root: Path | None = None,
) -> list[dict[str, Any]]:
    overrides = load_overrides_map(game_date, data_root=data_root)
    now = _utc_now_iso()

    for update in updates:
        if not isinstance(update, dict):
            continue
        game_id = update.get("game_id")
        player_id = update.get("player_id")
        if game_id is None or player_id is None:
            continue

        sticky_fields: list[str] | None = None
        sticky_fields_raw = update.get(STICKY_FIELDS_KEY)
        if isinstance(sticky_fields_raw, list):
            sticky_fields = [str(x) for x in sticky_fields_raw if x is not None]

        fields = update.get("fields") if isinstance(update.get("fields"), dict) else update
        normalized = _normalize_override_fields(fields)
        if not normalized:
            continue
        key = OverrideKey.from_values(game_id, player_id)
        existing = overrides.get(key)
        merged_fields = dict(existing.get("fields", {}) if existing else {})
        merged_fields.update(normalized)
        merged_sticky = list(existing.get(STICKY_FIELDS_KEY, []) if existing else [])
        if sticky_fields is not None:
            merged_sticky = sticky_fields
        record = {
            "game_id": str(game_id),
            "player_id": str(player_id),
            "fields": merged_fields,
            "updated_at": now,
            "note": note or (existing.get("note") if existing else None),
            STICKY_FIELDS_KEY: merged_sticky,
        }
        overrides[key] = record

    sorted_overrides = sorted(overrides.values(), key=lambda r: (r.get("game_id", ""), r.get("player_id", "")))
    payload = {
        "version": OPS_OVERRIDES_VERSION,
        "game_date": game_date.isoformat(),
        "updated_at": now,
        "overrides": sorted_overrides,
    }
    _atomic_write_json(overrides_path(game_date, data_root=data_root), payload)
    return payload["overrides"]


def clear_overrides(
    game_date: date,
    *,
    game_id: str | int | None = None,
    player_id: str | int | None = None,
    data_root: Path | None = None,
) -> list[dict[str, Any]]:
    existing = load_overrides_map(game_date, data_root=data_root)

    if game_id is None and player_id is None:
        path = overrides_path(game_date, data_root=data_root)
        if path.exists():
            path.unlink()
        return []

    game_id_s = str(game_id) if game_id is not None else None
    player_id_s = str(player_id) if player_id is not None else None

    kept: dict[OverrideKey, dict[str, Any]] = {}
    for key, record in existing.items():
        if game_id_s is not None and key.game_id != game_id_s:
            kept[key] = record
            continue
        if player_id_s is not None and key.player_id != player_id_s:
            kept[key] = record
            continue

    now = _utc_now_iso()
    payload = {
        "version": OPS_OVERRIDES_VERSION,
        "game_date": game_date.isoformat(),
        "updated_at": now,
        "overrides": sorted(kept.values(), key=lambda r: (r.get("game_id", ""), r.get("player_id", ""))),
    }
    _atomic_write_json(overrides_path(game_date, data_root=data_root), payload)
    return payload["overrides"]


def clear_override_fields(
    game_date: date,
    *,
    game_id: str | int,
    player_id: str | int,
    fields: Iterable[str],
    data_root: Path | None = None,
) -> dict[str, Any] | None:
    """Remove specific override fields for a single player/game (keeps other fields)."""
    overrides = load_overrides_map(game_date, data_root=data_root)
    key = OverrideKey.from_values(game_id, player_id)
    record = overrides.get(key)
    if not record:
        return None

    field_set = set(fields)
    current_fields = record.get("fields", {}) if isinstance(record.get("fields"), dict) else {}
    next_fields = {k: v for k, v in current_fields.items() if k not in field_set}
    record[STICKY_FIELDS_KEY] = [
        f for f in (record.get(STICKY_FIELDS_KEY) or []) if isinstance(f, str) and f not in field_set
    ]

    now = _utc_now_iso()
    if next_fields:
        record["fields"] = next_fields
        record["updated_at"] = now
        overrides[key] = record
    else:
        overrides.pop(key, None)

    payload = {
        "version": OPS_OVERRIDES_VERSION,
        "game_date": game_date.isoformat(),
        "updated_at": now,
        "overrides": sorted(overrides.values(), key=lambda r: (r.get("game_id", ""), r.get("player_id", ""))),
    }
    _atomic_write_json(overrides_path(game_date, data_root=data_root), payload)
    return record if next_fields else None


def auto_clear_nonsticky_confirmed_starters_from_rotowire(
    *,
    game_date: date,
    roster_df: pd.DataFrame,
    rotowire_df: pd.DataFrame,
    data_root: Path | None = None,
) -> int:
    """Auto-clear non-sticky is_confirmed_starter overrides when Rotowire confirms a lineup and conflicts.

    Rule: for any team with at least 1 `lineup_role == confirmed_starter`, treat that team as \"confirmed\".
    For those teams, any player on that team *not* in the confirmed starter names list will have
    `is_confirmed_starter` override cleared unless it's marked sticky.
    """
    if roster_df.empty or rotowire_df.empty:
        return 0
    if "lineup_role" not in rotowire_df.columns:
        return 0
    if "player_name" not in rotowire_df.columns:
        return 0
    if "team_abbreviation" not in rotowire_df.columns:
        return 0
    if "player_name" not in roster_df.columns or "player_id" not in roster_df.columns or "game_id" not in roster_df.columns:
        return 0

    role = rotowire_df["lineup_role"].fillna("").astype(str).str.strip().str.lower()
    confirmed = rotowire_df.loc[role.eq("confirmed_starter")].copy()
    if confirmed.empty:
        return 0

    teams_confirmed = set(confirmed["team_abbreviation"].dropna().astype(str).str.strip().str.upper().unique())
    confirmed_names = set(confirmed["player_name"].astype(str).str.strip().str.lower().unique())

    roster_work = roster_df.copy()
    team_col = "team_tricode" if "team_tricode" in roster_work.columns else None
    if team_col is None:
        return 0
    roster_teams = roster_work[team_col].astype(str).str.strip().str.upper()
    roster_names = roster_work["player_name"].astype(str).str.strip().str.lower()
    in_confirmed_team = roster_teams.isin(teams_confirmed)
    not_in_official_starters = ~roster_names.isin(confirmed_names)
    impacted_mask = in_confirmed_team & not_in_official_starters
    if not impacted_mask.any():
        return 0

    overrides = load_overrides_map(game_date, data_root=data_root)
    cleared = 0
    for _, row in roster_work.loc[impacted_mask, ["game_id", "player_id"]].drop_duplicates().iterrows():
        key = OverrideKey.from_values(row["game_id"], row["player_id"])
        record = overrides.get(key)
        if not record:
            continue
        fields = record.get("fields", {}) if isinstance(record.get("fields"), dict) else {}
        if "is_confirmed_starter" not in fields:
            continue
        sticky_fields = set(record.get(STICKY_FIELDS_KEY) or [])
        if "is_confirmed_starter" in sticky_fields:
            continue
        clear_override_fields(
            game_date,
            game_id=str(row["game_id"]),
            player_id=str(row["player_id"]),
            fields=["is_confirmed_starter"],
            data_root=data_root,
        )
        cleared += 1

    return cleared

def _overrides_as_frame(
    overrides: dict[OverrideKey, dict[str, Any]],
    *,
    include_fields: Iterable[str],
) -> pd.DataFrame:
    if not overrides:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    include = set(include_fields)
    for record in overrides.values():
        fields = record.get("fields", {}) if isinstance(record.get("fields"), dict) else {}
        selected = {k: v for k, v in fields.items() if k in include}
        row = {
            "_ops_key": f"{record.get('game_id')}|{record.get('player_id')}",
            **selected,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _distribute_delta_with_caps(
    base: pd.Series,
    delta: float,
    *,
    lower: float = 0.0,
    upper: float = 48.0,
) -> pd.Series:
    """Distribute delta across base proportionally, respecting per-element caps."""
    if base.empty or abs(delta) < 1e-9:
        return base

    values = base.astype(float).copy()
    remaining = float(delta)

    for _ in range(5):
        if abs(remaining) < 1e-6:
            break

        if remaining > 0:
            headroom = (upper - values).clip(lower=0.0)
            eligible = headroom > 1e-9
            if not eligible.any():
                break
            weights = values.where(eligible, 0.0)
            if weights.sum() <= 1e-9:
                weights = headroom.where(eligible, 0.0)
            share = weights / weights.sum()
            inc = (remaining * share).clip(upper=headroom)
            values = values + inc
            remaining -= float(inc.sum())
        else:
            reducible = (values - lower).clip(lower=0.0)
            eligible = reducible > 1e-9
            if not eligible.any():
                break
            weights = values.where(eligible, 0.0)
            if weights.sum() <= 1e-9:
                weights = reducible.where(eligible, 0.0)
            share = weights / weights.sum()
            dec = ((-remaining) * share).clip(upper=reducible)
            values = values - dec
            remaining += float(dec.sum())

    return values


def _reconcile_minutes_after_overrides(
    df: pd.DataFrame,
    *,
    locked_mask: pd.Series,
    target_team_minutes: float = 240.0,
) -> pd.DataFrame:
    """Reconcile team minutes back to 240 without changing locked players."""
    if df.empty or not {"game_id", "team_id", "minutes_p50"} <= set(df.columns):
        return df

    work = df.copy()
    work["minutes_p50"] = pd.to_numeric(work["minutes_p50"], errors="coerce").fillna(0.0).astype(float)
    locked_mask = locked_mask.reindex(work.index).fillna(False).astype(bool)

    status_lower = work.get("status")
    if status_lower is not None:
        status_lower = status_lower.astype(str).str.strip().str.lower()
    else:
        status_lower = pd.Series("", index=work.index, dtype=str)

    quant_cols = [
        c
        for c in (
            "minutes_p10",
            "minutes_p90",
            "minutes_p10_cond",
            "minutes_p50_cond",
            "minutes_p90_cond",
        )
        if c in work.columns
    ]
    for col in quant_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0).astype(float)

    for (_, _), group_idx in work.groupby(["game_id", "team_id"], sort=False).groups.items():
        idx = pd.Index(group_idx)
        g = work.loc[idx]
        g_locked = locked_mask.loc[idx]

        # Only adjust non-OUT players; keep OUT at 0.
        g_out = status_lower.loc[idx].eq("out")
        adjustable = (~g_locked) & (~g_out) & (g["minutes_p50"] > 0)
        if not adjustable.any():
            continue

        total = float(g["minutes_p50"].sum())
        delta = float(target_team_minutes - total)
        if abs(delta) < 1e-6:
            continue

        base = g.loc[adjustable, "minutes_p50"]
        new_vals = _distribute_delta_with_caps(base, delta, lower=0.0, upper=48.0)
        p50_delta = new_vals - base
        work.loc[new_vals.index, "minutes_p50"] = new_vals

        for col in quant_cols:
            work.loc[new_vals.index, col] = (work.loc[new_vals.index, col] + p50_delta).clip(lower=0.0, upper=48.0)

    if {"minutes_p10", "minutes_p50"} <= set(work.columns):
        work["minutes_p10"] = work["minutes_p10"].clip(upper=work["minutes_p50"])
        if "minutes_p10_cond" in work.columns:
            work["minutes_p10_cond"] = work["minutes_p10_cond"].clip(upper=work["minutes_p50"])
    if {"minutes_p90", "minutes_p50"} <= set(work.columns):
        work["minutes_p90"] = work["minutes_p90"].clip(lower=work["minutes_p50"])
        if "minutes_p90_cond" in work.columns:
            work["minutes_p90_cond"] = work["minutes_p90_cond"].clip(lower=work["minutes_p50"])
    if "minutes_p50_cond" in work.columns:
        work["minutes_p50_cond"] = work["minutes_p50_cond"].clip(lower=0.0, upper=48.0)

    return work


def apply_overrides_to_minutes_df(
    minutes_df: pd.DataFrame,
    *,
    game_date: date,
    data_root: Path | None = None,
    reconcile_team_minutes: bool = True,
) -> pd.DataFrame:
    """Apply authoritative ops overrides to a minutes projections frame.

    Note: if minutes overrides change team totals, we reconcile minutes_p50 back to 240
    (within each game/team) so downstream sims/validation remain stable.
    """
    overrides = load_overrides_map(game_date, data_root=data_root)
    if minutes_df.empty or not overrides:
        return minutes_df

    ops_df = _overrides_as_frame(overrides, include_fields=MINUTES_FIELDS)
    if ops_df.empty:
        return minutes_df

    work = minutes_df.copy()
    work["_ops_key"] = _ops_key_for_df(work)
    merged = work.merge(ops_df, on="_ops_key", how="left", suffixes=("", "_ops"))

    locked_mask = pd.Series(False, index=merged.index)

    # Track which quantiles were explicitly overridden (for downstream sim to preserve)
    for qcol in ("minutes_p10", "minutes_p90"):
        override_col = f"{qcol}_ops"
        flag_col = f"_{qcol}_overridden"
        if override_col in merged.columns:
            merged[flag_col] = merged[override_col].notna()
        else:
            merged[flag_col] = False

    for col in MINUTES_FIELDS:
        if col not in ops_df.columns:
            continue
        override_col = f"{col}_ops"
        if override_col not in merged.columns:
            continue
        if col in merged.columns:
            merged[col] = merged[override_col].where(merged[override_col].notna(), merged[col])
        else:
            merged[col] = merged[override_col]
        if col == "minutes_p50":
            locked_mask = locked_mask | merged[override_col].notna()

    # Apply minutes_delta as additive adjustment (takes precedence over raw quantile overrides)
    # Note: minutes_delta only exists in ops_df (never in original minutes_df), so after merge
    # it keeps its original name (no _ops suffix) because suffixes only apply to colliding columns.
    delta_col = "minutes_delta" if "minutes_delta" in merged.columns else "minutes_delta_ops"
    if delta_col in merged.columns:
        has_delta = merged[delta_col].notna()
        if has_delta.any():
            delta_vals = merged.loc[has_delta, delta_col].astype(float)
            for qcol in ("minutes_p10", "minutes_p50", "minutes_p90", "minutes_p10_cond", "minutes_p50_cond", "minutes_p90_cond"):
                if qcol in merged.columns:
                    merged.loc[has_delta, qcol] = (
                        merged.loc[has_delta, qcol].astype(float) + delta_vals
                    ).clip(0.0, 48.0)
            # Mark delta-adjusted players as locked and with overridden quantiles
            locked_mask = locked_mask | has_delta
            merged.loc[has_delta, "_minutes_p10_overridden"] = True
            merged.loc[has_delta, "_minutes_p90_overridden"] = True

    # If operator marks player OUT, force play_prob=0 and minutes=0 (keep row for downstream).
    status_raw = merged.get("status")
    if status_raw is not None:
        status_lower = status_raw.astype(str).str.strip().str.lower()
        is_out = status_lower.eq("out")
        if is_out.any():
            merged.loc[is_out, "play_prob"] = 0.0
            for mcol in ("minutes_p10", "minutes_p50", "minutes_p90", "minutes_p10_cond", "minutes_p50_cond", "minutes_p90_cond"):
                if mcol in merged.columns:
                    merged.loc[is_out, mcol] = 0.0
            locked_mask = locked_mask | is_out

    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_ops")] + ["_ops_key"], errors="ignore")

    if reconcile_team_minutes and {"game_id", "team_id", "minutes_p50"} <= set(merged.columns):
        merged = _reconcile_minutes_after_overrides(merged, locked_mask=locked_mask, target_team_minutes=240.0)

    return merged


def apply_overrides_to_rates_df(rates_df: pd.DataFrame, *, game_date: date, data_root: Path | None = None) -> pd.DataFrame:
    """Apply authoritative ops overrides to a rates_v1 predictions frame."""
    overrides = load_overrides_map(game_date, data_root=data_root)
    if rates_df.empty or not overrides:
        return rates_df

    ops_df = _overrides_as_frame(overrides, include_fields=USAGE_RATE_FIELDS)
    if ops_df.empty:
        return rates_df

    work = rates_df.copy()
    work["_ops_key"] = _ops_key_for_df(work)
    merged = work.merge(ops_df, on="_ops_key", how="left", suffixes=("", "_ops"))

    for col in USAGE_RATE_FIELDS:
        if col not in ops_df.columns:
            continue
        override_col = f"{col}_ops"
        if override_col not in merged.columns:
            continue
        if col in merged.columns:
            merged[col] = merged[override_col].where(merged[override_col].notna(), merged[col])
        else:
            merged[col] = merged[override_col]

    return merged.drop(columns=[c for c in merged.columns if c.endswith("_ops")] + ["_ops_key"], errors="ignore")
