from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from projections import paths
from projections.minutes.reconcile import (
    IN_ROTATION_THRESHOLD_MIN,
    MINUTES_CONTRACT_VERSION,
    minutes_contract_hash,
    reconcile_team_minutes,
)

OPS_OVERRIDES_VERSION = 1

# Gameview-only manual classification for UI + downstream heuristics.
# This is persisted alongside minutes/rates overrides so it survives future flows.
OPS_DEPTH_ROLE_FIELD = "ops_depth_role"
OPS_DEPTH_ROLE_VALUES: tuple[str, ...] = ("starter", "rotation", "deep_bench", "out")

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
    # Stage 1A: predictable hard targets + locks (authoritative control plane).
    "minutes_target",  # Absolute minutes target (0-48). Implies lock in effective layer + sim allocator.
    "minutes_lock",  # If True, fix minutes to target (or current minutes when target unset).
    "minutes_delta",  # Additive adjustment to model quantiles (e.g., +5 or -3)
    "play_prob",
    "status",
    "is_confirmed_starter",
    "is_projected_starter",
    OPS_DEPTH_ROLE_FIELD,
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

    if name == "minutes_lock":
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

    if name == OPS_DEPTH_ROLE_FIELD:
        if not isinstance(value, str):
            raise ValueError(f"Invalid {OPS_DEPTH_ROLE_FIELD}: {value!r}")
        normalized = value.strip().lower()
        if normalized not in set(OPS_DEPTH_ROLE_VALUES):
            raise ValueError(
                f"Invalid {OPS_DEPTH_ROLE_FIELD}: {value!r} (expected one of {OPS_DEPTH_ROLE_VALUES})"
            )
        return normalized

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
    log_diagnostics: bool = False,
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

    diags: list[tuple[Any, Any, Any]] = []
    for (_, _), group_idx in work.groupby(["game_id", "team_id"], sort=False).groups.items():
        idx = pd.Index(group_idx)
        g = work.loc[idx]
        g_locked = locked_mask.loc[idx]

        # Only adjust non-OUT players; keep OUT at 0.
        g_out = status_lower.loc[idx].eq("out")
        if g_out.any():
            work.loc[idx[g_out], "minutes_p50"] = 0.0
            for col in quant_cols:
                work.loc[idx[g_out], col] = 0.0

        minutes_before = work.loc[idx, "minutes_p50"].astype(float)
        reconciled, diag = reconcile_team_minutes(
            g,
            float(target_team_minutes),
            minutes_col="minutes_p50",
            cap_col=None,
            weight_col=None,
            state_col=None,
            locked_mask=(g_locked | g_out),
            in_rotation_threshold_min=float(IN_ROTATION_THRESHOLD_MIN),
            default_cap=48.0,
            max_passes=5,
            eps=1e-6,
        )
        work.loc[idx, "minutes_p50"] = reconciled
        p50_delta = reconciled - minutes_before
        for col in quant_cols:
            work.loc[idx, col] = (work.loc[idx, col].astype(float) + p50_delta).clip(lower=0.0, upper=48.0)
        if log_diagnostics:
            diags.append((g.iloc[0].get("game_id"), g.iloc[0].get("team_id"), diag))

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

    if log_diagnostics and diags:
        for gid, tid, d in diags:
            print(
                "[minutes-reconcile] game_id=%s team_id=%s pre=%.2f post=%.2f residual=%.4f "
                "passes=%d cap_infeasible=%s locked_infeasible=%s out=%d cameo=%d rotation=%d cap_hits=%d"
                % (
                    gid,
                    tid,
                    float(d.pre_sum),
                    float(d.post_sum),
                    float(d.residual),
                    int(d.passes),
                    bool(d.cap_infeasible),
                    bool(d.locked_infeasible),
                    int(d.n_out_or_dnp),
                    int(d.n_cameo),
                    int(d.n_rotation),
                    int(d.n_cap_hits),
                )
            )

    return work


def _reconcile_effective_minutes_after_overrides(
    df: pd.DataFrame,
    *,
    locked_mask: pd.Series,
    target_team_minutes: float = 240.0,
    log_diagnostics: bool = False,
) -> pd.DataFrame:
    """Reconcile effective minutes back to 240 without changing locked players.

    Notes:
    - We treat players with 0 effective_minutes (but not OUT) as DNP for reconcile purposes,
      so we don't "grow" the rotation when adjusting team totals.
    - This is primarily used for RMH outputs where the scorer produces an `effective_minutes`
      contract column alongside raw model quantiles.
    """
    if df.empty or not {"game_id", "team_id", "effective_minutes"} <= set(df.columns):
        return df

    work = df.copy()
    work["effective_minutes"] = (
        pd.to_numeric(work["effective_minutes"], errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    )
    locked_mask = locked_mask.reindex(work.index).fillna(False).astype(bool)

    status_raw = work.get("status")
    if status_raw is not None:
        status_lower = status_raw.astype(str).str.strip().str.lower()
    else:
        status_lower = pd.Series("", index=work.index, dtype=str)

    play_prob = pd.to_numeric(work.get("play_prob"), errors="coerce") if "play_prob" in work.columns else None
    if play_prob is None:
        play_prob = pd.Series(1.0, index=work.index, dtype=float)
    else:
        play_prob = play_prob.fillna(1.0).astype(float)

    is_out = status_lower.eq("out") | (play_prob <= 0.0)
    if is_out.any():
        work.loc[is_out, "effective_minutes"] = 0.0

    # Force a stable state semantics so 0-minute rows are treated as DNP.
    eff = work["effective_minutes"].astype(float)
    state = pd.Series("rotation", index=work.index, dtype="string")
    state.loc[is_out] = "out"
    state.loc[~is_out & (eff <= 0.0)] = "dnp"
    state.loc[~is_out & (eff > 0.0) & (eff < float(IN_ROTATION_THRESHOLD_MIN))] = "cameo"
    state.loc[~is_out & (eff >= float(IN_ROTATION_THRESHOLD_MIN))] = "rotation"
    work["_effective_minutes_state"] = state

    diags: list[tuple[Any, Any, Any]] = []
    for (_, _), group_idx in work.groupby(["game_id", "team_id"], sort=False).groups.items():
        idx = pd.Index(group_idx)
        g = work.loc[idx]
        g_locked = locked_mask.loc[idx] | is_out.loc[idx]

        reconciled, diag = reconcile_team_minutes(
            g,
            float(target_team_minutes),
            minutes_col="effective_minutes",
            cap_col=None,
            weight_col=None,
            state_col="_effective_minutes_state",
            locked_mask=g_locked,
            in_rotation_threshold_min=float(IN_ROTATION_THRESHOLD_MIN),
            default_cap=48.0,
            max_passes=5,
            eps=1e-6,
        )
        work.loc[idx, "effective_minutes"] = reconciled
        if log_diagnostics:
            diags.append((g.iloc[0].get("game_id"), g.iloc[0].get("team_id"), diag))

    work = work.drop(columns=["_effective_minutes_state"], errors="ignore")

    if log_diagnostics and diags:
        for gid, tid, d in diags:
            print(
                "[effective-minutes-reconcile] game_id=%s team_id=%s pre=%.2f post=%.2f residual=%.4f "
                "passes=%d cap_infeasible=%s locked_infeasible=%s out=%d cameo=%d rotation=%d cap_hits=%d"
                % (
                    gid,
                    tid,
                    float(d.pre_sum),
                    float(d.post_sum),
                    float(d.residual),
                    int(d.passes),
                    bool(d.cap_infeasible),
                    bool(d.locked_infeasible),
                    int(d.n_out_or_dnp),
                    int(d.n_cameo),
                    int(d.n_rotation),
                    int(d.n_cap_hits),
                )
            )

    return work


def _raise_locked_minutes_infeasible(
    df_team: pd.DataFrame,
    *,
    game_id: Any,
    team_id: Any,
    locked_sum: float,
    target_team_minutes: float,
) -> None:
    cols = [c for c in ("player_id", "player_name", "minutes_target_eff", "minutes_lock_eff", "minutes_delta", "status") if c in df_team.columns]
    preview = df_team.loc[df_team.get("minutes_lock_eff", False).astype(bool), cols].copy() if cols else pd.DataFrame()
    if not preview.empty and "minutes_target_eff" in preview.columns:
        preview["minutes_target_eff"] = pd.to_numeric(preview["minutes_target_eff"], errors="coerce")
        preview = preview.sort_values("minutes_target_eff", ascending=False)
    details = ""
    if not preview.empty:
        lines = []
        for _, row in preview.head(20).iterrows():
            pid = row.get("player_id")
            name = row.get("player_name") or ""
            tgt = row.get("minutes_target_eff")
            delta = row.get("minutes_delta")
            status = row.get("status")
            bits = [f"player_id={pid}", f"name={str(name)[:24]!r}"]
            if pd.notna(tgt):
                bits.append(f"target={float(tgt):.1f}")
            if pd.notna(delta):
                bits.append(f"delta={float(delta):+.1f}")
            if status:
                bits.append(f"status={status!r}")
            lines.append("  - " + " ".join(bits))
        details = "\n" + "\n".join(lines)
    raise ValueError(
        "[ops-overrides] infeasible locked minutes: "
        f"game_id={game_id} team_id={team_id} locked_sum={locked_sum:.1f} "
        f"target_team_minutes={float(target_team_minutes):.1f}. "
        "Reduce locked targets or unlock players." + details
    )


def _reconcile_minutes_after_overrides_hard_targets(
    df: pd.DataFrame,
    *,
    target_team_minutes: float = 240.0,
    log_diagnostics: bool = False,
) -> pd.DataFrame:
    """Reconcile team minutes back to 240, enforcing hard targets/locks.

    Rules:
    - Rows with minutes_lock_eff=True are treated as fixed at minutes_target_eff.
    - If sum(locked targets) > target_team_minutes for a team, raise (do not relax locks).
    - Remaining minutes are distributed across unlocked players via reconcile_team_minutes.
    """
    required = {"game_id", "team_id", "minutes_p50", "minutes_lock_eff", "minutes_target_eff"}
    if df.empty or not required <= set(df.columns):
        return df

    work = df.copy()
    work["minutes_p50"] = pd.to_numeric(work["minutes_p50"], errors="coerce").fillna(0.0).astype(float)
    work["minutes_lock_eff"] = work["minutes_lock_eff"].fillna(False).astype(bool)
    work["minutes_target_eff"] = pd.to_numeric(work["minutes_target_eff"], errors="coerce")

    status_raw = work.get("status")
    if status_raw is not None:
        status_lower = status_raw.astype(str).str.strip().str.lower()
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

    diags: list[tuple[Any, Any, Any]] = []
    for (_, _), group_idx in work.groupby(["game_id", "team_id"], sort=False).groups.items():
        idx = pd.Index(group_idx)
        g = work.loc[idx]

        # OUT rows are always fixed at 0.
        g_out = status_lower.loc[idx].eq("out")
        if g_out.any():
            work.loc[idx[g_out], "minutes_p50"] = 0.0
            for col in quant_cols:
                work.loc[idx[g_out], col] = 0.0

        lock = work.loc[idx, "minutes_lock_eff"].astype(bool) | g_out
        targets = work.loc[idx, "minutes_target_eff"]
        # If lock is set but target is missing, treat current minutes_p50 as the target.
        if (lock & targets.isna()).any():
            fill_mask = lock & targets.isna()
            work.loc[idx[fill_mask], "minutes_target_eff"] = work.loc[idx[fill_mask], "minutes_p50"]
            targets = work.loc[idx, "minutes_target_eff"]

        locked_sum = float(pd.to_numeric(targets.where(lock), errors="coerce").fillna(0.0).sum())
        if locked_sum > float(target_team_minutes) + 1e-6:
            _raise_locked_minutes_infeasible(
                work.loc[idx],
                game_id=g.iloc[0].get("game_id"),
                team_id=g.iloc[0].get("team_id"),
                locked_sum=locked_sum,
                target_team_minutes=float(target_team_minutes),
            )

        minutes_before = work.loc[idx, "minutes_p50"].astype(float)

        remaining = float(target_team_minutes) - locked_sum
        unlocked_mask = ~lock
        if unlocked_mask.any():
            sub_idx = idx[unlocked_mask.to_numpy(dtype=bool)]
            g_unlocked = work.loc[sub_idx]
            reconciled, diag = reconcile_team_minutes(
                g_unlocked,
                float(remaining),
                minutes_col="minutes_p50",
                cap_col=None,
                weight_col=None,
                state_col=None,
                locked_mask=None,
                in_rotation_threshold_min=float(IN_ROTATION_THRESHOLD_MIN),
                default_cap=48.0,
                max_passes=5,
                eps=1e-6,
            )
            work.loc[sub_idx, "minutes_p50"] = reconciled
            if log_diagnostics:
                diags.append((g.iloc[0].get("game_id"), g.iloc[0].get("team_id"), {"locked_sum": locked_sum, "remaining": remaining, "unlocked_diag": diag}))
        else:
            if log_diagnostics:
                diags.append((g.iloc[0].get("game_id"), g.iloc[0].get("team_id"), {"locked_sum": locked_sum, "remaining": remaining, "unlocked_diag": None}))

        # Locked players: enforce exact targets after any unlocked reconcile.
        if lock.any():
            work.loc[idx[lock], "minutes_p50"] = pd.to_numeric(
                work.loc[idx[lock], "minutes_target_eff"], errors="coerce"
            ).fillna(0.0).astype(float)

        p50_delta = work.loc[idx, "minutes_p50"].astype(float) - minutes_before
        for col in quant_cols:
            work.loc[idx, col] = (work.loc[idx, col].astype(float) + p50_delta).clip(lower=0.0, upper=48.0)

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

    if log_diagnostics and diags:
        for gid, tid, d in diags:
            locked_sum = float(d.get("locked_sum", 0.0))
            remaining = float(d.get("remaining", 0.0))
            unlocked_diag = d.get("unlocked_diag")
            base_bits = f"[minutes-reconcile-hard] game_id={gid} team_id={tid} locked_sum={locked_sum:.2f} remaining={remaining:.2f}"
            if unlocked_diag is None:
                print(base_bits + " unlocked_diag=n/a")
            else:
                ud = unlocked_diag
                print(
                    base_bits
                    + (
                        " unlocked_pre=%.2f unlocked_post=%.2f residual=%.4f passes=%d cap_infeasible=%s out=%d cameo=%d rotation=%d cap_hits=%d"
                        % (
                            float(ud.pre_sum),
                            float(ud.post_sum),
                            float(ud.residual),
                            int(ud.passes),
                            bool(ud.cap_infeasible),
                            int(ud.n_out_or_dnp),
                            int(ud.n_cameo),
                            int(ud.n_rotation),
                            int(ud.n_cap_hits),
                        )
                    )
                )

    return work


def _reconcile_effective_minutes_after_overrides_hard_targets(
    df: pd.DataFrame,
    *,
    target_team_minutes: float = 240.0,
    log_diagnostics: bool = False,
) -> pd.DataFrame:
    """Reconcile effective_minutes back to 240, enforcing hard targets/locks."""
    required = {"game_id", "team_id", "effective_minutes", "minutes_lock_eff", "minutes_target_eff"}
    if df.empty or not required <= set(df.columns):
        return df

    work = df.copy()
    work["effective_minutes"] = (
        pd.to_numeric(work["effective_minutes"], errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    )
    work["minutes_lock_eff"] = work["minutes_lock_eff"].fillna(False).astype(bool)
    work["minutes_target_eff"] = pd.to_numeric(work["minutes_target_eff"], errors="coerce")

    status_raw = work.get("status")
    if status_raw is not None:
        status_lower = status_raw.astype(str).str.strip().str.lower()
    else:
        status_lower = pd.Series("", index=work.index, dtype=str)

    play_prob = pd.to_numeric(work.get("play_prob"), errors="coerce") if "play_prob" in work.columns else None
    if play_prob is None:
        play_prob = pd.Series(1.0, index=work.index, dtype=float)
    else:
        play_prob = play_prob.fillna(1.0).astype(float)

    is_out = status_lower.eq("out") | (play_prob <= 0.0)
    if is_out.any():
        work.loc[is_out, "effective_minutes"] = 0.0
        work.loc[is_out, "minutes_target_eff"] = 0.0
        work.loc[is_out, "minutes_lock_eff"] = True

    # Fill missing targets for lock-only rows from current effective_minutes.
    needs_fill = work["minutes_lock_eff"].astype(bool) & work["minutes_target_eff"].isna()
    if needs_fill.any():
        work.loc[needs_fill, "minutes_target_eff"] = work.loc[needs_fill, "effective_minutes"]

    # Enforce targets on effective_minutes before team reconcile.
    lock_mask = work["minutes_lock_eff"].astype(bool) & work["minutes_target_eff"].notna()
    if lock_mask.any():
        work.loc[lock_mask, "effective_minutes"] = pd.to_numeric(
            work.loc[lock_mask, "minutes_target_eff"], errors="coerce"
        ).fillna(0.0).astype(float)

    # Stable state semantics for unlocked rows: treat 0-minute rows as DNP.
    eff = work["effective_minutes"].astype(float)
    state = pd.Series("rotation", index=work.index, dtype="string")
    state.loc[is_out] = "out"
    state.loc[~is_out & (eff <= 0.0)] = "dnp"
    state.loc[~is_out & (eff > 0.0) & (eff < float(IN_ROTATION_THRESHOLD_MIN))] = "cameo"
    state.loc[~is_out & (eff >= float(IN_ROTATION_THRESHOLD_MIN))] = "rotation"
    work["_effective_minutes_state"] = state

    diags: list[tuple[Any, Any, Any]] = []
    for (_, _), group_idx in work.groupby(["game_id", "team_id"], sort=False).groups.items():
        idx = pd.Index(group_idx)
        g = work.loc[idx]

        lock = work.loc[idx, "minutes_lock_eff"].astype(bool) | is_out.loc[idx]
        targets = work.loc[idx, "minutes_target_eff"]
        locked_sum = float(pd.to_numeric(targets.where(lock), errors="coerce").fillna(0.0).sum())
        if locked_sum > float(target_team_minutes) + 1e-6:
            _raise_locked_minutes_infeasible(
                work.loc[idx],
                game_id=g.iloc[0].get("game_id"),
                team_id=g.iloc[0].get("team_id"),
                locked_sum=locked_sum,
                target_team_minutes=float(target_team_minutes),
            )

        remaining = float(target_team_minutes) - locked_sum
        unlocked = ~lock
        if unlocked.any():
            sub_idx = idx[unlocked.to_numpy(dtype=bool)]
            g_unlocked = work.loc[sub_idx]
            reconciled, diag = reconcile_team_minutes(
                g_unlocked,
                float(remaining),
                minutes_col="effective_minutes",
                cap_col=None,
                weight_col=None,
                state_col="_effective_minutes_state",
                locked_mask=None,
                in_rotation_threshold_min=float(IN_ROTATION_THRESHOLD_MIN),
                default_cap=48.0,
                max_passes=5,
                eps=1e-6,
            )
            work.loc[sub_idx, "effective_minutes"] = reconciled
            if log_diagnostics:
                diags.append((g.iloc[0].get("game_id"), g.iloc[0].get("team_id"), {"locked_sum": locked_sum, "remaining": remaining, "unlocked_diag": diag}))
        else:
            if log_diagnostics:
                diags.append((g.iloc[0].get("game_id"), g.iloc[0].get("team_id"), {"locked_sum": locked_sum, "remaining": remaining, "unlocked_diag": None}))

        # Re-assert locked targets.
        if lock.any():
            work.loc[idx[lock], "effective_minutes"] = pd.to_numeric(
                work.loc[idx[lock], "minutes_target_eff"], errors="coerce"
            ).fillna(0.0).astype(float)

    work = work.drop(columns=["_effective_minutes_state"], errors="ignore")

    if log_diagnostics and diags:
        for gid, tid, d in diags:
            locked_sum = float(d.get("locked_sum", 0.0))
            remaining = float(d.get("remaining", 0.0))
            unlocked_diag = d.get("unlocked_diag")
            base_bits = f"[effective-minutes-reconcile-hard] game_id={gid} team_id={tid} locked_sum={locked_sum:.2f} remaining={remaining:.2f}"
            if unlocked_diag is None:
                print(base_bits + " unlocked_diag=n/a")
            else:
                ud = unlocked_diag
                print(
                    base_bits
                    + (
                        " unlocked_pre=%.2f unlocked_post=%.2f residual=%.4f passes=%d cap_infeasible=%s out=%d cameo=%d rotation=%d cap_hits=%d"
                        % (
                            float(ud.pre_sum),
                            float(ud.post_sum),
                            float(ud.residual),
                            int(ud.passes),
                            bool(ud.cap_infeasible),
                            int(ud.n_out_or_dnp),
                            int(ud.n_cameo),
                            int(ud.n_rotation),
                            int(ud.n_cap_hits),
                        )
                    )
                )

    return work


def apply_overrides_to_minutes_df(
    minutes_df: pd.DataFrame,
    *,
    game_date: date,
    data_root: Path | None = None,
    reconcile_team_minutes: bool = True,
    log_diagnostics: bool = False,
    force_reconcile: bool = False,
) -> pd.DataFrame:
    """Apply authoritative ops overrides to a minutes projections frame.

    Note: if minutes overrides change team totals, we reconcile minutes_p50 back to 240
    (within each game/team) so downstream sims/validation remain stable.
    """
    if minutes_df.empty:
        return minutes_df

    overrides = load_overrides_map(game_date, data_root=data_root)

    work = minutes_df.copy()
    # Preserve raw model outputs for debugging (ops overrides should only affect the effective layer).
    # If the caller passes an already-effective frame (with *_model columns), reset the
    # mutable columns back to their model values to keep this operation idempotent.
    for col in (
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
        "effective_minutes",
    ):
        model_col = f"{col}_model"
        if model_col in work.columns:
            work[col] = work[model_col]
        elif col in work.columns:
            work[model_col] = work[col]

    # Drop previously-materialized effective-layer fields so we can rebuild them deterministically.
    work = work.drop(
        columns=[
            OPS_DEPTH_ROLE_FIELD,
            "minutes_delta",
            "minutes_delta_applied",
            "minutes_target",
            "minutes_lock",
            "minutes_target_eff",
            "minutes_lock_eff",
            "ops_override_applied",
            "minutes_final",
            "minutes_contract_version",
            "minutes_contract_hash",
            "_minutes_p10_overridden",
            "_minutes_p90_overridden",
        ],
        errors="ignore",
    )

    if not overrides:
        # Still attach stable "effective layer" columns so downstream consumers can
        # prefer a single source of truth, even when no overrides exist.
        work["ops_override_applied"] = False
        work["minutes_delta_applied"] = False
        work["minutes_target"] = pd.NA
        work["minutes_lock"] = False
        work["minutes_target_eff"] = pd.NA
        work["minutes_lock_eff"] = False
        if "minutes_delta" not in work.columns:
            work["minutes_delta"] = pd.NA
        locked_mask = pd.Series(False, index=work.index)
        if force_reconcile and reconcile_team_minutes:
            if {"game_id", "team_id", "effective_minutes"} <= set(work.columns):
                work = _reconcile_effective_minutes_after_overrides(
                    work,
                    locked_mask=locked_mask,
                    target_team_minutes=240.0,
                    log_diagnostics=log_diagnostics,
                )
            elif {"game_id", "team_id", "minutes_p50"} <= set(work.columns):
                work = _reconcile_minutes_after_overrides(
                    work,
                    locked_mask=locked_mask,
                    target_team_minutes=240.0,
                    log_diagnostics=log_diagnostics,
                )
        if "minutes_final" not in work.columns:
            base_col = (
                "effective_minutes"
                if "effective_minutes" in work.columns
                else ("minutes_p50_cond" if "minutes_p50_cond" in work.columns else "minutes_p50")
            )
            work["minutes_final"] = pd.to_numeric(work.get(base_col), errors="coerce").fillna(0.0).astype(float)
        work["minutes_contract_version"] = int(MINUTES_CONTRACT_VERSION)
        work["minutes_contract_hash"] = minutes_contract_hash()
        return work

    ops_df = _overrides_as_frame(overrides, include_fields=MINUTES_FIELDS)
    if ops_df.empty:
        work["ops_override_applied"] = False
        work["minutes_delta_applied"] = False
        work["minutes_target"] = pd.NA
        work["minutes_lock"] = False
        work["minutes_target_eff"] = pd.NA
        work["minutes_lock_eff"] = False
        locked_mask = pd.Series(False, index=work.index)
        if force_reconcile and reconcile_team_minutes:
            if {"game_id", "team_id", "effective_minutes"} <= set(work.columns):
                work = _reconcile_effective_minutes_after_overrides(
                    work,
                    locked_mask=locked_mask,
                    target_team_minutes=240.0,
                    log_diagnostics=log_diagnostics,
                )
            elif {"game_id", "team_id", "minutes_p50"} <= set(work.columns):
                work = _reconcile_minutes_after_overrides(
                    work,
                    locked_mask=locked_mask,
                    target_team_minutes=240.0,
                    log_diagnostics=log_diagnostics,
                )
        if "minutes_final" not in work.columns:
            base_col = (
                "effective_minutes"
                if "effective_minutes" in work.columns
                else ("minutes_p50_cond" if "minutes_p50_cond" in work.columns else "minutes_p50")
            )
            work["minutes_final"] = pd.to_numeric(work.get(base_col), errors="coerce").fillna(0.0).astype(float)
        work["minutes_contract_version"] = int(MINUTES_CONTRACT_VERSION)
        work["minutes_contract_hash"] = minutes_contract_hash()
        return work

    work["_ops_key"] = _ops_key_for_df(work)
    merged = work.merge(ops_df, on="_ops_key", how="left", suffixes=("", "_ops"))

    if "minutes_target" not in merged.columns:
        merged["minutes_target"] = pd.NA
    if "minutes_lock" not in merged.columns:
        merged["minutes_lock"] = False

    locked_mask = pd.Series(False, index=merged.index)
    ops_applied = pd.Series(False, index=merged.index)

    # Track which quantiles were explicitly overridden (for downstream sim to preserve)
    for qcol in ("minutes_p10", "minutes_p90"):
        override_col = f"{qcol}_ops"
        flag_col = f"_{qcol}_overridden"
        if override_col in merged.columns:
            merged[flag_col] = merged[override_col].notna()
            ops_applied = ops_applied | merged[override_col].notna()
        else:
            merged[flag_col] = False

    for col in MINUTES_FIELDS:
        if col not in ops_df.columns:
            continue
        # Some fields only exist in ops_df (no collision with minutes_df), in which case
        # pandas keeps the original column name (no "_ops" suffix).
        override_col = f"{col}_ops" if f"{col}_ops" in merged.columns else col
        if override_col not in merged.columns:
            continue

        if override_col != col:
            # Typical case: base column exists and override column is suffixed.
            merged[col] = merged[override_col].where(merged[override_col].notna(), merged[col])

        if col == "minutes_p50":
            if "effective_minutes" in merged.columns:
                merged["effective_minutes"] = merged[override_col].where(
                    merged[override_col].notna(),
                    merged["effective_minutes"],
                )
            locked_mask = locked_mask | merged[override_col].notna()

        ops_applied = ops_applied | merged[override_col].notna()

    has_target = merged["minutes_target"].notna() if "minutes_target" in merged.columns else pd.Series(False, index=merged.index)

    if has_target.any():
        # Hard targets: treat as an absolute minutes center (shift all minutes columns by the delta
        # from current minutes_p50 so the distribution moves with the target).
        target_vals = pd.to_numeric(merged.loc[has_target, "minutes_target"], errors="coerce").fillna(0.0).astype(float)
        target_vals = target_vals.clip(lower=0.0, upper=48.0)
        base_p50 = pd.to_numeric(merged.loc[has_target, "minutes_p50"], errors="coerce").fillna(0.0).astype(float)
        delta_to_target = target_vals - base_p50
        for mcol in (
            "minutes_p10",
            "minutes_p50",
            "minutes_p90",
            "minutes_p10_cond",
            "minutes_p50_cond",
            "minutes_p90_cond",
        ):
            if mcol in merged.columns:
                merged.loc[has_target, mcol] = (
                    pd.to_numeric(merged.loc[has_target, mcol], errors="coerce").fillna(0.0).astype(float)
                    + delta_to_target
                ).clip(0.0, 48.0)
        if "effective_minutes" in merged.columns:
            merged.loc[has_target, "effective_minutes"] = (
                pd.to_numeric(merged.loc[has_target, "effective_minutes"], errors="coerce").fillna(0.0).astype(float)
                + delta_to_target
            ).clip(0.0, 48.0)
        locked_mask = locked_mask | has_target
        ops_applied = ops_applied | has_target

    # Apply minutes_delta as additive adjustment (takes precedence over raw quantile overrides)
    # Note: minutes_delta only exists in ops_df (never in original minutes_df), so after merge
    # it keeps its original name (no _ops suffix) because suffixes only apply to colliding columns.
    delta_col = "minutes_delta" if "minutes_delta" in merged.columns else "minutes_delta_ops"
    if delta_col in merged.columns:
        # Stage 1A: treat minutes_delta as sugar for a hard target when minutes_target is absent.
        has_delta = merged[delta_col].notna() & ~has_target
        if has_delta.any():
            delta_vals = merged.loc[has_delta, delta_col].astype(float)
            for qcol in ("minutes_p10", "minutes_p50", "minutes_p90", "minutes_p10_cond", "minutes_p50_cond", "minutes_p90_cond"):
                if qcol in merged.columns:
                    merged.loc[has_delta, qcol] = (
                        merged.loc[has_delta, qcol].astype(float) + delta_vals
                    ).clip(0.0, 48.0)
            if "effective_minutes" in merged.columns:
                merged.loc[has_delta, "effective_minutes"] = (
                    pd.to_numeric(merged.loc[has_delta, "effective_minutes"], errors="coerce").fillna(0.0).astype(float)
                    + delta_vals
                ).clip(0.0, 48.0)
            # Mark delta-adjusted players as locked and with overridden quantiles
            locked_mask = locked_mask | has_delta
            merged.loc[has_delta, "_minutes_p10_overridden"] = True
            merged.loc[has_delta, "_minutes_p90_overridden"] = True
            ops_applied = ops_applied | has_delta
            merged["minutes_delta_applied"] = has_delta
        else:
            merged["minutes_delta_applied"] = False
    else:
        merged["minutes_delta_applied"] = False

    # If operator sets ops depth role to OUT, normalize to status='out' so downstream consistently
    # zeros minutes/play_prob and any other consumers relying on status behave the same.
    if OPS_DEPTH_ROLE_FIELD in merged.columns:
        role = merged[OPS_DEPTH_ROLE_FIELD]
        role_lower = role.astype(str).str.strip().str.lower()
        role_out = role.notna() & role_lower.eq("out")
        if role_out.any():
            merged.loc[role_out, "status"] = "out"

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
            if "effective_minutes" in merged.columns:
                merged.loc[is_out, "effective_minutes"] = 0.0
            locked_mask = locked_mask | is_out
            ops_applied = ops_applied | is_out

    # Stage 1A: materialize effective hard-target columns for downstream reconcile + sim allocator.
    merged["minutes_target_eff"] = pd.to_numeric(merged.get("minutes_target"), errors="coerce") if "minutes_target" in merged.columns else pd.NA
    if "minutes_lock" in merged.columns:
        raw_lock = merged["minutes_lock"]
        minutes_lock_flag = raw_lock.fillna(False).astype(bool)
    else:
        minutes_lock_flag = pd.Series(False, index=merged.index)
    merged["minutes_lock_eff"] = (minutes_lock_flag | has_target | locked_mask).fillna(False).astype(bool)

    # OUT always implies a hard 0 target + lock.
    if status_raw is not None:
        status_lower = status_raw.astype(str).str.strip().str.lower()
        is_out = status_lower.eq("out")
        if is_out.any():
            merged.loc[is_out, "minutes_target_eff"] = 0.0
            merged.loc[is_out, "minutes_lock_eff"] = True

    # Fill missing targets for lock-only rows from the minutes contract column used downstream.
    base_target_col = (
        "effective_minutes"
        if "effective_minutes" in merged.columns
        else ("minutes_p50_cond" if "minutes_p50_cond" in merged.columns else "minutes_p50")
    )
    needs_target = merged["minutes_lock_eff"].astype(bool) & merged["minutes_target_eff"].isna()
    if needs_target.any():
        merged.loc[needs_target, "minutes_target_eff"] = (
            pd.to_numeric(merged.loc[needs_target, base_target_col], errors="coerce").fillna(0.0).astype(float).clip(0.0, 48.0)
        )

    # Enforce targets immediately so even reconcile_team_minutes=False consumers see fixed minutes.
    enforce_mask = merged["minutes_lock_eff"].astype(bool) & merged["minutes_target_eff"].notna()
    if enforce_mask.any():
        tgt_vals = pd.to_numeric(merged.loc[enforce_mask, "minutes_target_eff"], errors="coerce").fillna(0.0).astype(float).clip(0.0, 48.0)
        if base_target_col in merged.columns:
            merged.loc[enforce_mask, base_target_col] = tgt_vals
        if "effective_minutes" in merged.columns:
            merged.loc[enforce_mask, "effective_minutes"] = tgt_vals
        if "minutes_p50" in merged.columns:
            merged.loc[enforce_mask, "minutes_p50"] = tgt_vals

    merged["ops_override_applied"] = ops_applied
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_ops")] + ["_ops_key"], errors="ignore")

    if reconcile_team_minutes:
        if {"game_id", "team_id", "effective_minutes"} <= set(merged.columns):
            merged = _reconcile_effective_minutes_after_overrides_hard_targets(
                merged, target_team_minutes=240.0, log_diagnostics=log_diagnostics
            )
        elif {"game_id", "team_id", "minutes_p50"} <= set(merged.columns):
            merged = _reconcile_minutes_after_overrides_hard_targets(
                merged, target_team_minutes=240.0, log_diagnostics=log_diagnostics
            )

    # Single source of truth column for consumers.
    if "minutes_final" not in merged.columns:
        base_col = (
            "effective_minutes"
            if "effective_minutes" in merged.columns
            else ("minutes_p50_cond" if "minutes_p50_cond" in merged.columns else "minutes_p50")
        )
        merged["minutes_final"] = pd.to_numeric(merged.get(base_col), errors="coerce").fillna(0.0).astype(float)
    merged["minutes_contract_version"] = int(MINUTES_CONTRACT_VERSION)
    merged["minutes_contract_hash"] = minutes_contract_hash()

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
