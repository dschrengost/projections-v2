from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from projections.alloc.bounded_projection import ProjectionInfeasibleError, project_sum_with_bounds

# Explicit v2 fields (new schema) accepted in per-player override `fields`.
_EXPLICIT_FLOOR_FIELDS = {"lb_minutes", "minutes_lb", "minutes_min", "minutes_floor", "min_minutes", "floor"}
_EXPLICIT_CAP_FIELDS = {"ub_minutes", "minutes_ub", "minutes_max", "minutes_cap", "max_minutes", "cap"}
_EXPLICIT_TARGET_FIELDS = {"mu_minutes", "minutes_target", "target_minutes", "target", "minutes"}
_EXPLICIT_DELTA_FIELDS = {"minutes_delta", "delta"}
_EXPLICIT_LOCK_FIELDS = {"minutes_lock", "lock", "exact_lock", "hard_lock", "exact"}
_EXPLICIT_FORCE_ACTIVE_FIELDS = {"force_active", "must_play"}
_EXPLICIT_FORCE_INACTIVE_FIELDS = {"force_inactive", "inactive", "bench_zero", "must_sit"}
_EXPLICIT_ELIGIBLE_FIELDS = {"eligible", "is_eligible"}
_OUTLIKE_FLAG_FIELDS = {"out", "dnp", "is_out"}


@dataclass(frozen=True)
class MinutesOverrideV2Policy:
    """Config for compiling and enforcing minutes override constraints in worlds v2."""

    target_team_minutes: float = 240.0
    default_lb_minutes: float = 0.0
    default_ub_minutes: float = 48.0
    legacy_target_band_eps: float = 2.0
    starter_weight_bonus: float = 1.25
    minutes_weight_scale: float = 1.0
    override_infeasible: str = "error"  # error | relax | ignore
    relax_step_minutes: float = 0.5
    relax_max_steps: int = 96

    def validate(self) -> "MinutesOverrideV2Policy":
        mode = str(self.override_infeasible).strip().lower()
        if mode not in {"error", "relax", "ignore"}:
            raise ValueError(f"override_infeasible must be one of error|relax|ignore, got {self.override_infeasible!r}")
        if self.default_lb_minutes < 0.0 or self.default_ub_minutes <= 0.0:
            raise ValueError("default bounds must be non-negative and ub > 0")
        if self.default_lb_minutes > self.default_ub_minutes:
            raise ValueError("default_lb_minutes cannot exceed default_ub_minutes")
        return MinutesOverrideV2Policy(
            target_team_minutes=float(self.target_team_minutes),
            default_lb_minutes=float(self.default_lb_minutes),
            default_ub_minutes=float(self.default_ub_minutes),
            legacy_target_band_eps=float(max(0.0, self.legacy_target_band_eps)),
            starter_weight_bonus=float(max(0.0, self.starter_weight_bonus)),
            minutes_weight_scale=float(max(0.0, self.minutes_weight_scale)),
            override_infeasible=mode,
            relax_step_minutes=float(max(0.01, self.relax_step_minutes)),
            relax_max_steps=int(max(1, self.relax_max_steps)),
        )


@dataclass(frozen=True)
class _OverrideKey:
    game_id: str
    player_id: str


@dataclass(frozen=True)
class _CompiledConstraint:
    lb_minutes: float
    ub_minutes: float
    force_active: bool
    force_inactive: bool
    eligible: bool
    weight: float
    constraint_kind: str
    override_fields: dict[str, Any]


def _as_policy(policy: MinutesOverrideV2Policy | dict[str, Any] | None) -> MinutesOverrideV2Policy:
    if policy is None:
        return MinutesOverrideV2Policy().validate()
    if isinstance(policy, MinutesOverrideV2Policy):
        return policy.validate()
    if isinstance(policy, dict):
        return MinutesOverrideV2Policy(**policy).validate()
    raise TypeError(f"Unsupported policy type: {type(policy)!r}")


def _normalize_id_series(series: pd.Series) -> pd.Series:
    out = series.astype("string", copy=False).fillna("")
    numeric = pd.to_numeric(series, errors="coerce")
    int_like = numeric.notna() & (numeric % 1 == 0)
    if int_like.any():
        out = out.where(~int_like, numeric.where(int_like).astype("Int64").astype("string"))
    return out.str.replace(r"\.0$", "", regex=True)


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if np.isnan(float(value)):
            return None
        return bool(int(float(value) != 0.0))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off", ""}:
            return False
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _extract_override_items(overrides_payload: Any) -> list[dict[str, Any]]:
    """Return normalized override records: [{game_id, player_id, fields}, ...]."""

    if overrides_payload is None:
        return []

    payload: Any = overrides_payload
    if isinstance(overrides_payload, (str, Path)):
        path = Path(overrides_payload)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    if isinstance(payload, dict):
        if isinstance(payload.get("overrides"), list):
            raw_items = payload.get("overrides") or []
        elif isinstance(payload.get("updates"), list):
            raw_items = payload.get("updates") or []
        elif {"game_id", "player_id"}.issubset(payload.keys()):
            raw_items = [payload]
        else:
            raw_items = []
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = []

    out: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        game_id = item.get("game_id")
        player_id = item.get("player_id")
        if game_id is None or player_id is None:
            continue
        fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
        # Backward compatibility: some payloads place fields at top level.
        if not fields:
            fields = {
                k: v
                for k, v in item.items()
                if k
                not in {
                    "game_id",
                    "player_id",
                    "fields",
                    "updated_at",
                    "note",
                    "sticky_fields",
                }
            }
        out.append(
            {
                "game_id": str(game_id),
                "player_id": str(player_id),
                "fields": dict(fields),
                "updated_at": item.get("updated_at"),
                "note": item.get("note"),
            }
        )
    return out


def _merge_override_items(items: list[dict[str, Any]]) -> tuple[dict[_OverrideKey, dict[str, Any]], list[dict[str, Any]]]:
    merged: dict[_OverrideKey, dict[str, Any]] = {}
    normalized_items: list[dict[str, Any]] = []

    for item in items:
        key = _OverrideKey(game_id=str(item["game_id"]), player_id=str(item["player_id"]))
        fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
        clean_fields = {k: v for k, v in fields.items() if v is not None}
        if not clean_fields:
            continue
        existing = merged.get(key, {})
        merged[key] = {**existing, **clean_fields}
        normalized_items.append({"game_id": key.game_id, "player_id": key.player_id, "fields": dict(clean_fields)})

    return merged, normalized_items


def _resolve_baseline_minutes_col(df: pd.DataFrame) -> str:
    for candidate in ("b_minutes", "minutes_mean", "minutes_final", "effective_minutes", "minutes_p50_cond", "minutes_p50"):
        if candidate in df.columns:
            return candidate
    raise KeyError(
        "baseline_minutes_df missing baseline minutes column; expected one of "
        "b_minutes/minutes_mean/minutes_final/effective_minutes/minutes_p50_cond/minutes_p50"
    )


def _compute_weight(df: pd.DataFrame, policy: MinutesOverrideV2Policy) -> np.ndarray:
    b_col = _resolve_baseline_minutes_col(df)
    b = pd.to_numeric(df[b_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    starter_like = np.zeros(len(df), dtype=float)
    for col in ("is_confirmed_starter", "is_projected_starter", "is_starter", "starter_flag"):
        if col in df.columns:
            starter_like = np.maximum(
                starter_like,
                pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            )
    starter_mask = starter_like > 0.5

    base_component = 1.0 + policy.minutes_weight_scale * np.clip(b, 0.0, 48.0) / 48.0
    weight = base_component + policy.starter_weight_bonus * starter_mask.astype(float)
    return np.clip(weight, 0.05, None)


def _compile_constraint_for_row(
    row: pd.Series,
    *,
    row_key: _OverrideKey,
    override_fields: dict[str, Any],
    b_minutes: float,
    weight: float,
    policy: MinutesOverrideV2Policy,
    strict: bool,
) -> _CompiledConstraint:
    lb = float(policy.default_lb_minutes)
    ub = float(policy.default_ub_minutes)

    eligible = True
    if "eligible_flag" in row.index:
        raw_eligible = _to_bool(row.get("eligible_flag"))
        if raw_eligible is not None:
            eligible = bool(raw_eligible)

    force_active = False
    force_inactive = False
    constraint_kind = "none"

    # Parse explicit booleans.
    for key in _EXPLICIT_ELIGIBLE_FIELDS:
        if key in override_fields:
            raw = _to_bool(override_fields.get(key))
            if raw is not None:
                eligible = bool(raw)
                break
    for key in _EXPLICIT_FORCE_ACTIVE_FIELDS:
        if key in override_fields:
            raw = _to_bool(override_fields.get(key))
            if raw is not None and raw:
                force_active = True
                constraint_kind = "force_active"
                break
    for key in _EXPLICIT_FORCE_INACTIVE_FIELDS:
        if key in override_fields:
            raw = _to_bool(override_fields.get(key))
            if raw is not None and raw:
                force_inactive = True
                constraint_kind = "force_inactive"
                break

    # OUT-like semantics.
    out_like = False
    for key in _OUTLIKE_FLAG_FIELDS:
        if key in override_fields:
            raw = _to_bool(override_fields.get(key))
            if raw:
                out_like = True
                break

    status_val = override_fields.get("status")
    if isinstance(status_val, str) and status_val.strip().lower() in {"out", "dnp", "inactive", "suspended"}:
        out_like = True

    role_val = override_fields.get("ops_depth_role")
    if isinstance(role_val, str) and role_val.strip().lower() == "out":
        out_like = True

    if out_like:
        force_inactive = True
        eligible = False
        constraint_kind = "zero_lock"

    # Numeric floors/caps.
    for key in _EXPLICIT_FLOOR_FIELDS:
        if key in override_fields:
            val = _to_float(override_fields.get(key))
            if val is not None:
                lb = max(lb, float(val))
                constraint_kind = "floor"
    for key in _EXPLICIT_CAP_FIELDS:
        if key in override_fields:
            val = _to_float(override_fields.get(key))
            if val is not None:
                ub = min(ub, float(val))
                constraint_kind = "cap"

    # Target and delta semantics.
    target_val: float | None = None
    for key in _EXPLICIT_TARGET_FIELDS:
        if key in override_fields:
            target_val = _to_float(override_fields.get(key))
            if target_val is not None:
                break
    if target_val is None:
        for key in _EXPLICIT_DELTA_FIELDS:
            if key in override_fields:
                delta = _to_float(override_fields.get(key))
                if delta is not None:
                    target_val = float(b_minutes) + float(delta)
                    break

    lock_exact = False
    for key in _EXPLICIT_LOCK_FIELDS:
        if key in override_fields:
            raw = _to_bool(override_fields.get(key))
            if raw is not None:
                lock_exact = bool(raw)
                break

    if out_like:
        lock_exact = True
        target_val = 0.0

    if target_val is not None:
        t = float(np.clip(target_val, 0.0, policy.default_ub_minutes))
        if lock_exact:
            lb = max(lb, t)
            ub = min(ub, t)
            constraint_kind = "lock"
        else:
            band_eps = _to_float(override_fields.get("minutes_band_eps"))
            if band_eps is None:
                band_eps = float(policy.legacy_target_band_eps)
            band_eps = max(0.0, float(band_eps))
            lb = max(lb, t - band_eps)
            ub = min(ub, t + band_eps)
            constraint_kind = "band"

    lb = float(np.clip(lb, 0.0, policy.default_ub_minutes))
    ub = float(np.clip(ub, 0.0, policy.default_ub_minutes))

    if not eligible or force_inactive or ub <= 1e-12:
        eligible = False
        force_inactive = True
        force_active = False
        lb = 0.0
        ub = 0.0
        constraint_kind = "zero_lock"

    if lb > 0.0:
        force_active = True
        if constraint_kind == "none":
            constraint_kind = "floor"

    if lb > ub + 1e-8:
        if strict:
            raise ValueError(
                "Override bounds invalid after compile: "
                f"game_id={row_key.game_id} player_id={row_key.player_id} lb={lb:.3f} ub={ub:.3f}"
            )
        # Non-strict fallback: collapse to midpoint clipped to legal range.
        mid = float(np.clip(0.5 * (lb + ub), 0.0, policy.default_ub_minutes))
        lb = mid
        ub = mid
        constraint_kind = "lock"

    return _CompiledConstraint(
        lb_minutes=float(lb),
        ub_minutes=float(ub),
        force_active=bool(force_active),
        force_inactive=bool(force_inactive),
        eligible=bool(eligible),
        weight=float(max(0.05, weight)),
        constraint_kind=constraint_kind,
        override_fields=dict(override_fields),
    )


def _has_team_override(group: pd.DataFrame, policy: MinutesOverrideV2Policy) -> bool:
    if bool(group["override_present"].any()):
        return True
    if bool(group["force_active"].any()):
        return True
    if bool(group["force_inactive"].any()):
        return True
    if bool((~group["eligible"]).any()):
        return True
    default_lb = float(policy.default_lb_minutes)
    default_ub = float(policy.default_ub_minutes)
    if bool((group["lb_minutes"] > default_lb + 1e-9).any()):
        return True
    if bool((group["ub_minutes"] < default_ub - 1e-9).any()):
        return True
    return False


def _relax_and_project(
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    w: np.ndarray,
    *,
    policy: MinutesOverrideV2Policy,
) -> tuple[np.ndarray, dict[str, Any]]:
    lock_mask = np.abs(lb - ub) <= 1e-8

    # Never violate locks in relax mode.
    sum_locked = float(lb[lock_mask].sum())
    if sum_locked > float(policy.target_team_minutes) + 1e-8:
        raise ProjectionInfeasibleError(
            reason="locked_minutes_exceed_target",
            target_sum=float(policy.target_team_minutes),
            sum_lb=float(lb.sum()),
            sum_ub=float(ub.sum()),
            n_items=int(len(b)),
        )

    lb_rel = lb.copy()
    ub_rel = ub.copy()
    non_locked = ~lock_mask

    relax_steps = 0
    while relax_steps <= int(policy.relax_max_steps):
        sum_lb = float(lb_rel.sum())
        sum_ub = float(ub_rel.sum())
        if sum_lb <= float(policy.target_team_minutes) + 1e-8 and sum_ub >= float(policy.target_team_minutes) - 1e-8:
            y = project_sum_with_bounds(
                b,
                float(policy.target_team_minutes),
                lb_rel,
                ub_rel,
                w,
            )
            return y, {
                "relax_steps": int(relax_steps),
                "sum_lb_relaxed": float(sum_lb),
                "sum_ub_relaxed": float(sum_ub),
            }

        relax_steps += 1
        pad = float(relax_steps) * float(policy.relax_step_minutes)
        lb_rel[non_locked] = np.maximum(0.0, lb[non_locked] - pad)
        ub_rel[non_locked] = np.minimum(float(policy.default_ub_minutes), ub[non_locked] + pad)

    raise ProjectionInfeasibleError(
        reason="relax_failed_to_find_feasible_bounds",
        target_sum=float(policy.target_team_minutes),
        sum_lb=float(lb_rel.sum()),
        sum_ub=float(ub_rel.sum()),
        n_items=int(len(b)),
    )


def apply_minutes_overrides_v2(
    baseline_minutes_df: pd.DataFrame,
    overrides_payload: Any,
    *,
    policy: MinutesOverrideV2Policy | dict[str, Any] | None = None,
    seed: int | None = None,
    strict: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compile and apply minutes overrides as deterministic constraints.

    The output is a resolved per-player minutes constraint frame containing:
    `game_id, team_id, player_id, b_minutes, mu_minutes, lb_minutes, ub_minutes,
    eligible, force_active, force_inactive, weight`.

    Legacy payload compatibility:
    - `minutes_delta` / `minutes_target` are compiled into lock or band constraints.
    - `status=out`, `ops_depth_role=out`, `out=true`, `dnp=true` compile to unbreakable zero-locks.
    """

    if baseline_minutes_df.empty:
        empty = baseline_minutes_df.copy()
        for col in (
            "b_minutes",
            "mu_minutes",
            "lb_minutes",
            "ub_minutes",
            "eligible",
            "force_active",
            "force_inactive",
            "weight",
            "constraint_kind",
            "override_present",
            "override_fields",
        ):
            if col not in empty.columns:
                empty[col] = pd.Series(dtype=float if col.endswith("_minutes") or col == "weight" else object)
        return empty, {
            "policy": asdict(_as_policy(policy)),
            "team_diagnostics": [],
            "unknown_overrides": [],
            "seed": seed,
        }

    cfg = _as_policy(policy)
    work = baseline_minutes_df.copy()

    for required in ("game_id", "team_id", "player_id"):
        if required not in work.columns:
            raise KeyError(f"baseline_minutes_df missing required column: {required!r}")

    b_col = _resolve_baseline_minutes_col(work)
    work["b_minutes"] = pd.to_numeric(work[b_col], errors="coerce").fillna(0.0).astype(float)
    work["b_minutes"] = work["b_minutes"].clip(lower=0.0, upper=float(cfg.default_ub_minutes))

    work["_game_id_key"] = _normalize_id_series(work["game_id"])
    work["_player_id_key"] = _normalize_id_series(work["player_id"])
    work["_team_sort_key"] = _normalize_id_series(work["team_id"])

    items = _extract_override_items(overrides_payload)
    merged_map, normalized_items = _merge_override_items(items)

    known_keys = set(zip(work["_game_id_key"], work["_player_id_key"]))
    unknown_overrides = [
        {"game_id": item["game_id"], "player_id": item["player_id"], "fields": dict(item["fields"])}
        for item in normalized_items
        if (str(item["game_id"]), str(item["player_id"])) not in known_keys
    ]

    weight_vec = _compute_weight(work, cfg)

    compiled_rows: list[dict[str, Any]] = []
    for pos, (_, row) in enumerate(work.iterrows()):
        row_key = _OverrideKey(game_id=str(row["_game_id_key"]), player_id=str(row["_player_id_key"]))
        override_fields = merged_map.get(row_key, {})
        compiled = _compile_constraint_for_row(
            row,
            row_key=row_key,
            override_fields=override_fields,
            b_minutes=float(row["b_minutes"]),
            weight=float(weight_vec[pos]),
            policy=cfg,
            strict=bool(strict),
        )
        compiled_rows.append(
            {
                "lb_minutes": compiled.lb_minutes,
                "ub_minutes": compiled.ub_minutes,
                "force_active": compiled.force_active,
                "force_inactive": compiled.force_inactive,
                "eligible": compiled.eligible,
                "weight": compiled.weight,
                "constraint_kind": compiled.constraint_kind,
                "override_present": bool(len(compiled.override_fields) > 0),
                "override_fields": compiled.override_fields,
            }
        )

    compiled_df = pd.DataFrame(compiled_rows, index=work.index)
    work = pd.concat([work, compiled_df], axis=1)

    # Deterministic order before per-team projection.
    work = work.sort_values(["_game_id_key", "_team_sort_key", "_player_id_key"], kind="mergesort").reset_index(drop=True)

    work["mu_minutes"] = work["b_minutes"].astype(float)

    team_diags: list[dict[str, Any]] = []

    for (game_key, team_key), idx in work.groupby(["_game_id_key", "_team_sort_key"], sort=False).groups.items():
        group_idx = pd.Index(idx)
        g = work.loc[group_idx]
        b = g["b_minutes"].to_numpy(dtype=float)
        lb = g["lb_minutes"].to_numpy(dtype=float)
        ub = g["ub_minutes"].to_numpy(dtype=float)
        w = g["weight"].to_numpy(dtype=float)

        has_override = _has_team_override(g, cfg)
        infeasible_reason: str | None = None
        action = "none"
        relax_meta: dict[str, Any] = {}

        if has_override:
            try:
                mu = project_sum_with_bounds(
                    b,
                    float(cfg.target_team_minutes),
                    lb,
                    ub,
                    w,
                )
                action = "project"
            except ProjectionInfeasibleError as exc:
                infeasible_reason = exc.reason
                if cfg.override_infeasible == "error" or strict:
                    raise
                if cfg.override_infeasible == "ignore":
                    mu = b.copy()
                    action = "ignore_to_baseline"
                    # Fully drop compiled bounds/forces for this team in ignore mode.
                    work.loc[group_idx, "lb_minutes"] = float(cfg.default_lb_minutes)
                    work.loc[group_idx, "ub_minutes"] = float(cfg.default_ub_minutes)
                    work.loc[group_idx, "force_active"] = False
                    work.loc[group_idx, "force_inactive"] = False
                    work.loc[group_idx, "eligible"] = True
                    work.loc[group_idx, "constraint_kind"] = "none"
                    lb = np.full_like(b, float(cfg.default_lb_minutes), dtype=float)
                    ub = np.full_like(b, float(cfg.default_ub_minutes), dtype=float)
                else:
                    mu, relax_meta = _relax_and_project(b, lb, ub, w, policy=cfg)
                    action = "relax_and_project"
        else:
            mu = b.copy()
            action = "no_override_baseline"

        work.loc[group_idx, "mu_minutes"] = mu

        lock_mask = np.abs(lb - ub) <= 1e-8
        hit_floor = np.flatnonzero(np.isclose(mu, lb, atol=1e-6) | (mu < lb + 1e-6))
        hit_cap = np.flatnonzero(np.isclose(mu, ub, atol=1e-6) | (mu > ub - 1e-6))
        player_ids = g["player_id"].astype(str).to_numpy(dtype=str)

        diag_row: dict[str, Any] = {
            "game_id": g["game_id"].iloc[0],
            "team_id": g["team_id"].iloc[0],
            "sum_lb": float(lb.sum()),
            "sum_ub": float(ub.sum()),
            "sum_mu": float(mu.sum()),
            "locked_minutes_total": float(mu[lock_mask].sum()),
            "remaining_to_fill": float(cfg.target_team_minutes - mu[lock_mask].sum()),
            "hit_floor_player_ids": [player_ids[j] for j in hit_floor.tolist()],
            "hit_cap_player_ids": [player_ids[j] for j in hit_cap.tolist()],
            "n_players": int(len(g)),
            "n_overrides": int(g["override_present"].sum()),
            "infeasibility_reason": infeasible_reason,
            "infeasible_action": action,
        }
        if relax_meta:
            diag_row.update(relax_meta)
        team_diags.append(diag_row)

    resolved_cols = [
        "game_id",
        "team_id",
        "player_id",
        "b_minutes",
        "mu_minutes",
        "lb_minutes",
        "ub_minutes",
        "eligible",
        "force_active",
        "force_inactive",
        "weight",
        "constraint_kind",
        "override_present",
        "override_fields",
    ]
    resolved = work[resolved_cols].copy()

    diag: dict[str, Any] = {
        "policy": asdict(cfg),
        "seed": seed,
        "n_input_overrides": int(len(normalized_items)),
        "unknown_overrides": unknown_overrides,
        "team_diagnostics": team_diags,
    }

    return resolved, diag


__all__ = ["MinutesOverrideV2Policy", "apply_minutes_overrides_v2"]
