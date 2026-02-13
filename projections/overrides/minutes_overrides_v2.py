from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from projections.alloc.bounded_projection import ProjectionInfeasibleError, project_sum_with_bounds

# Explicit v2 fields (new schema) accepted in per-player override `fields`.
_EXPLICIT_FLOOR_FIELDS = {"lb_minutes", "minutes_lb", "minutes_min", "minutes_floor", "min_minutes", "floor"}
_EXPLICIT_CAP_FIELDS = {"ub_minutes", "minutes_ub", "minutes_max", "minutes_cap", "max_minutes", "cap"}
_EXPLICIT_MEAN_LB_FIELDS = (
    "mean_lb_minutes",
    "minutes_mean_lb",
    "mean_min_minutes",
    "mean_floor_minutes",
    "lb_minutes",
    "minutes_lb",
    "minutes_min",
    "minutes_floor",
    "min_minutes",
    "floor",
)
_EXPLICIT_MEAN_UB_FIELDS = (
    "mean_ub_minutes",
    "minutes_mean_ub",
    "mean_max_minutes",
    "mean_cap_minutes",
    "ub_minutes",
    "minutes_ub",
    "minutes_max",
    "minutes_cap",
    "max_minutes",
    "cap",
)
_EXPLICIT_WORLD_LB_FIELDS = ("world_lb_minutes", "minutes_world_lb", "world_floor_minutes")
_EXPLICIT_WORLD_UB_FIELDS = (
    "world_ub_minutes",
    "minutes_world_ub",
    "world_cap_minutes",
    "hard_cap_minutes",
    "hard_ub_minutes",
)
_OVERRIDE_MODE_FIELDS = ("override_mode", "mode")
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
    mean_lb_minutes: float
    mean_ub_minutes: float
    world_lb_minutes: float
    world_ub_minutes: float
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
    mean_lb = float(policy.default_lb_minutes)
    mean_ub = float(policy.default_ub_minutes)
    world_lb = float(policy.default_lb_minutes)
    world_ub = float(policy.default_ub_minutes)

    eligible = True
    if "eligible_flag" in row.index:
        raw_eligible = _to_bool(row.get("eligible_flag"))
        if raw_eligible is not None:
            eligible = bool(raw_eligible)

    force_active = False
    force_inactive = False
    constraint_kind = "none"
    override_mode: str | None = None
    for key in _OVERRIDE_MODE_FIELDS:
        raw_mode = override_fields.get(key)
        if isinstance(raw_mode, str) and raw_mode.strip():
            override_mode = raw_mode.strip().lower()
            break

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

    def _first_numeric(fields: dict[str, Any], keys: Iterable[str]) -> tuple[str | None, float | None]:
        for key in keys:
            if key not in fields:
                continue
            val = _to_float(fields.get(key))
            if val is not None:
                return key, float(val)
        return None, None

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

    explicit_mean_lb_key, explicit_mean_lb = _first_numeric(override_fields, _EXPLICIT_MEAN_LB_FIELDS)
    explicit_mean_ub_key, explicit_mean_ub = _first_numeric(override_fields, _EXPLICIT_MEAN_UB_FIELDS)
    _, explicit_world_lb = _first_numeric(override_fields, _EXPLICIT_WORLD_LB_FIELDS)
    _, explicit_world_ub = _first_numeric(override_fields, _EXPLICIT_WORLD_UB_FIELDS)

    if explicit_world_lb is not None:
        world_lb = max(world_lb, explicit_world_lb)
    if explicit_world_ub is not None:
        world_ub = min(world_ub, explicit_world_ub)

    if override_mode == "zero":
        out_like = True
        lock_exact = True
        target_val = 0.0
        force_inactive = True
        eligible = False
        constraint_kind = "zero_lock"
    elif override_mode == "force_inactive":
        force_inactive = True
        constraint_kind = "force_inactive"
    elif override_mode == "force_active":
        force_active = True
        constraint_kind = "force_active"

    # Inference path for payloads that only send bound values.
    inferred_lock = False
    inferred_band = False
    if override_mode in {"lock", "band"}:
        inferred_lock = override_mode == "lock"
        inferred_band = override_mode == "band"
    elif target_val is None and lock_exact:
        inferred_lock = True
    elif target_val is None and explicit_mean_lb is not None and explicit_mean_ub is not None:
        # Legacy payloads with lb/ub values can represent lock/band.
        if abs(float(explicit_mean_lb) - float(explicit_mean_ub)) <= 1e-8:
            inferred_lock = True
        else:
            inferred_band = True
    elif target_val is None and explicit_mean_ub is not None and explicit_mean_lb is None and override_mode == "cap":
        world_ub = min(world_ub, float(explicit_mean_ub))
        constraint_kind = "cap"
    elif target_val is None and explicit_mean_lb is not None and explicit_mean_ub is None and override_mode == "floor":
        # Legacy floor mode -> mean floor only (no world floor).
        mean_lb = max(mean_lb, float(explicit_mean_lb))
        constraint_kind = "band"

    if target_val is not None:
        t = float(np.clip(target_val, 0.0, policy.default_ub_minutes))
        if lock_exact or inferred_lock:
            mean_lb = max(mean_lb, t)
            mean_ub = min(mean_ub, t)
            constraint_kind = "lock"
        else:
            band_eps = _to_float(override_fields.get("minutes_band_eps"))
            if band_eps is None:
                band_eps = float(policy.legacy_target_band_eps)
            band_eps = max(0.0, float(band_eps))
            mean_lb = max(mean_lb, t - band_eps)
            mean_ub = min(mean_ub, t + band_eps)
            constraint_kind = "band"
    elif inferred_lock:
        if explicit_mean_lb is None and explicit_mean_ub is None:
            lock_target = float(np.clip(b_minutes, 0.0, policy.default_ub_minutes))
        elif explicit_mean_lb is None:
            lock_target = float(np.clip(explicit_mean_ub, 0.0, policy.default_ub_minutes))
        else:
            lock_target = float(np.clip(explicit_mean_lb, 0.0, policy.default_ub_minutes))
        mean_lb = max(mean_lb, lock_target)
        mean_ub = min(mean_ub, lock_target)
        constraint_kind = "lock"
    elif inferred_band:
        if explicit_mean_lb is not None:
            mean_lb = max(mean_lb, float(explicit_mean_lb))
        if explicit_mean_ub is not None:
            mean_ub = min(mean_ub, float(explicit_mean_ub))
        constraint_kind = "band"
    else:
        # Fallback compatibility behavior.
        if explicit_mean_lb is not None and explicit_mean_ub is not None:
            mean_lb = max(mean_lb, float(explicit_mean_lb))
            mean_ub = min(mean_ub, float(explicit_mean_ub))
            if abs(mean_lb - mean_ub) <= 1e-8:
                constraint_kind = "lock"
            else:
                constraint_kind = "band"
        elif explicit_mean_ub is not None:
            world_ub = min(world_ub, float(explicit_mean_ub))
            constraint_kind = "cap"
        elif explicit_mean_lb is not None:
            mean_lb = max(mean_lb, float(explicit_mean_lb))
            constraint_kind = "band"

    if override_mode == "cap":
        cap_val = _to_float(override_fields.get("cap_value"))
        if cap_val is None:
            if explicit_mean_ub is not None:
                cap_val = explicit_mean_ub
            else:
                _, cap_val = _first_numeric(override_fields, _EXPLICIT_CAP_FIELDS)
        if cap_val is not None:
            world_ub = min(world_ub, float(cap_val))
            constraint_kind = "cap"
    elif override_mode is None:
        # Legacy payload path where a lone cap field should remain a world cap.
        if explicit_mean_lb_key in _EXPLICIT_FLOOR_FIELDS and explicit_mean_ub_key is None and explicit_mean_lb is not None:
            mean_lb = max(mean_lb, float(explicit_mean_lb))
            if constraint_kind == "none":
                constraint_kind = "band"
        if explicit_mean_ub_key in _EXPLICIT_CAP_FIELDS and explicit_mean_lb_key is None and explicit_mean_ub is not None:
            world_ub = min(world_ub, float(explicit_mean_ub))
            constraint_kind = "cap"

    mean_lb = float(np.clip(mean_lb, 0.0, policy.default_ub_minutes))
    mean_ub = float(np.clip(mean_ub, 0.0, policy.default_ub_minutes))
    world_lb = float(np.clip(world_lb, 0.0, policy.default_ub_minutes))
    world_ub = float(np.clip(world_ub, 0.0, policy.default_ub_minutes))

    if not eligible or force_inactive or world_ub <= 1e-12:
        eligible = False
        force_inactive = True
        force_active = False
        mean_lb = 0.0
        mean_ub = 0.0
        world_lb = 0.0
        world_ub = 0.0
        constraint_kind = "zero_lock"

    if mean_lb > 0.0:
        force_active = True
        if constraint_kind == "none":
            constraint_kind = "band"

    if mean_lb > mean_ub + 1e-8:
        if strict:
            raise ValueError(
                "Override mean bounds invalid after compile: "
                f"game_id={row_key.game_id} player_id={row_key.player_id} "
                f"mean_lb={mean_lb:.3f} mean_ub={mean_ub:.3f}"
            )
        # Non-strict fallback: collapse to midpoint clipped to legal range.
        mid = float(np.clip(0.5 * (mean_lb + mean_ub), 0.0, policy.default_ub_minutes))
        mean_lb = mid
        mean_ub = mid
        constraint_kind = "lock"

    if world_lb > world_ub + 1e-8:
        if strict:
            raise ValueError(
                "Override world bounds invalid after compile: "
                f"game_id={row_key.game_id} player_id={row_key.player_id} "
                f"world_lb={world_lb:.3f} world_ub={world_ub:.3f}"
            )
        world_mid = float(np.clip(0.5 * (world_lb + world_ub), 0.0, policy.default_ub_minutes))
        world_lb = world_mid
        world_ub = world_mid

    return _CompiledConstraint(
        mean_lb_minutes=float(mean_lb),
        mean_ub_minutes=float(mean_ub),
        world_lb_minutes=float(world_lb),
        world_ub_minutes=float(world_ub),
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
    if bool((group["mean_lb_minutes"] > default_lb + 1e-9).any()):
        return True
    if bool((group["mean_ub_minutes"] < default_ub - 1e-9).any()):
        return True
    if bool((group["world_lb_minutes"] > default_lb + 1e-9).any()):
        return True
    if bool((group["world_ub_minutes"] < default_ub - 1e-9).any()):
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
    """Compile and apply minutes overrides as deterministic mean constraints.

    The output is a resolved per-player minutes constraint frame containing:
    `game_id, team_id, player_id, b_minutes, mu_minutes, mean_lb_minutes, mean_ub_minutes,
    world_lb_minutes, world_ub_minutes, eligible, force_active, force_inactive, weight`.
    Compatibility aliases are kept as:
    - `lb_minutes = mean_lb_minutes`
    - `ub_minutes = mean_ub_minutes`

    Legacy payload compatibility:
    - `minutes_delta` / `minutes_target` are compiled into lock or band constraints.
    - `status=out`, `ops_depth_role=out`, `out=true`, `dnp=true` compile to unbreakable zero-locks.
    """

    if baseline_minutes_df.empty:
        empty = baseline_minutes_df.copy()
        for col in (
            "b_minutes",
            "mu_minutes",
            "mean_lb_minutes",
            "mean_ub_minutes",
            "world_lb_minutes",
            "world_ub_minutes",
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
                "mean_lb_minutes": compiled.mean_lb_minutes,
                "mean_ub_minutes": compiled.mean_ub_minutes,
                "world_lb_minutes": compiled.world_lb_minutes,
                "world_ub_minutes": compiled.world_ub_minutes,
                # Backward compatibility aliases.
                "lb_minutes": compiled.mean_lb_minutes,
                "ub_minutes": compiled.mean_ub_minutes,
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
        mean_lb = g["mean_lb_minutes"].to_numpy(dtype=float)
        mean_ub = g["mean_ub_minutes"].to_numpy(dtype=float)
        world_lb = g["world_lb_minutes"].to_numpy(dtype=float)
        world_ub = g["world_ub_minutes"].to_numpy(dtype=float)
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
                    mean_lb,
                    mean_ub,
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
                    work.loc[group_idx, "mean_lb_minutes"] = float(cfg.default_lb_minutes)
                    work.loc[group_idx, "mean_ub_minutes"] = float(cfg.default_ub_minutes)
                    work.loc[group_idx, "world_lb_minutes"] = float(cfg.default_lb_minutes)
                    work.loc[group_idx, "world_ub_minutes"] = float(cfg.default_ub_minutes)
                    work.loc[group_idx, "lb_minutes"] = float(cfg.default_lb_minutes)
                    work.loc[group_idx, "ub_minutes"] = float(cfg.default_ub_minutes)
                    work.loc[group_idx, "force_active"] = False
                    work.loc[group_idx, "force_inactive"] = False
                    work.loc[group_idx, "eligible"] = True
                    work.loc[group_idx, "constraint_kind"] = "none"
                    mean_lb = np.full_like(b, float(cfg.default_lb_minutes), dtype=float)
                    mean_ub = np.full_like(b, float(cfg.default_ub_minutes), dtype=float)
                    world_lb = np.full_like(b, float(cfg.default_lb_minutes), dtype=float)
                    world_ub = np.full_like(b, float(cfg.default_ub_minutes), dtype=float)
                else:
                    mu, relax_meta = _relax_and_project(b, mean_lb, mean_ub, w, policy=cfg)
                    action = "relax_and_project"
        else:
            mu = b.copy()
            action = "no_override_baseline"

        work.loc[group_idx, "mu_minutes"] = mu

        lock_mask = np.abs(mean_lb - mean_ub) <= 1e-8
        hit_floor = np.flatnonzero(np.isclose(mu, mean_lb, atol=1e-6) | (mu < mean_lb + 1e-6))
        hit_cap = np.flatnonzero(np.isclose(mu, mean_ub, atol=1e-6) | (mu > mean_ub - 1e-6))
        bounds_diff_mask = (np.abs(mean_lb - world_lb) > 1e-8) | (np.abs(mean_ub - world_ub) > 1e-8)
        player_ids = g["player_id"].astype(str).to_numpy(dtype=str)

        diag_row: dict[str, Any] = {
            "game_id": g["game_id"].iloc[0],
            "team_id": g["team_id"].iloc[0],
            "sum_mean_lb": float(mean_lb.sum()),
            "sum_mean_ub": float(mean_ub.sum()),
            "sum_world_lb": float(world_lb.sum()),
            "sum_world_ub": float(world_ub.sum()),
            # Backward-compatible aliases for existing dashboards/log parsing.
            "sum_lb": float(mean_lb.sum()),
            "sum_ub": float(mean_ub.sum()),
            "sum_mu": float(mu.sum()),
            "locked_minutes_total": float(mu[lock_mask].sum()),
            "remaining_to_fill": float(cfg.target_team_minutes - mu[lock_mask].sum()),
            "hit_floor_player_ids": [player_ids[j] for j in hit_floor.tolist()],
            "hit_cap_player_ids": [player_ids[j] for j in hit_cap.tolist()],
            "mean_world_bounds_differ": bool(bounds_diff_mask.any()),
            "mean_world_bounds_differ_player_ids": [player_ids[j] for j in np.flatnonzero(bounds_diff_mask).tolist()],
            "mean_world_bounds_note": (
                "mean bounds apply to mu projection only; world bounds apply per-world clamp/project"
                if bool(bounds_diff_mask.any())
                else None
            ),
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
        "mean_lb_minutes",
        "mean_ub_minutes",
        "world_lb_minutes",
        "world_ub_minutes",
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
