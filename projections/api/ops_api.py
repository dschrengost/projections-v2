from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi import BackgroundTasks
from pydantic import BaseModel

from projections import paths
from projections.artifacts.unified_projections import load_projections_df, load_summary, resolve_unified_run_dir
from projections.ops.manual_availability import (
    apply_manual_overrides_to_frame,
    clear_manual_override,
    list_manual_overrides,
    load_manual_overrides_df,
    upsert_manual_override,
)
from projections.ops.overrides import (
    OPS_OVERRIDES_VERSION,
    OverrideKey,
    USAGE_RATE_FIELDS,
    apply_overrides_to_minutes_df,
    apply_overrides_to_rates_df,
    clear_overrides,
    list_overrides,
    load_overrides_map,
    overrides_path,
    upsert_overrides,
)
from projections.overrides import MinutesOverrideV2Policy, apply_minutes_overrides_v2
from projections.ops.worlds_patch import patch_worlds_matrix_for_game
from projections.pipeline.effective_inputs import write_effective_minutes_layer

router = APIRouter(prefix="/api/ops", tags=["ops"])


def _normalize_id_str_series(series: pd.Series) -> pd.Series:
    """Normalize identifier values to stable string tokens (e.g., 123.0 -> '123')."""

    if series.empty:
        return pd.Series([], index=series.index, dtype="string")
    out = series.astype("string", copy=False).fillna("")
    numeric = pd.to_numeric(series, errors="coerce")
    int_like = numeric.notna() & (numeric % 1 == 0)
    if int_like.any():
        out = out.where(~int_like, numeric.where(int_like).astype("Int64").astype("string"))
    return out.str.replace(r"\.0$", "", regex=True)


def _parse_date(value: str | None) -> date:
    if not value:
        raise HTTPException(status_code=400, detail="Missing date (YYYY-MM-DD).")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.") from exc


def _read_pointer_run_id(pointer_path: Path) -> str | None:
    if not pointer_path.exists():
        return None
    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    run_id = payload.get("run_id")
    return str(run_id) if run_id else None


def _resolve_run_dir(base_dir: Path, *, run_id: str | None, parquet_name: str) -> tuple[Path, str | None]:
    if run_id:
        candidate = base_dir / f"run={run_id}"
        if (candidate / parquet_name).exists():
            return candidate, run_id
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found under {base_dir}.")

    latest = _read_pointer_run_id(base_dir / "latest_run.json")
    if latest:
        candidate = base_dir / f"run={latest}"
        if (candidate / parquet_name).exists():
            return candidate, latest

    direct = base_dir / parquet_name
    if direct.exists():
        return base_dir, None

    run_dirs = sorted([p for p in base_dir.glob("run=*") if p.is_dir()], reverse=True)
    for candidate in run_dirs:
        if (candidate / parquet_name).exists():
            resolved = candidate.name.split("=", 1)[1] if candidate.name.startswith("run=") else None
            return candidate, resolved

    raise HTTPException(status_code=404, detail=f"No artifact found under {base_dir}.")


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _to_json_compatible(value: Any) -> Any:
    """Recursively coerce pandas/numpy scalars into JSON-native Python types."""

    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_compatible(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_json_compatible(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _to_json_compatible(value.item())
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _parse_iso_minutes(value: Any) -> float:
    """Parse NBA API minute strings like PT31M27.00S into float minutes."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    if text.startswith("PT"):
        mins = 0.0
        try:
            body = text[2:]
            if "M" in body:
                mins_part, rest = body.split("M", 1)
                mins += float(mins_part or 0)
            else:
                rest = body
            if "S" in rest:
                secs_part = rest.replace("S", "")
                if secs_part:
                    mins += float(secs_part) / 60.0
            return mins
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _dk_fpts_from_stats(
    *,
    pts: float,
    reb: float,
    ast: float,
    stl: float,
    blk: float,
    tov: float,
) -> float:
    base = pts + 1.25 * reb + 1.5 * ast + 2.0 * stl + 2.0 * blk - 0.5 * tov
    stats_10 = sum(1 for x in (pts, reb, ast, stl, blk) if x >= 10)
    if stats_10 >= 3:
        return base + 3.0
    if stats_10 >= 2:
        return base + 1.5
    return base


def _iter_boxscore_partitions_desc(data_root: Path, *, before_date: date) -> list[tuple[date, Path]]:
    """Return unique boxscore day partitions before a date, newest first."""
    root = data_root / "bronze" / "boxscores_raw"
    if not root.exists():
        return []
    by_day: dict[date, tuple[int, Path]] = {}
    for season_dir in root.glob("season=*"):
        if not season_dir.is_dir():
            continue
        season_name = season_dir.name.split("=", 1)[1] if "=" in season_dir.name else ""
        try:
            season_rank = int(season_name)
        except ValueError:
            season_rank = -1
        for day_dir in season_dir.glob("date=*"):
            if not day_dir.is_dir():
                continue
            day_str = day_dir.name.split("=", 1)[1] if "=" in day_dir.name else day_dir.name
            try:
                day = date.fromisoformat(day_str)
            except ValueError:
                continue
            if day >= before_date:
                continue
            box_path = day_dir / "boxscores_raw.parquet"
            if box_path.exists():
                prev = by_day.get(day)
                if prev is None or season_rank >= prev[0]:
                    by_day[day] = (season_rank, box_path)
    out = [(day, item[1]) for day, item in by_day.items()]
    return sorted(out, key=lambda item: item[0], reverse=True)


def _extract_player_line_from_payload(
    *,
    payload: dict[str, Any],
    game_date: date,
    player_id: str,
) -> dict[str, Any] | None:
    target_pid = str(player_id).strip()
    if not target_pid:
        return None
    game_id = str(payload.get("game_id") or "").strip()
    if not game_id:
        return None

    for side in ("away", "home"):
        team_data = payload.get(side, {}) or {}
        opp_data = payload.get("home" if side == "away" else "away", {}) or {}
        players = team_data.get("players", [])
        for player in players:
            pid = str(player.get("person_id") or player.get("player_id") or "").strip()
            if pid != target_pid:
                continue

            status = str(player.get("status") or "").upper()
            stats = player.get("statistics", {}) or {}
            minutes = _parse_iso_minutes(stats.get("minutes"))

            # Skip inactive-with-zero rows to keep this aligned with meaningful game logs.
            if status and status != "ACTIVE" and minutes <= 0:
                return None

            pts = float(pd.to_numeric(stats.get("points", 0), errors="coerce") or 0.0)
            reb = float(pd.to_numeric(stats.get("reboundsTotal", 0), errors="coerce") or 0.0)
            ast = float(pd.to_numeric(stats.get("assists", 0), errors="coerce") or 0.0)
            stl = float(pd.to_numeric(stats.get("steals", 0), errors="coerce") or 0.0)
            blk = float(pd.to_numeric(stats.get("blocks", 0), errors="coerce") or 0.0)
            tov = float(pd.to_numeric(stats.get("turnovers", 0), errors="coerce") or 0.0)

            return {
                "game_date": game_date.isoformat(),
                "game_id": game_id,
                "team_tricode": team_data.get("team_tricode"),
                "opponent_tricode": opp_data.get("team_tricode"),
                "minutes": round(minutes, 2),
                "pts": round(pts, 1),
                "reb": round(reb, 1),
                "ast": round(ast, 1),
                "stl": round(stl, 1),
                "blk": round(blk, 1),
                "to": round(tov, 1),
                "fpts": round(
                    _dk_fpts_from_stats(pts=pts, reb=reb, ast=ast, stl=stl, blk=blk, tov=tov),
                    2,
                ),
            }
    return None


def _load_player_last_games(
    *,
    data_root: Path,
    player_id: str,
    before_date: date,
    limit: int,
) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    seen_game_ids: set[str] = set()
    for game_day, box_path in _iter_boxscore_partitions_desc(data_root, before_date=before_date):
        if len(games) >= limit:
            break
        try:
            raw_df = pd.read_parquet(box_path)
        except Exception:
            continue
        if raw_df.empty or "payload" not in raw_df.columns:
            continue

        for _, row in raw_df.iterrows():
            payload_raw = row.get("payload")
            if payload_raw is None:
                continue
            try:
                payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            record = _extract_player_line_from_payload(
                payload=payload,
                game_date=game_day,
                player_id=player_id,
            )
            if record is not None:
                game_id = str(record.get("game_id") or "").strip()
                if game_id and game_id in seen_game_ids:
                    continue
                if game_id:
                    seen_game_ids.add(game_id)
                games.append(record)
                break
    return games[:limit]


_V2_MODE_VALUES = {
    "none",
    "lock",
    "band",
    "cap",
    "floor",
    "zero",
    "force_active",
    "force_inactive",
}
_LEGACY_MINUTES_FIELDS = {
    "minutes_target",
    "minutes_lock",
    "minutes_delta",
    "minutes_p10",
    "minutes_p50",
    "minutes_p90",
    "minutes_p10_cond",
    "minutes_p50_cond",
    "minutes_p90_cond",
    "ops_depth_role",
    "status",
}
_V2_FIELDS = {
    "override_mode",
    "mean_lb_minutes",
    "mean_ub_minutes",
    "world_lb_minutes",
    "world_ub_minutes",
    "lb_minutes",
    "ub_minutes",
    "force_active",
    "force_inactive",
    "eligible",
    "protect_weight",
    "weight",
}
_V2_CONFLICT_FIELDS = _LEGACY_MINUTES_FIELDS | _V2_FIELDS


def _save_overrides_payload(
    *,
    game_date: date,
    records: list[dict[str, Any]],
    data_root: Path,
) -> None:
    payload = {
        "version": OPS_OVERRIDES_VERSION,
        "game_date": game_date.isoformat(),
        "updated_at": _utc_now_iso(),
        "overrides": sorted(records, key=lambda r: (str(r.get("game_id", "")), str(r.get("player_id", "")))),
    }
    path = overrides_path(game_date, data_root=data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.v2.json")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _baseline_minutes_column(df: pd.DataFrame) -> str:
    for col in ("minutes_final", "minutes_p50_cond", "minutes_p50", "minutes_sim_uncond_mean", "minutes_mean"):
        if col in df.columns:
            return col
    raise HTTPException(
        status_code=400,
        detail="No baseline minutes column found for v2 compile.",
    )


class ManualAvailabilityOverrideRequest(BaseModel):
    date: str
    game_id: str
    player_id: str
    override_type: str
    entered_by: str
    reason_code: str | None = None
    reason_text: str | None = None
    source_label: str | None = None
    expires_ts: str | None = None
    player_name: str | None = None
    team_id: int | None = None
    team_tricode: str | None = None


class OpsPlayerOverrideUpdate(BaseModel):
    game_id: str
    player_id: str

    sticky_fields: list[str] | None = None

    ops_depth_role: str | None = None

    status: str | None = None
    is_confirmed_starter: bool | None = None
    is_projected_starter: bool | None = None
    play_prob: float | None = None

    minutes_p10: float | None = None
    minutes_p50: float | None = None
    minutes_p90: float | None = None
    minutes_p10_cond: float | None = None
    minutes_p50_cond: float | None = None
    minutes_p90_cond: float | None = None
    minutes_target: float | None = None  # Absolute minutes target (0-48). Implies lock.
    minutes_lock: bool | None = None  # If True, fix minutes to target (or current minutes when target unset).
    minutes_delta: float | None = None  # Additive adjustment to model quantiles (e.g., +5 or -3)

    pred_fga2_per_min: float | None = None
    pred_fga3_per_min: float | None = None
    pred_fta_per_min: float | None = None
    pred_ast_per_min: float | None = None
    pred_tov_per_min: float | None = None
    pred_oreb_per_min: float | None = None
    pred_dreb_per_min: float | None = None
    pred_stl_per_min: float | None = None
    pred_blk_per_min: float | None = None
    pred_fg2_pct: float | None = None
    pred_fg3_pct: float | None = None
    pred_ft_pct: float | None = None


class OpsUpsertOverridesRequest(BaseModel):
    date: str
    updates: list[OpsPlayerOverrideUpdate]
    note: str | None = None


class OpsV2PlayerOverride(BaseModel):
    player_id: str
    mode: str = "none"
    lock_value: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    cap_value: float | None = None
    floor_value: float | None = None
    protect_weight: bool | None = None


class OpsV2ApplyRequest(BaseModel):
    date: str
    game_id: str
    run_id: str | None = None
    override_infeasible: str = "error"
    overrides: list[OpsV2PlayerOverride]


def _resolve_manual_override_player_context(
    *,
    game_date: date,
    game_id: str,
    player_id: str,
    data_root: Path,
    player_name: str | None,
    team_id: int | None,
    team_tricode: str | None,
) -> dict[str, Any]:
    if player_name and team_id is not None and team_tricode:
        return {
            "player_name": str(player_name),
            "team_id": int(team_id),
            "team_tricode": str(team_tricode).strip().upper(),
        }

    run_dir, _ = resolve_unified_run_dir(data_root, game_date, run_id=None)
    if run_dir is not None:
        unified_df = load_projections_df(run_dir)
        if not unified_df.empty:
            work = unified_df.copy()
            work["game_id"] = _normalize_id_str_series(work["game_id"])
            work["player_id"] = _normalize_id_str_series(work["player_id"])
            match = work.loc[
                work["game_id"].eq(str(game_id)) & work["player_id"].eq(str(player_id))
            ].copy()
            if not match.empty:
                row = match.iloc[0]
                resolved_team_id = pd.to_numeric(
                    pd.Series([row.get("team_id")]), errors="coerce"
                ).iloc[0]
                if pd.isna(resolved_team_id):
                    resolved_team_id = team_id
                if resolved_team_id is None or pd.isna(resolved_team_id):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Could not resolve team_id for manual availability override. "
                            "Provide team_id explicitly or ensure unified projections contain it."
                        ),
                    )
                return {
                    "player_name": str(row.get("player_name") or player_name or ""),
                    "team_id": int(resolved_team_id),
                    "team_tricode": str(row.get("team_tricode") or team_tricode or "").strip().upper(),
                }

    if player_name and team_id is not None and team_tricode:
        return {
            "player_name": str(player_name),
            "team_id": int(team_id),
            "team_tricode": str(team_tricode).strip().upper(),
        }

    raise HTTPException(
        status_code=400,
        detail=(
            "Could not resolve player/team context for manual availability override. "
            "Provide player_name, team_id, and team_tricode or ensure unified projections exist for the slate."
        ),
    )


@router.get("/overrides")
def get_overrides(date: str = Query(...)) -> dict[str, Any]:
    game_date = _parse_date(date)
    return {"date": game_date.isoformat(), "overrides": list_overrides(game_date)}


@router.post("/overrides")
def post_overrides(req: OpsUpsertOverridesRequest) -> dict[str, Any]:
    game_date = _parse_date(req.date)
    updates = [item.model_dump(exclude_none=True) for item in req.updates]
    data_root = paths.data_path()
    path = overrides_path(game_date, data_root=data_root)
    previous_text: str | None = None
    if path.exists():
        try:
            previous_text = path.read_text(encoding="utf-8")
        except OSError:
            previous_text = None

    def _restore_previous() -> None:
        if previous_text is None:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.restore.json")
            tmp.write_text(previous_text, encoding="utf-8")
            tmp.replace(path)
        except OSError:
            # Best-effort: fall back to non-atomic write.
            path.write_text(previous_text, encoding="utf-8")

    merged = upsert_overrides(game_date, updates, note=req.note, data_root=data_root)

    # Validate hard targets/locks immediately so the UI can surface infeasibility
    # (e.g., sum locked targets > 240) without waiting for a pipeline run.
    try:
        minutes_base_dir = data_root / "artifacts" / "minutes_v1" / "daily" / game_date.isoformat()
        minutes_dir, _ = _resolve_run_dir(minutes_base_dir, run_id=None, parquet_name="minutes.parquet")
        minutes_path = minutes_dir / "minutes.parquet"
        minutes_df = pd.read_parquet(minutes_dir / "minutes.parquet")
        apply_overrides_to_minutes_df(
            minutes_df,
            game_date=game_date,
            data_root=data_root,
            force_reconcile=True,
        )

        # Materialize the effective minutes layer in-place so /api/minutes reflects the change immediately.
        write_effective_minutes_layer(
            game_date=game_date,
            minutes_path=minutes_path,
            out_dir=minutes_dir,
            data_root=data_root,
            source="gameview",
        )
    except FileNotFoundError:
        # Minutes artifact missing; skip validation (dev/backfill scenarios).
        pass
    except ValueError as exc:
        _restore_previous()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        _restore_previous()
        raise

    return {"date": game_date.isoformat(), "overrides": merged}


@router.get("/manual-availability-overrides")
def get_manual_availability_overrides(
    date: str = Query(...),
    active_only: bool = Query(True),
) -> dict[str, Any]:
    game_date = _parse_date(date)
    items = list_manual_overrides(
        game_date,
        data_root=paths.data_path(),
        active_only=bool(active_only),
    )
    return {"date": game_date.isoformat(), "overrides": items}


@router.post("/manual-availability-overrides")
def post_manual_availability_override(
    req: ManualAvailabilityOverrideRequest,
) -> dict[str, Any]:
    game_date = _parse_date(req.date)
    data_root = paths.data_path()
    context = _resolve_manual_override_player_context(
        game_date=game_date,
        game_id=str(req.game_id),
        player_id=str(req.player_id),
        data_root=data_root,
        player_name=req.player_name,
        team_id=req.team_id,
        team_tricode=req.team_tricode,
    )
    try:
        created = upsert_manual_override(
            game_date,
            game_id=req.game_id,
            player_id=req.player_id,
            player_name=context["player_name"],
            team_id=context["team_id"],
            team_tricode=context["team_tricode"],
            override_type=req.override_type,
            entered_by=req.entered_by,
            reason_code=req.reason_code,
            reason_text=req.reason_text,
            source_label=req.source_label,
            expires_ts=req.expires_ts,
            data_root=data_root,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "date": game_date.isoformat(),
        "override": created,
        "material_change_detected": True,
        "active_overrides": list_manual_overrides(
            game_date,
            data_root=data_root,
            active_only=True,
        ),
    }


def _infer_v2_mode(fields: dict[str, Any]) -> str:
    if not fields:
        return "none"
    mean_lb = pd.to_numeric(
        pd.Series([fields.get("mean_lb_minutes", fields.get("lb_minutes"))]), errors="coerce"
    ).fillna(0.0).iloc[0]
    mean_ub = pd.to_numeric(
        pd.Series([fields.get("mean_ub_minutes", fields.get("ub_minutes"))]), errors="coerce"
    ).fillna(48.0).iloc[0]
    world_ub = pd.to_numeric(
        pd.Series([fields.get("world_ub_minutes", fields.get("ub_minutes"))]), errors="coerce"
    ).fillna(48.0).iloc[0]
    force_active = bool(fields.get("force_active"))
    force_inactive = bool(fields.get("force_inactive"))
    eligible = fields.get("eligible")
    if force_inactive and mean_ub <= 0.0 and world_ub <= 0.0 and eligible is False:
        return "zero"
    if force_inactive:
        return "force_inactive"
    if force_active and mean_lb <= 0.0 and mean_ub >= 48.0 and world_ub >= 48.0:
        return "force_active"
    if mean_lb > 0.0 and mean_ub < 48.0:
        return "lock" if abs(mean_lb - mean_ub) <= 1e-6 else "band"
    if world_ub < 48.0 and mean_lb <= 0.0 and mean_ub >= 48.0:
        return "cap"
    if mean_lb > 0.0:
        # Legacy floor is represented as a lower-bound mean band.
        return "band"
    return "none"


def _compile_v2_fields(ovr: OpsV2PlayerOverride) -> dict[str, Any]:
    mode = str(ovr.mode or "none").strip().lower()
    if mode not in _V2_MODE_VALUES:
        raise HTTPException(status_code=400, detail=f"Unsupported v2 override mode: {ovr.mode!r}")

    fields: dict[str, Any] = {}
    if mode == "lock":
        if ovr.lock_value is None:
            raise HTTPException(status_code=400, detail="lock mode requires lock_value")
        val = float(max(0.0, min(48.0, ovr.lock_value)))
        fields["mean_lb_minutes"] = val
        fields["mean_ub_minutes"] = val
        # Compatibility aliases consumed by older readers.
        fields["lb_minutes"] = val
        fields["ub_minutes"] = val
    elif mode == "band":
        if ovr.min_value is None or ovr.max_value is None:
            raise HTTPException(status_code=400, detail="band mode requires min_value and max_value")
        lb = float(max(0.0, min(48.0, ovr.min_value)))
        ub = float(max(0.0, min(48.0, ovr.max_value)))
        if lb > ub:
            raise HTTPException(status_code=400, detail="band mode requires min_value <= max_value")
        fields["mean_lb_minutes"] = lb
        fields["mean_ub_minutes"] = ub
        # Compatibility aliases consumed by older readers.
        fields["lb_minutes"] = lb
        fields["ub_minutes"] = ub
    elif mode == "cap":
        if ovr.cap_value is None:
            raise HTTPException(status_code=400, detail="cap mode requires cap_value")
        cap = float(max(0.0, min(48.0, ovr.cap_value)))
        fields["world_ub_minutes"] = cap
        # Compatibility alias consumed by older readers.
        fields["ub_minutes"] = cap
    elif mode == "floor":
        if ovr.floor_value is None:
            raise HTTPException(status_code=400, detail="floor mode requires floor_value")
        floor = float(max(0.0, min(48.0, ovr.floor_value)))
        fields["mean_lb_minutes"] = floor
        # Compatibility alias consumed by older readers.
        fields["lb_minutes"] = floor
    elif mode == "zero":
        fields["mean_lb_minutes"] = 0.0
        fields["mean_ub_minutes"] = 0.0
        fields["world_lb_minutes"] = 0.0
        fields["world_ub_minutes"] = 0.0
        fields["lb_minutes"] = 0.0
        fields["ub_minutes"] = 0.0
        fields["force_inactive"] = True
        fields["eligible"] = False
    elif mode == "force_active":
        fields["force_active"] = True
    elif mode == "force_inactive":
        fields["mean_lb_minutes"] = 0.0
        fields["mean_ub_minutes"] = 0.0
        fields["world_lb_minutes"] = 0.0
        fields["world_ub_minutes"] = 0.0
        fields["lb_minutes"] = 0.0
        fields["ub_minutes"] = 0.0
        fields["force_inactive"] = True
    elif mode == "none":
        fields = {}

    if fields:
        fields["override_mode"] = mode
        if ovr.protect_weight is not None:
            fields["protect_weight"] = bool(ovr.protect_weight)
    return fields


@router.delete("/overrides")
def delete_overrides(
    date: str = Query(...),
    game_id: str | None = Query(None),
    player_id: str | None = Query(None),
) -> dict[str, Any]:
    game_date = _parse_date(date)
    data_root = paths.data_path()
    path = overrides_path(game_date, data_root=data_root)
    previous_text: str | None = None
    if path.exists():
        try:
            previous_text = path.read_text(encoding="utf-8")
        except OSError:
            previous_text = None

    def _restore_previous() -> None:
        if previous_text is None:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.restore.json")
            tmp.write_text(previous_text, encoding="utf-8")
            tmp.replace(path)
        except OSError:
            # Best-effort: fall back to non-atomic write.
            path.write_text(previous_text, encoding="utf-8")

    remaining = clear_overrides(game_date, game_id=game_id, player_id=player_id, data_root=data_root)

    # Materialize effective minutes after clearing so /api/minutes reflects the change immediately.
    try:
        minutes_base_dir = data_root / "artifacts" / "minutes_v1" / "daily" / game_date.isoformat()
        minutes_dir, _ = _resolve_run_dir(minutes_base_dir, run_id=None, parquet_name="minutes.parquet")
        minutes_path = minutes_dir / "minutes.parquet"
        write_effective_minutes_layer(
            game_date=game_date,
            minutes_path=minutes_path,
            out_dir=minutes_dir,
            data_root=data_root,
            source="gameview",
        )
    except FileNotFoundError:
        # Minutes artifact missing; skip in dev/backfill scenarios.
        pass
    except Exception as exc:
        _restore_previous()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write effective minutes layer after clearing overrides: {exc}",
        ) from exc

    return {"date": game_date.isoformat(), "overrides": remaining}


@router.delete("/manual-availability-overrides/{override_id}")
def delete_manual_availability_override(
    override_id: str,
    date: str = Query(...),
    cleared_by: str = Query(...),
) -> dict[str, Any]:
    game_date = _parse_date(date)
    data_root = paths.data_path()
    try:
        cleared = clear_manual_override(
            game_date,
            override_id=override_id,
            cleared_by=cleared_by,
            data_root=data_root,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if cleared is None:
        raise HTTPException(status_code=404, detail=f"Manual override {override_id} not found.")
    return {
        "date": game_date.isoformat(),
        "cleared_override": cleared,
        "material_change_detected": True,
        "active_overrides": list_manual_overrides(
            game_date,
            data_root=data_root,
            active_only=True,
        ),
    }


@router.get("/overrides-v2")
def get_overrides_v2(
    date: str = Query(...),
    game_id: str | None = Query(None),
) -> dict[str, Any]:
    game_date = _parse_date(date)
    gid = str(game_id) if game_id is not None else None
    overrides_map = load_overrides_map(game_date, data_root=paths.data_path())

    items: list[dict[str, Any]] = []
    for key, record in overrides_map.items():
        if gid is not None and key.game_id != gid:
            continue
        fields = record.get("fields", {}) if isinstance(record.get("fields"), dict) else {}
        v2_fields = {k: v for k, v in fields.items() if k in _V2_FIELDS}
        legacy_fields = sorted(k for k in fields.keys() if k in _LEGACY_MINUTES_FIELDS)
        if not v2_fields and not legacy_fields:
            continue
        items.append(
            {
                "game_id": key.game_id,
                "player_id": key.player_id,
                "mode": _infer_v2_mode(v2_fields),
                "fields": v2_fields,
                "legacy_fields_present": legacy_fields,
                "updated_at": record.get("updated_at"),
                "note": record.get("note"),
            }
        )

    items.sort(key=lambda r: (str(r.get("game_id", "")), str(r.get("player_id", ""))))
    return {"date": game_date.isoformat(), "game_id": gid, "overrides": items}


@router.get("/player-last-games")
def get_player_last_games(
    date: str = Query(..., description="Slate date (YYYY-MM-DD). Returns games strictly before this date."),
    player_id: str = Query(..., description="NBA player_id"),
    limit: int = Query(5, ge=1, le=20, description="Number of recent games to return"),
) -> dict[str, Any]:
    slate_day = _parse_date(date)
    pid = str(player_id).strip()
    if not pid:
        raise HTTPException(status_code=400, detail="player_id is required.")

    data_root = paths.data_path()
    games = _load_player_last_games(
        data_root=data_root,
        player_id=pid,
        before_date=slate_day,
        limit=limit,
    )
    return {
        "date": slate_day.isoformat(),
        "player_id": pid,
        "limit": limit,
        "games": games,
    }


@router.post("/overrides-v2/apply")
def post_overrides_v2_apply(req: OpsV2ApplyRequest) -> dict[str, Any]:
    game_date = _parse_date(req.date)
    gid = str(req.game_id)
    data_root = paths.data_path()
    now_iso = _utc_now_iso()

    mode = str(req.override_infeasible or "error").strip().lower()
    if mode not in {"error", "relax", "ignore"}:
        raise HTTPException(status_code=400, detail="override_infeasible must be one of error|relax|ignore")

    overrides_map = load_overrides_map(game_date, data_root=data_root)
    mutable: dict[OverrideKey, dict[str, Any]] = {k: dict(v) for k, v in overrides_map.items()}

    for item in req.overrides:
        key = OverrideKey.from_values(gid, item.player_id)
        compiled = _compile_v2_fields(item)
        existing = mutable.get(
            key,
            {
                "game_id": gid,
                "player_id": str(item.player_id),
                "fields": {},
                "updated_at": now_iso,
                "note": None,
                "sticky_fields": [],
            },
        )
        existing_fields = existing.get("fields", {}) if isinstance(existing.get("fields"), dict) else {}
        retained = {k: v for k, v in existing_fields.items() if k not in _V2_CONFLICT_FIELDS}
        merged = {**retained, **compiled}

        if merged:
            existing["game_id"] = gid
            existing["player_id"] = str(item.player_id)
            existing["fields"] = merged
            existing["updated_at"] = now_iso
            mutable[key] = existing
        else:
            mutable.pop(key, None)

    projections_run_dir, ctx = resolve_unified_run_dir(data_root, game_date, run_id=req.run_id)
    if projections_run_dir is None:
        raise HTTPException(status_code=404, detail=f"No unified projections found for {game_date.isoformat()}.")
    unified_df = load_projections_df(projections_run_dir)
    if unified_df.empty:
        raise HTTPException(status_code=404, detail=f"Unified projections empty for {game_date.isoformat()}.")

    work = unified_df.copy()
    work["game_id"] = _normalize_id_str_series(work["game_id"])
    work["player_id"] = _normalize_id_str_series(work["player_id"])
    game_df = work.loc[work["game_id"] == gid].copy()
    if game_df.empty:
        raise HTTPException(status_code=404, detail=f"Game {gid} not found in unified projections.")
    baseline_col = _baseline_minutes_column(game_df)
    game_df["minutes_mean"] = pd.to_numeric(game_df[baseline_col], errors="coerce").fillna(0.0).clip(0.0, 48.0)

    payload_overrides = []
    for record in mutable.values():
        if str(record.get("game_id")) != gid:
            continue
        fields = record.get("fields", {}) if isinstance(record.get("fields"), dict) else {}
        if not fields:
            continue
        payload_overrides.append(
            {
                "game_id": gid,
                "player_id": str(record.get("player_id")),
                "fields": fields,
            }
        )

    try:
        resolved_df, diag = apply_minutes_overrides_v2(
            game_df,
            {"overrides": payload_overrides},
            policy=MinutesOverrideV2Policy(override_infeasible=mode),
            seed=None,
            strict=(mode == "error"),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    resolved_df = resolved_df.copy()
    resolved_df["game_id"] = _normalize_id_str_series(resolved_df["game_id"])
    resolved_df["player_id"] = _normalize_id_str_series(resolved_df["player_id"])
    resolved_rows = resolved_df.loc[resolved_df["game_id"] == gid].to_dict(orient="records")

    team_diags = []
    for td in diag.get("team_diagnostics", []) or []:
        if str(td.get("game_id")) == gid:
            team_diags.append(td)

    persisted = []
    for record in mutable.values():
        if str(record.get("game_id")) != gid:
            continue
        fields = record.get("fields", {}) if isinstance(record.get("fields"), dict) else {}
        v2_fields = {k: v for k, v in fields.items() if k in _V2_FIELDS}
        legacy_fields = sorted(k for k in fields.keys() if k in _LEGACY_MINUTES_FIELDS)
        if not v2_fields and not legacy_fields:
            continue
        persisted.append(
            {
                "game_id": gid,
                "player_id": str(record.get("player_id")),
                "mode": _infer_v2_mode(v2_fields),
                "fields": v2_fields,
                "legacy_fields_present": legacy_fields,
                "updated_at": record.get("updated_at"),
            }
        )

    _save_overrides_payload(game_date=game_date, records=list(mutable.values()), data_root=data_root)

    return _to_json_compatible(
        {
            "date": game_date.isoformat(),
            "game_id": gid,
            "applied_at": now_iso,
            "run_context": {"projections_run_id": ctx.resolved_run_id},
            "override_infeasible": mode,
            "resolved_players": resolved_rows,
            "team_diagnostics": team_diags,
            "diag": diag,
            "overrides": sorted(persisted, key=lambda r: str(r.get("player_id", ""))),
        }
    )


class OpsRunWorldsRequest(BaseModel):
    date: str
    game_id: int
    base_run_id: str | None = None
    pin: bool = False
    background: bool = True
    minutes_override_mode: str = "v2"
    override_infeasible: str = "error"


@router.post("/run-worlds")
def post_run_worlds(req: OpsRunWorldsRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    game_date = _parse_date(req.date)
    mode = str(req.minutes_override_mode or "v2").strip().lower()
    if mode not in {"legacy", "v2"}:
        raise HTTPException(status_code=400, detail="minutes_override_mode must be one of legacy|v2")
    infeasible = str(req.override_infeasible or "error").strip().lower()
    if infeasible not in {"error", "relax", "ignore"}:
        raise HTTPException(status_code=400, detail="override_infeasible must be one of error|relax|ignore")
    run_ts = _utc_now_iso()
    if req.background:
        background_tasks.add_task(
            patch_worlds_matrix_for_game,
            game_date=game_date,
            game_id=int(req.game_id),
            base_projections_run_id=req.base_run_id,
            data_root=paths.data_path(),
            pin_projections_run=bool(req.pin),
            minutes_override_mode=mode,
            override_infeasible=infeasible,
            status_run_ts=run_ts,
        )
        return {
            "status": "triggered",
            "date": game_date.isoformat(),
            "game_id": int(req.game_id),
            "run_ts": run_ts,
            "minutes_override_mode": mode,
            "override_infeasible": infeasible,
            "message": "Worlds patch started in background; poll /api/pipeline/status?stage=ops",
        }
    return patch_worlds_matrix_for_game(
        game_date=game_date,
        game_id=int(req.game_id),
        base_projections_run_id=req.base_run_id,
        data_root=paths.data_path(),
        pin_projections_run=bool(req.pin),
        minutes_override_mode=mode,
        override_infeasible=infeasible,
        status_run_ts=run_ts,
    )


@router.get("/game")
def get_game_ops(
    date: str = Query(..., description="Slate date (YYYY-MM-DD)"),
    game_id: str = Query(..., description="NBA game_id to inspect"),
    run_id: str | None = Query(None, description="Optional unified projections run_id"),
) -> dict[str, Any]:
    slate_day = _parse_date(date)
    data_root = paths.data_path()

    projections_run_dir, ctx = resolve_unified_run_dir(data_root, slate_day, run_id=run_id)
    if projections_run_dir is None:
        raise HTTPException(status_code=404, detail=f"No unified projections found for {slate_day.isoformat()}.")

    summary = load_summary(projections_run_dir) or {}
    unified_df = load_projections_df(projections_run_dir)
    if unified_df.empty:
        raise HTTPException(status_code=404, detail=f"Unified projections empty for {slate_day.isoformat()}.")

    gid = str(game_id)
    unified_df = unified_df.copy()
    unified_df["game_id"] = _normalize_id_str_series(unified_df["game_id"])
    unified_game = unified_df.loc[unified_df["game_id"] == gid].copy()
    if unified_game.empty:
        raise HTTPException(status_code=404, detail=f"Game {gid} not found in unified projections.")

    minutes_run_id = summary.get("minutes_run_id")
    rates_run_id = summary.get("rates_run_id")
    sim_run_id = summary.get("sim_run_id")

    minutes_base_dir = data_root / "artifacts" / "minutes_v1" / "daily" / slate_day.isoformat()
    minutes_dir, resolved_minutes_run_id = _resolve_run_dir(
        minutes_base_dir, run_id=str(minutes_run_id) if minutes_run_id else None, parquet_name="minutes.parquet"
    )
    minutes_df = pd.read_parquet(minutes_dir / "minutes.parquet")
    minutes_df = minutes_df.copy()
    minutes_df["game_id"] = _normalize_id_str_series(minutes_df["game_id"])
    minutes_game = minutes_df.loc[minutes_df["game_id"] == gid].copy()

    rates_base_dir = data_root / "gold" / "rates_v1_live" / slate_day.isoformat()
    rates_dir, resolved_rates_run_id = _resolve_run_dir(
        rates_base_dir, run_id=str(rates_run_id) if rates_run_id else None, parquet_name="rates.parquet"
    )
    rates_df = pd.read_parquet(rates_dir / "rates.parquet")
    rates_df = rates_df.copy()
    rates_df["game_id"] = _normalize_id_str_series(rates_df["game_id"])
    rates_game = rates_df.loc[rates_df["game_id"] == gid].copy()

    minutes_effective = apply_overrides_to_minutes_df(
        minutes_game,
        game_date=slate_day,
        data_root=data_root,
        force_reconcile=True,
    )
    manual_overrides_df = load_manual_overrides_df(slate_day, data_root=data_root)
    minutes_effective, _ = apply_manual_overrides_to_frame(
        minutes_effective,
        overrides_df=manual_overrides_df,
    )
    rates_effective = apply_overrides_to_rates_df(rates_game, game_date=slate_day, data_root=data_root)

    game_manual_overrides: dict[str, dict[str, Any]] = {}
    for record in list_manual_overrides(
        slate_day,
        data_root=data_root,
        active_only=True,
    ):
        if str(record.get("game_id")) != gid:
            continue
        game_manual_overrides[str(record.get("player_id"))] = record

    overrides_map = load_overrides_map(slate_day, data_root=data_root)
    game_overrides: dict[str, dict[str, Any]] = {}
    for key, record in overrides_map.items():
        if key.game_id != gid:
            continue
        fields = record.get("fields", {}) if isinstance(record.get("fields"), dict) else {}
        game_overrides[str(key.player_id)] = {
            "game_id": key.game_id,
            "player_id": key.player_id,
            "fields": fields,
            "updated_at": record.get("updated_at"),
            "note": record.get("note"),
        }

    # Build per-player rows keyed by player_id (string).
    out_players: list[dict[str, Any]] = []
    unified_game["player_id"] = _normalize_id_str_series(unified_game["player_id"])
    if not minutes_game.empty:
        minutes_game["player_id"] = _normalize_id_str_series(minutes_game["player_id"])
        minutes_effective["player_id"] = _normalize_id_str_series(minutes_effective["player_id"])
    if not rates_game.empty:
        rates_game["player_id"] = _normalize_id_str_series(rates_game["player_id"])
        rates_effective["player_id"] = _normalize_id_str_series(rates_effective["player_id"])

    minutes_base_by_pid = (
        minutes_game.set_index("player_id").to_dict(orient="index") if not minutes_game.empty else {}
    )
    minutes_eff_by_pid = (
        minutes_effective.set_index("player_id").to_dict(orient="index") if not minutes_effective.empty else {}
    )
    rates_base_by_pid = rates_game.set_index("player_id").to_dict(orient="index") if not rates_game.empty else {}
    rates_eff_by_pid = rates_effective.set_index("player_id").to_dict(orient="index") if not rates_effective.empty else {}

    for _, row in unified_game.iterrows():
        pid = str(row.get("player_id"))
        base_minutes = minutes_base_by_pid.get(pid) or {}
        eff_minutes = minutes_eff_by_pid.get(pid) or {}
        base_rates = rates_base_by_pid.get(pid) or {}
        eff_rates = rates_eff_by_pid.get(pid) or {}
        override = game_overrides.get(pid)
        manual_override = game_manual_overrides.get(pid)

        def _pick(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
            return {k: source.get(k) for k in keys if k in source}

        rates_keys = tuple(k for k in USAGE_RATE_FIELDS if k in base_rates or k in eff_rates)

        out_players.append(
            {
                "player_id": pid,
                "player_name": row.get("player_name"),
                "team_id": row.get("team_id"),
                "team_tricode": row.get("team_tricode"),
                "status": row.get("status"),
                "is_confirmed_starter": row.get("is_confirmed_starter"),
                "is_projected_starter": row.get("is_projected_starter"),
                "effective": row.to_dict(),
                "minutes_baseline": _pick(base_minutes, (
                    "status",
                    "play_prob",
                    "is_confirmed_starter",
                    "is_projected_starter",
                    "minutes_final",
                    "minutes_p10",
                    "minutes_p50",
                    "minutes_p90",
                    "minutes_p10_cond",
                    "minutes_p50_cond",
                    "minutes_p90_cond",
                    "minutes_delta",
                    "minutes_delta_applied",
                    "ops_override_applied",
                    "minutes_contract_version",
                    "minutes_contract_hash",
                    "rotation_prob",
                    "p_rot",
                    "mu_cond",
                    "eligible_flag",
                )),
                "minutes_effective": _pick(eff_minutes, (
                    "status",
                    "play_prob",
                    "is_confirmed_starter",
                    "is_projected_starter",
                    "minutes_final",
                    "minutes_p10",
                    "minutes_p50",
                    "minutes_p90",
                    "minutes_p10_cond",
                    "minutes_p50_cond",
                    "minutes_p90_cond",
                    "minutes_delta",
                    "minutes_delta_applied",
                    "ops_override_applied",
                    "minutes_contract_version",
                    "minutes_contract_hash",
                    "manual_override_id",
                    "manual_override_type",
                    "manual_override_reason_code",
                    "manual_override_reason_text",
                    "manual_override_source_label",
                    "manual_override_entered_by",
                    "manual_override_active",
                    "manual_override_used",
                )),
                "rates_baseline": _pick(base_rates, rates_keys),
                "rates_effective": _pick(eff_rates, rates_keys),
                "override": override,
                "manual_override": manual_override,
            }
        )

    return {
        "date": slate_day.isoformat(),
        "game_id": gid,
        "run_context": {
            "projections_run_id": ctx.resolved_run_id,
            "minutes_run_id": resolved_minutes_run_id or minutes_run_id,
            "rates_run_id": resolved_rates_run_id or rates_run_id,
            "sim_run_id": sim_run_id,
            "blessed_run_id": ctx.blessed_run_id,
            "pinned_run_id": ctx.pinned_run_id,
            "latest_run_id": ctx.latest_run_id,
        },
        "players": out_players,
    }
