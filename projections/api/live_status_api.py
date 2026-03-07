from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from projections import paths
from projections.api.optimizer_service import build_player_pool, get_slates_for_date
from projections.api.slate_analytics_service import load_or_compute_slate_player_analytics
from projections.ops.manual_availability import manual_override_report

router = APIRouter(prefix="/api/live", tags=["live"])
logger = logging.getLogger(__name__)

_RUN_ROOTS: tuple[str, ...] = ("nba_live_v3", "nba_live")
_PUBLISHED_POINTER_CANDIDATES: tuple[str, ...] = ("latest_run.json", "LATEST/current.json")
_SOURCE_KEYS: tuple[str, ...] = ("injuries", "lineups", "odds", "props", "roster")


def _safe_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _resolve_default_draft_group_id(game_date: str, requested_draft_group_id: int | None) -> int:
    if requested_draft_group_id is not None:
        return int(requested_draft_group_id)

    slates = get_slates_for_date(game_date, slate_type="all")
    if not slates:
        raise HTTPException(status_code=404, detail=f"No slates found for {game_date}.")

    def _rank(slate: dict[str, Any]) -> tuple[int, int, int, str]:
        slate_type = str(slate.get("slate_type") or "").strip().lower()
        games = slate.get("games") if isinstance(slate.get("games"), list) else []
        n_games = len(games)
        n_contests = int(slate.get("n_contests") or 0)
        earliest = str(slate.get("earliest_start") or "")
        # Prefer main slates, then broadest slate, then earliest-starting.
        return (0 if slate_type == "main" else 1, -n_games, -n_contests, earliest)

    selected = sorted(slates, key=_rank)[0]
    draft_group_id = selected.get("draft_group_id")
    if draft_group_id is None:
        raise HTTPException(status_code=404, detail=f"Unable to resolve draft group for {game_date}.")
    return int(draft_group_id)


def _build_metric_leaders(
    players: list[dict[str, Any]],
    *,
    key: str,
    top_n: int,
    reverse: bool = True,
) -> list[dict[str, Any]]:
    ranked = [
        row
        for row in players
        if isinstance(row, dict) and _safe_float(row.get(key)) is not None
    ]
    ranked.sort(key=lambda row: float(row.get(key) or 0.0), reverse=reverse)
    return ranked[: max(1, int(top_n))]


def _parse_date(value: str | None) -> date:
    if not value:
        return date.today()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.") from exc


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_run_report_dir(*, data_root: Path, game_date: str, run_id: str) -> Path | None:
    for root_name in _RUN_ROOTS:
        candidate = data_root / "artifacts" / "runs" / root_name / f"game_date={game_date}" / f"run={run_id}"
        if candidate.exists():
            return candidate
    return None


def _resolve_run_manifest(*, data_root: Path, game_date: str, run_id: str | None) -> dict[str, Any] | None:
    if not run_id:
        return None
    for root_name in ("nba_live", "nba_live_v3"):
        manifest_path = data_root / "artifacts" / "runs" / root_name / f"game_date={game_date}" / f"run={run_id}" / "manifest.json"
        payload = _read_json(manifest_path)
        if payload is not None:
            return payload
    return None


def _load_latest_published_pointer(*, data_root: Path, game_date: str) -> dict[str, Any] | None:
    dataset_dir = data_root / "artifacts" / "projections" / game_date
    for candidate_name in _PUBLISHED_POINTER_CANDIDATES:
        payload = _read_json(dataset_dir / candidate_name)
        if payload is not None:
            return payload
    return None


def _list_run_ids(*, data_root: Path, game_date: str) -> list[str]:
    run_ids: set[str] = set()
    for root_name in _RUN_ROOTS:
        date_dir = data_root / "artifacts" / "runs" / root_name / f"game_date={game_date}"
        if not date_dir.exists():
            continue
        for candidate in date_dir.iterdir():
            if candidate.is_dir() and candidate.name.startswith("run="):
                run_ids.add(candidate.name.split("=", 1)[1])
    return sorted(run_ids)


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _iso_or_none(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _status_reason_from_skip(skip_report: dict[str, Any] | None) -> str:
    reason = str((skip_report or {}).get("reason") or "").strip()
    return reason or "no_new_candidate_needed"


def _classify_run_status(
    *,
    run_id: str,
    published_run_id: str | None,
    published_as_of_ts: str | None,
    manifest: dict[str, Any] | None,
    report_dir: Path | None,
) -> tuple[str, str]:
    if run_id == published_run_id:
        return "published", "matches_latest_published_pointer"

    skip_report = None if report_dir is None else _read_json(report_dir / "skip_report.json")
    stale_report = None if report_dir is None else _read_json(report_dir / "stale_publish_report.json")
    superseded_report = None if report_dir is None else _read_json(report_dir / "publish_superseded_report.json")

    if skip_report is not None:
        return "waiting_for_fresh_input", _status_reason_from_skip(skip_report)

    if stale_report is not None and bool(stale_report.get("stale")):
        return "stale_relative_to_newer_input", "newer_authoritative_input_detected"

    if superseded_report is not None and bool(superseded_report.get("superseded")):
        return "superseded", str(superseded_report.get("reason") or "newer_run_already_published")

    candidate_as_of_ts = None if manifest is None else manifest.get("as_of_ts")
    candidate_as_of = _parse_timestamp(candidate_as_of_ts)
    published_as_of = _parse_timestamp(published_as_of_ts)

    if published_run_id and published_run_id != run_id and candidate_as_of and published_as_of:
        if published_as_of >= candidate_as_of:
            report_ready = report_dir is not None and (report_dir / "postflight_report.json").exists()
            if report_ready:
                return "superseded", "newer_or_equal_published_as_of_ts"

    if report_dir is not None and (
        (report_dir / "preflight_report.json").exists()
        or (report_dir / "unified_artifacts_report.json").exists()
        or (report_dir / "postflight_report.json").exists()
    ):
        return "in_progress", "awaiting_publish_completion"

    return "blocked", "missing_run_reports"


def _format_source_summary(source_freshness: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("injuries", "lineups", "odds", "props"):
        payload = source_freshness.get(key)
        if not isinstance(payload, dict):
            continue
        source_used = str(payload.get("source_used") or "unknown")
        latest_as_of_ts = payload.get("latest_as_of_ts")
        if latest_as_of_ts:
            parts.append(f"{key}:{source_used}@{latest_as_of_ts}")
        else:
            parts.append(f"{key}:{source_used}")
    return " | ".join(parts)


def _build_warning_badges(
    *,
    changed_sources: list[str],
    manual_override_active: bool,
    candidate_status: str,
    report_window_active: bool,
    report_window_blocking: bool,
) -> list[str]:
    badges: list[str] = []
    if manual_override_active:
        badges.append("manual-override")
    if "injuries" in changed_sources:
        badges.append("injury-change")
    if "lineups" in changed_sources:
        badges.append("lineup-change")
    if "odds" in changed_sources:
        badges.append("odds-change")
    if "props" in changed_sources:
        badges.append("props-change")
    if candidate_status in {"blocked", "stale_relative_to_newer_input", "superseded"}:
        badges.append(candidate_status)
    elif candidate_status == "in_progress":
        badges.append("candidate-running")
    if report_window_active and report_window_blocking:
        badges.append("report-window")
    return badges


@router.get("/status")
def get_live_status(
    date: str | None = Query(None, description="Slate date (YYYY-MM-DD). Defaults to today."),
) -> dict[str, Any]:
    slate_day = _parse_date(date)
    game_date = slate_day.isoformat()
    data_root = paths.data_path()

    published_pointer = _load_latest_published_pointer(data_root=data_root, game_date=game_date) or {}
    latest_published_run_id = (
        str(published_pointer.get("run_id")) if published_pointer.get("run_id") else None
    )
    latest_published_as_of_ts = published_pointer.get("as_of_ts")

    run_ids = _list_run_ids(data_root=data_root, game_date=game_date)
    candidate_run_id: str | None = None
    candidate_manifest: dict[str, Any] | None = None
    candidate_report_dir: Path | None = None
    candidate_status = "waiting_for_fresh_input"
    candidate_status_reason = "no_runs_found"

    if run_ids:
        classified_runs: list[tuple[str, dict[str, Any] | None, Path | None, str, str]] = []
        for run_id in reversed(run_ids):
            manifest = _resolve_run_manifest(
                data_root=data_root,
                game_date=game_date,
                run_id=run_id,
            )
            report_dir = _resolve_run_report_dir(data_root=data_root, game_date=game_date, run_id=run_id)
            status, reason = _classify_run_status(
                run_id=run_id,
                published_run_id=latest_published_run_id,
                published_as_of_ts=latest_published_as_of_ts,
                manifest=manifest,
                report_dir=report_dir,
            )
            classified_runs.append((run_id, manifest, report_dir, status, reason))

        # Ignore ad-hoc/manual runs that have no run reports and would otherwise
        # mask a healthy published or in-progress pipeline candidate.
        selected = next(
            (item for item in classified_runs if not (item[3] == "blocked" and item[4] == "missing_run_reports")),
            None,
        )
        if selected is None:
            selected = classified_runs[0]

        candidate_run_id, candidate_manifest, candidate_report_dir, candidate_status, candidate_status_reason = selected

    has_distinct_candidate = bool(candidate_run_id and candidate_run_id != latest_published_run_id)

    published_manifest = _resolve_run_manifest(
        data_root=data_root,
        game_date=game_date,
        run_id=latest_published_run_id,
    )
    effective_manifest = candidate_manifest or published_manifest or {}

    source_freshness = dict(effective_manifest.get("source_freshness") or {})
    per_game = dict(source_freshness.get("per_game") or {})
    input_change_set = dict(effective_manifest.get("input_change_set") or {})
    per_game_digests = dict(input_change_set.get("per_game_digests") or {})
    if per_game_digests:
        for game_id, payload in per_game_digests.items():
            if not isinstance(payload, dict):
                continue
            digest_payload = payload.get("payload")
            if isinstance(digest_payload, dict):
                per_game.setdefault(str(game_id), digest_payload)
    rerun_plan = dict(effective_manifest.get("rerun_plan") or input_change_set.get("rerun_plan") or {})
    freshness_gates = dict(effective_manifest.get("freshness_gates") or {})
    lock_window_gate = dict(freshness_gates.get("lock_window") or {})
    report_window_gate = dict(freshness_gates.get("report_window") or {})

    changed_games = {
        str(item.get("game_id")): item
        for item in input_change_set.get("changed_games", [])
        if isinstance(item, dict) and item.get("game_id") is not None
    }
    changed_game_ids = {str(game_id) for game_id in input_change_set.get("changed_game_ids", [])}
    rerun_mode = str(rerun_plan.get("mode") or "")
    rerun_targets = {str(game_id) for game_id in rerun_plan.get("target_game_ids", [])}
    blocking_games = {str(game_id) for game_id in report_window_gate.get("blocking_games", [])}

    override_summary = manual_override_report(slate_day, data_root=data_root)
    overrides_by_game: dict[str, int] = {}
    for game_id, payload in dict(override_summary.get("per_game") or {}).items():
        if not isinstance(payload, dict):
            continue
        overrides_by_game[str(game_id)] = int(payload.get("active_override_count") or 0)

    now = datetime.now(tz=UTC)
    game_ids = sorted(set(per_game.keys()) | set(changed_games.keys()) | set(overrides_by_game.keys()))
    games: list[dict[str, Any]] = []
    for game_id in game_ids:
        payload = per_game.get(game_id)
        game_payload = payload if isinstance(payload, dict) else {}
        changed = changed_games.get(game_id) or {}
        tip_ts = game_payload.get("tip_ts") or changed.get("tip_ts")
        tip_dt = _parse_timestamp(tip_ts)
        minutes_to_tip: int | None = None
        if tip_dt is not None:
            minutes_to_tip = int((tip_dt - now).total_seconds() // 60)

        source_container = (
            dict(game_payload.get("sources"))
            if isinstance(game_payload.get("sources"), dict)
            else game_payload
        )
        source_payload = {
            key: value
            for key in _SOURCE_KEYS
            if isinstance((value := source_container.get(key)), dict)
        }
        manual_override_active = overrides_by_game.get(game_id, 0) > 0
        changed_sources = [str(item) for item in changed.get("changed_sources", []) if item]
        rerun_targeted = rerun_mode == "full_slate" or game_id in rerun_targets
        report_window_blocking = game_id in blocking_games
        warning_badges = _build_warning_badges(
            changed_sources=changed_sources,
            manual_override_active=manual_override_active,
            candidate_status=candidate_status,
            report_window_active=bool(report_window_gate.get("active")),
            report_window_blocking=report_window_blocking,
        )

        games.append(
            {
                "game_id": game_id,
                "tip_ts": tip_ts,
                "minutes_to_tip": minutes_to_tip,
                "source_freshness": source_payload,
                "freshness_gates": {
                    "lock_window_ok": bool(lock_window_gate.get("ok", True)),
                    "report_window_active": bool(report_window_gate.get("active")),
                    "report_window_label": report_window_gate.get("label"),
                    "report_window_blocking": report_window_blocking,
                },
                "affected_by_change_set": game_id in changed_game_ids,
                "rerun_targeted": rerun_targeted,
                "manual_override_active": manual_override_active,
                "manual_override_count": overrides_by_game.get(game_id, 0),
                "changed_sources": changed_sources,
                "latest_effective_status_source_summary": _format_source_summary(source_payload),
                "warning_badges": warning_badges,
                "status_source_run_id": candidate_run_id or latest_published_run_id,
                "status_source_label": "candidate" if has_distinct_candidate else "published",
            }
        )

    recent_runs = run_ids[-6:]
    run_event_strip: list[dict[str, Any]] = []
    for run_id in reversed(recent_runs):
        manifest = _resolve_run_manifest(data_root=data_root, game_date=game_date, run_id=run_id)
        report_dir = _resolve_run_report_dir(data_root=data_root, game_date=game_date, run_id=run_id)
        status, reason = _classify_run_status(
            run_id=run_id,
            published_run_id=latest_published_run_id,
            published_as_of_ts=latest_published_as_of_ts,
            manifest=manifest,
            report_dir=report_dir,
        )
        rerun_plan_payload = dict((manifest or {}).get("rerun_plan") or {})
        run_event_strip.append(
            {
                "run_id": run_id,
                "status": status,
                "reason": reason,
                "as_of_ts": None if manifest is None else manifest.get("as_of_ts"),
                "updated_at": None if manifest is None else manifest.get("created_at"),
                "target_game_ids": rerun_plan_payload.get("target_game_ids") or [],
            }
        )

    return {
        "game_date": game_date,
        "latest_published_run_id": latest_published_run_id,
        "latest_published_as_of_ts": latest_published_as_of_ts,
        "candidate_run_id": candidate_run_id,
        "candidate_status": candidate_status,
        "candidate_status_reason": candidate_status_reason,
        "publish_status": "published" if latest_published_run_id else candidate_status,
        "updated_at": _iso_or_none(_parse_timestamp((candidate_manifest or {}).get("created_at"))),
        "status_source_run_id": candidate_run_id or latest_published_run_id,
        "status_source_label": "candidate" if has_distinct_candidate else "published",
        "games": games,
        "run_event_strip": run_event_strip,
    }


@router.get("/slate-analytics")
def get_slate_analytics(
    date: str | None = Query(None, description="Slate date (YYYY-MM-DD). Defaults to today."),
    draft_group_id: int | None = Query(None, description="Optional DK draft group id"),
    run_id: str | None = Query(None, description="Optional projections run_id"),
    top_n: int = Query(8, ge=3, le=30, description="Leaderboard depth per metric"),
) -> dict[str, Any]:
    slate_day = _parse_date(date)
    game_date = slate_day.isoformat()
    data_root = paths.data_path()
    dg_id = _resolve_default_draft_group_id(game_date, draft_group_id)

    try:
        pool = build_player_pool(
            game_date=game_date,
            draft_group_id=dg_id,
            site="dk",
            run_id=run_id,
            include_slate_analytics=False,
        )
        analytics_payload = load_or_compute_slate_player_analytics(
            game_date=game_date,
            draft_group_id=dg_id,
            pool_rows=pool,
            run_id=run_id,
            data_root=data_root,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to build slate analytics for %s/dg=%s: %s", game_date, dg_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to build slate analytics: {exc}") from exc

    players = [
        row
        for row in (analytics_payload.get("players") or [])
        if isinstance(row, dict)
    ]

    return {
        "game_date": game_date,
        "draft_group_id": int(dg_id),
        "run_id": analytics_payload.get("run_id"),
        "generated_at": analytics_payload.get("generated_at"),
        "sample_worlds": analytics_payload.get("sample_worlds"),
        "sample_seed": analytics_payload.get("sample_seed"),
        "optimal_worlds_solved": analytics_payload.get("optimal_worlds_solved"),
        "players": players,
        "leaders": {
            "optimal_pct": _build_metric_leaders(players, key="optimal_pct", top_n=top_n, reverse=True),
            "ceiling_leverage": _build_metric_leaders(players, key="ceiling_leverage", top_n=top_n, reverse=True),
            "boom_pct": _build_metric_leaders(players, key="boom_pct", top_n=top_n, reverse=True),
            "bust_pct": _build_metric_leaders(players, key="bust_pct", top_n=top_n, reverse=True),
        },
    }
