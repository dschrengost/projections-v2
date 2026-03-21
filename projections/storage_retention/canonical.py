"""Canonical run selector and per-run retention decisions."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from projections.storage_retention.common import (
    floor_bucket_utc,
    parse_any_ts,
    parse_run_id_ts,
    utc_now_compact,
    write_json,
)
from projections.storage_retention.config import RetentionPolicy
from projections.storage_retention.paths import retention_decision_dir, retention_reports_dir


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _load_manifest(run_path: Path) -> dict[str, Any] | None:
    path = run_path / "manifest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_tip_ts(manifest: dict[str, Any] | None) -> datetime | None:
    if not manifest:
        return None
    slate = manifest.get("slate") if isinstance(manifest.get("slate"), dict) else {}
    direct = parse_any_ts(slate.get("first_tip_ts"))
    if direct is not None:
        return direct

    source_freshness = manifest.get("source_freshness")
    per_game = {}
    if isinstance(source_freshness, dict):
        maybe = source_freshness.get("per_game")
        if isinstance(maybe, dict):
            per_game = maybe

    tips: list[datetime] = []
    for row in per_game.values():
        if not isinstance(row, dict):
            continue
        parsed = parse_any_ts(row.get("tip_ts"))
        if parsed is not None:
            tips.append(parsed)
    if not tips:
        return None
    return min(tips)


def _manifest_slate_signature(manifest: dict[str, Any] | None) -> str:
    if not manifest:
        return "unknown"
    slate = manifest.get("slate") if isinstance(manifest.get("slate"), dict) else {}
    provided = slate.get("slate_signature")
    if provided:
        return str(provided)

    source_freshness = manifest.get("source_freshness")
    per_game = {}
    if isinstance(source_freshness, dict):
        maybe = source_freshness.get("per_game")
        if isinstance(maybe, dict):
            per_game = maybe

    game_rows: list[tuple[str, str]] = []
    for key, row in per_game.items():
        if not isinstance(row, dict):
            continue
        game_id = row.get("game_id") if row.get("game_id") is not None else key
        tip = row.get("tip_ts")
        game_rows.append((str(game_id), str(tip or "")))
    if not game_rows:
        return "unknown"
    return _stable_digest(sorted(game_rows))


def _canonical_key(
    *,
    family: str,
    game_date: str,
    first_tip_ts: datetime | None,
    slate_signature: str,
    bucket_minutes: int,
) -> str:
    if first_tip_ts is None:
        bucket = "unknown"
    else:
        bucket = floor_bucket_utc(first_tip_ts.astimezone(UTC), bucket_minutes).strftime(
            "%Y-%m-%dT%H:%MZ"
        )
    return f"{family}|{game_date}|{slate_signature}|{bucket}"


def classify_inventory_runs(
    *,
    inventory: dict[str, Any],
    retention_policy: RetentionPolicy,
    today_utc: date | None = None,
) -> dict[str, Any]:
    rows = [dict(row) for row in list(inventory.get("runs") or [])]
    today = today_utc or datetime.now(tz=UTC).date()

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        family = str(row.get("family") or "")
        game_date = str(row.get("game_date") or "")
        grouped.setdefault((family, game_date), []).append(row)

    decisions: list[dict[str, Any]] = []
    canonical_map: dict[str, dict[str, Any]] = {}

    for (family, game_date), day_rows in sorted(grouped.items()):
        parsed_game_date: date | None
        try:
            parsed_game_date = date.fromisoformat(game_date)
        except ValueError:
            parsed_game_date = None

        for row in day_rows:
            run_path = Path(str(row.get("run_path")))
            manifest = _load_manifest(run_path)
            run_ts = parse_run_id_ts(str(row.get("run_id") or ""))
            as_of_ts = parse_any_ts((manifest or {}).get("as_of_ts")) or run_ts
            first_tip_ts = _manifest_tip_ts(manifest)
            slate_sig = _manifest_slate_signature(manifest)
            row["_manifest"] = manifest
            row["_run_ts"] = run_ts
            row["_as_of_ts"] = as_of_ts
            row["_first_tip_ts"] = first_tip_ts
            row["_slate_signature"] = slate_sig
            row["_canonical_key"] = _canonical_key(
                family=family,
                game_date=game_date,
                first_tip_ts=first_tip_ts,
                slate_signature=slate_sig,
                bucket_minutes=retention_policy.start_time_bucket_minutes,
            )

        runs_by_key: dict[str, list[dict[str, Any]]] = {}
        for row in day_rows:
            runs_by_key.setdefault(str(row["_canonical_key"]), []).append(row)

        canonical_run_ids: set[str] = set()
        for key, key_rows in runs_by_key.items():
            first_tip = next((r.get("_first_tip_ts") for r in key_rows if r.get("_first_tip_ts") is not None), None)
            cutoff = (
                first_tip - timedelta(minutes=int(retention_policy.lead_time_minutes))
                if first_tip is not None
                else None
            )
            pre_tip = [
                r
                for r in key_rows
                if r.get("_as_of_ts") is not None and cutoff is not None and r["_as_of_ts"] <= cutoff
            ]
            degraded = False
            if pre_tip:
                selected = max(
                    pre_tip,
                    key=lambda r: (
                        r.get("_as_of_ts") or datetime.min.replace(tzinfo=UTC),
                        str(r.get("run_id") or ""),
                    ),
                )
            else:
                typed = [r for r in key_rows if r.get("_as_of_ts") is not None]
                if typed:
                    selected = min(
                        typed,
                        key=lambda r: (
                            r.get("_as_of_ts") or datetime.max.replace(tzinfo=UTC),
                            str(r.get("run_id") or ""),
                        ),
                    )
                    degraded = True
                else:
                    selected = max(key_rows, key=lambda r: str(r.get("run_id") or ""))
                    degraded = True

            selected_run_id = str(selected.get("run_id") or "")
            canonical_run_ids.add(selected_run_id)
            canonical_map[key] = {
                "family": family,
                "game_date": game_date,
                "canonical_key": key,
                "canonical_run_id": selected_run_id,
                "canonical_degraded": bool(degraded),
                "first_tip_ts": None
                if selected.get("_first_tip_ts") is None
                else selected["_first_tip_ts"].isoformat(),
                "slate_signature": str(selected.get("_slate_signature") or "unknown"),
            }

        latest_sorted = sorted(
            day_rows,
            key=lambda r: (
                r.get("_as_of_ts") or datetime.min.replace(tzinfo=UTC),
                str(r.get("run_id") or ""),
            ),
            reverse=True,
        )
        debug_keep: set[str] = set(
            str(r.get("run_id") or "")
            for r in latest_sorted[: int(retention_policy.keep_latest_debug_runs)]
        )

        for row in day_rows:
            run_id = str(row.get("run_id") or "")
            pointer_protected = bool(
                row.get("is_pointer_latest_current")
                or row.get("is_pointer_latest_run")
                or row.get("is_pointer_pinned")
            )
            is_today_protected = bool(
                retention_policy.protect_current_day
                and parsed_game_date is not None
                and parsed_game_date == today
            )
            unknown_time = row.get("_as_of_ts") is None
            is_canonical = run_id in canonical_run_ids
            is_debug_keep = run_id in debug_keep

            classification = "noncanonical"
            if is_canonical:
                classification = "canonical"
            elif pointer_protected:
                classification = "pointer_protected"
            elif is_debug_keep:
                classification = "debug_keep"
            elif unknown_time:
                classification = "unknown"

            protected = bool(
                pointer_protected
                or is_today_protected
                or unknown_time
                or is_canonical
                or is_debug_keep
            )

            decisions.append(
                {
                    "family": family,
                    "game_date": game_date,
                    "run_id": run_id,
                    "run_path": str(row.get("run_path") or ""),
                    "size_bytes": int(row.get("size_bytes") or 0),
                    "file_count": int(row.get("file_count") or 0),
                    "canonical_key": str(row.get("_canonical_key") or ""),
                    "slate_signature": str(row.get("_slate_signature") or "unknown"),
                    "first_tip_ts": None
                    if row.get("_first_tip_ts") is None
                    else row["_first_tip_ts"].isoformat(),
                    "as_of_ts": None
                    if row.get("_as_of_ts") is None
                    else row["_as_of_ts"].isoformat(),
                    "classification": classification,
                    "protected": protected,
                    "protection_reasons": {
                        "pointer": pointer_protected,
                        "current_day": is_today_protected,
                        "unknown_time": unknown_time,
                        "canonical": is_canonical,
                        "debug_keep": is_debug_keep,
                    },
                    "pointer_refs": {
                        "latest_current": bool(row.get("is_pointer_latest_current")),
                        "latest_run": bool(row.get("is_pointer_latest_run")),
                        "pinned": bool(row.get("is_pointer_pinned")),
                    },
                }
            )

    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "hot_root": inventory.get("hot_root"),
        "retention_policy": {
            "lead_time_minutes": int(retention_policy.lead_time_minutes),
            "start_time_bucket_minutes": int(retention_policy.start_time_bucket_minutes),
            "keep_latest_debug_runs": int(retention_policy.keep_latest_debug_runs),
            "protect_current_day": bool(retention_policy.protect_current_day),
        },
        "canonical_map": canonical_map,
        "decisions": decisions,
    }


def write_decision_reports(
    *,
    canonical_output: dict[str, Any],
    hot_root: Path,
    write_per_run_decisions: bool,
) -> dict[str, str]:
    ts = utc_now_compact()
    reports_dir = retention_reports_dir(hot_root=hot_root)
    reports_dir.mkdir(parents=True, exist_ok=True)

    map_path = reports_dir / f"{ts}_storage_canonical_map.json"
    write_json(map_path, canonical_output)

    if write_per_run_decisions:
        for row in list(canonical_output.get("decisions") or []):
            decision_path = retention_decision_dir(
                hot_root=hot_root,
                family=str(row.get("family") or ""),
                game_date=str(row.get("game_date") or ""),
                run_id=str(row.get("run_id") or ""),
            ) / "decision.json"
            write_json(decision_path, row)

    return {
        "canonical_map_json": str(map_path),
    }
