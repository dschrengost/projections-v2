from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer

from projections import paths
from projections.api.entry_manager_api import _build_dk_maps, _extract_draftable_id

app = typer.Typer(help="Backfill export manifest contest-sim lineage by matching exports to saved portfolio builds.")

DK_NBA_SLOTS = ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL")


def _parse_iso_utc(value: object) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    cleaned = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_draftable_to_internal(game_date: str, draft_group_id: int) -> Dict[int, str]:
    internal_to_dk, _, draftable_ids_by_player, _ = _build_dk_maps(game_date, draft_group_id)
    dk_to_internal = {int(dk_id): str(pid) for pid, dk_id in internal_to_dk.items()}
    draftable_to_internal: Dict[int, str] = {}
    for dk_player_id, slot_map in draftable_ids_by_player.items():
        internal_id = dk_to_internal.get(int(dk_player_id))
        if not internal_id:
            continue
        for draftable_id in slot_map.values():
            draftable_to_internal[int(draftable_id)] = internal_id
    return draftable_to_internal


def _read_export_lineups(
    export_csv_path: Path,
    *,
    draftable_to_internal: Dict[int, str],
) -> Tuple[List[List[str]], int]:
    lineups: List[List[str]] = []
    unmapped_rows = 0
    with export_csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lineup: List[str] = []
            failed = False
            for slot in DK_NBA_SLOTS:
                draftable_id = _extract_draftable_id(str(row.get(slot, "")).strip())
                internal_id = draftable_to_internal.get(draftable_id) if draftable_id is not None else None
                if internal_id is None:
                    failed = True
                    break
                lineup.append(str(internal_id))
            if failed:
                unmapped_rows += 1
                continue
            lineups.append(sorted(lineup))
    return lineups, unmapped_rows


def _portfolio_build_paths(game_date: str, draft_group_id: int) -> List[Path]:
    root = paths.data_path("builds", "contest_sim", game_date)
    if not root.exists():
        return []
    out: List[Path] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if str(payload.get("kind") or "") != "portfolio":
            continue
        if int(payload.get("draft_group_id") or 0) != int(draft_group_id):
            continue
        out.append(path)
    return out


def _counter_from_lineups(lineups: List[List[str]]) -> Counter[Tuple[str, ...]]:
    return Counter(tuple(sorted(str(pid) for pid in lineup if str(pid).strip())) for lineup in lineups)


def _score_match(
    export_counter: Counter[Tuple[str, ...]],
    build_counter: Counter[Tuple[str, ...]],
) -> Tuple[int, float, bool]:
    matched = sum((export_counter & build_counter).values())
    total = sum(export_counter.values())
    ratio = float(matched) / float(total) if total else 0.0
    exact = export_counter == build_counter
    return matched, ratio, exact


def _choose_match(
    *,
    manifest_path: Path,
    manifest: dict,
    export_lineups: List[List[str]],
) -> Optional[dict]:
    game_date = str(manifest.get("game_date") or "").strip()
    draft_group_id = int(manifest.get("draft_group_id") or 0)
    export_created = _parse_iso_utc(manifest.get("created_at_utc"))
    if not game_date or draft_group_id <= 0 or export_created is None:
        return None

    export_counter = _counter_from_lineups(export_lineups)
    total_export = sum(export_counter.values())
    candidates: List[dict] = []
    for build_path in _portfolio_build_paths(game_date, draft_group_id):
        build = _load_json(build_path)
        build_created = _parse_iso_utc(build.get("created_at"))
        if build_created is None:
            continue
        if abs((export_created - build_created).total_seconds()) > 6 * 3600:
            continue
        build_lineups = list(build.get("lineups") or [])
        build_counter = _counter_from_lineups(build_lineups)
        matched, ratio, exact = _score_match(export_counter, build_counter)
        if matched <= 0:
            continue
        source_run_build_id = None
        if isinstance(build.get("request"), dict):
            raw = build["request"].get("source_build_id")
            source_run_build_id = str(raw) if raw else None
        candidates.append(
            {
                "build_path": str(build_path),
                "build_id": str(build.get("build_id") or build_path.stem),
                "build_name": str(build.get("name") or "") or None,
                "source_run_build_id": source_run_build_id,
                "selection_mode": (
                    str(build["request"].get("selection_mode"))
                    if isinstance(build.get("request"), dict) and build["request"].get("selection_mode")
                    else None
                ),
                "matched": matched,
                "ratio": ratio,
                "exact": exact,
                "created_delta_seconds": abs((export_created - build_created).total_seconds()),
                "portfolio_lineup_count": sum(build_counter.values()),
                "export_lineup_count": total_export,
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            1 if item["exact"] else 0,
            item["ratio"],
            -item["matched"],
            -1 * item["created_delta_seconds"],
        ),
        reverse=True,
    )
    best = candidates[0]
    if not best["exact"] and best["ratio"] < 1.0:
        return None
    return best


def _manifest_paths_in_range(start_date: str, end_date: str) -> List[Path]:
    root = paths.data_path("contests", "dk")
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    out: List[Path] = []
    current = start
    while current <= end:
        date_str = current.isoformat()
        out.extend(sorted(root.glob(f"game_date={date_str}/dg=*/exports/*_manifest.json")))
        current += timedelta(days=1)
    return out


@app.command()
def backfill(
    start_date: str = typer.Option(..., help="Inclusive start date YYYY-MM-DD"),
    end_date: str = typer.Option(..., help="Inclusive end date YYYY-MM-DD"),
    apply: bool = typer.Option(False, help="Write matched provenance back to manifests"),
) -> None:
    manifests = _manifest_paths_in_range(start_date, end_date)
    typer.echo(f"[lineage] scanning {len(manifests)} export manifests")
    matched_count = 0
    for manifest_path in manifests:
        manifest = _load_json(manifest_path)
        if manifest.get("source_run_build_id"):
            continue
        game_date = str(manifest.get("game_date") or "").strip()
        draft_group_id = int(manifest.get("draft_group_id") or 0)
        export_csv_path = Path(str(manifest.get("export_csv_path") or ""))
        if not game_date or draft_group_id <= 0 or not export_csv_path.exists():
            continue
        try:
            draftable_to_internal = _build_draftable_to_internal(game_date, draft_group_id)
            export_lineups, unmapped_rows = _read_export_lineups(
                export_csv_path,
                draftable_to_internal=draftable_to_internal,
            )
        except Exception as exc:
            typer.echo(f"[lineage] ERROR {manifest_path}: {exc}")
            continue
        if not export_lineups:
            typer.echo(f"[lineage] SKIP {manifest_path.name}: no mapped export lineups (unmapped_rows={unmapped_rows})")
            continue
        best = _choose_match(
            manifest_path=manifest_path,
            manifest=manifest,
            export_lineups=export_lineups,
        )
        if best is None:
            typer.echo(
                f"[lineage] NO_MATCH {manifest_path.name}: export_lineups={len(export_lineups)} unmapped_rows={unmapped_rows}"
            )
            continue
        matched_count += 1
        typer.echo(
            "[lineage] MATCH "
            f"{manifest_path.name}: portfolio={best['build_id']} "
            f"source_run={best['source_run_build_id']} "
            f"ratio={best['ratio']:.3f} exact={best['exact']} "
            f"delta_s={int(best['created_delta_seconds'])}"
        )
        if apply:
            manifest["source_build_source"] = "contest-sim"
            manifest["source_build_id"] = best["build_id"]
            manifest["source_build_kind"] = "portfolio"
            manifest["source_build_name"] = best["build_name"]
            manifest["source_portfolio_build_id"] = best["build_id"]
            manifest["source_run_build_id"] = best["source_run_build_id"]
            manifest["source_selection_mode"] = best["selection_mode"]
            manifest["lineage_backfill"] = {
                "matched_at_utc": datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "matched_by": "backfill_export_lineage",
                "match_ratio": best["ratio"],
                "exact_lineup_multiset_match": bool(best["exact"]),
                "created_delta_seconds": best["created_delta_seconds"],
                "mapped_export_lineup_count": len(export_lineups),
                "unmapped_export_rows": unmapped_rows,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    typer.echo(f"[lineage] matched {matched_count} manifests")


if __name__ == "__main__":
    app()
