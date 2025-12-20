"""FastAPI router for DraftKings entry management."""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from projections import paths
from projections.api.optimizer_api import DK_NBA_SLOTS, _load_dk_nba_draftable_ids_by_player
from projections.api.optimizer_service import build_player_pool, load_saved_build
from projections.optimizer.cpsat_solver import solve_cpsat_iterative_counts
from projections.optimizer.optimizer_types import Constraints, OwnershipPenaltySettings

logger = logging.getLogger(__name__)
router = APIRouter()


def _entries_dir(game_date: str) -> Path:
    return paths.data_path() / "entries" / game_date / "dk"


def _entry_path(game_date: str, contest_id: str) -> Path:
    return _entries_dir(game_date) / f"{contest_id}.json"


@dataclass
class EntryRow:
    entry_id: str
    contest_id: str
    contest_name: str
    entry_fee: str
    slots: Dict[str, str]


def _parse_entry_csv(content: str) -> tuple[List[str], List[EntryRow]]:
    reader = csv.DictReader(io.StringIO(content))
    rows: List[EntryRow] = []
    if not reader.fieldnames:
        raise ValueError("Missing CSV header")
    header = reader.fieldnames
    for row in reader:
        if not row:
            continue
        contest_id = str(row.get("Contest ID", "")).strip()
        if not contest_id:
            continue
        entry_id = str(row.get("Entry ID", "")).strip()
        rows.append(
            EntryRow(
                entry_id=entry_id,
                contest_id=contest_id,
                contest_name=str(row.get("Contest Name", "")).strip(),
                entry_fee=str(row.get("Entry Fee", "")).strip(),
                slots={slot: str(row.get(slot, "")).strip() for slot in DK_NBA_SLOTS},
            )
        )
    return header, rows


def _build_dk_maps(
    game_date: str,
    draft_group_id: int,
    player_pool: Optional[List[Dict[str, object]]] = None,
) -> tuple[Dict[str, int], Dict[str, str], Dict[int, Dict[str, int]], Dict[int, str]]:
    pool = player_pool or build_player_pool(
        game_date=game_date, draft_group_id=draft_group_id, site="dk"
    )

    internal_to_dk_player_id: Dict[str, int] = {}
    internal_to_name: Dict[str, str] = {}
    for p in pool:
        pid = str(p.get("player_id"))
        internal_to_name[pid] = str(p.get("name") or pid)
        dk_id_raw = p.get("dk_id")
        if not dk_id_raw:
            continue
        try:
            internal_to_dk_player_id[pid] = int(dk_id_raw)
        except (TypeError, ValueError):
            continue

    draftable_ids_by_player, dk_names_by_player = _load_dk_nba_draftable_ids_by_player(draft_group_id)
    return internal_to_dk_player_id, internal_to_name, draftable_ids_by_player, dk_names_by_player


def _assign_lineup_to_slots_with_maps(
    lineups: List[str],
    internal_to_dk_player_id: Dict[str, int],
    internal_to_name: Dict[str, str],
    draftable_ids_by_player: Dict[int, Dict[str, int]],
    dk_names_by_player: Dict[int, str],
) -> Dict[str, str]:
    pids = [str(pid) for pid in lineups]
    if len(pids) != len(DK_NBA_SLOTS):
        return {}

    adj: Dict[str, List[str]] = {}
    for pid in pids:
        dk_pid = internal_to_dk_player_id.get(pid)
        if dk_pid is None:
            adj[pid] = []
            continue
        adj[pid] = list(draftable_ids_by_player.get(dk_pid, {}).keys())

    match_r: Dict[str, Optional[str]] = {s: None for s in DK_NBA_SLOTS}
    match_l: Dict[str, Optional[str]] = {pid: None for pid in pids}

    def dfs(pid: str, seen: set[str]) -> bool:
        for s in adj.get(pid, []):
            if s in seen:
                continue
            seen.add(s)
            if match_r[s] is None or dfs(match_r[s], seen):
                match_r[s] = pid
                match_l[pid] = s
                return True
        return False

    changed = True
    while changed:
        changed = False
        for pid in pids:
            if match_l[pid] is None and dfs(pid, set()):
                changed = True

    slot_to_pid = {slot: pid for slot, pid in match_r.items() if pid is not None}
    slot_values: Dict[str, str] = {}
    for slot in DK_NBA_SLOTS:
        internal_pid = slot_to_pid.get(slot)
        if not internal_pid:
            slot_values[slot] = ""
            continue
        dk_player_id = internal_to_dk_player_id.get(internal_pid)
        if dk_player_id is None:
            slot_values[slot] = ""
            continue
        draftable_id = draftable_ids_by_player.get(dk_player_id, {}).get(slot)
        if not draftable_id:
            slot_values[slot] = ""
            continue
        name = dk_names_by_player.get(dk_player_id) or internal_to_name.get(internal_pid) or str(internal_pid)
        slot_values[slot] = f"{name} ({draftable_id})"
    return slot_values


def _assign_lineup_to_slots(
    lineups: List[str],
    draft_group_id: int,
    game_date: str,
) -> Dict[str, str]:
    maps = _build_dk_maps(game_date, draft_group_id)
    return _assign_lineup_to_slots_with_maps(lineups, *maps)


def _extract_draftable_id(value: str) -> Optional[int]:
    if not value:
        return None
    match = re.search(r"\((\d+)\)\s*$", value)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _parse_game_start(value: str) -> Optional[datetime]:
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


class EntryFileSummary(BaseModel):
    contest_id: str
    contest_name: str
    entry_count: int
    created_at: str
    updated_at: str


class EntryFileState(BaseModel):
    game_date: str
    draft_group_id: int
    site: str = "dk"
    contest_id: str
    contest_name: str
    entry_fee: str
    created_at: str
    updated_at: str
    client_revision: int
    header: List[str]
    entries: List[Dict[str, str]]


class ApplyBuildRequest(BaseModel):
    build_source: Optional[str] = Field(default=None, description="optimizer|contest-sim")
    build_id: Optional[str] = None
    lineups: Optional[List[List[str]]] = None


class LateSwapRequest(BaseModel):
    use_user_overrides: bool = True
    ownership_mode: str = "renormalize"
    run_id: Optional[str] = None


class LateSwapResult(BaseModel):
    entry_state: EntryFileState
    locked_count: int
    updated_entries: int
    missing_locked_ids: List[str] = Field(default_factory=list)
    locked_slots_by_entry_id: Dict[str, List[str]] = Field(default_factory=dict)


class ExportEntriesRequest(BaseModel):
    contest_ids: List[str]


@router.post("/entries/upload", response_model=List[EntryFileSummary])
async def upload_entries(
    date: str,
    draft_group_id: int,
    file: UploadFile = File(...),
):
    """Upload DK entry CSV and persist per-contest state."""
    content = (await file.read()).decode("utf-8")
    header, rows = _parse_entry_csv(content)
    if not rows:
        raise HTTPException(status_code=400, detail="No entries found in CSV")

    entries_by_contest: Dict[str, List[EntryRow]] = {}
    for row in rows:
        entries_by_contest.setdefault(row.contest_id, []).append(row)

    summaries: List[EntryFileSummary] = []
    for contest_id, contest_rows in entries_by_contest.items():
        contest_name = contest_rows[0].contest_name
        entry_fee = contest_rows[0].entry_fee
        now = datetime.utcnow().isoformat()
        entry_state = EntryFileState(
            game_date=date,
            draft_group_id=draft_group_id,
            contest_id=contest_id,
            contest_name=contest_name,
            entry_fee=entry_fee,
            created_at=now,
            updated_at=now,
            client_revision=1,
            header=header,
            entries=[
                {
                    "entry_id": r.entry_id,
                    "contest_id": r.contest_id,
                    "contest_name": r.contest_name,
                    "entry_fee": r.entry_fee,
                    **{slot: r.slots.get(slot, "") for slot in DK_NBA_SLOTS},
                }
                for r in contest_rows
            ],
        )
        path = _entry_path(date, contest_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(entry_state.model_dump_json(indent=2))
        summaries.append(
            EntryFileSummary(
                contest_id=contest_id,
                contest_name=contest_name,
                entry_count=len(contest_rows),
                created_at=now,
                updated_at=now,
            )
        )

    return summaries


@router.get("/entries", response_model=List[EntryFileSummary])
async def list_entries(date: str):
    """List entry files for a date."""
    root = _entries_dir(date)
    if not root.exists():
        return []
    summaries: List[EntryFileSummary] = []
    for path in sorted(root.glob("*.json"), reverse=True):
        try:
            data = EntryFileState.model_validate_json(path.read_text())
            summaries.append(
                EntryFileSummary(
                    contest_id=data.contest_id,
                    contest_name=data.contest_name,
                    entry_count=len(data.entries),
                    created_at=data.created_at,
                    updated_at=data.updated_at,
                )
            )
        except Exception as exc:
            logger.warning("Failed to read entry file %s: %s", path, exc)
            continue
    return summaries


@router.get("/entries/{contest_id}", response_model=EntryFileState)
async def get_entry_file(contest_id: str, date: str):
    path = _entry_path(date, contest_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Entry file {contest_id} not found for {date}")
    return EntryFileState.model_validate_json(path.read_text())


@router.post("/entries/{contest_id}/apply-build", response_model=EntryFileState)
async def apply_build(contest_id: str, date: str, request: ApplyBuildRequest):
    path = _entry_path(date, contest_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Entry file {contest_id} not found for {date}")
    entry_state = EntryFileState.model_validate_json(path.read_text())

    if request.lineups:
        lineups = request.lineups
    elif request.build_source == "optimizer":
        if not request.build_id:
            raise HTTPException(status_code=400, detail="build_id required for optimizer source")
        build = load_saved_build(date, request.build_id)
        if not build or "lineups" not in build:
            raise HTTPException(status_code=404, detail="Optimizer build not found")
        lineups = [lu["player_ids"] for lu in build["lineups"]]
    elif request.build_source == "contest-sim":
        if not request.build_id:
            raise HTTPException(status_code=400, detail="build_id required for contest-sim source")
        build_path = paths.data_path() / "builds" / "contest_sim" / date / f"{request.build_id}.json"
        if not build_path.exists():
            raise HTTPException(status_code=404, detail="Contest sim build not found")
        import json
        build = json.loads(build_path.read_text())
        lineups = build.get("lineups", [])
    else:
        raise HTTPException(status_code=400, detail="Must provide lineups or build_source/build_id")

    if len(lineups) < len(entry_state.entries):
        raise HTTPException(status_code=400, detail="Not enough lineups to populate entries")

    updated_entries = []
    for idx, entry in enumerate(entry_state.entries):
        slot_values = _assign_lineup_to_slots(
            lineups[idx],
            draft_group_id=entry_state.draft_group_id,
            game_date=entry_state.game_date,
        )
        updated_entries.append(
            {
                "entry_id": entry.get("entry_id", ""),
                "contest_id": entry_state.contest_id,
                "contest_name": entry_state.contest_name,
                "entry_fee": entry_state.entry_fee,
                **{slot: slot_values.get(slot, "") for slot in DK_NBA_SLOTS},
            }
        )

    entry_state.entries = updated_entries
    entry_state.client_revision += 1
    entry_state.updated_at = datetime.utcnow().isoformat()
    path.write_text(entry_state.model_dump_json(indent=2))
    return entry_state


@router.post("/entries/{contest_id}/late-swap", response_model=LateSwapResult)
async def late_swap_entries(contest_id: str, date: str, request: LateSwapRequest):
    path = _entry_path(date, contest_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Entry file {contest_id} not found for {date}")
    entry_state = EntryFileState.model_validate_json(path.read_text())

    try:
        player_pool = build_player_pool(
            game_date=entry_state.game_date,
            draft_group_id=entry_state.draft_group_id,
            site=entry_state.site,
            run_id=request.run_id,
            use_user_overrides=request.use_user_overrides,
            ownership_mode=request.ownership_mode,
        )
    except Exception as exc:
        logger.exception("Failed to build player pool for late swap: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    if len(player_pool) < 8:
        raise HTTPException(status_code=400, detail="Player pool too small for late swap")

    internal_to_dk_player_id, internal_to_name, draftable_ids_by_player, dk_names_by_player = _build_dk_maps(
        entry_state.game_date,
        entry_state.draft_group_id,
        player_pool,
    )
    dk_to_internal = {dk_id: pid for pid, dk_id in internal_to_dk_player_id.items()}

    draftable_to_internal: Dict[int, str] = {}
    for dk_id, slot_map in draftable_ids_by_player.items():
        internal_id = dk_to_internal.get(dk_id)
        if not internal_id:
            continue
        for draftable_id in slot_map.values():
            draftable_to_internal.setdefault(int(draftable_id), internal_id)

    start_times: Dict[str, Optional[datetime]] = {}
    for p in player_pool:
        pid = str(p.get("player_id"))
        start_raw = str(p.get("game_start_utc") or "")
        start_times[pid] = _parse_game_start(start_raw)

    now_utc = datetime.now(timezone.utc)
    available_ids = {str(p.get("player_id")) for p in player_pool}
    missing_locked_ids: List[str] = []
    updated_entries: List[Dict[str, str]] = []
    locked_slots_by_entry_id: Dict[str, List[str]] = {}
    total_locked = 0

    for entry in entry_state.entries:
        locked_ids: List[str] = []
        locked_slots: List[str] = []
        for slot in DK_NBA_SLOTS:
            draftable_id = _extract_draftable_id(entry.get(slot, ""))
            if draftable_id is None:
                continue
            internal_id = draftable_to_internal.get(draftable_id)
            if not internal_id:
                missing_locked_ids.append(str(draftable_id))
                continue
            start_time = start_times.get(internal_id)
            if start_time and start_time <= now_utc:
                locked_ids.append(internal_id)
                locked_slots.append(slot)

        locked_ids = [pid for pid in dict.fromkeys(locked_ids) if pid in available_ids]
        total_locked += len(locked_ids)

        constraints = Constraints(N_lineups=1, unique_players=1)
        constraints.lock_ids = locked_ids
        constraints.ban_ids = []
        constraints.ownership_penalty = OwnershipPenaltySettings(enabled=False)
        constraints.validate(entry_state.site, stddev_available=any("stddev" in p for p in player_pool))

        lineups, diagnostics = solve_cpsat_iterative_counts(
            player_pool,
            constraints,
            seed=0,
            site=entry_state.site,
        )
        if not lineups:
            detail = diagnostics.get("message") if isinstance(diagnostics, dict) else None
            raise HTTPException(status_code=400, detail=detail or "Late swap optimizer failed to generate lineups")

        lineup_player_ids = [str(p.player_id) for p in lineups[0].players]
        slot_values = _assign_lineup_to_slots_with_maps(
            lineup_player_ids,
            internal_to_dk_player_id,
            internal_to_name,
            draftable_ids_by_player,
            dk_names_by_player,
        )
        if not slot_values:
            raise HTTPException(status_code=400, detail="Late swap failed to assign lineup to DK slots")

        final_locked_slots: List[str] = []
        for slot in DK_NBA_SLOTS:
            draftable_id = _extract_draftable_id(slot_values.get(slot, ""))
            if draftable_id is None:
                continue
            internal_id = draftable_to_internal.get(draftable_id)
            if not internal_id:
                continue
            start_time = start_times.get(internal_id)
            if start_time and start_time <= now_utc:
                final_locked_slots.append(slot)

        entry_id = str(entry.get("entry_id", ""))
        if entry_id:
            locked_slots_by_entry_id[entry_id] = final_locked_slots

        updated_entries.append(
            {
                "entry_id": entry.get("entry_id", ""),
                "contest_id": entry_state.contest_id,
                "contest_name": entry_state.contest_name,
                "entry_fee": entry_state.entry_fee,
                **{slot: slot_values.get(slot, "") for slot in DK_NBA_SLOTS},
            }
        )

    entry_state.entries = updated_entries
    entry_state.client_revision += 1
    entry_state.updated_at = datetime.utcnow().isoformat()
    path.write_text(entry_state.model_dump_json(indent=2))

    return LateSwapResult(
        entry_state=entry_state,
        locked_count=total_locked,
        updated_entries=len(updated_entries),
        missing_locked_ids=sorted(set(missing_locked_ids)),
        locked_slots_by_entry_id=locked_slots_by_entry_id,
    )


@router.post("/entries/{contest_id}/export")
async def export_entry_file(contest_id: str, date: str):
    path = _entry_path(date, contest_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Entry file {contest_id} not found for {date}")
    entry_state = EntryFileState.model_validate_json(path.read_text())

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(entry_state.header)
    for entry in entry_state.entries:
        row = [
            entry.get("entry_id", ""),
            entry.get("contest_name", ""),
            entry.get("contest_id", ""),
            entry.get("entry_fee", ""),
        ]
        for slot in DK_NBA_SLOTS:
            row.append(entry.get(slot, ""))
        writer.writerow(row)

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=entries_{date}_{contest_id}.csv"},
    )


@router.post("/entries/export")
async def export_entries_batch(date: str, request: ExportEntriesRequest):
    """Export multiple contests into a single CSV."""
    if not request.contest_ids:
        raise HTTPException(status_code=400, detail="No contest_ids provided")
    output = io.StringIO()
    writer = csv.writer(output)
    header_written = False
    for contest_id in request.contest_ids:
        path = _entry_path(date, contest_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Entry file {contest_id} not found for {date}")
        entry_state = EntryFileState.model_validate_json(path.read_text())
        if not header_written:
            writer.writerow(entry_state.header)
            header_written = True
        for entry in entry_state.entries:
            row = [
                entry.get("entry_id", ""),
                entry.get("contest_name", ""),
                entry.get("contest_id", ""),
                entry.get("entry_fee", ""),
            ]
            for slot in DK_NBA_SLOTS:
                row.append(entry.get(slot, ""))
            writer.writerow(row)

    filename = f"entries_{date}_combined.csv"
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
