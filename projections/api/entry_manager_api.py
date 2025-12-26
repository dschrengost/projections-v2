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
    # Handle DK's "(LOCKED)" suffix during live slates
    # Match the FIRST numeric-only parenthesized value (the draftable ID)
    # e.g., "Moses Moody (41322706) (LOCKED)" -> 41322706
    match = re.search(r"\((\d+)\)", value)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _parse_game_start(value: str) -> Optional[datetime]:
    if not value:
        return None
    cleaned = value.strip().replace("Z", "+00:00")
    # Be robust to fractional seconds with more than 6 digits (DK sometimes emits 7).
    cleaned = re.sub(r"(\.\d{6})\d+(?=[+-])", r"\1", cleaned)
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


_DK_LOCKED_RE = re.compile(r"\blocked\b", re.IGNORECASE)


def _is_dk_locked(value: str) -> bool:
    """Return True if DK marked this roster cell as locked."""
    if not value:
        return False
    return bool(_DK_LOCKED_RE.search(value))


def _slot_values_from_lineup_players(
    lineup_players: List[object],
    internal_to_dk_player_id: Dict[str, int],
    internal_to_name: Dict[str, str],
    draftable_ids_by_player: Dict[int, Dict[str, int]],
    dk_names_by_player: Dict[int, str],
) -> Dict[str, str]:
    """Render slot -> 'Name (draftableId)' using solver-assigned slots."""
    slot_values: Dict[str, str] = {slot: "" for slot in DK_NBA_SLOTS}
    for p in lineup_players:
        internal_pid = str(getattr(p, "player_id", ""))
        slot = str(getattr(p, "pos", ""))
        if slot not in slot_values:
            return {}
        dk_pid = internal_to_dk_player_id.get(internal_pid)
        if dk_pid is None:
            return {}
        draftable_id = draftable_ids_by_player.get(dk_pid, {}).get(slot)
        if not draftable_id:
            return {}
        name = dk_names_by_player.get(dk_pid) or internal_to_name.get(internal_pid) or internal_pid
        slot_values[slot] = f"{name} ({draftable_id})"

    if any(not slot_values[s] for s in DK_NBA_SLOTS):
        return {}
    return slot_values


def _refresh_draftables_for_late_swap(draft_group_id: int, max_age_seconds: int = 120) -> bool:
    """Fetch DK draftables if missing or stale; return True if refreshed."""
    import json
    from projections.dk.api import fetch_draftables

    data_root = paths.data_path()
    bronze_path = (
        data_root
        / "bronze"
        / "dk"
        / "draftables"
        / f"draftables_raw_{draft_group_id}.json"
    )
    if bronze_path.exists():
        age = datetime.now(timezone.utc).timestamp() - bronze_path.stat().st_mtime
        if age <= max_age_seconds:
            return False

    try:
        payload = fetch_draftables(draft_group_id)
    except Exception as exc:
        logger.warning("Late swap draftables refresh failed for dg=%s: %s", draft_group_id, exc)
        return False

    try:
        bronze_path.parent.mkdir(parents=True, exist_ok=True)
        bronze_path.write_text(json.dumps(payload))
        return True
    except Exception as exc:
        logger.warning("Failed to write draftables JSON for dg=%s: %s", draft_group_id, exc)
        return False


def _load_draftable_start_times(draft_group_id: int) -> Dict[int, datetime]:
    """Return {draftable_id -> game start time (UTC)} from bronze draftables."""
    import json

    bronze_path = (
        paths.data_path()
        / "bronze"
        / "dk"
        / "draftables"
        / f"draftables_raw_{draft_group_id}.json"
    )
    if not bronze_path.exists():
        logger.warning("Draftables not found for dg=%s; late swap locks may be incomplete", draft_group_id)
        return {}

    try:
        payload = json.loads(bronze_path.read_text())
    except Exception as exc:
        logger.warning("Failed to parse draftables JSON for dg=%s: %s", draft_group_id, exc)
        return {}

    comp_start: Dict[int, datetime] = {}
    for comp in payload.get("competitions", []):
        comp_id = comp.get("competitionId")
        if comp_id is None:
            continue
        start_time = _parse_game_start(comp.get("startTime"))
        if start_time:
            try:
                comp_start[int(comp_id)] = start_time
            except (TypeError, ValueError):
                continue

    start_times: Dict[int, datetime] = {}
    for draftable in payload.get("draftables", []):
        if not isinstance(draftable, dict):
            continue
        draftable_id = draftable.get("draftableId") or draftable.get("id")
        if draftable_id is None:
            continue
        comp_id = None
        comp_obj = draftable.get("competition")
        if isinstance(comp_obj, dict):
            comp_id = comp_obj.get("competitionId")
        if comp_id is None:
            comps = draftable.get("competitions")
            if isinstance(comps, list) and comps:
                comp_id = comps[0].get("competitionId") if isinstance(comps[0], dict) else None
        try:
            draftable_id_i = int(draftable_id)
        except (TypeError, ValueError):
            continue
        if comp_id is None:
            continue
        try:
            comp_id_i = int(comp_id)
        except (TypeError, ValueError):
            continue
        start_time = comp_start.get(comp_id_i)
        if start_time:
            start_times[draftable_id_i] = start_time

    return start_times


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
    n_alternatives: int = Field(default=5, ge=1, le=20, description="Number of lineup alternatives to generate")
    randomness_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Randomness percentage for variance-aware noise")


class PlayerSwap(BaseModel):
    """Represents a single player swap in a slot."""
    slot: str
    old_player: str  # "Name (draftable_id)"
    new_player: str
    old_proj: Optional[float] = None
    new_proj: Optional[float] = None


class LineupAlternative(BaseModel):
    """A single lineup alternative with projected score."""
    lineup_idx: int
    projected_score: float
    slot_values: Dict[str, str]  # Slot -> "Player Name (DraftableID)"
    player_swaps: List[PlayerSwap]  # Diffs from original lineup


class EntryAlternatives(BaseModel):
    """Alternatives for a single entry."""
    entry_id: str
    locked_slots: List[str]
    alternatives: List[LineupAlternative]
    selected_idx: int  # Which alternative was auto-applied (0 = best)


class LateSwapSummary(BaseModel):
    """High-level summary of late swap selections."""
    entries_total: int
    entries_swapped: int
    entries_held: int
    entries_unmapped: int
    entries_unknown: int


class SolverSummary(BaseModel):
    """Aggregate solver diagnostics."""
    status_counts: Dict[str, int]
    avg_gap: Optional[float] = None
    max_gap: Optional[float] = None


class LateSwapResult(BaseModel):
    entry_state: EntryFileState
    locked_count: int
    updated_entries: int
    missing_locked_ids: List[str] = Field(default_factory=list)
    locked_slots_by_entry_id: Dict[str, List[str]] = Field(default_factory=dict)
    alternatives_by_entry_id: Dict[str, EntryAlternatives] = Field(default_factory=dict)
    selection_summary: Optional[LateSwapSummary] = None
    solver_summary: Optional[SolverSummary] = None


class ExportEntriesRequest(BaseModel):
    contest_ids: List[str]


class SelectAlternativeRequest(BaseModel):
    """Request to select a specific alternative for an entry."""
    entry_id: str
    alternative_idx: int
    slot_values: Dict[str, str]


def _compute_player_swaps(
    original_entry: Dict[str, str],
    new_slot_values: Dict[str, str],
    player_pool: List[Dict],
    draftable_to_internal: Dict[int, str],
) -> List[PlayerSwap]:
    """Compute player-level diffs between original and new lineup."""
    # Build lookup for projections by internal player_id
    proj_by_internal = {str(p.get("player_id")): p.get("proj", 0.0) for p in player_pool}

    swaps = []
    for slot in DK_NBA_SLOTS:
        old_val = original_entry.get(slot, "")
        new_val = new_slot_values.get(slot, "")
        if old_val != new_val:
            old_draftable = _extract_draftable_id(old_val)
            new_draftable = _extract_draftable_id(new_val)
            old_internal = draftable_to_internal.get(old_draftable) if old_draftable else None
            new_internal = draftable_to_internal.get(new_draftable) if new_draftable else None
            swaps.append(
                PlayerSwap(
                    slot=slot,
                    old_player=old_val,
                    new_player=new_val,
                    old_proj=proj_by_internal.get(old_internal) if old_internal else None,
                    new_proj=proj_by_internal.get(new_internal) if new_internal else None,
                )
            )
    return swaps


def _compute_entry_projection(
    entry: Dict[str, str],
    proj_by_internal: Dict[str, float],
    draftable_to_internal: Dict[int, str],
) -> Optional[float]:
    total = 0.0
    for slot in DK_NBA_SLOTS:
        slot_value = entry.get(slot, "")
        draftable_id = _extract_draftable_id(slot_value)
        if draftable_id is None:
            return None
        internal_id = draftable_to_internal.get(draftable_id)
        if not internal_id:
            return None
        proj = proj_by_internal.get(internal_id)
        if proj is None:
            return None
        total += proj
    return total


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
                    "entry_key": r.entry_id or f"row-{idx + 1}",
                    "contest_id": r.contest_id,
                    "contest_name": r.contest_name,
                    "entry_fee": r.entry_fee,
                    **{slot: r.slots.get(slot, "") for slot in DK_NBA_SLOTS},
                }
                for idx, r in enumerate(contest_rows)
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
        entry_key = entry.get("entry_key") or entry.get("entry_id") or f"row-{idx + 1}"
        slot_values = _assign_lineup_to_slots(
            lineups[idx],
            draft_group_id=entry_state.draft_group_id,
            game_date=entry_state.game_date,
        )
        updated_entries.append(
            {
                "entry_id": entry.get("entry_id", ""),
                "entry_key": entry_key,
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
        _refresh_draftables_for_late_swap(entry_state.draft_group_id)
        player_pool = build_player_pool(
            game_date=entry_state.game_date,
            draft_group_id=entry_state.draft_group_id,
            site=entry_state.site,
            run_id=request.run_id,
            use_user_overrides=request.use_user_overrides,
            ownership_mode=request.ownership_mode,
            include_unmatched_salaries=True,
            allow_zero_projections=True,
            exclude_inactive_players=False,
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
    draftable_start_times = _load_draftable_start_times(entry_state.draft_group_id)

    inactive_ids: set[str] = set()
    for p in player_pool:
        pid = str(p.get("player_id"))
        # When overrides are enabled, the pool may contain inactive/out players; ban them unless locked.
        if p.get("is_active") is False or p.get("is_out") is True:
            inactive_ids.add(pid)

    now_utc = datetime.now(timezone.utc)
    available_ids = {str(p.get("player_id")) for p in player_pool}
    missing_locked_ids: List[str] = []
    updated_entries: List[Dict[str, str]] = []
    locked_slots_by_entry_id: Dict[str, List[str]] = {}
    alternatives_by_entry_id: Dict[str, EntryAlternatives] = {}
    total_locked = 0
    entries_total = 0
    entries_swapped = 0
    entries_held = 0
    entries_unmapped = 0
    entries_unknown = 0
    solver_status_counts: Dict[str, int] = {}
    solver_gaps: List[float] = []

    for idx, entry in enumerate(entry_state.entries):
        entry_key = entry.get("entry_key") or entry.get("entry_id") or f"row-{idx + 1}"
        entries_total += 1
        locked_ids: List[str] = []
        locked_slots: List[str] = []
        lock_slots: Dict[str, str] = {}
        for slot in DK_NBA_SLOTS:
            slot_value = entry.get(slot, "")
            draftable_id = _extract_draftable_id(slot_value)
            if draftable_id is None and not slot_value:
                continue
            
            # Check if DK marked this player as locked (from earlier games)
            is_dk_locked = _is_dk_locked(slot_value)
            
            internal_id = draftable_to_internal.get(draftable_id) if draftable_id else None
            slot_start = draftable_start_times.get(draftable_id) if draftable_id else None
            if slot_start is None and internal_id:
                slot_start = start_times.get(internal_id)
            
            # Lock if: DK says locked OR game has started
            is_locked = bool(is_dk_locked or (slot_start and slot_start <= now_utc))
            
            if is_locked:
                locked_slots.append(slot)
                if internal_id:
                    locked_ids.append(internal_id)
                    lock_slots[slot] = internal_id
                elif draftable_id is not None:
                    missing_locked_ids.append(str(draftable_id))
                logger.debug(
                    "Late swap: Locking slot=%s draftable=%s (dk_locked=%s, game_started=%s)",
                    slot, draftable_id, is_dk_locked, slot_start <= now_utc if slot_start else False,
                )
            elif slot_start is None and draftable_id and not is_dk_locked:
                logger.warning(
                    "Late swap: No start time for slot=%s draftable=%s internal=%s entry=%s",
                    slot, draftable_id, internal_id, entry_key,
                )

        locked_ids = [pid for pid in dict.fromkeys(locked_ids) if pid in available_ids]
        total_locked += len(locked_slots)

        locked_slots_by_entry_id[entry_key] = locked_slots

        # If we can't map any locked slots to a player_id, do not attempt to late swap this entry.
        if any(slot not in lock_slots for slot in locked_slots):
            updated_entries.append(entry)
            entries_unmapped += 1
            continue

        # Generate N alternatives instead of just 1
        constraints = Constraints(
            N_lineups=request.n_alternatives,
            unique_players=1,
            randomness_pct=request.randomness_pct or 0.0,
            min_salary=0,
        )
        constraints.lock_ids = locked_ids
        constraints.lock_slots = lock_slots
        started_ids = {
            pid for pid, start_time in start_times.items()
            if start_time and start_time <= now_utc
        }
        ban_ids = (started_ids | inactive_ids) - set(locked_ids)
        constraints.ban_ids = sorted(ban_ids)
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

        if isinstance(diagnostics, dict):
            status = diagnostics.get("status")
            if status:
                solver_status_counts[status] = solver_status_counts.get(status, 0) + 1
            gap = diagnostics.get("achieved_gap")
            if isinstance(gap, (float, int)):
                solver_gaps.append(float(gap))

        # Build alternatives list with projected scores
        alternatives: List[LineupAlternative] = []
        for lineup_idx, lineup in enumerate(lineups):
            slot_values = _slot_values_from_lineup_players(
                lineup.players,
                internal_to_dk_player_id,
                internal_to_name,
                draftable_ids_by_player,
                dk_names_by_player,
            )
            if not slot_values:
                continue  # Skip unassignable lineups

            # Overlay locked slots with original entry values
            for slot in locked_slots:
                slot_values[slot] = entry.get(slot, "")

            # Compute player-level swaps vs original entry (only for non-locked slots)
            player_swaps = _compute_player_swaps(entry, slot_values, player_pool, draftable_to_internal)
            # Filter out swaps for locked slots
            player_swaps = [swap for swap in player_swaps if swap.slot not in locked_slots]

            alternatives.append(
                LineupAlternative(
                    lineup_idx=lineup_idx,
                    projected_score=lineup.total_proj,
                    slot_values=slot_values,
                    player_swaps=player_swaps,
                )
            )

        proj_by_internal = {str(p.get("player_id")): float(p.get("proj", 0.0)) for p in player_pool}
        current_proj = _compute_entry_projection(entry, proj_by_internal, draftable_to_internal)
        if current_proj is not None:
            alternatives.append(
                LineupAlternative(
                    lineup_idx=-1,
                    projected_score=current_proj,
                    slot_values={slot: entry.get(slot, "") for slot in DK_NBA_SLOTS},
                    player_swaps=[],
                )
            )

        if not alternatives:
            raise HTTPException(status_code=400, detail="Late swap failed to assign any lineup to DK slots")

        # Sort by projected score and auto-select the best (first).
        alternatives.sort(key=lambda a: a.projected_score, reverse=True)
        best_idx = 0
        best_alternative = alternatives[0]

        alternatives_by_entry_id[entry_key] = EntryAlternatives(
            entry_id=entry_key,
            locked_slots=locked_slots,
            alternatives=alternatives,
            selected_idx=best_idx,
        )

        if current_proj is None:
            entries_unknown += 1

        if best_alternative.player_swaps:
            entries_swapped += 1
        else:
            entries_held += 1

        updated_entries.append(
            {
                "entry_id": entry.get("entry_id", ""),
                "entry_key": entry_key,
                "contest_id": entry_state.contest_id,
                "contest_name": entry_state.contest_name,
                "entry_fee": entry_state.entry_fee,
                # Preserve original slot values for locked slots, use optimizer output for others
                **{
                    slot: (
                        entry.get(slot, "")  # Keep original for locked slots
                        if slot in locked_slots
                        else best_alternative.slot_values.get(slot, "")  # Use optimizer for unlocked
                    )
                    for slot in DK_NBA_SLOTS
                },
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
        alternatives_by_entry_id=alternatives_by_entry_id,
        selection_summary=LateSwapSummary(
            entries_total=entries_total,
            entries_swapped=entries_swapped,
            entries_held=entries_held,
            entries_unmapped=entries_unmapped,
            entries_unknown=entries_unknown,
        ),
        solver_summary=SolverSummary(
            status_counts=solver_status_counts,
            avg_gap=(sum(solver_gaps) / len(solver_gaps)) if solver_gaps else None,
            max_gap=max(solver_gaps) if solver_gaps else None,
        ),
    )


@router.post("/entries/{contest_id}/select-alternative", response_model=EntryFileState)
async def select_alternative(
    contest_id: str,
    date: str,
    request: SelectAlternativeRequest,
):
    """Apply a specific alternative to an entry."""
    path = _entry_path(date, contest_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Entry file {contest_id} not found for {date}")

    entry_state = EntryFileState.model_validate_json(path.read_text())

    # Find and update the specific entry
    updated = False
    for i, entry in enumerate(entry_state.entries):
        entry_key = entry.get("entry_key") or entry.get("entry_id")
        if str(entry.get("entry_id", "")) == request.entry_id or str(entry_key or "") == request.entry_id:
            entry_state.entries[i] = {
                "entry_id": entry.get("entry_id", ""),
                "entry_key": entry.get("entry_key", entry.get("entry_id", "")),
                "contest_id": entry_state.contest_id,
                "contest_name": entry_state.contest_name,
                "entry_fee": entry_state.entry_fee,
                **{slot: request.slot_values.get(slot, "") for slot in DK_NBA_SLOTS},
            }
            updated = True
            break

    if not updated:
        raise HTTPException(status_code=404, detail=f"Entry {request.entry_id} not found")

    entry_state.client_revision += 1
    entry_state.updated_at = datetime.utcnow().isoformat()
    path.write_text(entry_state.model_dump_json(indent=2))

    return entry_state


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
