"""FastAPI router for DraftKings entry management."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import secrets
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from projections import paths
from projections.api.optimizer_api import (
    DK_NBA_ROSTER_SLOT_ID_TO_SLOT,
    DK_NBA_SLOTS,
    _load_dk_nba_draftable_ids_by_player,
)
from projections.api.optimizer_service import build_player_pool, load_saved_build
from projections.dk.slates import build_contest_id_to_draft_group
from projections.late_swap.candidate_generation import (
    CandidateGenerationInput,
    generate_candidates_for_entries,
)
from projections.late_swap.diagnostics import (
    build_exposure_diagnostics,
    derive_target_count_by_player,
    exposure_counts_from_entries,
    exposure_counts_from_selection,
    summarize_selection,
    validate_policy_feasibility,
)
from projections.late_swap.lock_state import build_lock_state
from projections.late_swap.models import (
    LateSwapCandidateSummary,
    LateSwapCandidate,
    LateSwapCommitRequest,
    LateSwapExportRequest,
    LateSwapLockStateSummary,
    LateSwapPinCandidatesRequest,
    LateSwapPolicy,
    LateSwapPolicyUpdateRequest,
    LateSwapPreviewRequest,
    LateSwapPreviewResponse,
    LateSwapSourceProfile,
    LateSwapSession,
    LateSwapSessionCreateRequest,
    utc_now_iso,
)
from projections.late_swap.portfolio_selector import (
    SelectorInput,
    select_grouped_portfolio,
)
from projections.late_swap.scoring import apply_candidate_scores
from projections.late_swap import session_store
from projections.optimizer.cpsat_solver import solve_cpsat_iterative_counts  # noqa: F401

logger = logging.getLogger(__name__)
router = APIRouter()

_EXPORT_ID_SUFFIX_BYTES = 3  # 6 hex chars


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


def _is_dk_nba_classic_entry_header(header: List[str]) -> bool:
    cols = {str(c).strip() for c in header if c is not None}
    return all(slot in cols for slot in DK_NBA_SLOTS)


def _export_header_for_entry_state(entry_state: "EntryFileState") -> List[str]:
    header = [str(col) if col is not None else "" for col in (entry_state.header or [])]
    if _is_dk_nba_classic_entry_header(header):
        return header
    return ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"] + list(DK_NBA_SLOTS)


def _export_row_for_header(entry: Dict[str, str], header: List[str]) -> List[str]:
    row: List[str] = []
    for col in header:
        if col == "Entry ID":
            row.append(entry.get("entry_id", ""))
        elif col == "Contest Name":
            row.append(entry.get("contest_name", ""))
        elif col == "Contest ID":
            row.append(entry.get("contest_id", ""))
        elif col == "Entry Fee":
            row.append(entry.get("entry_fee", ""))
        elif col in DK_NBA_SLOTS:
            row.append(entry.get(col, ""))
        else:
            row.append("")
    return row


def _draft_group_looks_like_dk_nba_classic(draft_group_id: int, *, game_date: str) -> tuple[bool, int]:
    bronze_path = (
        paths.data_path()
        / "bronze"
        / "dk"
        / "draftables"
        / f"draftables_raw_{draft_group_id}.json"
    )
    if not bronze_path.exists():
        return False, 0
    try:
        payload = json.loads(bronze_path.read_text(encoding="utf-8"))
        draftables = payload.get("draftables", [])
        if not isinstance(draftables, list) or not draftables:
            return False, 0
        classic_slot_ids = set(DK_NBA_ROSTER_SLOT_ID_TO_SLOT.keys())
        roster_slot_ids = set()
        for d in draftables[:500]:
            if not isinstance(d, dict):
                continue
            rs = d.get("rosterSlotId")
            if rs is None:
                continue
            try:
                roster_slot_ids.add(int(rs))
            except (TypeError, ValueError):
                continue
        looks_classic = bool(roster_slot_ids.intersection(classic_slot_ids))
        return looks_classic, len(draftables)
    except Exception:
        return False, 0


def _guess_best_classic_draft_group_id(*, game_date: str) -> int | None:
    """Best-effort guess for NBA Classic DK slate DG for this date."""
    root = paths.data_path() / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}"
    candidates: List[int] = []
    if root.exists():
        for dg_dir in root.iterdir():
            if not dg_dir.is_dir() or not dg_dir.name.startswith("draft_group_id="):
                continue
            try:
                candidates.append(int(dg_dir.name.split("=", 1)[1]))
            except (ValueError, IndexError):
                continue
    if not candidates:
        return None

    best: tuple[int, int] | None = None  # (draftables_count, dg)
    for dg in candidates:
        looks_classic, n_draftables = _draft_group_looks_like_dk_nba_classic(dg, game_date=game_date)
        if not looks_classic:
            continue
        score = (n_draftables, dg)
        if best is None or score > best:
            best = score
    return best[1] if best else None


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
    draftable_to_dk_player_id: Optional[Dict[int, int]] = None,
) -> Dict[str, str]:
    pids = [str(pid) for pid in lineups]
    if len(pids) != len(DK_NBA_SLOTS):
        return {}

    resolved_dk_player_ids: Dict[str, int] = {}
    if draftable_to_dk_player_id is None:
        draftable_to_dk_player_id = {}
        for dk_player_id, slot_map in draftable_ids_by_player.items():
            for draftable_id in slot_map.values():
                draftable_to_dk_player_id.setdefault(int(draftable_id), int(dk_player_id))

    def resolve_dk_player_id(pid: str) -> Optional[int]:
        dk_pid = internal_to_dk_player_id.get(pid)
        if dk_pid is not None:
            return dk_pid
        try:
            numeric_pid = int(pid)
        except (TypeError, ValueError):
            return None
        if numeric_pid in draftable_ids_by_player:
            return numeric_pid
        return draftable_to_dk_player_id.get(numeric_pid)

    adj: Dict[str, List[str]] = {}
    for pid in pids:
        dk_pid = resolve_dk_player_id(pid)
        if dk_pid is None:
            adj[pid] = []
            continue
        resolved_dk_player_ids[pid] = dk_pid
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
        dk_player_id = resolved_dk_player_ids.get(internal_pid)
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


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _generate_export_id() -> str:
    ts = datetime.now(tz=timezone.utc).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(_EXPORT_ID_SUFFIX_BYTES)
    return f"{ts}_{suffix}"


def _contest_root_for_export(*, site: str, game_date: str, draft_group_id: int | None) -> Path:
    dg_part = f"dg={int(draft_group_id)}" if draft_group_id is not None else "dg=UNKNOWN"
    return paths.data_path("contests", site, f"game_date={game_date}", dg_part)


def _safe_git_sha() -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=str(paths.get_project_root()),
        ).strip()
        return sha or None
    except Exception:
        return None


def _resolve_latest_sim_v2_worlds(*, game_date: str) -> dict[str, object]:
    """Best-effort resolve of the current base worlds run for this date."""
    base_dir = paths.data_path("artifacts", "sim_v2", "worlds_fpts_v2", f"game_date={game_date}")
    worlds_variant = "sim_v2"
    if not base_dir.exists():
        gtv2_dir = paths.data_path("artifacts", "gtv2_worlds", f"game_date={game_date}")
        if gtv2_dir.exists():
            base_dir = gtv2_dir
            worlds_variant = "gtv2"

    out: dict[str, object] = {
        "base_worlds_run_id": None,
        "base_worlds_path": None,
        "base_sim_manifest_path": None,
        "base_sim_manifest_sha256": None,
        "sim_profile": None,
        "worlds_variant": worlds_variant,
    }
    if not base_dir.exists():
        return out

    run_id: str | None = None
    pointer = base_dir / "latest_run.json"
    if pointer.exists():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                run_id = str(payload.get("run_id") or "").strip() or None
        except Exception:
            run_id = None

    run_dir: Path | None = None
    if run_id:
        candidate = base_dir / f"run={run_id}"
        if candidate.exists():
            run_dir = candidate

    if run_dir is None:
        run_dirs = sorted(
            [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run=")],
            reverse=True,
        )
        if run_dirs:
            run_dir = run_dirs[0]
            run_id = run_dir.name.replace("run=", "", 1)

    if run_dir is None:
        return out

    worlds_matrix = run_dir / "worlds_matrix.parquet"
    gtv2_worlds = run_dir / "worlds.parquet"
    base_worlds_path = worlds_matrix if worlds_matrix.exists() else gtv2_worlds if gtv2_worlds.exists() else run_dir

    out["base_worlds_run_id"] = run_id
    out["base_worlds_path"] = str(base_worlds_path.resolve())

    sim_manifest_path = run_dir / "sim_manifest.json"
    if sim_manifest_path.exists():
        out["base_sim_manifest_path"] = str(sim_manifest_path.resolve())
        try:
            raw = sim_manifest_path.read_bytes()
            out["base_sim_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
            sim_manifest = json.loads(raw.decode("utf-8"))
            if isinstance(sim_manifest, dict):
                out["sim_profile"] = sim_manifest.get("profile") or sim_manifest.get("sim_profile")
        except Exception:
            pass

    return out


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


@dataclass(frozen=True)
class DraftGroupCandidate:
    """Candidate DraftKings draft group match for an entry file."""

    draft_group_id: int
    match_count: int
    slate_type: str


def _sample_entry_draftable_ids(entry_state: "EntryFileState", max_entries: int = 20) -> list[int]:
    """Sample draftable IDs from a DK entry file payload."""
    out: list[int] = []
    if not entry_state.entries:
        return out

    seen: set[int] = set()
    for entry in entry_state.entries[:max_entries]:
        for slot in DK_NBA_SLOTS:
            draftable_id = _extract_draftable_id(str(entry.get(slot, "")).strip())
            if draftable_id is None or draftable_id in seen:
                continue
            seen.add(draftable_id)
            out.append(draftable_id)
    return out


def _detect_draft_group_candidates(
    sample_draftable_ids: list[int],
    *,
    game_date: str | None = None,
    max_files: int = 50,
    min_match_count: int = 5,
) -> list[DraftGroupCandidate]:
    """Detect likely draft group IDs by matching sampled draftable IDs to bronze draftables files."""
    import json

    ids: set[int] = set()
    for raw in sample_draftable_ids or []:
        try:
            ids.add(int(raw))
        except (TypeError, ValueError):
            continue
    if not ids:
        return []

    bronze_dir = paths.data_path() / "bronze" / "dk" / "draftables"
    if not bronze_dir.exists():
        return []

    files = sorted(bronze_dir.glob("draftables_raw_*.json"), reverse=True)[:max_files]
    candidates: list[DraftGroupCandidate] = []

    for path in files:
        m = re.search(r"draftables_raw_(\d+)\.json$", path.name)
        if not m:
            continue
        try:
            draft_group_id = int(m.group(1))
        except ValueError:
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        competitions = payload.get("competitions") or []
        slate_type = "showdown" if isinstance(competitions, list) and len(competitions) == 1 else "classic"

        if game_date:
            start_times: list[datetime] = []
            for comp in competitions:
                if not isinstance(comp, dict):
                    continue
                start_time = _parse_game_start(str(comp.get("startTime") or ""))
                if start_time:
                    start_times.append(start_time)
            if start_times:
                comp_game_dates = {t.date().isoformat() for t in start_times}
                if game_date not in comp_game_dates:
                    continue

        draftables = payload.get("draftables") or []
        draftable_ids: set[int] = set()
        for d in draftables:
            if not isinstance(d, dict):
                continue
            raw = d.get("draftableId") or d.get("id")
            if raw is None:
                continue
            try:
                draftable_ids.add(int(raw))
            except (TypeError, ValueError):
                continue

        match_count = len(ids.intersection(draftable_ids))
        if match_count >= min_match_count:
            candidates.append(
                DraftGroupCandidate(
                    draft_group_id=draft_group_id,
                    match_count=match_count,
                    slate_type=slate_type,
                )
            )

    candidates.sort(key=lambda c: (c.match_count, c.draft_group_id), reverse=True)
    return candidates


class EntryFileSummary(BaseModel):
    contest_id: str
    contest_name: str
    draft_group_id: int
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
    source_build_source: Optional[str] = None
    source_build_id: Optional[str] = None
    source_build_kind: Optional[str] = None
    source_build_name: Optional[str] = None
    source_portfolio_build_id: Optional[str] = None
    source_run_build_id: Optional[str] = None
    source_selection_mode: Optional[str] = None
    source_late_swap_session_id: Optional[str] = None
    source_late_swap_mode: Optional[str] = None
    source_late_swap_committed_at: Optional[str] = None


class ApplyBuildRequest(BaseModel):
    build_source: Optional[str] = Field(default=None, description="optimizer|contest-sim")
    build_id: Optional[str] = None
    lineups: Optional[List[List[str]]] = None


def _entry_state_source_payload(entry_state: EntryFileState) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "source_build_source": entry_state.source_build_source,
        "source_build_id": entry_state.source_build_id,
        "source_build_kind": entry_state.source_build_kind,
        "source_build_name": entry_state.source_build_name,
        "source_portfolio_build_id": entry_state.source_portfolio_build_id,
        "source_run_build_id": entry_state.source_run_build_id,
        "source_selection_mode": entry_state.source_selection_mode,
        "source_late_swap_session_id": entry_state.source_late_swap_session_id,
        "source_late_swap_mode": entry_state.source_late_swap_mode,
        "source_late_swap_committed_at": entry_state.source_late_swap_committed_at,
    }
    return {key: value for key, value in payload.items() if value not in (None, "", [])}


def _aggregate_export_sources(entry_states: List[EntryFileState]) -> Dict[str, object]:
    per_contest_sources: List[Dict[str, object]] = []
    for entry_state in entry_states:
        payload = _entry_state_source_payload(entry_state)
        if not payload:
            continue
        payload["contest_id"] = str(entry_state.contest_id)
        per_contest_sources.append(payload)

    if not per_contest_sources:
        return {}

    aggregated: Dict[str, object] = {"source_entry_files": per_contest_sources}
    for field in [
        "source_build_source",
        "source_build_id",
        "source_build_kind",
        "source_build_name",
        "source_portfolio_build_id",
        "source_run_build_id",
        "source_selection_mode",
        "source_late_swap_session_id",
        "source_late_swap_mode",
        "source_late_swap_committed_at",
    ]:
        values = sorted({str(item[field]) for item in per_contest_sources if item.get(field) not in (None, "")})
        if len(values) == 1:
            aggregated[field] = values[0]
        elif values:
            aggregated[f"{field}s"] = values
    return aggregated


class LateSwapRequest(BaseModel):
    use_user_overrides: bool = False
    ownership_mode: str = "renormalize"
    run_id: Optional[str] = None
    n_alternatives: int = Field(default=5, ge=1, le=20, description="Number of lineup alternatives to generate")
    randomness_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Randomness percentage for variance-aware noise")
    only_out_lineups: bool = Field(default=False, description="Only swap lineups containing at least one OUT player")


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
    entries_skipped_no_out: int = 0


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


def _new_late_swap_session_id() -> str:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{secrets.token_hex(4)}"


def _entry_scoped_id(contest_id: str, entry: Dict[str, str], idx: int) -> str:
    entry_key = str(entry.get("entry_key") or entry.get("entry_id") or f"row-{idx + 1}")
    return f"{contest_id}:{entry_key}"


def _split_scoped_entry_id(scoped_entry_id: str) -> tuple[str, str]:
    if ":" not in scoped_entry_id:
        raise ValueError(f"Invalid scoped entry id: {scoped_entry_id}")
    contest_id, entry_key = scoped_entry_id.split(":", 1)
    return contest_id, entry_key


def _load_entry_states_for_contests(game_date: str, contest_ids: List[str]) -> List[EntryFileState]:
    states: List[EntryFileState] = []
    for contest_id in contest_ids:
        path = _entry_path(game_date, contest_id)
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Entry file {contest_id} not found for {game_date}",
            )
        states.append(EntryFileState.model_validate_json(path.read_text()))
    return states


def _build_source_profile(entry_states: List[EntryFileState]) -> dict[str, object]:
    source_build_ids = sorted(
        {
            str(state.source_build_id)
            for state in entry_states
            if state.source_build_id not in (None, "")
        }
    )
    source_portfolio_build_ids = sorted(
        {
            str(state.source_portfolio_build_id)
            for state in entry_states
            if state.source_portfolio_build_id not in (None, "")
        }
    )
    source_selection_modes = sorted(
        {
            str(state.source_selection_mode)
            for state in entry_states
            if state.source_selection_mode not in (None, "")
        }
    )
    source_kinds = {
        (
            str(state.source_build_source or ""),
            str(state.source_build_kind or ""),
            str(state.source_portfolio_build_id or ""),
        )
        for state in entry_states
    }
    return {
        "entries_total": sum(len(state.entries) for state in entry_states),
        "contests_total": len(entry_states),
        "source_build_ids": source_build_ids,
        "source_portfolio_build_ids": source_portfolio_build_ids,
        "source_selection_modes": source_selection_modes,
        "mixed_sources": len(source_kinds) > 1,
    }


def _merge_candidate_summaries(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    out = dict(left)
    numeric_fields = [
        "entries_total",
        "entries_with_candidates",
        "requested_total",
        "generated_total",
        "deduped_total",
        "rejected_unassignable_total",
        "rejected_salary_total",
        "rejected_swap_limit_total",
    ]
    for field in numeric_fields:
        out[field] = int(out.get(field, 0)) + int(right.get(field, 0))
    pass_counts = dict(out.get("pass_counts", {}))
    for key, value in dict(right.get("pass_counts", {})).items():
        pass_counts[str(key)] = int(pass_counts.get(str(key), 0)) + int(value)
    out["pass_counts"] = pass_counts
    return out


def _session_preview(
    *,
    session: LateSwapSession,
    request: LateSwapPreviewRequest,
) -> tuple[LateSwapSession, dict[str, list[LateSwapCandidate]]]:
    entry_states = _load_entry_states_for_contests(session.game_date, session.contest_ids)
    stale_reasons: list[str] = []
    for state in entry_states:
        expected_revision = int(session.source_entry_revisions.get(str(state.contest_id), -1))
        if expected_revision >= 0 and int(state.client_revision) != expected_revision:
            stale_reasons.append(
                f"contest {state.contest_id} revision drifted ({expected_revision} -> {state.client_revision})"
            )

    entries_by_id: dict[str, dict[str, str]] = {}
    contest_by_entry_id: dict[str, str] = {}
    draft_group_by_entry_id: dict[str, int] = {}
    player_name_by_id: dict[str, str] = {}
    player_team_by_id: dict[str, str] = {}
    player_game_by_id: dict[str, str] = {}

    candidates_by_entry_id: dict[str, list[LateSwapCandidate]] = {}
    candidate_summary: dict[str, object] = {}
    lock_summary: dict[str, Any] = {
        "entries_total": 0,
        "locked_slots_total": 0,
        "entries_fully_locked": 0,
        "entries_with_unmapped_locked_slots": 0,
        "locked_slots_by_entry_id": {},
        "unmapped_locked_by_entry_id": {},
        "player_locked_floor_count": {},
    }
    current_counts_total: dict[str, int] = {}
    out_ids_total: set[str] = set()

    states_by_dg: dict[int, list[tuple[EntryFileState, str, dict[str, str]]]] = {}
    for state in entry_states:
        for idx, entry in enumerate(state.entries):
            scoped_id = _entry_scoped_id(str(state.contest_id), entry, idx)
            entries_by_id[scoped_id] = entry
            contest_by_entry_id[scoped_id] = str(state.contest_id)
            draft_group_by_entry_id[scoped_id] = int(state.draft_group_id)
            states_by_dg.setdefault(int(state.draft_group_id), []).append((state, scoped_id, entry))

    now_utc = datetime.now(timezone.utc)
    for draft_group_id, items in states_by_dg.items():
        try:
            _refresh_draftables_for_late_swap(draft_group_id)
        except Exception:
            pass
        player_pool = build_player_pool(
            game_date=session.game_date,
            draft_group_id=draft_group_id,
            site="dk",
            run_id=request.run_id,
            use_user_overrides=bool(request.use_user_overrides),
            ownership_mode=request.ownership_mode,
            include_unmatched_salaries=True,
            allow_zero_projections=True,
            exclude_inactive_players=False,
        )
        if len(player_pool) < 8:
            raise HTTPException(
                status_code=400,
                detail=f"Player pool too small for draft_group_id={draft_group_id}",
            )

        (
            internal_to_dk_player_id,
            internal_to_name,
            draftable_ids_by_player,
            dk_names_by_player,
        ) = _build_dk_maps(session.game_date, draft_group_id, player_pool)
        dk_to_internal = {dk_id: pid for pid, dk_id in internal_to_dk_player_id.items()}
        draftable_to_internal: Dict[int, str] = {}
        for dk_id, slot_map in draftable_ids_by_player.items():
            internal_id = dk_to_internal.get(dk_id)
            if not internal_id:
                continue
            for draftable_id in slot_map.values():
                draftable_to_internal.setdefault(int(draftable_id), internal_id)

        internal_start_times: Dict[str, Optional[datetime]] = {}
        banned_ids_global: set[str] = set()
        out_ids: set[str] = set()
        for player in player_pool:
            pid = str(player.get("player_id"))
            player_name_by_id[pid] = str(player.get("name") or pid)
            player_team_by_id[pid] = str(player.get("team") or "")
            player_game_by_id[pid] = str(player.get("matchup") or "")
            start_raw = str(player.get("game_start_utc") or "")
            start_time = _parse_game_start(start_raw)
            internal_start_times[pid] = start_time
            if start_time and start_time <= now_utc:
                banned_ids_global.add(pid)
            if player.get("is_active") is False or player.get("is_out") is True:
                banned_ids_global.add(pid)
            if player.get("is_out") is True:
                out_ids.add(pid)
        out_ids_total.update(out_ids)

        draftable_start_times = _load_draftable_start_times(draft_group_id)
        group_entries = [entry for _state, _entry_id, entry in items]
        group_ids = [entry_id for _state, entry_id, _entry in items]

        lock_states, group_lock_summary = build_lock_state(
            entries=group_entries,
            dk_slots=DK_NBA_SLOTS,
            entry_id_resolver=lambda _entry, idx: group_ids[idx],
            extract_draftable_id=_extract_draftable_id,
            is_dk_locked=_is_dk_locked,
            draftable_to_internal=draftable_to_internal,
            draftable_start_times=draftable_start_times,
            internal_start_times=internal_start_times,
            now_utc=now_utc,
        )
        lock_state_by_id = {item.entry_id: item for item in lock_states}

        generated, generated_summary = generate_candidates_for_entries(
            CandidateGenerationInput(
                entries_by_entry_id={entry_id: entries_by_id[entry_id] for entry_id in group_ids},
                contest_by_entry_id=contest_by_entry_id,
                lock_state_by_entry_id=lock_state_by_id,
                policy=session.policy,
                dk_slots=DK_NBA_SLOTS,
                player_pool=player_pool,
                internal_to_dk_player_id=internal_to_dk_player_id,
                internal_to_name=internal_to_name,
                draftable_ids_by_player=draftable_ids_by_player,
                dk_names_by_player=dk_names_by_player,
                draftable_to_internal=draftable_to_internal,
                extract_draftable_id=_extract_draftable_id,
                banned_ids_global=banned_ids_global,
                out_ids=out_ids,
                only_out_lineups=request.only_out_lineups,
            )
        )
        for entry_id, cands in generated.items():
            candidates_by_entry_id[entry_id] = cands

        candidate_summary = _merge_candidate_summaries(
            candidate_summary,
            generated_summary.model_dump(mode="json"),
        )
        lock_summary["entries_total"] += int(group_lock_summary.entries_total)
        lock_summary["locked_slots_total"] += int(group_lock_summary.locked_slots_total)
        lock_summary["entries_fully_locked"] += int(group_lock_summary.entries_fully_locked)
        lock_summary["entries_with_unmapped_locked_slots"] += int(
            group_lock_summary.entries_with_unmapped_locked_slots
        )
        lock_summary["locked_slots_by_entry_id"].update(group_lock_summary.locked_slots_by_entry_id)
        lock_summary["unmapped_locked_by_entry_id"].update(group_lock_summary.unmapped_locked_by_entry_id)
        for pid, count in group_lock_summary.player_locked_floor_count.items():
            lock_summary["player_locked_floor_count"][pid] = (
                int(lock_summary["player_locked_floor_count"].get(pid, 0)) + int(count)
            )

        current_counts = exposure_counts_from_entries(
            entries_by_entry_id={entry_id: entries_by_id[entry_id] for entry_id in group_ids},
            dk_slots=DK_NBA_SLOTS,
            extract_draftable_id=_extract_draftable_id,
            draftable_to_internal=draftable_to_internal,
        )
        for pid, count in current_counts.items():
            current_counts_total[pid] = int(current_counts_total.get(pid, 0)) + int(count)

    candidates_by_entry_id = apply_candidate_scores(
        candidates_by_entry_id=candidates_by_entry_id,
        policy=session.policy,
    )

    team_keys_by_candidate_id: dict[str, set[str]] = {}
    game_keys_by_candidate_id: dict[str, set[str]] = {}
    coverage_by_player: dict[str, int] = {}
    for entry_id, candidates in candidates_by_entry_id.items():
        entry_presence: dict[str, bool] = {}
        for candidate in candidates:
            team_keys_by_candidate_id[candidate.candidate_id] = {
                player_team_by_id.get(pid, "")
                for pid in candidate.player_ids
                if player_team_by_id.get(pid, "")
            }
            game_keys_by_candidate_id[candidate.candidate_id] = {
                player_game_by_id.get(pid, "")
                for pid in candidate.player_ids
                if player_game_by_id.get(pid, "")
            }
            for pid in set(candidate.player_ids):
                entry_presence[str(pid)] = True
        for pid in entry_presence:
            coverage_by_player[pid] = int(coverage_by_player.get(pid, 0)) + 1

    total_entries = len(entries_by_id)
    source_target_counts = dict(current_counts_total)
    target_counts = derive_target_count_by_player(
        policy=session.policy,
        total_entries=total_entries,
        source_target_count_by_player=source_target_counts,
        current_committed_count_by_player=current_counts_total,
    )

    feasibility_errors, feasibility_warnings = validate_policy_feasibility(
        policy=session.policy,
        total_entries=total_entries,
        locked_floor_count_by_player=lock_summary["player_locked_floor_count"],
        candidate_coverage_count_by_player=coverage_by_player,
    )

    selector_result = select_grouped_portfolio(
        SelectorInput(
            candidates_by_entry_id=candidates_by_entry_id,
            policy=session.policy,
            locked_floor_count_by_player=lock_summary["player_locked_floor_count"],
            target_count_by_player=target_counts,
            pinned_candidates_by_entry_id=session.pinned_candidates_by_entry_id,
            team_keys_by_candidate_id=team_keys_by_candidate_id,
            game_keys_by_candidate_id=game_keys_by_candidate_id,
        )
    )

    proposed_counts = exposure_counts_from_selection(
        selected_candidate_ids_by_entry_id=selector_result.selected_candidate_ids_by_entry_id,
        candidates_by_entry_id=candidates_by_entry_id,
    )
    diagnostics = build_exposure_diagnostics(
        policy=session.policy,
        total_entries=total_entries,
        player_name_by_id=player_name_by_id,
        source_target_count_by_player=source_target_counts,
        locked_floor_count_by_player=lock_summary["player_locked_floor_count"],
        current_committed_count_by_player=current_counts_total,
        proposed_final_count_by_player=proposed_counts,
    )
    diagnostics.errors.extend(feasibility_errors)
    diagnostics.warnings.extend(feasibility_warnings)
    diagnostics.warnings.extend(selector_result.warnings)
    diagnostics.selector_notes.extend(selector_result.notes)
    if stale_reasons:
        diagnostics.stale_reasons.extend(stale_reasons)

    selection_summary = summarize_selection(
        selected_candidate_ids_by_entry_id=selector_result.selected_candidate_ids_by_entry_id,
        candidates_by_entry_id=candidates_by_entry_id,
        objective_value=selector_result.objective_value,
        status=selector_result.status,
        current_committed_count_by_player=current_counts_total,
        proposed_final_count_by_player=proposed_counts,
    )
    if feasibility_errors:
        selection_summary.infeasibility_count = len(feasibility_errors)
        if selector_result.status == "infeasible":
            selection_summary.status = "infeasible"

    lock_summary["locked_slots_pct"] = (
        100.0
        * float(lock_summary["locked_slots_total"])
        / float(max(1, total_entries * len(DK_NBA_SLOTS)))
    )

    session.lock_state = LateSwapLockStateSummary.model_validate(lock_summary)
    session.candidate_summary = LateSwapCandidateSummary.model_validate(candidate_summary or {})
    session.selected_candidates_by_entry_id = dict(selector_result.selected_candidate_ids_by_entry_id)
    session.selection_summary = selection_summary
    session.diagnostics = diagnostics
    session.warnings = list(dict.fromkeys([*session.warnings, *diagnostics.warnings]))
    session.updated_at = utc_now_iso()
    session.status = "preview_ready" if not selector_result.infeasible else "failed"

    return session, candidates_by_entry_id


def _load_session_any_date(session_id: str) -> LateSwapSession:
    root = paths.data_path() / "late_swap"
    if not root.exists():
        raise FileNotFoundError(f"Late swap session not found: {session_id}")
    for date_dir in sorted(root.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        try:
            return session_store.load_session(
                game_date=str(date_dir.name),
                session_id=session_id,
                site="dk",
            )
        except Exception:
            continue
    raise FileNotFoundError(f"Late swap session not found: {session_id}")


class ExportEntriesRequest(BaseModel):
    contest_ids: List[str]


class ExportEntrySelectionRequest(BaseModel):
    entry_ids: List[str]


class SelectAlternativeRequest(BaseModel):
    """Request to select a specific alternative for an entry."""
    entry_id: str
    alternative_idx: int
    slot_values: Dict[str, str]


class EntryValidationIssue(BaseModel):
    """A single validation issue for an entry."""
    entry_id: str
    severity: str  # "error" | "warning"
    issue_type: str  # "empty_slot" | "invalid_draftable" | "duplicate" | "salary_exceeded"
    message: str
    slot: Optional[str] = None


class ExportValidationResult(BaseModel):
    """Result of validating entries before export."""
    valid: bool
    entry_count: int
    issues: List[EntryValidationIssue] = Field(default_factory=list)
    warnings_count: int = 0
    errors_count: int = 0
    duplicate_lineup_count: int = 0
    empty_slot_count: int = 0


def _validate_entries_for_export(
    entries: List[Dict[str, str]],
    draft_group_id: int,
) -> ExportValidationResult:
    """Validate entries before export to catch issues early."""
    issues: List[EntryValidationIssue] = []
    seen_lineups: Dict[str, str] = {}  # hash -> first entry_id that had it

    for entry in entries:
        entry_id = entry.get("entry_id", "unknown")

        # Check each slot
        slot_ids: List[int] = []
        for slot in DK_NBA_SLOTS:
            slot_value = entry.get(slot, "").strip()

            if not slot_value:
                issues.append(EntryValidationIssue(
                    entry_id=entry_id,
                    severity="error",
                    issue_type="empty_slot",
                    message=f"Empty slot {slot}",
                    slot=slot,
                ))
                continue

            draftable_id = _extract_draftable_id(slot_value)
            if draftable_id is None:
                issues.append(EntryValidationIssue(
                    entry_id=entry_id,
                    severity="error",
                    issue_type="invalid_draftable",
                    message=f"Cannot parse draftable ID from '{slot_value}'",
                    slot=slot,
                ))
            else:
                slot_ids.append(draftable_id)

        # Check for duplicate lineups (same set of draftable IDs)
        if len(slot_ids) == len(DK_NBA_SLOTS):
            lineup_hash = ",".join(str(x) for x in sorted(slot_ids))
            if lineup_hash in seen_lineups:
                issues.append(EntryValidationIssue(
                    entry_id=entry_id,
                    severity="warning",
                    issue_type="duplicate",
                    message=f"Duplicate lineup (same as entry {seen_lineups[lineup_hash]})",
                ))
            else:
                seen_lineups[lineup_hash] = entry_id

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    duplicates = [i for i in issues if i.issue_type == "duplicate"]
    empty_slots = [i for i in issues if i.issue_type == "empty_slot"]

    return ExportValidationResult(
        valid=len(errors) == 0,
        entry_count=len(entries),
        issues=issues,
        warnings_count=len(warnings),
        errors_count=len(errors),
        duplicate_lineup_count=len(duplicates),
        empty_slot_count=len(empty_slots),
    )


def _compute_player_swaps(
    original_entry: Dict[str, str],
    new_slot_values: Dict[str, str],
    player_pool: List[Dict],
    draftable_to_internal: Dict[int, str],
) -> List[PlayerSwap]:
    """Compute player-level diffs between original and new lineup (ignore slot shuffles)."""
    # Build lookup for projections by internal player_id
    proj_by_internal = {str(p.get("player_id")): p.get("proj", 0.0) for p in player_pool}

    old_by_id: Dict[int, str] = {}
    old_slot_by_id: Dict[int, str] = {}
    new_by_id: Dict[int, str] = {}

    for slot in DK_NBA_SLOTS:
        old_val = original_entry.get(slot, "")
        old_id = _extract_draftable_id(old_val)
        if old_id is not None:
            old_by_id[old_id] = old_val
            old_slot_by_id[old_id] = slot

        new_val = new_slot_values.get(slot, "")
        new_id = _extract_draftable_id(new_val)
        if new_id is not None:
            new_by_id[new_id] = new_val

    out_ids: List[int] = []
    for slot in DK_NBA_SLOTS:
        old_val = original_entry.get(slot, "")
        old_id = _extract_draftable_id(old_val)
        if old_id is not None and old_id not in new_by_id:
            out_ids.append(old_id)

    in_ids: List[int] = []
    for slot in DK_NBA_SLOTS:
        new_val = new_slot_values.get(slot, "")
        new_id = _extract_draftable_id(new_val)
        if new_id is not None and new_id not in old_by_id:
            in_ids.append(new_id)

    if not out_ids and not in_ids:
        return []

    swaps: List[PlayerSwap] = []
    max_len = max(len(out_ids), len(in_ids))
    for idx in range(max_len):
        out_id = out_ids[idx] if idx < len(out_ids) else None
        in_id = in_ids[idx] if idx < len(in_ids) else None
        old_val = old_by_id.get(out_id, "") if out_id is not None else ""
        new_val = new_by_id.get(in_id, "") if in_id is not None else ""
        old_internal = draftable_to_internal.get(out_id) if out_id is not None else None
        new_internal = draftable_to_internal.get(in_id) if in_id is not None else None
        slot = old_slot_by_id.get(out_id, "swap") if out_id is not None else "swap"
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
    draft_group_id: int | None = Query(default=None),
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

    contest_dg_map: Dict[str, int] = {}
    try:
        contest_dg_map = build_contest_id_to_draft_group(date)
    except Exception as exc:
        logger.warning("Contest->draft_group map lookup failed for %s: %s", date, exc)

    classic_entry = _is_dk_nba_classic_entry_header(header)
    guessed_classic_dg = _guess_best_classic_draft_group_id(game_date=date) if classic_entry else None

    mapped_values = [contest_dg_map.get(str(cid)) for cid in entries_by_contest.keys()]
    mapped_unique = {dg for dg in mapped_values if dg is not None}
    fallback_mapped_dg = next(iter(mapped_unique)) if len(mapped_unique) == 1 else None

    upload_ts = datetime.utcnow().isoformat()
    summaries: List[EntryFileSummary] = []
    for contest_id, contest_rows in entries_by_contest.items():
        contest_name = contest_rows[0].contest_name
        entry_fee = contest_rows[0].entry_fee
        requested_dg: int | None = int(draft_group_id) if draft_group_id is not None else None
        mapped_dg = contest_dg_map.get(str(contest_id))
        if mapped_dg is None and requested_dg is None and fallback_mapped_dg is not None:
            logger.warning(
                "Entry upload contest %s not found in lobby; falling back to mapped dg=%s from other contests in file",
                contest_id,
                fallback_mapped_dg,
            )
            mapped_dg = int(fallback_mapped_dg)
        if mapped_dg is None and guessed_classic_dg is not None:
            # Post-lock DK lobby may omit contests. If entry format is NBA Classic, prefer classic DG from disk.
            looks_requested_classic = False
            if requested_dg is not None:
                looks_requested_classic, _ = _draft_group_looks_like_dk_nba_classic(
                    int(requested_dg), game_date=date
                )
            if requested_dg is None or not looks_requested_classic:
                logger.warning(
                    "Entry upload contest %s using guessed classic dg=%s (requested=%s)",
                    contest_id,
                    guessed_classic_dg,
                    requested_dg,
                )
                mapped_dg = int(guessed_classic_dg)
        resolved_dg: int | None = mapped_dg if mapped_dg is not None else requested_dg
        if mapped_dg is not None and requested_dg is not None and int(mapped_dg) != int(requested_dg):
            logger.warning(
                "Entry upload slate mismatch for contest %s: requested=%s mapped=%s; using mapped",
                contest_id,
                requested_dg,
                mapped_dg,
            )
        entry_state = EntryFileState(
            game_date=date,
            draft_group_id=resolved_dg if resolved_dg is not None else -1,
            contest_id=contest_id,
            contest_name=contest_name,
            entry_fee=entry_fee,
            created_at=upload_ts,
            updated_at=upload_ts,
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
        try:
            # Contest mapping is authoritative (works even with empty roster slots).
            if mapped_dg is not None:
                entry_state.draft_group_id = int(mapped_dg)
            elif resolved_dg is None and requested_dg is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Could not determine slate for contest {contest_id}. "
                        "Pick a slate and retry, or ensure DK contests are available for this date."
                    ),
                )

            sample_ids = _sample_entry_draftable_ids(entry_state)
            candidates = _detect_draft_group_candidates(sample_ids, game_date=date)
            if mapped_dg is None and candidates:
                detected = candidates[0].draft_group_id
                if requested_dg is None:
                    entry_state.draft_group_id = int(detected)
                elif int(detected) != int(entry_state.draft_group_id):
                    logger.warning(
                        "Entry file %s draft_group_id override: %s -> %s (match_count=%s)",
                        contest_id,
                        entry_state.draft_group_id,
                        detected,
                        candidates[0].match_count,
                    )
                    entry_state.draft_group_id = int(detected)
            elif requested_dg is None and mapped_dg is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Could not auto-detect slate for contest {contest_id}. "
                        "Pick a slate and retry, or ensure DK draftables are present for this date."
                    ),
                )
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise
            logger.warning("Entry file %s draft_group_id detection failed: %s", contest_id, exc)
        path = _entry_path(date, contest_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(entry_state.model_dump_json(indent=2))
        summaries.append(
            EntryFileSummary(
                contest_id=contest_id,
                contest_name=contest_name,
                draft_group_id=int(entry_state.draft_group_id),
                entry_count=len(contest_rows),
                created_at=upload_ts,
                updated_at=upload_ts,
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
                    draft_group_id=int(data.draft_group_id),
                    entry_count=len(data.entries),
                    created_at=data.created_at,
                    updated_at=data.updated_at,
                )
            )
        except Exception as exc:
            logger.warning("Failed to read entry file %s: %s", path, exc)
            continue
    return summaries


@router.post("/entries/repair-dg", response_model=List[EntryFileSummary])
async def repair_entries_draft_group_ids(date: str):
    """Repair stored entry file draft_group_id values using DK contest->dg mapping."""
    root = _entries_dir(date)
    if not root.exists():
        return []

    contest_dg_map: Dict[str, int] = {}
    try:
        contest_dg_map = build_contest_id_to_draft_group(date)
    except Exception as exc:
        logger.warning("Contest->draft_group map lookup failed for %s: %s", date, exc)

    guessed_classic_dg = _guess_best_classic_draft_group_id(game_date=date)

    items: List[tuple[Path, EntryFileState, str]] = []
    for path in sorted(root.glob("*.json"), reverse=True):
        try:
            state = EntryFileState.model_validate_json(path.read_text())
        except Exception as exc:
            logger.warning("Failed to read entry file %s: %s", path, exc)
            continue
        batch_key = str(state.created_at)[:19]  # second granularity
        items.append((path, state, batch_key))

    batches: Dict[str, List[tuple[Path, EntryFileState]]] = {}
    for path, state, batch_key in items:
        batches.setdefault(batch_key, []).append((path, state))

    now = datetime.utcnow().isoformat()
    updated = 0
    for batch_key, batch_items in batches.items():
        known_dgs = {
            contest_dg_map.get(str(state.contest_id))
            for _, state in batch_items
        }
        known_dgs = {dg for dg in known_dgs if dg is not None}
        batch_dg = next(iter(known_dgs)) if len(known_dgs) == 1 else None
        if batch_dg is None and guessed_classic_dg is not None:
            # If DK lobby no longer lists these contests, fall back to classic DG guess from disk.
            # Only apply if the entry header looks like NBA Classic.
            any_classic = any(_is_dk_nba_classic_entry_header(state.header) for _, state in batch_items)
            if any_classic:
                batch_dg = int(guessed_classic_dg)

        for path, state in batch_items:
            desired = contest_dg_map.get(str(state.contest_id)) or batch_dg
            if desired is None:
                continue
            if int(state.draft_group_id) == int(desired):
                continue
            logger.warning(
                "Repairing entry file %s: dg %s -> %s (batch=%s)",
                state.contest_id,
                state.draft_group_id,
                desired,
                batch_key,
            )
            state.draft_group_id = int(desired)
            state.updated_at = now
            state.client_revision = int(state.client_revision) + 1
            path.write_text(state.model_dump_json(indent=2))
            updated += 1

    logger.info("Repaired %d entry files for %s", updated, date)
    return await list_entries(date)


@router.get("/entries/{contest_id}", response_model=EntryFileState)
async def get_entry_file(contest_id: str, date: str):
    path = _entry_path(date, contest_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Entry file {contest_id} not found for {date}")
    return EntryFileState.model_validate_json(path.read_text())


@router.delete("/entries/{contest_id}")
async def delete_entry_file(contest_id: str, date: str):
    path = _entry_path(date, contest_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Entry file {contest_id} not found for {date}")
    path.unlink()
    return {"status": "deleted", "contest_id": contest_id}


@router.post("/entries/{contest_id}/apply-build", response_model=EntryFileState)
async def apply_build(contest_id: str, date: str, request: ApplyBuildRequest):
    path = _entry_path(date, contest_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Entry file {contest_id} not found for {date}")
    entry_state = EntryFileState.model_validate_json(path.read_text())
    entry_state.source_build_source = None
    entry_state.source_build_id = None
    entry_state.source_build_kind = None
    entry_state.source_build_name = None
    entry_state.source_portfolio_build_id = None
    entry_state.source_run_build_id = None
    entry_state.source_selection_mode = None
    entry_state.source_late_swap_session_id = None
    entry_state.source_late_swap_mode = None
    entry_state.source_late_swap_committed_at = None

    if request.lineups:
        lineups = request.lineups
    elif request.build_source == "optimizer":
        if not request.build_id:
            raise HTTPException(status_code=400, detail="build_id required for optimizer source")
        build = load_saved_build(date, request.build_id)
        if not build or "lineups" not in build:
            raise HTTPException(status_code=404, detail="Optimizer build not found")
        build_draft_group_id = build.get("draft_group_id")
        if (
            build_draft_group_id is not None
            and int(build_draft_group_id) != int(entry_state.draft_group_id)
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Optimizer build draft_group_id "
                    f"{build_draft_group_id} does not match entry draft_group_id "
                    f"{entry_state.draft_group_id}"
                ),
            )
        lineups = [lu["player_ids"] for lu in build["lineups"]]
        entry_state.source_build_source = "optimizer"
        entry_state.source_build_id = str(request.build_id)
        entry_state.source_build_kind = str(build.get("kind") or "run")
        entry_state.source_build_name = str(build.get("name") or "") or None
    elif request.build_source == "contest-sim":
        if not request.build_id:
            raise HTTPException(status_code=400, detail="build_id required for contest-sim source")
        build_path = paths.data_path() / "builds" / "contest_sim" / date / f"{request.build_id}.json"
        if not build_path.exists():
            raise HTTPException(status_code=404, detail="Contest sim build not found")
        import json
        build = json.loads(build_path.read_text())
        build_draft_group_id = build.get("draft_group_id")
        if (
            build_draft_group_id is not None
            and int(build_draft_group_id) != int(entry_state.draft_group_id)
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Contest sim build draft_group_id "
                    f"{build_draft_group_id} does not match entry draft_group_id "
                    f"{entry_state.draft_group_id}"
                ),
            )
        lineups = build.get("lineups", [])
        entry_state.source_build_source = "contest-sim"
        entry_state.source_build_id = str(request.build_id)
        entry_state.source_build_kind = str(build.get("kind") or "run")
        entry_state.source_build_name = str(build.get("name") or "") or None
        if str(build.get("kind") or "") == "portfolio":
            entry_state.source_portfolio_build_id = str(request.build_id)
            source_run_build_id = (
                build.get("request", {}).get("source_build_id")
                if isinstance(build.get("request"), dict)
                else None
            )
            entry_state.source_run_build_id = str(source_run_build_id) if source_run_build_id else None
            selection_mode = (
                build.get("request", {}).get("selection_mode")
                if isinstance(build.get("request"), dict)
                else None
            )
            entry_state.source_selection_mode = str(selection_mode) if selection_mode else None
        else:
            entry_state.source_portfolio_build_id = None
            entry_state.source_run_build_id = str(request.build_id)
            entry_state.source_selection_mode = None
    else:
        raise HTTPException(status_code=400, detail="Must provide lineups or build_source/build_id")

    if len(lineups) < len(entry_state.entries):
        raise HTTPException(status_code=400, detail="Not enough lineups to populate entries")

    # Refresh draftables to ensure we have current DK data (handles locked games)
    try:
        _refresh_draftables_for_late_swap(entry_state.draft_group_id)
    except Exception as exc:
        logger.warning("Failed to refresh draftables for apply_build: %s", exc)

    # Pre-build maps once for all entries
    try:
        maps = _build_dk_maps(entry_state.game_date, entry_state.draft_group_id)
        internal_to_dk, internal_to_name, draftable_ids_by_player, dk_names = maps
        draftable_to_dk_player_id: Dict[int, int] = {}
        for dk_player_id, slot_map in draftable_ids_by_player.items():
            for draftable_id in slot_map.values():
                draftable_to_dk_player_id.setdefault(int(draftable_id), int(dk_player_id))
    except Exception as exc:
        logger.exception("Failed to build DK maps for apply_build")
        raise HTTPException(status_code=500, detail=f"Failed to map players to DK IDs: {exc}")

    unmapped_players: set[str] = set()
    updated_entries = []
    for idx, entry in enumerate(entry_state.entries):
        entry_key = entry.get("entry_key") or entry.get("entry_id") or f"row-{idx + 1}"
        slot_values = _assign_lineup_to_slots_with_maps(
            lineups[idx],
            internal_to_dk,
            internal_to_name,
            draftable_ids_by_player,
            dk_names,
            draftable_to_dk_player_id,
        )
        # Track players that couldn't be mapped
        for pid in lineups[idx]:
            pid_str = str(pid)
            dk_id = internal_to_dk.get(pid_str)
            if dk_id is None:
                try:
                    numeric_pid = int(pid_str)
                except (TypeError, ValueError):
                    numeric_pid = None
                if numeric_pid is not None and numeric_pid in draftable_ids_by_player:
                    dk_id = numeric_pid
                elif numeric_pid is not None:
                    dk_id = draftable_to_dk_player_id.get(numeric_pid)
            if dk_id is None:
                unmapped_players.add(pid_str)
                continue
            if dk_id not in draftable_ids_by_player:
                unmapped_players.add(f"{pid_str}(dk={dk_id})")
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

    if unmapped_players:
        logger.warning(
            "apply_build: %d players could not be mapped to DK draftable IDs: %s",
            len(unmapped_players),
            sorted(unmapped_players)[:20],  # Log first 20 to avoid spam
        )

    entry_state.entries = updated_entries
    entry_state.client_revision += 1
    entry_state.updated_at = datetime.utcnow().isoformat()
    path.write_text(entry_state.model_dump_json(indent=2))
    return entry_state


def _build_preview_response(
    session: LateSwapSession,
    candidates_by_entry_id: dict[str, list[LateSwapCandidate]],
) -> LateSwapPreviewResponse:
    return LateSwapPreviewResponse(
        session=session,
        candidates_by_entry_id=candidates_by_entry_id,
        selected_candidates_by_entry_id=session.selected_candidates_by_entry_id,
    )


@router.post("/late-swap/sessions", response_model=LateSwapSession)
async def create_late_swap_session(request: LateSwapSessionCreateRequest):
    contest_ids = [str(cid) for cid in request.contest_ids if str(cid).strip()]
    if not contest_ids:
        raise HTTPException(status_code=400, detail="contest_ids cannot be empty")
    entry_states = _load_entry_states_for_contests(request.date, contest_ids)
    session = LateSwapSession(
        session_id=_new_late_swap_session_id(),
        game_date=request.date,
        site="dk",
        contest_ids=contest_ids,
        draft_group_ids=sorted({int(state.draft_group_id) for state in entry_states}),
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
        status="draft",
        source_entry_revisions={
            str(state.contest_id): int(state.client_revision)
            for state in entry_states
        },
        source_profile=LateSwapSourceProfile.model_validate(_build_source_profile(entry_states)),
        policy=request.policy or LateSwapPolicy.with_mode_defaults("preserve_targets"),
    )
    session_store.create_session(
        session,
        request_payload=request.model_dump(mode="json", by_alias=True),
    )
    return session


@router.get("/late-swap/sessions", response_model=List[LateSwapSession])
async def list_late_swap_sessions(date: str, limit: int = 30):
    return session_store.list_sessions(game_date=date, site="dk", limit=limit)


@router.get("/late-swap/sessions/{session_id}", response_model=LateSwapPreviewResponse)
async def get_late_swap_session(session_id: str, date: str | None = None):
    try:
        session = (
            session_store.load_session(game_date=str(date), session_id=session_id, site="dk")
            if date
            else _load_session_any_date(session_id)
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    candidates = session_store.load_candidates(
        game_date=session.game_date,
        session_id=session.session_id,
        site=session.site,
    )
    return _build_preview_response(session, candidates)


@router.post("/late-swap/sessions/{session_id}/preview", response_model=LateSwapPreviewResponse)
async def preview_late_swap_session(
    session_id: str,
    date: str | None = None,
    request: LateSwapPreviewRequest = Body(default=LateSwapPreviewRequest()),
):
    try:
        session = (
            session_store.load_session(game_date=str(date), session_id=session_id, site="dk")
            if date
            else _load_session_any_date(session_id)
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        updated_session, candidates = _session_preview(session=session, request=request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Late swap preview failed for session=%s", session_id)
        session.status = "failed"
        session.updated_at = utc_now_iso()
        session.warnings = [*session.warnings, f"preview_failed: {exc}"]
        session_store.save_session(session)
        raise HTTPException(status_code=500, detail=f"Late swap preview failed: {exc}") from exc

    session_store.save_candidates(
        game_date=updated_session.game_date,
        session_id=updated_session.session_id,
        site=updated_session.site,
        candidates_by_entry_id=candidates,
    )
    session_store.save_session(updated_session)
    return _build_preview_response(updated_session, candidates)


@router.post("/late-swap/sessions/{session_id}/pin-candidates", response_model=LateSwapPreviewResponse)
async def pin_late_swap_candidates(
    session_id: str,
    date: str | None = None,
    request: LateSwapPinCandidatesRequest = Body(...),
):
    try:
        session = (
            session_store.load_session(game_date=str(date), session_id=session_id, site="dk")
            if date
            else _load_session_any_date(session_id)
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    candidates = session_store.load_candidates(
        game_date=session.game_date,
        session_id=session.session_id,
        site=session.site,
    )
    candidate_ids_by_entry = {
        entry_id: {candidate.candidate_id for candidate in entry_candidates}
        for entry_id, entry_candidates in candidates.items()
    }
    next_pins = {} if request.clear_existing else dict(session.pinned_candidates_by_entry_id)
    for entry_id, candidate_id in request.pins.items():
        if candidate_id not in candidate_ids_by_entry.get(entry_id, set()):
            raise HTTPException(
                status_code=400,
                detail=f"Pinned candidate not found for entry {entry_id}: {candidate_id}",
            )
        next_pins[str(entry_id)] = str(candidate_id)

    session.pinned_candidates_by_entry_id = next_pins
    session.updated_at = utc_now_iso()
    session.status = "stale"
    session_store.save_session(session)
    updated_session, candidates = _session_preview(session=session, request=LateSwapPreviewRequest())
    session_store.save_candidates(
        game_date=updated_session.game_date,
        session_id=updated_session.session_id,
        site=updated_session.site,
        candidates_by_entry_id=candidates,
    )
    session_store.save_session(updated_session)
    return _build_preview_response(updated_session, candidates)


@router.post("/late-swap/sessions/{session_id}/policy", response_model=LateSwapSession)
async def update_late_swap_policy(
    session_id: str,
    date: str | None = None,
    request: LateSwapPolicyUpdateRequest = Body(...),
):
    try:
        session = (
            session_store.load_session(game_date=str(date), session_id=session_id, site="dk")
            if date
            else _load_session_any_date(session_id)
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.policy = request.policy
    session.status = "stale"
    session.updated_at = utc_now_iso()
    session_store.save_session(session)
    return session


@router.post("/late-swap/sessions/{session_id}/commit", response_model=LateSwapSession)
async def commit_late_swap_session(
    session_id: str,
    date: str | None = None,
    request: LateSwapCommitRequest = Body(default=LateSwapCommitRequest()),
):
    try:
        session = (
            session_store.load_session(game_date=str(date), session_id=session_id, site="dk")
            if date
            else _load_session_any_date(session_id)
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    candidates = session_store.load_candidates(
        game_date=session.game_date,
        session_id=session.session_id,
        site=session.site,
    )
    candidate_lookup: dict[str, LateSwapCandidate] = {}
    for entry_candidates in candidates.values():
        for candidate in entry_candidates:
            candidate_lookup[candidate.candidate_id] = candidate
    if not session.selected_candidates_by_entry_id:
        raise HTTPException(status_code=400, detail="Session has no selected preview to commit")

    entry_states = {
        str(state.contest_id): state
        for state in _load_entry_states_for_contests(session.game_date, session.contest_ids)
    }
    now_iso = utc_now_iso()
    updated_contests: set[str] = set()
    for scoped_entry_id, candidate_id in session.selected_candidates_by_entry_id.items():
        candidate = candidate_lookup.get(candidate_id)
        if candidate is None:
            continue
        contest_id, entry_key = _split_scoped_entry_id(scoped_entry_id)
        state = entry_states.get(contest_id)
        if state is None:
            continue
        for idx, entry in enumerate(state.entries):
            scoped = _entry_scoped_id(contest_id, entry, idx)
            if scoped != scoped_entry_id:
                continue
            for slot in DK_NBA_SLOTS:
                entry[slot] = str(candidate.slot_values.get(slot, entry.get(slot, "")))
            updated_contests.add(contest_id)
            break

    for contest_id in updated_contests:
        state = entry_states[contest_id]
        state.client_revision = int(state.client_revision) + 1
        state.updated_at = now_iso
        state.source_late_swap_session_id = session.session_id
        state.source_late_swap_mode = session.policy.mode
        state.source_late_swap_committed_at = now_iso
        _entry_path(session.game_date, contest_id).write_text(state.model_dump_json(indent=2))
        session.source_entry_revisions[contest_id] = int(state.client_revision)

    session.status = "committed"
    session.updated_at = now_iso
    if request.note:
        session.warnings = [*session.warnings, f"commit_note: {request.note}"]
    session_store.save_session(session)
    return session


@router.post("/late-swap/sessions/{session_id}/export")
async def export_late_swap_session(
    session_id: str,
    date: str | None = None,
    request: LateSwapExportRequest = Body(default=LateSwapExportRequest()),
):
    try:
        session = (
            session_store.load_session(game_date=str(date), session_id=session_id, site="dk")
            if date
            else _load_session_any_date(session_id)
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    contest_ids = [str(cid) for cid in (request.contest_ids or session.contest_ids)]
    if not contest_ids:
        raise HTTPException(status_code=400, detail="No contests available for export")

    if not request.include_uncommitted_preview:
        return await export_entries_batch(
            date=session.game_date,
            request=ExportEntriesRequest(contest_ids=contest_ids),
            force=False,
        )

    candidates = session_store.load_candidates(
        game_date=session.game_date,
        session_id=session.session_id,
        site=session.site,
    )
    candidate_lookup: dict[str, LateSwapCandidate] = {}
    for entry_candidates in candidates.values():
        for candidate in entry_candidates:
            candidate_lookup[candidate.candidate_id] = candidate

    entry_states = _load_entry_states_for_contests(session.game_date, contest_ids)
    output = io.StringIO()
    writer = csv.writer(output)

    header = _export_header_for_entry_state(entry_states[0])
    writer.writerow(header)
    exported_entries = 0
    for state in entry_states:
        for idx, entry in enumerate(state.entries):
            scoped_id = _entry_scoped_id(str(state.contest_id), entry, idx)
            candidate_id = session.selected_candidates_by_entry_id.get(scoped_id)
            if candidate_id:
                candidate = candidate_lookup.get(candidate_id)
                if candidate:
                    temp_entry = dict(entry)
                    for slot in DK_NBA_SLOTS:
                        temp_entry[slot] = str(candidate.slot_values.get(slot, temp_entry.get(slot, "")))
                    writer.writerow(_export_row_for_header(temp_entry, header))
                    exported_entries += 1
                    continue
            writer.writerow(_export_row_for_header(entry, header))
            exported_entries += 1

    csv_text = output.getvalue()
    session_root = session_store.session_dir(
        game_date=session.game_date,
        session_id=session.session_id,
        site=session.site,
    )
    session_root.mkdir(parents=True, exist_ok=True)
    preview_csv = session_root / "preview_export.csv"
    preview_manifest = session_root / "preview_export_manifest.json"
    preview_csv.write_text(csv_text, encoding="utf-8")
    preview_manifest.write_text(
        json.dumps(
            {
                "session_id": session.session_id,
                "created_at": utc_now_iso(),
                "game_date": session.game_date,
                "contest_ids": contest_ids,
                "lineup_count": exported_entries,
                "policy_mode": session.policy.mode,
                "target_source": session.policy.target_source,
                "warnings": session.warnings,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=entries_{session.game_date}_{session.session_id}_preview.csv",
            "X-Late-Swap-Session-Id": session.session_id,
            "X-Entry-Count": str(exported_entries),
            "Access-Control-Expose-Headers": "X-Late-Swap-Session-Id, X-Entry-Count",
        },
    )


@router.post("/entries/{contest_id}/late-swap", response_model=LateSwapResult)
async def late_swap_entries(contest_id: str, date: str, request: LateSwapRequest):
    path = _entry_path(date, contest_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Entry file {contest_id} not found for {date}")

    entry_state_before = EntryFileState.model_validate_json(path.read_text())
    policy = LateSwapPolicy.with_mode_defaults("preserve_targets")
    policy.candidate_count_per_entry = max(6, min(20, int(request.n_alternatives) + 1))

    session = LateSwapSession(
        session_id=_new_late_swap_session_id(),
        game_date=date,
        site="dk",
        contest_ids=[contest_id],
        draft_group_ids=[int(entry_state_before.draft_group_id)],
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
        status="draft",
        source_entry_revisions={contest_id: int(entry_state_before.client_revision)},
        source_profile=LateSwapSourceProfile.model_validate(_build_source_profile([entry_state_before])),
        policy=policy,
    )
    session_store.create_session(
        session,
        request_payload={
            "legacy": True,
            "contest_id": contest_id,
            "date": date,
            "request": request.model_dump(mode="json"),
        },
    )

    preview_request = LateSwapPreviewRequest(
        run_id=request.run_id,
        ownership_mode=request.ownership_mode,
        use_user_overrides=True,
        only_out_lineups=request.only_out_lineups,
    )
    session, candidates_by_entry_id = _session_preview(session=session, request=preview_request)
    session_store.save_candidates(
        game_date=session.game_date,
        session_id=session.session_id,
        site=session.site,
        candidates_by_entry_id=candidates_by_entry_id,
    )
    session_store.save_session(session)

    # Legacy compatibility keeps auto-commit behavior.
    await commit_late_swap_session(
        session_id=session.session_id,
        date=session.game_date,
        request=LateSwapCommitRequest(note="legacy_auto_commit"),
    )
    entry_state_after = EntryFileState.model_validate_json(path.read_text())

    alternatives_by_entry_id: Dict[str, EntryAlternatives] = {}
    locked_slots_by_entry_id: Dict[str, List[str]] = {}
    skipped_no_out = 0
    for scoped_entry_id, candidates in candidates_by_entry_id.items():
        _contest, local_entry_id = _split_scoped_entry_id(scoped_entry_id)
        selected_candidate_id = session.selected_candidates_by_entry_id.get(scoped_entry_id)
        selected_idx = 0
        alternatives: List[LineupAlternative] = []
        for idx, candidate in enumerate(candidates):
            if candidate.candidate_id == selected_candidate_id:
                selected_idx = idx
            alternatives.append(
                LineupAlternative(
                    lineup_idx=idx,
                    projected_score=float(candidate.projected_score or 0.0),
                    slot_values=dict(candidate.slot_values),
                    player_swaps=[],
                )
            )
        hold_candidate = next((cand for cand in candidates if cand.generated_by == "hold"), None)
        if hold_candidate and "skipped_only_out_filter" in hold_candidate.reason_codes:
            skipped_no_out += 1
        locked_slots = list(session.lock_state.locked_slots_by_entry_id.get(scoped_entry_id, []))
        alternatives_by_entry_id[local_entry_id] = EntryAlternatives(
            entry_id=local_entry_id,
            locked_slots=locked_slots,
            alternatives=alternatives,
            selected_idx=selected_idx,
        )
        locked_slots_by_entry_id[local_entry_id] = locked_slots

    selection_summary = session.selection_summary
    entries_total = int(selection_summary.entries_total) if selection_summary else len(entry_state_after.entries)
    entries_swapped = int(selection_summary.entries_swapped) if selection_summary else 0
    entries_held = int(selection_summary.entries_held) if selection_summary else entries_total
    status_key = selection_summary.status if selection_summary else "fallback_hold"

    return LateSwapResult(
        entry_state=entry_state_after,
        locked_count=int(session.lock_state.locked_slots_total),
        updated_entries=len(entry_state_after.entries),
        missing_locked_ids=[],
        locked_slots_by_entry_id=locked_slots_by_entry_id,
        alternatives_by_entry_id=alternatives_by_entry_id,
        selection_summary=LateSwapSummary(
            entries_total=entries_total,
            entries_swapped=entries_swapped,
            entries_held=entries_held,
            entries_unmapped=int(session.lock_state.entries_with_unmapped_locked_slots),
            entries_unknown=0,
            entries_skipped_no_out=skipped_no_out,
        ),
        solver_summary=SolverSummary(status_counts={str(status_key): 1}, avg_gap=None, max_gap=None),
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


@router.get("/entries/{contest_id}/validate", response_model=ExportValidationResult)
async def validate_entry_file(contest_id: str, date: str):
    """Validate entries before export - check for empty slots, invalid IDs, duplicates."""
    path = _entry_path(date, contest_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Entry file {contest_id} not found for {date}")
    entry_state = EntryFileState.model_validate_json(path.read_text())

    return _validate_entries_for_export(entry_state.entries, entry_state.draft_group_id)


@router.post("/entries/{contest_id}/export")
async def export_entry_file(
    contest_id: str,
    date: str,
    force: bool = False,
    request: ExportEntrySelectionRequest | None = Body(default=None),
):
    """Export entries to CSV for DraftKings upload.

    Args:
        contest_id: Contest ID to export
        date: Game date
        force: If True, export even with validation errors (default False)
    """
    path = _entry_path(date, contest_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Entry file {contest_id} not found for {date}")
    entry_state = EntryFileState.model_validate_json(path.read_text())
    entries = entry_state.entries

    selected_ids: set[str] | None = None
    if request and request.entry_ids:
        selected_ids = {str(entry_id) for entry_id in request.entry_ids}
        filtered_entries: list[dict[str, str]] = []
        for idx, entry in enumerate(entries):
            entry_id = str(entry.get("entry_id", ""))
            entry_key = str(entry.get("entry_key") or entry_id or f"row-{idx + 1}")
            if entry_id in selected_ids or entry_key in selected_ids:
                filtered_entries.append(entry)
        entries = filtered_entries
        if len(entries) == 0:
            raise HTTPException(status_code=400, detail="No selected entries found for export")

    # Validate before export
    validation = _validate_entries_for_export(entries, entry_state.draft_group_id)
    if not validation.valid and not force:
        error_details = "; ".join(f"{i.entry_id}: {i.message}" for i in validation.issues[:5])
        if len(validation.issues) > 5:
            error_details += f" (and {len(validation.issues) - 5} more)"
        raise HTTPException(
            status_code=400,
            detail=f"Validation failed with {validation.errors_count} errors: {error_details}. Use force=true to export anyway.",
        )

    export_id = _generate_export_id()
    contest_root = _contest_root_for_export(
        site=str(entry_state.site or "dk"),
        game_date=date,
        draft_group_id=int(entry_state.draft_group_id),
    )
    exports_dir = contest_root / "exports"
    eval_dir = contest_root / "eval_pre" / f"export_{export_id}"
    exports_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    output = io.StringIO()
    writer = csv.writer(output)
    export_header = _export_header_for_entry_state(entry_state)
    writer.writerow(export_header)
    for entry in entries:
        writer.writerow(_export_row_for_header(entry, export_header))

    csv_text = output.getvalue()
    export_csv_path = exports_dir / f"export_{export_id}.csv"
    manifest_path = exports_dir / f"export_{export_id}_manifest.json"
    status_path = eval_dir / "eval_status.json"

    worlds_info = _resolve_latest_sim_v2_worlds(game_date=date)
    manifest: dict[str, object] = {
        "export_id": export_id,
        "created_at_utc": _utc_now_iso(),
        "site": str(entry_state.site or "dk"),
        "game_date": date,
        "draft_group_id": int(entry_state.draft_group_id),
        "contest_ids": [str(contest_id)],
        "export_csv_path": str(export_csv_path.resolve()),
        "lineup_count": int(len(entries)),
        "git_sha": _safe_git_sha(),
        # Evaluation config defaults (passed explicitly by runner).
        "train_frac": 0.7,
        "eval_seed": 123,
        "k_runtime_holdouts": 3,
        "num_worlds_runtime": 10000,
        **worlds_info,
        **_entry_state_source_payload(entry_state),
    }

    try:
        export_csv_path.write_text(csv_text, encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        status_payload = {
            "status": "PENDING",
            "export_id": export_id,
            "created_at": manifest["created_at_utc"],
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "return_code": None,
            "report_dir": str(eval_dir.resolve()),
            "error_message": None,
            "warnings": [],
        }
        status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True), encoding="utf-8")

        subprocess.Popen(
            [sys.executable, "-m", "projections.jobs.eval_runner", "--manifest", str(manifest_path)],
            cwd=str(paths.get_project_root()),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        logger.exception("Export eval setup failed (export will still succeed): %s", exc)
        try:
            status_payload = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            if not isinstance(status_payload, dict):
                status_payload = {}
            status_payload.update(
                {
                    "status": "FAILED",
                    "export_id": export_id,
                    "finished_at": _utc_now_iso(),
                    "error_message": str(exc),
                    "report_dir": str(eval_dir.resolve()),
                }
            )
            status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            pass

    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=entries_{date}_{contest_id}.csv",
            "X-Export-Id": export_id,
            "X-Validation-Warnings": str(validation.warnings_count),
            "X-Validation-Duplicates": str(validation.duplicate_lineup_count),
            "X-Entry-Count": str(len(entries)),
            "Access-Control-Expose-Headers": "X-Export-Id, X-Validation-Warnings, X-Validation-Duplicates, X-Entry-Count",
        },
    )


@router.post("/entries/export")
async def export_entries_batch(date: str, request: ExportEntriesRequest, force: bool = False):
    """Export multiple contests into a single CSV.

    Args:
        date: Game date
        request: Contest IDs to export
        force: If True, export even with validation errors (default False)
    """
    if not request.contest_ids:
        raise HTTPException(status_code=400, detail="No contest_ids provided")

    # First pass: load all entries and validate
    all_entries: List[Dict[str, str]] = []
    entry_states: List[EntryFileState] = []
    draft_group_ids: set[int] = set()
    sites: set[str] = set()
    export_header: List[str] | None = None
    for contest_id in request.contest_ids:
        path = _entry_path(date, contest_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Entry file {contest_id} not found for {date}")
        entry_state = EntryFileState.model_validate_json(path.read_text())
        entry_states.append(entry_state)
        contest_header = _export_header_for_entry_state(entry_state)
        if export_header is None:
            export_header = contest_header
        elif contest_header != export_header:
            export_header = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"] + list(DK_NBA_SLOTS)
        draft_group_ids.add(int(entry_state.draft_group_id))
        sites.add(str(entry_state.site or "dk"))
        all_entries.extend(entry_state.entries)

    # Validate all entries
    draft_group_id: int | None = next(iter(draft_group_ids)) if len(draft_group_ids) == 1 else None
    validation = _validate_entries_for_export(all_entries, draft_group_id or 0)
    if not validation.valid and not force:
        error_details = "; ".join(f"{i.entry_id}: {i.message}" for i in validation.issues[:5])
        if len(validation.issues) > 5:
            error_details += f" (and {len(validation.issues) - 5} more)"
        raise HTTPException(
            status_code=400,
            detail=f"Validation failed with {validation.errors_count} errors: {error_details}. Use force=true to export anyway.",
        )

    # Second pass: write CSV
    output = io.StringIO()
    writer = csv.writer(output)
    resolved_header = export_header or ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"] + list(DK_NBA_SLOTS)
    writer.writerow(resolved_header)
    total_entries = 0
    for entry in all_entries:
        writer.writerow(_export_row_for_header(entry, resolved_header))
        total_entries += 1

    export_id = _generate_export_id()
    site = next(iter(sites)) if len(sites) == 1 else "dk"
    contest_root = _contest_root_for_export(site=site, game_date=date, draft_group_id=draft_group_id)
    exports_dir = contest_root / "exports"
    eval_dir = contest_root / "eval_pre" / f"export_{export_id}"
    exports_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    filename = f"entries_{date}_combined.csv"
    csv_text = output.getvalue()
    export_csv_path = exports_dir / f"export_{export_id}.csv"
    manifest_path = exports_dir / f"export_{export_id}_manifest.json"
    status_path = eval_dir / "eval_status.json"

    worlds_info = _resolve_latest_sim_v2_worlds(game_date=date)
    manifest: dict[str, object] = {
        "export_id": export_id,
        "created_at_utc": _utc_now_iso(),
        "site": site,
        "game_date": date,
        "draft_group_id": int(draft_group_id) if draft_group_id is not None else None,
        "draft_group_id_candidates": sorted(draft_group_ids),
        "contest_ids": [str(cid) for cid in request.contest_ids],
        "export_csv_path": str(export_csv_path.resolve()),
        "lineup_count": int(total_entries),
        "git_sha": _safe_git_sha(),
        # Evaluation config defaults (passed explicitly by runner).
        "train_frac": 0.7,
        "eval_seed": 123,
        "k_runtime_holdouts": 3,
        "num_worlds_runtime": 10000,
        **worlds_info,
        **_aggregate_export_sources(entry_states),
    }

    try:
        export_csv_path.write_text(csv_text, encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        status_payload = {
            "status": "PENDING",
            "export_id": export_id,
            "created_at": manifest["created_at_utc"],
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "return_code": None,
            "report_dir": str(eval_dir.resolve()),
            "error_message": None,
            "warnings": [],
        }
        if draft_group_id is None:
            status_payload["status"] = "FAILED"
            status_payload["finished_at"] = _utc_now_iso()
            status_payload["error_message"] = f"ambiguous draft_group_id candidates={sorted(draft_group_ids)}"
        status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True), encoding="utf-8")

        if draft_group_id is not None:
            subprocess.Popen(
                [sys.executable, "-m", "projections.jobs.eval_runner", "--manifest", str(manifest_path)],
                cwd=str(paths.get_project_root()),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    except Exception as exc:
        logger.exception("Export eval setup failed (export will still succeed): %s", exc)
        try:
            status_payload = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            if not isinstance(status_payload, dict):
                status_payload = {}
            status_payload.update(
                {
                    "status": "FAILED",
                    "export_id": export_id,
                    "finished_at": _utc_now_iso(),
                    "error_message": str(exc),
                    "report_dir": str(eval_dir.resolve()),
                }
            )
            status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            pass

    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "X-Export-Id": export_id,
            "X-Validation-Warnings": str(validation.warnings_count),
            "X-Validation-Duplicates": str(validation.duplicate_lineup_count),
            "X-Entry-Count": str(total_entries),
            "Access-Control-Expose-Headers": "X-Export-Id, X-Validation-Warnings, X-Validation-Duplicates, X-Entry-Count",
        },
    )


def _find_export_manifest(export_id: str) -> Path:
    root = paths.data_path("contests")
    pattern = f"export_{export_id}_manifest.json"
    matches = list(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"export_id not found: {export_id}")
    if len(matches) > 1:
        raise RuntimeError(f"export_id is not unique on disk: {export_id} ({len(matches)} matches)")
    return matches[0]


def _tail_text(path: Path, *, max_lines: int) -> str:
    max_lines = max(1, min(int(max_lines), 2000))
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            start = max(0, end - 256_000)
            f.seek(start, os.SEEK_SET)
            data = f.read()
        text = data.decode("utf-8", errors="replace")
    except Exception:
        text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()[-max_lines:]
    return "\n".join(lines) + "\n"


@router.get("/exports/{export_id}/eval-status")
async def get_export_eval_status(export_id: str):
    try:
        manifest_path = _find_export_manifest(export_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    eval_dir = manifest_path.parent.parent / "eval_pre" / f"export_{export_id}"
    status_path = eval_dir / "eval_status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing eval_status.json for export_id={export_id}")

    # Stale-run detection: if RUNNING too long and PID is gone, mark FAILED.
    try:
        from projections.jobs.eval_runner import _maybe_mark_stale_running

        status = _maybe_mark_stale_running(status_path)
    except Exception:
        status = json.loads(status_path.read_text(encoding="utf-8"))

    return status


@router.get("/exports/{export_id}/eval-report")
async def get_export_eval_report(export_id: str, format: str = "md"):
    try:
        manifest_path = _find_export_manifest(export_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    eval_dir = manifest_path.parent.parent / "eval_pre" / f"export_{export_id}"
    if format.lower() == "json":
        report_path = eval_dir / "eval_report.json"
        if not report_path.exists():
            raise HTTPException(status_code=404, detail=f"Missing eval_report.json for export_id={export_id}")
        return json.loads(report_path.read_text(encoding="utf-8"))

    report_path = eval_dir / "eval_report.md"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing eval_report.md for export_id={export_id}")
    return Response(content=report_path.read_text(encoding="utf-8"), media_type="text/markdown")


@router.get("/exports/{export_id}/eval-log")
async def get_export_eval_log(export_id: str, lines: int = 200):
    try:
        manifest_path = _find_export_manifest(export_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    eval_dir = manifest_path.parent.parent / "eval_pre" / f"export_{export_id}"
    log_path = eval_dir / "eval.log"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing eval.log for export_id={export_id}")

    return Response(content=_tail_text(log_path, max_lines=lines), media_type="text/plain")
