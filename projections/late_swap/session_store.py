"""Persistence helpers for late swap sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from projections import paths
from projections.late_swap.models import (
    LateSwapCandidate,
    LateSwapSession,
    LateSwapStatus,
)

SESSION_STATE_FILENAME = "session_state.json"
REQUEST_FILENAME = "request.json"
CANDIDATES_FILENAME = "candidates.json"
SELECTION_FILENAME = "selection_result.json"
DIAGNOSTICS_FILENAME = "diagnostics.json"


def late_swap_date_root(game_date: str, site: str = "dk") -> Path:
    return paths.data_path() / "late_swap" / str(game_date) / str(site)


def session_dir(*, game_date: str, session_id: str, site: str = "dk") -> Path:
    return late_swap_date_root(game_date=game_date, site=site) / f"session={session_id}"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(loaded, (dict, list)):
        return loaded
    return None


def create_session(session: LateSwapSession, request_payload: dict[str, object]) -> Path:
    root = session_dir(game_date=session.game_date, session_id=session.session_id, site=session.site)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / REQUEST_FILENAME, request_payload)
    _write_json(root / SESSION_STATE_FILENAME, session.model_dump(mode="json"))
    return root


def save_session(session: LateSwapSession) -> Path:
    root = session_dir(game_date=session.game_date, session_id=session.session_id, site=session.site)
    _write_json(root / SESSION_STATE_FILENAME, session.model_dump(mode="json"))
    _write_json(root / DIAGNOSTICS_FILENAME, session.diagnostics.model_dump(mode="json"))
    if session.selection_summary is not None:
        _write_json(root / SELECTION_FILENAME, session.selection_summary.model_dump(mode="json"))
    return root


def load_session(*, game_date: str, session_id: str, site: str = "dk") -> LateSwapSession:
    root = session_dir(game_date=game_date, session_id=session_id, site=site)
    path = root / SESSION_STATE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Late swap session not found: {session_id}")
    return LateSwapSession.model_validate_json(path.read_text(encoding="utf-8"))


def list_sessions(*, game_date: str, site: str = "dk", limit: int = 50) -> list[LateSwapSession]:
    root = late_swap_date_root(game_date=game_date, site=site)
    if not root.exists():
        return []
    sessions: list[LateSwapSession] = []
    for item in sorted(root.glob("session=*"), reverse=True):
        path = item / SESSION_STATE_FILENAME
        if not path.exists():
            continue
        try:
            sessions.append(LateSwapSession.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:
            continue
        if len(sessions) >= int(limit):
            break
    return sessions


def save_candidates(
    *,
    game_date: str,
    session_id: str,
    site: str,
    candidates_by_entry_id: dict[str, list[LateSwapCandidate]],
) -> None:
    root = session_dir(game_date=game_date, session_id=session_id, site=site)
    payload = {
        entry_id: [c.model_dump(mode="json") for c in candidates]
        for entry_id, candidates in candidates_by_entry_id.items()
    }
    _write_json(root / CANDIDATES_FILENAME, payload)


def load_candidates(
    *,
    game_date: str,
    session_id: str,
    site: str,
) -> dict[str, list[LateSwapCandidate]]:
    root = session_dir(game_date=game_date, session_id=session_id, site=site)
    payload = _read_json(root / CANDIDATES_FILENAME)
    if not isinstance(payload, dict):
        return {}
    out: dict[str, list[LateSwapCandidate]] = {}
    for entry_id, raw_candidates in payload.items():
        if not isinstance(raw_candidates, list):
            continue
        parsed: list[LateSwapCandidate] = []
        for candidate in raw_candidates:
            if not isinstance(candidate, dict):
                continue
            try:
                parsed.append(LateSwapCandidate.model_validate(candidate))
            except Exception:
                continue
        out[str(entry_id)] = parsed
    return out


def mark_session_status(
    *,
    game_date: str,
    session_id: str,
    site: str,
    status: LateSwapStatus,
    warning: str | None = None,
) -> LateSwapSession:
    session = load_session(game_date=game_date, session_id=session_id, site=site)
    session.status = status
    if warning:
        session.warnings = [*session.warnings, warning]
    save_session(session)
    return session


def all_session_ids(*, game_date: str, site: str = "dk") -> Iterable[str]:
    root = late_swap_date_root(game_date=game_date, site=site)
    if not root.exists():
        return []
    session_ids: list[str] = []
    for p in sorted(root.glob("session=*"), reverse=True):
        token = p.name.split("session=", 1)[-1].strip()
        if token:
            session_ids.append(token)
    return session_ids
