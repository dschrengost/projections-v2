from __future__ import annotations

import difflib
import json
import logging
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from unidecode import unidecode

from projections.api.contest_service import parse_contest_csv, parse_lineup
from projections.api.optimizer_api import _load_dk_nba_draftable_ids_by_player
from projections.api.optimizer_service import build_player_pool
from projections.contest_sim.contest_sim_service import load_player_worlds, run_contest_simulation
from projections.contest_sim.field_library import FieldLibrary, save_field_library
from projections.names import normalize_player_name
from projections.paths import get_data_root
from projections.post_contest.replay_models import (
    ContestReplayEntry,
    ContestReplayMeta,
    ContestReplayRun,
    PreparedReplayContext,
    ResolvedContestReplayEntry,
)

logger = logging.getLogger(__name__)

_ENTRY_ID_COLUMNS = ("EntryId", "Entry ID", "entry_id")
_ENTRY_NAME_COLUMNS = ("EntryName", "Entry Name", "entry_name")
_RANK_COLUMNS = ("Rank", "rank")
_POINT_COLUMNS = ("Points", "points")
_LINEUP_COLUMNS = ("Lineup", "lineup", "Lineup String")
_PRIZE_COLUMNS = ("Prize", "prize", "Winnings", "Payout")
_INVENTORY_COLUMNS = [
    "date",
    "contest_id",
    "contest_name",
    "entry_fee",
    "prize_pool",
    "first_place_prize",
    "current_entries_meta",
    "max_entries_per_user",
    "draft_group_id",
    "start_time",
    "results_path",
    "contest_class",
    "entry_limit_bucket",
    "is_low_stakes",
    "is_flagship",
]


def _normalize_name(name: str) -> str:
    return normalize_player_name(unidecode(str(name or "")).strip())


def _name_signature(normalized_name: str) -> str:
    parts = [part for part in str(normalized_name or "").split() if part]
    if len(parts) < 2:
        return ""
    return f"{parts[0][0]} {parts[-1]}"


def _last_name(normalized_name: str) -> str:
    parts = [part for part in str(normalized_name or "").split() if part]
    return parts[-1] if parts else ""


def _alias_override_path(data_root: Optional[Path]) -> Path:
    root = data_root or get_data_root()
    return root / "control_plane" / "contest_results" / "player_alias_overrides.json"


def _load_alias_overrides(data_root: Optional[Path]) -> Dict[str, str]:
    path = _alias_override_path(data_root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, str] = {}
    for raw_name, canonical_name in payload.items():
        normalized_raw = _normalize_name(str(raw_name))
        normalized_canonical = _normalize_name(str(canonical_name))
        if normalized_raw and normalized_canonical:
            out[normalized_raw] = normalized_canonical
    return out


def _coerce_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _canonicalize_player_id(value: object) -> str:
    coerced = _coerce_int(value)
    if coerced is not None:
        return str(coerced)
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace("$", "").replace(",", "").replace("%", "")
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _first_present(row: pd.Series, columns: Sequence[str]) -> object:
    for column in columns:
        if column in row.index:
            return row.get(column)
    return None


def _raw_results_root(data_root: Path) -> Path:
    return data_root / "bronze" / "dk_contests" / "nba_gpp_data"


def _inventory_path(data_root: Path) -> Path:
    return data_root / "analytics" / "contest_results" / "contest_inventory.parquet"


@lru_cache(maxsize=16)
def _cached_world_player_ids(
    *,
    game_date: str,
    data_root_str: str,
    run_id: Optional[str],
    worlds_source: str,
) -> Tuple[str, ...]:
    worlds = load_player_worlds(
        game_date=game_date,
        data_root=Path(data_root_str),
        run_id=run_id,
        worlds_source=worlds_source,  # type: ignore[arg-type]
    )
    return tuple(str(player_id) for player_id in worlds.player_index.keys())


@lru_cache(maxsize=4)
def _load_inventory_frame(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=_INVENTORY_COLUMNS)
    return pd.read_parquet(path, columns=_INVENTORY_COLUMNS)


def _load_inventory_row(
    *,
    contest_id: str,
    game_date: Optional[str],
    data_root: Path,
) -> Optional[Dict[str, Any]]:
    inventory = _load_inventory_frame(str(_inventory_path(data_root)))
    if inventory.empty:
        return None
    contest_mask = inventory["contest_id"].astype(str) == str(contest_id)
    if game_date:
        contest_mask &= inventory["date"].astype(str) == str(game_date)
    matches = inventory.loc[contest_mask]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def resolve_results_path(
    *,
    contest_id: str,
    game_date: Optional[str] = None,
    data_root: Optional[Path] = None,
) -> Path:
    data_root = data_root or get_data_root()
    inventory_row = _load_inventory_row(contest_id=str(contest_id), game_date=game_date, data_root=data_root)
    if inventory_row:
        results_path = inventory_row.get("results_path")
        if isinstance(results_path, str) and results_path:
            path = Path(results_path)
            if path.exists():
                return path

    base = _raw_results_root(data_root)
    candidates: List[Path] = []
    if game_date:
        candidates.extend(
            [
                base / game_date / "results" / f"contest_{contest_id}_results.csv",
                base / game_date / "results" / f"contest_{contest_id}_standings.csv",
                base / game_date / f"contest_{contest_id}_results.csv",
                base / game_date / f"contest_{contest_id}_standings.csv",
            ]
        )
    candidates.extend(
        [
            base / f"contest_{contest_id}_results.csv",
            base / f"contest_{contest_id}_standings.csv",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate results CSV for contest_id={contest_id} game_date={game_date or 'unknown'}"
    )


def normalized_entries_path(
    *,
    game_date: str,
    contest_id: str,
    data_root: Optional[Path] = None,
) -> Path:
    root = data_root or get_data_root()
    return (
        root
        / "silver"
        / "post_contest"
        / "contest_entries"
        / f"date={game_date}"
        / f"contest_id={contest_id}"
        / "entries.parquet"
    )


def replay_output_dir(
    *,
    game_date: str,
    contest_id: str,
    user_pattern: str,
    data_root: Optional[Path] = None,
) -> Path:
    root = data_root or get_data_root()
    user_slug = re.sub(r"[^a-z0-9]+", "-", _normalize_name(user_pattern)).strip("-") or "user"
    return (
        root
        / "analytics"
        / "contest_flashback"
        / f"date={game_date}"
        / f"contest_id={contest_id}"
        / f"user={user_slug}"
    )


def field_library_output_path(
    *,
    game_date: str,
    contest_id: str,
    data_root: Optional[Path] = None,
) -> Path:
    root = data_root or get_data_root()
    return (
        root
        / "builds"
        / "contest_flashback"
        / f"game_date={game_date}"
        / f"contest_id={contest_id}"
        / "field_library_actual.json"
    )


def load_contest_entries(
    *,
    contest_id: str,
    game_date: Optional[str] = None,
    data_root: Optional[Path] = None,
) -> Tuple[ContestReplayMeta, List[ContestReplayEntry]]:
    data_root = data_root or get_data_root()
    inventory_row = _load_inventory_row(contest_id=str(contest_id), game_date=game_date, data_root=data_root)
    results_path = resolve_results_path(contest_id=str(contest_id), game_date=game_date, data_root=data_root)
    df = parse_contest_csv(results_path)

    entry_id_col = next((col for col in _ENTRY_ID_COLUMNS if col in df.columns), None)
    if not entry_id_col:
        raise ValueError(f"Contest results CSV missing entry id column: {results_path}")
    lineup_col = next((col for col in _LINEUP_COLUMNS if col in df.columns), None)
    if not lineup_col:
        raise ValueError(f"Contest results CSV missing lineup column: {results_path}")

    deduped = df.drop_duplicates(subset=[entry_id_col]).copy()
    entries: List[ContestReplayEntry] = []
    for _, row in deduped.iterrows():
        raw_lineup = str(_first_present(row, _LINEUP_COLUMNS) or "").strip()
        lineup_names = parse_lineup(raw_lineup)
        lineup_key = "|".join(sorted(_normalize_name(name) for name in lineup_names))
        entries.append(
            ContestReplayEntry(
                entry_id=str(_first_present(row, _ENTRY_ID_COLUMNS) or ""),
                entry_name=str(_first_present(row, _ENTRY_NAME_COLUMNS) or "").strip(),
                rank=_coerce_int(_first_present(row, _RANK_COLUMNS)),
                points=_coerce_float(_first_present(row, _POINT_COLUMNS)),
                lineup_names=lineup_names,
                raw_lineup=raw_lineup,
                lineup_key=lineup_key,
                prize=_coerce_float(_first_present(row, _PRIZE_COLUMNS)),
            )
        )

    inferred_game_date = str(game_date or (inventory_row or {}).get("date") or results_path.parent.parent.name)
    if inferred_game_date == "nba_gpp_data":
        inferred_game_date = ""
    field_size = _coerce_int((inventory_row or {}).get("current_entries_meta")) or len(entries)
    draft_group_id = _coerce_int((inventory_row or {}).get("draft_group_id"))
    entry_fee = _coerce_float((inventory_row or {}).get("entry_fee")) or 0.0
    contest_name = str((inventory_row or {}).get("contest_name") or f"Contest {contest_id}")
    extra = {
        "prize_pool": _coerce_float((inventory_row or {}).get("prize_pool")),
        "first_place_prize": _coerce_float((inventory_row or {}).get("first_place_prize")),
        "contest_class": (inventory_row or {}).get("contest_class"),
        "entry_limit_bucket": (inventory_row or {}).get("entry_limit_bucket"),
        "results_row_count": int(len(df)),
        "deduped_entry_count": int(len(entries)),
    }
    meta = ContestReplayMeta(
        game_date=inferred_game_date,
        contest_id=str(contest_id),
        contest_name=contest_name,
        draft_group_id=draft_group_id,
        entry_fee=entry_fee,
        field_size=int(field_size),
        results_path=str(results_path),
        source="raw_results_csv",
        source_mode="exact_replay",
        extra=extra,
    )
    return meta, entries


def _build_name_to_internal_map(
    *,
    game_date: str,
    draft_group_id: int,
    data_root: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
    pool = build_player_pool(
        game_date=game_date,
        draft_group_id=draft_group_id,
        site="dk",
        run_id=run_id,
        data_root=data_root,
        include_unmatched_salaries=True,
        allow_zero_projections=True,
        exclude_inactive_players=False,
    )

    internal_to_dk_player_id: Dict[str, int] = {}
    internal_to_name: Dict[str, str] = {}
    for player in pool:
        player_id = _canonicalize_player_id(player.get("player_id"))
        if not player_id:
            continue
        internal_to_name[player_id] = str(player.get("name") or player_id)
        dk_id_raw = player.get("dk_id")
        dk_id = _coerce_int(dk_id_raw)
        if dk_id is not None:
            internal_to_dk_player_id[player_id] = dk_id

    dk_names_by_player: Dict[int, str] = {}
    try:
        _, dk_names_by_player = _load_dk_nba_draftable_ids_by_player(int(draft_group_id))
    except FileNotFoundError:
        logger.warning("Draftables file missing for draft_group_id=%s", draft_group_id)

    candidate_ids: Dict[str, set[str]] = defaultdict(set)
    signature_ids: Dict[str, set[str]] = defaultdict(set)
    for player_id, player_name in internal_to_name.items():
        normalized = _normalize_name(player_name)
        if normalized:
            candidate_ids[normalized].add(player_id)
            signature = _name_signature(normalized)
            if signature:
                signature_ids[signature].add(player_id)
        dk_player_id = internal_to_dk_player_id.get(player_id)
        if dk_player_id is not None:
            dk_name = dk_names_by_player.get(dk_player_id)
            normalized_dk = _normalize_name(dk_name) if dk_name else ""
            if normalized_dk:
                candidate_ids[normalized_dk].add(player_id)
                signature = _name_signature(normalized_dk)
                if signature:
                    signature_ids[signature].add(player_id)

    alias_overrides = _load_alias_overrides(data_root)
    for raw_alias, canonical_name in alias_overrides.items():
        player_id = None
        if canonical_name in candidate_ids and len(candidate_ids[canonical_name]) == 1:
            player_id = next(iter(candidate_ids[canonical_name]))
        elif canonical_name in internal_to_name.values():
            matching = [pid for pid, name in internal_to_name.items() if _normalize_name(name) == canonical_name]
            if len(matching) == 1:
                player_id = matching[0]
        if player_id is not None:
            candidate_ids[raw_alias].add(player_id)

    resolved: Dict[str, str] = {}
    ambiguous: Dict[str, List[str]] = {}
    for normalized_name, player_ids in candidate_ids.items():
        ordered = sorted(player_ids)
        if len(ordered) == 1:
            resolved[normalized_name] = ordered[0]
        else:
            ambiguous[normalized_name] = ordered

    resolved_signatures: Dict[str, str] = {}
    for signature, player_ids in signature_ids.items():
        ordered = sorted(player_ids)
        if len(ordered) == 1:
            resolved_signatures[signature] = ordered[0]
    return resolved, ambiguous, internal_to_name, resolved_signatures


def _resolve_name_to_player_id(
    *,
    raw_name: str,
    resolved_name_map: Dict[str, str],
    ambiguous_name_map: Dict[str, List[str]],
    resolved_signatures: Dict[str, str],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    normalized = _normalize_name(raw_name)
    if not normalized:
        return None, {"method": "empty", "raw_name": raw_name}

    player_id = resolved_name_map.get(normalized)
    if player_id is not None:
        return player_id, {"method": "exact", "raw_name": raw_name, "normalized_name": normalized}

    if normalized in ambiguous_name_map:
        return None, {
            "method": "ambiguous_exact",
            "raw_name": raw_name,
            "normalized_name": normalized,
            "candidate_player_ids": ambiguous_name_map[normalized],
        }

    signature = _name_signature(normalized)
    if signature and signature in resolved_signatures:
        return resolved_signatures[signature], {
            "method": "signature",
            "raw_name": raw_name,
            "normalized_name": normalized,
            "signature": signature,
        }

    candidates = difflib.get_close_matches(normalized, list(resolved_name_map.keys()), n=2, cutoff=0.84)
    if candidates:
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        best_score = difflib.SequenceMatcher(None, normalized, best).ratio()
        second_score = difflib.SequenceMatcher(None, normalized, second).ratio() if second else 0.0
        same_last_name = _last_name(normalized) and _last_name(normalized) == _last_name(best)
        if best_score >= 0.93 or (same_last_name and best_score >= 0.87 and (best_score - second_score) >= 0.03):
            return resolved_name_map[best], {
                "method": "fuzzy",
                "raw_name": raw_name,
                "normalized_name": normalized,
                "matched_name": best,
                "score": round(best_score, 4),
            }
        return None, {
            "method": "ambiguous_fuzzy",
            "raw_name": raw_name,
            "normalized_name": normalized,
            "matched_name": best,
            "score": round(best_score, 4),
            "runner_up": second,
            "runner_up_score": round(second_score, 4) if second else None,
        }

    return None, {"method": "unresolved", "raw_name": raw_name, "normalized_name": normalized}


def resolve_entries_to_internal_ids(
    entries: Sequence[ContestReplayEntry],
    *,
    game_date: str,
    draft_group_id: int,
    data_root: Optional[Path] = None,
    run_id: Optional[str] = None,
    canonical_player_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[ResolvedContestReplayEntry], Dict[str, Any]]:
    resolved_name_map, ambiguous_name_map, internal_to_name, resolved_signatures = _build_name_to_internal_map(
        game_date=game_date,
        draft_group_id=draft_group_id,
        data_root=data_root,
        run_id=run_id,
    )
    canonical_ids = {str(player_id) for player_id in (canonical_player_ids or [])}

    resolved_entries: List[ResolvedContestReplayEntry] = []
    unresolved_examples: List[Dict[str, Any]] = []
    ambiguous_examples: List[Dict[str, Any]] = []
    fuzzy_examples: List[Dict[str, Any]] = []
    outside_world_examples: List[Dict[str, Any]] = []
    resolved_slot_count = 0
    unresolved_slot_count = 0
    outside_world_slot_count = 0

    for entry in entries:
        player_ids: List[str] = []
        unresolved_names: List[str] = []
        for name in entry.lineup_names:
            player_id, diag = _resolve_name_to_player_id(
                raw_name=name,
                resolved_name_map=resolved_name_map,
                ambiguous_name_map=ambiguous_name_map,
                resolved_signatures=resolved_signatures,
            )
            if player_id is not None:
                if canonical_ids and player_id not in canonical_ids:
                    unresolved_names.append(name)
                    unresolved_slot_count += 1
                    outside_world_slot_count += 1
                    outside_world_examples.append(
                        {
                            "entry_id": entry.entry_id,
                            "entry_name": entry.entry_name,
                            "method": "outside_worlds_namespace",
                            "raw_name": name,
                            "resolved_player_id": player_id,
                        }
                    )
                    continue
                player_ids.append(player_id)
                resolved_slot_count += 1
                if diag and diag.get("method") == "fuzzy":
                    fuzzy_examples.append(
                        {
                            "entry_id": entry.entry_id,
                            "entry_name": entry.entry_name,
                            **diag,
                        }
                    )
                continue
            unresolved_names.append(name)
            unresolved_slot_count += 1
            if diag and diag.get("method") in {"ambiguous_exact", "ambiguous_fuzzy"}:
                ambiguous_examples.append(
                    {
                        "entry_id": entry.entry_id,
                        "entry_name": entry.entry_name,
                        **diag,
                    }
                )
            else:
                unresolved_examples.append(
                    {
                        "entry_id": entry.entry_id,
                        "entry_name": entry.entry_name,
                        **(diag or {"name": name}),
                    }
                )
        resolved_entries.append(
            ResolvedContestReplayEntry(
                **entry.to_dict(),
                player_ids=player_ids,
                unresolved_names=unresolved_names,
            )
        )

    stats = {
        "entry_count": int(len(entries)),
        "resolved_entry_count": int(sum(1 for entry in resolved_entries if not entry.unresolved_names)),
        "unresolved_entry_count": int(sum(1 for entry in resolved_entries if entry.unresolved_names)),
        "ambiguous_name_count": int(len(ambiguous_examples)),
        "resolved_slot_count": int(resolved_slot_count),
        "unresolved_slot_count": int(unresolved_slot_count),
        "slot_resolution_rate": float(resolved_slot_count / max(resolved_slot_count + unresolved_slot_count, 1)),
        "fuzzy_match_count": int(len(fuzzy_examples)),
        "outside_worlds_slot_count": int(outside_world_slot_count),
        "unresolved_examples": unresolved_examples[:10],
        "ambiguous_examples": ambiguous_examples[:10],
        "fuzzy_examples": fuzzy_examples[:10],
        "outside_worlds_examples": outside_world_examples[:10],
        "resolved_name_count": int(len(resolved_name_map)),
        "player_pool_size": int(len(internal_to_name)),
        "canonical_world_player_count": int(len(canonical_ids)),
    }
    return resolved_entries, stats


def _aggregate_lineups(entries: Iterable[ResolvedContestReplayEntry]) -> Tuple[List[List[str]], List[int]]:
    counts: Counter[Tuple[str, ...]] = Counter()
    ordered_lineups: Dict[Tuple[str, ...], List[str]] = {}
    for entry in entries:
        lineup = tuple(sorted(str(player_id) for player_id in entry.player_ids))
        if not lineup:
            continue
        counts[lineup] += 1
        ordered_lineups.setdefault(lineup, list(lineup))
    unique_lineups = [ordered_lineups[key] for key in counts.keys()]
    weights = [int(counts[key]) for key in counts.keys()]
    return unique_lineups, weights


def build_actual_field_library(
    entries: Sequence[ResolvedContestReplayEntry],
    *,
    meta: ContestReplayMeta,
) -> FieldLibrary:
    lineups, weights = _aggregate_lineups(entries)
    library = FieldLibrary(
        lineups=lineups,
        weights=weights,
        meta={
            "source": "actual_contest_results",
            "source_mode": meta.source_mode,
            "contest_id": meta.contest_id,
            "contest_name": meta.contest_name,
            "game_date": meta.game_date,
            "draft_group_id": meta.draft_group_id,
            "entry_fee": meta.entry_fee,
            "field_size": meta.field_size,
            "results_path": meta.results_path,
            "unique_lineup_count": len(lineups),
            "observed_entry_count": int(sum(weights)),
        },
    )
    library.validate()
    return library


def _select_user_entries(
    entries: Sequence[ResolvedContestReplayEntry],
    *,
    user_pattern: str,
) -> List[ResolvedContestReplayEntry]:
    pattern = user_pattern.lower().strip()
    if not pattern:
        return []
    return [entry for entry in entries if pattern in entry.entry_name.lower()]


def prepare_post_contest_replay(
    *,
    contest_id: str,
    game_date: Optional[str] = None,
    user_pattern: str,
    draft_group_id: Optional[int] = None,
    data_root: Optional[Path] = None,
    run_id: Optional[str] = None,
    worlds_source: str = "gtv2",
    strict_resolution: bool = True,
) -> PreparedReplayContext:
    data_root = data_root or get_data_root()
    meta, entries = load_contest_entries(contest_id=contest_id, game_date=game_date, data_root=data_root)
    resolved_draft_group_id = int(draft_group_id or meta.draft_group_id or 0)
    if resolved_draft_group_id <= 0:
        raise ValueError(
            f"draft_group_id is required for replay preparation (contest_id={contest_id})"
        )
    if meta.field_size > len(entries):
        raise ValueError(
            "Partial contest field detected; anchored emulation is not implemented yet "
            f"(observed_entries={len(entries)}, expected_field_size={meta.field_size})"
        )

    canonical_player_ids = _cached_world_player_ids(
        game_date=meta.game_date,
        data_root_str=str(data_root),
        run_id=run_id,
        worlds_source=worlds_source,
    )
    resolved_entries, resolution_stats = resolve_entries_to_internal_ids(
        entries,
        game_date=meta.game_date,
        draft_group_id=resolved_draft_group_id,
        data_root=data_root,
        run_id=run_id,
        canonical_player_ids=canonical_player_ids,
    )
    unresolved_entries = [entry for entry in resolved_entries if entry.unresolved_names]
    if strict_resolution and unresolved_entries:
        sample = unresolved_entries[0]
        raise ValueError(
            "Could not resolve all lineup names to internal player IDs. "
            f"First unresolved entry_id={sample.entry_id} names={sample.unresolved_names}; "
            f"outside_worlds_slot_count={resolution_stats.get('outside_worlds_slot_count', 0)}"
        )

    user_entries = _select_user_entries(resolved_entries, user_pattern=user_pattern)
    if not user_entries:
        raise ValueError(f"No contest entries matched user_pattern={user_pattern!r}")

    user_entry_ids = {entry.entry_id for entry in user_entries}
    opponent_entries = [
        entry for entry in resolved_entries if entry.entry_id not in user_entry_ids and not entry.unresolved_names
    ]
    user_entries_resolved = [entry for entry in user_entries if not entry.unresolved_names]
    if not user_entries_resolved:
        raise ValueError("Matched user entries exist, but none resolved cleanly to internal player IDs")

    user_lineups, user_weights = _aggregate_lineups(user_entries_resolved)
    opponent_library = build_actual_field_library(opponent_entries, meta=meta)
    resolution_stats = dict(resolution_stats)
    resolution_stats.update(
        {
            "user_pattern": user_pattern,
            "user_entry_count": int(len(user_entries)),
            "resolved_user_entry_count": int(len(user_entries_resolved)),
            "opponent_entry_count": int(len(opponent_entries)),
            "observed_field_size": int(len(entries)),
            "unresolved_entry_count_total": int(len(unresolved_entries)),
            "worlds_source": worlds_source,
        }
    )

    meta = ContestReplayMeta(
        game_date=meta.game_date,
        contest_id=meta.contest_id,
        contest_name=meta.contest_name,
        draft_group_id=resolved_draft_group_id,
        entry_fee=meta.entry_fee,
        field_size=meta.field_size,
        results_path=meta.results_path,
        source=meta.source,
        source_mode=meta.source_mode,
        extra=dict(meta.extra),
    )
    return PreparedReplayContext(
        meta=meta,
        entries=entries,
        resolved_entries=resolved_entries,
        user_entries=user_entries_resolved,
        user_lineups=user_lineups,
        user_weights=user_weights,
        opponent_field_library=opponent_library,
        resolution_stats=resolution_stats,
    )


def write_resolved_entries_parquet(
    prepared: PreparedReplayContext,
    *,
    data_root: Optional[Path] = None,
    path: Optional[Path] = None,
) -> Path:
    target_path = path or normalized_entries_path(
        game_date=prepared.meta.game_date,
        contest_id=prepared.meta.contest_id,
        data_root=data_root,
    )
    rows = []
    for entry in prepared.resolved_entries:
        rows.append(
            {
                "game_date": prepared.meta.game_date,
                "contest_id": prepared.meta.contest_id,
                "contest_name": prepared.meta.contest_name,
                "draft_group_id": prepared.meta.draft_group_id,
                "entry_id": entry.entry_id,
                "entry_name": entry.entry_name,
                "rank": entry.rank,
                "points": entry.points,
                "prize": entry.prize,
                "raw_lineup": entry.raw_lineup,
                "lineup_key": entry.lineup_key,
                "lineup_names_json": json.dumps(entry.lineup_names),
                "player_ids_json": json.dumps(entry.player_ids),
                "unresolved_names_json": json.dumps(entry.unresolved_names),
                "is_user_entry": any(user.entry_id == entry.entry_id for user in prepared.user_entries),
            }
        )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target_path, index=False)
    return target_path


def save_actual_field_library(
    prepared: PreparedReplayContext,
    *,
    data_root: Optional[Path] = None,
    path: Optional[Path] = None,
) -> Path:
    target_path = path or field_library_output_path(
        game_date=prepared.meta.game_date,
        contest_id=prepared.meta.contest_id,
        data_root=data_root,
    )
    save_field_library(prepared.opponent_field_library, target_path)
    return target_path


def run_post_contest_replay(
    *,
    contest_id: str,
    game_date: Optional[str] = None,
    user_pattern: str,
    draft_group_id: Optional[int] = None,
    run_id: Optional[str] = None,
    entry_fee: Optional[float] = None,
    archetype: str = "medium",
    worlds_source: str = "gtv2",
    ownership_mode: str = "field_only",
    data_root: Optional[Path] = None,
) -> ContestReplayRun:
    prepared = prepare_post_contest_replay(
        contest_id=contest_id,
        game_date=game_date,
        user_pattern=user_pattern,
        draft_group_id=draft_group_id,
        data_root=data_root,
        run_id=run_id,
        worlds_source=worlds_source,
    )
    resolved_entry_fee = float(entry_fee if entry_fee is not None else prepared.meta.entry_fee)
    if resolved_entry_fee <= 0:
        raise ValueError(
            "Replay requires a positive entry_fee. Pass --entry-fee when contest inventory metadata is missing."
        )
    simulation = run_contest_simulation(
        user_lineups=prepared.user_lineups,
        user_weights=prepared.user_weights,
        game_date=prepared.meta.game_date,
        draft_group_id=prepared.meta.draft_group_id,
        run_id=run_id,
        archetype=archetype,
        entry_fee=resolved_entry_fee,
        field_lineups=prepared.opponent_field_library.lineups,
        field_weights=prepared.opponent_field_library.weights,
        field_size_override=prepared.meta.field_size,
        data_root=data_root,
        ownership_mode=ownership_mode,
        worlds_source=worlds_source,
    )
    run_meta = {
        "mode": "exact_replay",
        "contest_id": prepared.meta.contest_id,
        "game_date": prepared.meta.game_date,
        "draft_group_id": prepared.meta.draft_group_id,
        "user_pattern": user_pattern,
        "run_id": run_id,
        "entry_fee": resolved_entry_fee,
        "worlds_source": worlds_source,
        "ownership_mode": ownership_mode,
    }
    return ContestReplayRun(prepared=prepared, simulation=simulation, run_meta=run_meta)
