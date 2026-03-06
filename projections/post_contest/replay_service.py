from __future__ import annotations

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
from projections.contest_sim.contest_sim_service import run_contest_simulation
from projections.contest_sim.field_library import FieldLibrary, save_field_library
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
    text = unidecode(str(name or "")).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, str]]:
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
        player_id = str(player.get("player_id"))
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
    for player_id, player_name in internal_to_name.items():
        normalized = _normalize_name(player_name)
        if normalized:
            candidate_ids[normalized].add(player_id)
        dk_player_id = internal_to_dk_player_id.get(player_id)
        if dk_player_id is not None:
            dk_name = dk_names_by_player.get(dk_player_id)
            normalized_dk = _normalize_name(dk_name) if dk_name else ""
            if normalized_dk:
                candidate_ids[normalized_dk].add(player_id)

    resolved: Dict[str, str] = {}
    ambiguous: Dict[str, List[str]] = {}
    for normalized_name, player_ids in candidate_ids.items():
        ordered = sorted(player_ids)
        if len(ordered) == 1:
            resolved[normalized_name] = ordered[0]
        else:
            ambiguous[normalized_name] = ordered
    return resolved, ambiguous, internal_to_name


def resolve_entries_to_internal_ids(
    entries: Sequence[ContestReplayEntry],
    *,
    game_date: str,
    draft_group_id: int,
    data_root: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> Tuple[List[ResolvedContestReplayEntry], Dict[str, Any]]:
    resolved_name_map, ambiguous_name_map, internal_to_name = _build_name_to_internal_map(
        game_date=game_date,
        draft_group_id=draft_group_id,
        data_root=data_root,
        run_id=run_id,
    )

    resolved_entries: List[ResolvedContestReplayEntry] = []
    unresolved_examples: List[Dict[str, Any]] = []
    ambiguous_examples: List[Dict[str, Any]] = []

    for entry in entries:
        player_ids: List[str] = []
        unresolved_names: List[str] = []
        for name in entry.lineup_names:
            normalized = _normalize_name(name)
            player_id = resolved_name_map.get(normalized)
            if player_id is not None:
                player_ids.append(player_id)
                continue
            unresolved_names.append(name)
            if normalized in ambiguous_name_map:
                ambiguous_examples.append(
                    {
                        "entry_id": entry.entry_id,
                        "entry_name": entry.entry_name,
                        "name": name,
                        "candidate_player_ids": ambiguous_name_map[normalized],
                    }
                )
            else:
                unresolved_examples.append(
                    {
                        "entry_id": entry.entry_id,
                        "entry_name": entry.entry_name,
                        "name": name,
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
        "unresolved_examples": unresolved_examples[:10],
        "ambiguous_examples": ambiguous_examples[:10],
        "resolved_name_count": int(len(resolved_name_map)),
        "player_pool_size": int(len(internal_to_name)),
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

    resolved_entries, resolution_stats = resolve_entries_to_internal_ids(
        entries,
        game_date=meta.game_date,
        draft_group_id=resolved_draft_group_id,
        data_root=data_root,
        run_id=run_id,
    )
    unresolved_entries = [entry for entry in resolved_entries if entry.unresolved_names]
    if strict_resolution and unresolved_entries:
        sample = unresolved_entries[0]
        raise ValueError(
            "Could not resolve all lineup names to internal player IDs. "
            f"First unresolved entry_id={sample.entry_id} names={sample.unresolved_names}"
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
