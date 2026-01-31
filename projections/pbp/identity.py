from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from unidecode import unidecode

_PUNCT_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    """Normalize a player name into a stable matching key.

    Rules (minimal + deterministic):
    - Unicode normalize + accent fold
    - Lowercase
    - Strip punctuation to spaces
    - Collapse whitespace
    - Remove common suffix tokens (jr/sr/ii/iii/iv/v) when last token
    """
    if name is None:
        return ""
    s = str(name).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = unidecode(s)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return ""
    tokens = s.split(" ")
    if tokens and tokens[-1] in _SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens)


def _players_dim_columns() -> list[str]:
    return [
        "player_id",
        "canonical_name",
        "normalized_name",
        "vendor_name",
        "season_first_seen",
        "season_last_seen",
    ]


def load_players_dim(players_dim_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(players_dim_path)
    missing = [c for c in _players_dim_columns() if c not in df.columns]
    if missing:
        raise ValueError(
            f"players_dim missing required columns: {missing}. "
            f"Got columns={list(df.columns)} from {players_dim_path}"
        )
    return df[_players_dim_columns()].copy()


def validate_no_normalized_collisions(players_dim: pd.DataFrame) -> None:
    """Fail fast if normalized_name maps to multiple player_id values."""
    dupes = players_dim.groupby("normalized_name", dropna=False)["player_id"].nunique()
    bad = dupes[dupes > 1]
    if bad.empty:
        return
    examples = (
        players_dim[players_dim["normalized_name"].isin(bad.index)]
        .sort_values(["normalized_name", "player_id"])
        .to_dict(orient="records")
    )
    msg = (
        "Normalized-name collision in players_dim: "
        f"{len(bad)} normalized_name values map to multiple player_id values. "
        "This is a hard error (no silent collisions).\n"
        f"Examples: {json.dumps(examples[:20], ensure_ascii=False)}"
    )
    raise ValueError(msg)


@dataclass
class IdentityResolutionResult:
    name_to_player_id: dict[str, int]
    players_dim: pd.DataFrame
    unmapped_players: pd.DataFrame


def resolve_player_ids(
    vendor_names: Iterable[str],
    *,
    season_id: str,
    prev_players_dim: Optional[pd.DataFrame],
) -> IdentityResolutionResult:
    """Resolve vendor-provided names into stable integer player IDs.

    Mapping behavior:
    - If prev_players_dim provided: map by normalized_name to existing player_id.
    - If unmapped: create new player_id sequentially.
    - If prev_players_dim has normalized collisions: fail (no silent collisions).
    """
    prev = None
    if prev_players_dim is not None and not prev_players_dim.empty:
        prev = prev_players_dim.copy()
        validate_no_normalized_collisions(prev)
        prev = prev[_players_dim_columns()].copy()
        prev["normalized_name"] = prev["normalized_name"].fillna("")

    normalized = []
    raw = []
    for n in vendor_names:
        if n is None:
            continue
        s = str(n).strip()
        if not s or s.lower() in {"nan", "none"}:
            continue
        raw.append(s)
        normalized.append(normalize_name(s))

    if not raw:
        base = prev if prev is not None else pd.DataFrame(columns=_players_dim_columns())
        return IdentityResolutionResult({}, base, pd.DataFrame(columns=["vendor_name", "normalized_name"]))

    unique_pairs = (
        pd.DataFrame({"vendor_name": raw, "normalized_name": normalized})
        .drop_duplicates()
        .query("normalized_name != ''")
        .copy()
    )

    existing_map: dict[str, int] = {}
    players_dim = None
    if prev is not None:
        existing_map = dict(zip(prev["normalized_name"], prev["player_id"]))
        players_dim = prev.copy()
        next_id = int(players_dim["player_id"].max()) + 1 if len(players_dim) else 1
    else:
        players_dim = pd.DataFrame(columns=_players_dim_columns())
        next_id = 1

    name_to_player_id: dict[str, int] = {}
    newly_created_rows: list[dict] = []

    for _, row in unique_pairs.iterrows():
        vendor_name = row["vendor_name"]
        norm = row["normalized_name"]
        if norm in existing_map:
            pid = int(existing_map[norm])
            name_to_player_id[vendor_name] = pid
            continue
        pid = next_id
        next_id += 1
        name_to_player_id[vendor_name] = pid
        newly_created_rows.append(
            {
                "player_id": pid,
                "canonical_name": vendor_name,
                "normalized_name": norm,
                "vendor_name": vendor_name,
                "season_first_seen": season_id,
                "season_last_seen": season_id,
            }
        )
        existing_map[norm] = pid

    # Update last seen for all normalized names we encountered (existing + new).
    if len(players_dim) == 0 and newly_created_rows:
        players_dim = pd.DataFrame(newly_created_rows, columns=_players_dim_columns())
    elif newly_created_rows:
        players_dim = pd.concat(
            [players_dim, pd.DataFrame(newly_created_rows, columns=_players_dim_columns())],
            ignore_index=True,
        )

    encountered_norms = set(unique_pairs["normalized_name"].tolist())
    if len(players_dim):
        mask = players_dim["normalized_name"].isin(encountered_norms)
        players_dim.loc[mask, "season_last_seen"] = season_id

    unmapped = pd.DataFrame(newly_created_rows)[
        ["player_id", "vendor_name", "normalized_name"]
    ] if newly_created_rows else pd.DataFrame(columns=["player_id", "vendor_name", "normalized_name"])

    return IdentityResolutionResult(
        name_to_player_id=name_to_player_id,
        players_dim=players_dim[_players_dim_columns()].copy(),
        unmapped_players=unmapped,
    )

