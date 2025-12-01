"""Helpers and schema notes for gold DraftKings salaries.

Gold layout (per site/date/draft group):
    <root>/gold/dk_salaries/site=<site>/game_date=<YYYY-MM-DD>/draft_group_id=<id>/salaries.parquet

Schema (v1):
    - site: str ("dk")
    - game_date: str (YYYY-MM-DD)
    - draft_group_id: int64
    - dk_player_id: int64
    - display_name: str
    - positions: list[str]
    - salary: int64
    - team_abbrev: str
    - status: str | None
    - is_swappable: bool
    - is_disabled: bool
    - raw_competition_ids: list[int]
    - raw_data: dict (representative raw draftables row)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def dk_salaries_gold_path(root: Path, site: str, game_date: str, draft_group_id: int | str) -> Path:
    return (
        root
        / "gold"
        / "dk_salaries"
        / f"site={site}"
        / f"game_date={game_date}"
        / f"draft_group_id={draft_group_id}"
        / "salaries.parquet"
    )


def normalize_positions(values: Iterable[str] | str) -> List[str]:
    """
    Given one or more position strings like "PF/C" or "PG", split on
    '/' or ',' and return a sorted, unique list of position codes.
    """

    items: list[str]
    if isinstance(values, str):
        items = [values]
    else:
        items = list(values)

    tokens: set[str] = set()
    for raw in items:
        if raw is None:
            continue
        for piece in str(raw).replace(",", "/").split("/"):
            cleaned = piece.strip()
            if cleaned:
                tokens.add(cleaned)
    return sorted(tokens)
