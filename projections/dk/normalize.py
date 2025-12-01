from __future__ import annotations

"""Normalization helpers for converting DK draftables into gold salaries."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .salaries_schema import dk_salaries_gold_path, normalize_positions


def _flatten_draftable(row: Dict[str, Any], draft_group_id: int | str | None = None) -> Dict[str, Any]:
    dg_val = draft_group_id
    if dg_val is None:
        for key in ("draftGroupId", "draft_group_id", "draftGroupID", "dg"):
            if key in row:
                dg_val = row.get(key)
                break

    competition = row.get("competition") or {}
    competitions = row.get("competitions") or []
    if not competition and competitions:
        competition = competitions[0] or {}

    comp_id = competition.get("competitionId") if isinstance(competition, dict) else None
    comp_name = competition.get("name") if isinstance(competition, dict) else None
    comp_start = competition.get("startTime") if isinstance(competition, dict) else None

    return {
        "draft_group_id": dg_val,
        "draftable_id": row.get("draftableId") or row.get("id"),
        "dk_player_id": row.get("playerId") or row.get("playerDkId"),
        "display_name": row.get("displayName"),
        "position": row.get("position"),
        "salary": row.get("salary"),
        "team_abbrev": row.get("teamAbbreviation") or row.get("teamAbbrev"),
        "status": row.get("status"),
        "is_swappable": row.get("isSwappable"),
        "is_disabled": row.get("isDisabled"),
        "competition_id": comp_id,
        "competition_name": comp_name,
        "competition_start_time": comp_start,
        "roster_slot_id": row.get("rosterSlotId"),
        "raw": row,
    }


def draftables_json_to_df(raw: Dict[str, Any], draft_group_id: int | str | None = None) -> pd.DataFrame:
    """
    Flatten a draftables payload into a DataFrame with the columns consumed by
    normalize_draftables_to_salaries.
    """

    draftables = raw.get("draftables") if isinstance(raw, dict) else None
    if not isinstance(draftables, list):
        raise RuntimeError("DraftKings draftables payload missing 'draftables' list")
    rows: list[dict[str, Any]] = []
    for item in draftables:
        rows.append(_flatten_draftable(item, draft_group_id))
    df = pd.DataFrame(rows)
    if draft_group_id is not None:
        df["draft_group_id"] = int(draft_group_id)
    return df


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Draftables DataFrame missing required columns: {', '.join(missing)}")


def normalize_draftables_to_salaries(
    root: Path,
    site: str,
    game_date: str,
    draft_group_id: int | str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert a raw draftables DataFrame (one row per roster slot) into
    a dk_salaries gold DataFrame (one row per dk_player_id).

    Required columns in df: dk_player_id, display_name, position, salary, team_abbrev.
    """

    _validate_columns(df, ["dk_player_id", "display_name", "position", "salary", "team_abbrev"])

    df = df.copy()
    df["dk_player_id"] = pd.to_numeric(df["dk_player_id"], errors="coerce")
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    if "competition_id" in df.columns:
        df["competition_id"] = pd.to_numeric(df["competition_id"], errors="coerce")

    df = df[df["dk_player_id"].notna()]
    if df.empty:
        raise RuntimeError("Draftables DataFrame is empty after filtering invalid dk_player_id values")

    grouped_rows: List[Dict[str, Any]] = []
    for dk_player_id, group in df.groupby("dk_player_id"):
        positions = normalize_positions(group["position"].dropna().tolist())

        salary_values = group["salary"].dropna().unique()
        salary = None
        if len(salary_values) > 0:
            salary = int(max(salary_values))
            if len(salary_values) > 1:
                print(
                    f"[dk-normalize] warning: multiple salaries for dk_player_id={dk_player_id}: {salary_values}; using {salary}"
                )

        team_values = group["team_abbrev"].dropna().unique()
        team_abbrev = None
        if len(team_values) > 0:
            team_abbrev = team_values[0]
            if len(team_values) > 1:
                print(
                    f"[dk-normalize] warning: multiple teams for dk_player_id={dk_player_id}: {team_values}; using {team_abbrev}"
                )

        status = None
        if "status" in group.columns:
            non_null = group["status"].dropna()
            if not non_null.empty:
                status = non_null.iloc[0]

        is_swappable = bool(group.get("is_swappable", False).astype(bool).any())
        is_disabled = bool(group.get("is_disabled", False).astype(bool).any())

        comp_ids: list[int] = []
        if "competition_id" in group.columns:
            parsed_ids = []
            for raw_val in group["competition_id"].dropna().unique():
                try:
                    parsed_ids.append(int(raw_val))
                except (TypeError, ValueError):
                    continue
            comp_ids = sorted(set(parsed_ids))

        raw_data_dict = group.iloc[0].to_dict()
        try:
            raw_data = json.dumps(raw_data_dict)
        except Exception:
            raw_data = str(raw_data_dict)

        grouped_rows.append(
            {
                "site": site,
                "game_date": game_date,
                "draft_group_id": int(draft_group_id),
                "dk_player_id": int(dk_player_id),
                "display_name": group["display_name"].iloc[0],
                "positions": positions,
                "salary": salary,
                "team_abbrev": team_abbrev,
                "status": status,
                "is_swappable": is_swappable,
                "is_disabled": is_disabled,
                "raw_competition_ids": comp_ids,
                "raw_data": raw_data,
            }
        )

    result = pd.DataFrame(grouped_rows)
    return result


def write_salaries_gold(
    root: Path,
    site: str,
    game_date: str,
    draft_group_id: int | str,
    salaries_df: pd.DataFrame,
) -> Path:
    path = dk_salaries_gold_path(root, site, game_date, draft_group_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    salaries_df.to_parquet(path, index=False)
    return path
