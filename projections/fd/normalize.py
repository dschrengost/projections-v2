"""Normalization helpers for FanDuel fixture-list players payloads."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from projections.dk.salaries_schema import normalize_positions


def _coerce_numeric_salary(value: object) -> int | None:
    salary = pd.to_numeric(value, errors="coerce")
    if pd.isna(salary):
        return None
    try:
        return int(salary)
    except Exception:
        return None


def _parse_start(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = pd.to_datetime(text, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()
    except Exception:
        return None


def _extract_status(player_row: dict[str, Any]) -> str | None:
    direct_candidates = [
        player_row.get("injury_indicator"),
        player_row.get("injury_status"),
        player_row.get("status"),
    ]
    for candidate in direct_candidates:
        if candidate is None:
            continue
        text = str(candidate).strip().upper()
        if text and text not in {"NONE", "NAN", "NULL"}:
            return text

    indicators = player_row.get("injury_indicators")
    if isinstance(indicators, list):
        for indicator in indicators:
            if not isinstance(indicator, dict):
                continue
            for key in ("status", "code", "label", "name"):
                raw = indicator.get(key)
                if raw is None:
                    continue
                text = str(raw).strip().upper()
                if text and text not in {"NONE", "NAN", "NULL"}:
                    return text
    return None


def _team_code_from_ref(team_ref: object, team_by_id: dict[str, str]) -> str | None:
    def _extract_ref_id(raw: object) -> str | None:
        if raw is None:
            return None
        if isinstance(raw, dict):
            if "id" in raw and raw.get("id") is not None:
                text = str(raw.get("id")).strip()
                if text:
                    return text
            members = raw.get("_members")
            if isinstance(members, list) and members:
                text = str(members[0]).strip()
                if text:
                    return text
            return None
        text = str(raw).strip()
        return text or None

    if isinstance(team_ref, dict):
        if isinstance(team_ref.get("team"), dict):
            team_ref = team_ref.get("team")

        for key in ("code", "abbreviation", "abbr", "short_name", "name"):
            raw = team_ref.get(key)
            if raw:
                return str(raw).strip().upper()
        team_ref = _extract_ref_id(team_ref)

    ref_id = _extract_ref_id(team_ref)
    if ref_id is None:
        return None

    key = str(ref_id).strip()
    if not key:
        return None

    mapped = team_by_id.get(key)
    if mapped:
        return mapped

    return key.upper()


def _fixture_id_from_player(player_row: dict[str, Any]) -> str | None:
    for key in ("fixture", "fixture_id", "fixtureId"):
        raw = player_row.get(key)
        if raw is None:
            continue
        if isinstance(raw, dict):
            if raw.get("id") is not None:
                raw = raw.get("id")
            else:
                members = raw.get("_members")
                if isinstance(members, list) and members:
                    raw = members[0]
                else:
                    raw = None
        text = str(raw).strip()
        if text:
            return text
    return None


def _build_team_map(teams: Iterable[dict[str, Any]]) -> dict[str, str]:
    team_by_id: dict[str, str] = {}
    for team in teams:
        if not isinstance(team, dict):
            continue
        team_id = team.get("id")
        if team_id is None:
            continue

        code = None
        for key in ("code", "abbreviation", "abbr", "short_name", "name"):
            raw = team.get(key)
            if raw:
                code = str(raw).strip().upper()
                break

        if not code:
            continue
        team_by_id[str(team_id)] = code
    return team_by_id


def _build_fixture_map(
    fixtures: Iterable[dict[str, Any]],
    team_by_id: dict[str, str],
) -> dict[str, dict[str, Any]]:
    fixture_map: dict[str, dict[str, Any]] = {}
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            continue
        fixture_id = fixture.get("id")
        if fixture_id is None:
            continue

        home_ref = fixture.get("home_team")
        if home_ref is None:
            home_ref = fixture.get("homeTeam")
        if home_ref is None:
            home_ref = fixture.get("home_team_id")

        away_ref = fixture.get("away_team")
        if away_ref is None:
            away_ref = fixture.get("awayTeam")
        if away_ref is None:
            away_ref = fixture.get("away_team_id")

        home_code = _team_code_from_ref(home_ref, team_by_id)
        away_code = _team_code_from_ref(away_ref, team_by_id)
        matchup = None
        if away_code and home_code:
            matchup = f"{away_code}@{home_code}"

        comp_id: int | None = None
        try:
            comp_id = int(str(fixture_id).strip())
        except Exception:
            comp_id = None

        fixture_map[str(fixture_id)] = {
            "matchup": matchup,
            "start": _parse_start(fixture.get("start_date") or fixture.get("startTime")),
            "competition_id": comp_id,
        }
    return fixture_map


def _first_non_null_text(values: Iterable[object]) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.upper() not in {"NONE", "NAN", "NULL"}:
            return text
    return None


def players_json_to_df(
    payload: dict[str, Any],
    *,
    fixture_list_id: int | str | None = None,
    fixture_detail: dict[str, Any] | None = None,
    contests_payload: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Flatten FanDuel players payload to a tabular DataFrame."""
    _ = contests_payload  # Reserved for future contest-level enrichments.

    players = payload.get("players") if isinstance(payload, dict) else None
    if not isinstance(players, list):
        raise RuntimeError("FanDuel players payload missing 'players' list")

    teams = payload.get("teams") if isinstance(payload, dict) else None
    teams_list = [t for t in (teams or []) if isinstance(t, dict)]

    fixtures = payload.get("fixtures") if isinstance(payload, dict) else None
    fixtures_list = [f for f in (fixtures or []) if isinstance(f, dict)]
    if not fixtures_list and isinstance(fixture_detail, dict):
        detail_fixtures = fixture_detail.get("fixtures")
        if isinstance(detail_fixtures, list):
            fixtures_list = [f for f in detail_fixtures if isinstance(f, dict)]

    team_by_id = _build_team_map(teams_list)
    fixture_map = _build_fixture_map(fixtures_list, team_by_id)

    rows: list[dict[str, Any]] = []
    for player in players:
        if not isinstance(player, dict):
            continue

        raw_player_id = player.get("id")
        if raw_player_id is None:
            continue
        fd_player_id = str(raw_player_id).strip()
        if not fd_player_id:
            continue

        salary = _coerce_numeric_salary(player.get("salary"))
        if salary is None:
            continue

        display_name = _first_non_null_text(
            [
                player.get("nickname"),
                player.get("display_name"),
                player.get("name"),
                " ".join(
                    [
                        str(player.get("first_name") or "").strip(),
                        str(player.get("last_name") or "").strip(),
                    ]
                ).strip(),
            ]
        )
        if not display_name:
            continue

        position_raw = player.get("position") or player.get("positions")
        positions = normalize_positions(position_raw or "")
        if not positions:
            continue

        team_code = _team_code_from_ref(player.get("team") or player.get("team_id"), team_by_id)
        if not team_code:
            continue

        fixture_id = _fixture_id_from_player(player)
        fixture_info = fixture_map.get(fixture_id or "")

        raw_competition_ids: list[int] = []
        if fixture_info and fixture_info.get("competition_id") is not None:
            raw_competition_ids = [int(fixture_info["competition_id"])]

        rows.append(
            {
                "draft_group_id": fixture_list_id,
                "fd_player_id": fd_player_id,
                "site_player_id": fd_player_id,
                "display_name": display_name,
                "first_name": _first_non_null_text([player.get("first_name")]),
                "last_name": _first_non_null_text([player.get("last_name")]),
                "position": "/".join(positions),
                "positions": positions,
                "salary": salary,
                "team_abbrev": team_code,
                "status": _extract_status(player),
                "is_swappable": True,
                "is_disabled": False,
                "raw_competition_ids": raw_competition_ids,
                "game_matchup": fixture_info.get("matchup") if fixture_info else None,
                "game_start_utc": fixture_info.get("start") if fixture_info else None,
                "raw": player,
            }
        )

    return pd.DataFrame(rows)


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"FanDuel players DataFrame missing required columns: {', '.join(missing)}")


def normalize_fd_players_to_salaries(
    root: Path,
    site: str,
    game_date: str,
    draft_group_id: int | str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert flattened FD players DataFrame to gold salaries schema."""
    _ = root  # Normalizer signature matches DK normalizer.

    _validate_columns(df, ["fd_player_id", "display_name", "position", "salary", "team_abbrev"])

    work = df.copy()
    work["fd_player_id"] = work["fd_player_id"].astype(str).str.strip()
    work = work[work["fd_player_id"] != ""]
    work["salary"] = pd.to_numeric(work["salary"], errors="coerce")
    work = work[work["salary"].notna()]

    if work.empty:
        raise RuntimeError("FanDuel players DataFrame is empty after filtering invalid rows")

    grouped_rows: list[dict[str, Any]] = []
    for fd_player_id, group in work.groupby("fd_player_id"):
        positions = normalize_positions(group["position"].dropna().tolist())
        salary = int(group["salary"].max())

        team_abbrev = _first_non_null_text(group["team_abbrev"].tolist())
        if not team_abbrev:
            continue

        status = _first_non_null_text(group.get("status", pd.Series(dtype="object")).tolist())

        comp_ids: list[int] = []
        if "raw_competition_ids" in group.columns:
            for raw_val in group["raw_competition_ids"].tolist():
                values = raw_val if isinstance(raw_val, list) else [raw_val]
                for value in values:
                    try:
                        comp_ids.append(int(value))
                    except Exception:
                        continue
        comp_ids = sorted(set(comp_ids))

        game_matchup = _first_non_null_text(group.get("game_matchup", pd.Series(dtype="object")).tolist())
        game_start = None
        if "game_start_utc" in group.columns:
            starts = [s for s in group["game_start_utc"].tolist() if s is not None and str(s) != "NaT"]
            if starts:
                game_start = starts[0]

        raw_data_dict = group.iloc[0].get("raw")
        try:
            raw_data = json.dumps(raw_data_dict, default=str)
        except Exception:
            raw_data = str(raw_data_dict)

        grouped_rows.append(
            {
                "site": site,
                "game_date": game_date,
                "draft_group_id": draft_group_id,
                "site_player_id": str(fd_player_id),
                "fd_player_id": str(fd_player_id),
                "display_name": group["display_name"].iloc[0],
                "positions": positions,
                "salary": salary,
                "team_abbrev": team_abbrev,
                "status": status,
                "is_swappable": bool(group.get("is_swappable", pd.Series([True])).astype(bool).any()),
                "is_disabled": bool(group.get("is_disabled", pd.Series([False])).astype(bool).any()),
                "raw_competition_ids": comp_ids,
                "game_matchup": game_matchup,
                "game_start_utc": game_start,
                "raw_data": raw_data,
            }
        )

    return pd.DataFrame(grouped_rows)
