"""FanDuel slate discovery and filtering."""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Literal

import pandas as pd
from zoneinfo import ZoneInfo

from .api import fetch_fixture_lists

SlateType = Literal["main", "night", "turbo", "early", "showdown", "all"]

_EASTERN = ZoneInfo("America/New_York")


def _parse_start_datetime(value: object) -> dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        cleaned = cleaned.replace("Z", "+00:00")
        try:
            parsed = dt.datetime.fromisoformat(cleaned)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed
    if isinstance(value, (int, float)):
        millis = float(value)
        if pd.isna(millis):
            return None
        if millis > 1e12:
            millis /= 1000
        return dt.datetime.fromtimestamp(millis, tz=dt.timezone.utc)
    return None


def _infer_slate_type(label: str, *, n_games: int | None = None) -> str:
    label_lower = str(label or "").lower()
    if re.search(r"\bturbo\b", label_lower):
        return "turbo"
    if re.search(r"\b(late|night)\b", label_lower):
        return "night"
    if re.search(r"\bearly\b", label_lower):
        return "early"
    if "showdown" in label_lower or "single game" in label_lower:
        return "showdown"
    if n_games == 1:
        return "showdown"
    return "main"


def _coerce_draft_group_id(raw: object) -> int | str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _extract_fixture_lists(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    fixture_lists = payload.get("fixture_lists")
    if isinstance(fixture_lists, list):
        return [item for item in fixture_lists if isinstance(item, dict)]
    return []


def list_fixture_lists_for_date(
    game_date: str,
    slate_type: SlateType = "all",
    *,
    fixture_lists_payload: dict[str, Any] | None = None,
    client_id: str | None = None,
) -> pd.DataFrame:
    """Return one row per FanDuel fixture list for a game date."""

    try:
        target_date = dt.date.fromisoformat(game_date)
    except ValueError as exc:
        raise ValueError(f"Invalid game_date format (expected YYYY-MM-DD): {game_date}") from exc

    payload = fixture_lists_payload if fixture_lists_payload is not None else fetch_fixture_lists(client_id=client_id)
    rows = _extract_fixture_lists(payload)

    records: list[dict[str, Any]] = []
    for row in rows:
        sport = str(row.get("sport") or "").lower()
        if sport and sport != "nba":
            continue

        draft_group_id = _coerce_draft_group_id(row.get("id"))
        if draft_group_id is None:
            continue

        start_dt = _parse_start_datetime(row.get("start_date"))
        if start_dt is None:
            continue
        start_local = start_dt.astimezone(_EASTERN)
        if start_local.date() != target_date:
            continue

        label = str(row.get("label") or row.get("name") or f"FanDuel Fixture {draft_group_id}")

        fixtures_count_raw = row.get("fixtures")
        n_games = None
        if fixtures_count_raw is not None:
            try:
                n_games = int(fixtures_count_raw)
            except Exception:
                n_games = None

        inferred = _infer_slate_type(label, n_games=n_games)
        if slate_type != "all" and inferred != slate_type:
            continue

        n_contests = 0
        contests_raw = row.get("contests")
        if contests_raw is not None:
            try:
                n_contests = int(contests_raw)
            except Exception:
                n_contests = 0

        records.append(
            {
                "game_date": target_date.isoformat(),
                "slate_type": inferred,
                "draft_group_id": draft_group_id,
                "n_contests": n_contests,
                "earliest_start": start_local,
                "latest_start": start_local,
                "example_contest_name": label,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "game_date",
                "slate_type",
                "draft_group_id",
                "n_contests",
                "earliest_start",
                "latest_start",
                "example_contest_name",
            ]
        )

    out = pd.DataFrame(records)
    out = out.sort_values(by=["earliest_start", "draft_group_id"]).reset_index(drop=True)
    return out
