from __future__ import annotations

"""Minimal DraftKings NBA HTTP wrappers."""

from typing import Any, Dict

import requests

DEFAULT_TIMEOUT = 10

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


def _get_json(url: str, *, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    resp = requests.get(url, timeout=timeout, headers=DEFAULT_HEADERS)
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Failed to parse JSON from {url}") from exc


def fetch_nba_contests() -> Dict[str, Any]:
    """
    Call the DK lobby contests endpoint for NBA and return the raw JSON.

    Under the hood this hits:
        https://www.draftkings.com/lobby/getcontests?sport=NBA

    The response contains a 'Contests' list. Each contest has a 'dg' key
    (Draft Group ID) and metadata like name, start time, etc.
    """

    url = "https://www.draftkings.com/lobby/getcontests?sport=NBA"
    return _get_json(url)


def fetch_draftables(draft_group_id: int | str) -> Dict[str, Any]:
    """
    Fetch draftables (players + salaries) for a given Draft Group ID.

    Under the hood this hits:
        https://api.draftkings.com/draftgroups/v1/draftgroups/{draft_group_id}/draftables

    The response is expected to have a 'draftables' list. We return raw JSON.
    """

    url = (
        "https://api.draftkings.com/draftgroups/v1/draftgroups/"
        f"{draft_group_id}/draftables"
    )
    return _get_json(url)
