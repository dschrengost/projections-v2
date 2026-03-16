"""FanDuel API helpers for fixture lists, players, and contests."""

from __future__ import annotations

import os
import re
from typing import Any

import requests

DEFAULT_TIMEOUT = 20
API_BASE = "https://api.fanduel.com"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.fanduel.com",
    "Referer": "https://www.fanduel.com/contests",
}

_CLIENT_ID_PATTERNS = [
    re.compile(r'"clientId"\s*:\s*"([A-Za-z0-9+/=_:-]+)"'),
    re.compile(r"'clientId'\s*:\s*'([A-Za-z0-9+/=_:-]+)'"),
    re.compile(r"clientId=([A-Za-z0-9+/=_:-]+)"),
]

_CLIENT_ID_DISCOVERY_URLS = [
    "https://www.fanduel.com/contests",
    "https://account.www.fanduel.com/login?external-referrer-next=contests",
    "https://www.fanduel.com/games",
]

_cached_client_id: str | None = None


class FanDuelApiError(RuntimeError):
    """Raised when FanDuel API calls fail."""


def _normalize_client_id(value: str) -> str:
    text = str(value).strip()
    if text.lower().startswith("basic "):
        text = text[6:].strip()
    return text


def _extract_client_id_from_text(text: str) -> str | None:
    for pattern in _CLIENT_ID_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def _looks_like_perimeterx_block(text: str) -> bool:
    lowered = text.lower()
    return "px-captcha" in lowered or "perimeterx" in lowered or "press & hold" in lowered


def _response_snippet(resp: requests.Response, limit: int = 400) -> str:
    try:
        text = resp.text
    except Exception:
        return ""
    text = text.replace("\n", " ").replace("\r", " ").strip()
    return text[:limit]


def resolve_client_id(
    client_id: str | None = None,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    force_refresh: bool = False,
) -> str:
    """Resolve FanDuel clientId used for Basic auth.

    Resolution order:
    1. Explicit `client_id`
    2. `FANDUEL_CLIENT_ID` / `FD_CLIENT_ID`
    3. `FANDUEL_API_AUTH_BASIC` (accepts "Basic <id>" or raw id)
    4. Extract from FanDuel public HTML
    """

    global _cached_client_id

    if client_id:
        resolved = _normalize_client_id(client_id)
        if not resolved:
            raise FanDuelApiError("Provided FanDuel client_id is empty")
        _cached_client_id = resolved
        return resolved

    if not force_refresh and _cached_client_id:
        return _cached_client_id

    env_candidates = [
        os.environ.get("FANDUEL_CLIENT_ID"),
        os.environ.get("FD_CLIENT_ID"),
        os.environ.get("FANDUEL_API_AUTH_BASIC"),
    ]
    for candidate in env_candidates:
        if not candidate:
            continue
        resolved = _normalize_client_id(candidate)
        if resolved:
            _cached_client_id = resolved
            return resolved

    session = requests.Session()
    for url in _CLIENT_ID_DISCOVERY_URLS:
        try:
            resp = session.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
        except requests.RequestException:
            continue

        if resp.status_code >= 400 and not resp.text:
            continue

        candidate = _extract_client_id_from_text(resp.text or "")
        if candidate:
            _cached_client_id = candidate
            return candidate

    raise FanDuelApiError(
        "Unable to resolve FanDuel clientId automatically. "
        "Set FANDUEL_CLIENT_ID (or FD_CLIENT_ID) in the environment."
    )


def _get_json(
    path_or_url: str,
    *,
    client_id: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_client_id = resolve_client_id(client_id, timeout=timeout)
    headers = dict(DEFAULT_HEADERS)
    headers["Authorization"] = f"Basic {resolved_client_id}"

    url = path_or_url
    if not path_or_url.startswith("http"):
        url = f"{API_BASE.rstrip('/')}/{path_or_url.lstrip('/')}"

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise FanDuelApiError(f"FanDuel request failed for {url}: {exc}") from exc

    if resp.status_code == 403 and _looks_like_perimeterx_block(resp.text or ""):
        raise FanDuelApiError(
            "FanDuel blocked this request with PerimeterX (HTTP 403). "
            "Retry later or run from an allowlisted network/session."
        )

    if resp.status_code in {401, 403}:
        snippet = _response_snippet(resp)
        raise FanDuelApiError(
            f"FanDuel auth failed for {url} (status={resp.status_code}). "
            f"Response snippet: {snippet}"
        )

    if resp.status_code >= 400:
        snippet = _response_snippet(resp)
        raise FanDuelApiError(
            f"FanDuel API request failed for {url} (status={resp.status_code}). "
            f"Response snippet: {snippet}"
        )

    try:
        payload = resp.json()
    except ValueError as exc:
        snippet = _response_snippet(resp)
        raise FanDuelApiError(
            f"FanDuel API returned non-JSON for {url}. Response snippet: {snippet}"
        ) from exc

    if not isinstance(payload, dict):
        raise FanDuelApiError(f"FanDuel API returned unexpected payload type for {url}: {type(payload)!r}")

    return payload


def fetch_fixture_lists(*, client_id: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Fetch fixture lists payload."""
    return _get_json("/fixture-lists", client_id=client_id, timeout=timeout)


def fetch_fixture_list_detail(
    fixture_list_id: int | str,
    *,
    client_id: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Fetch fixture list detail for one slate."""
    return _get_json(f"/fixture-lists/{fixture_list_id}", client_id=client_id, timeout=timeout)


def fetch_fixture_list_players(
    fixture_list_id: int | str,
    *,
    client_id: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Fetch players payload for one fixture list."""
    return _get_json(f"/fixture-lists/{fixture_list_id}/players", client_id=client_id, timeout=timeout)


def fetch_contests(
    *,
    fixture_list_id: int | str,
    status: str = "open",
    client_id: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Fetch contests for a fixture list."""
    params = {"fixture_list": str(fixture_list_id)}
    if status:
        params["status"] = status
    return _get_json("/contests", client_id=client_id, timeout=timeout, params=params)
