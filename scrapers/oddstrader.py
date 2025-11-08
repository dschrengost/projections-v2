from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import quote, urlparse
import uuid


ODDSTRADER_BASE_URL = "https://www.oddstrader.com"
NBA_SCORES_PATH = "/nba/"

FANDUEL_PROVIDER_ID = 98
CAESARS_PROVIDER_ID = 101
BOOK_PRIORITY = (FANDUEL_PROVIDER_ID, CAESARS_PROVIDER_ID)
BOOK_LABELS = {
    FANDUEL_PROVIDER_ID: "FanDuel",
    CAESARS_PROVIDER_ID: "Caesars",
}

MARKET_TYPES = {
    401: "spread",
    402: "total",
    83: "moneyline",
}

TEAM_MARKET_TYPE_IDS = {401, 83}
TOTAL_PARTICIPANT_LABELS = {15143: "over", 15144: "under"}

CURRENT_LINES_QUERY = """
query CurrentLines($eventIds:[Int!]!, $providerIds:[Int!]!, $marketTypeIds:[Int!]!) {
  currentLines(paid:$providerIds, eid:$eventIds, mtid:$marketTypeIds) {
    eid
    mtid
    partid
    paid
    ap
    adj
    tim
    mtgrp
    boid
  }
}
""".strip()


@dataclass(frozen=True)
class MarketLine:
    """Normalized view of a sportsbook offer."""

    market: str
    selection: str
    price: int
    point: float | None
    book: str
    updated_at: datetime


@dataclass(frozen=True)
class EventOdds:
    """Scraped lines for a single NBA event."""

    event_id: int
    scheduled: datetime
    home_team: str
    away_team: str
    markets: Dict[str, Dict[str, MarketLine]] = field(default_factory=dict)


@dataclass(frozen=True)
class ParticipantInfo:
    """Metadata describing a team in the Oddstrader payload."""

    partid: int
    name: str
    abbreviation: str
    is_home: bool


@dataclass(frozen=True)
class EventMetadata:
    """Minimal event info needed to map line payloads to teams."""

    eid: int
    scheduled: datetime
    home: ParticipantInfo
    away: ParticipantInfo


class OddstraderScraper:
    """Scrape FanDuel (with Caesars fallback) odds from Oddstrader."""

    def __init__(self, *, timeout: float = 10.0, user_agent: str | None = None) -> None:
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )

    def fetch_daily_odds(self, target_date: date | None = None) -> List[EventOdds]:
        """Return normalized odds for a single date."""

        day = target_date or datetime.now(timezone.utc).date()
        return self._scrape_single_date(day)

    def fetch_range(self, start_date: date, end_date: date) -> List[EventOdds]:
        """Scrape an inclusive range of dates."""

        if end_date < start_date:
            raise ValueError("end_date must be greater than or equal to start_date")

        results: List[EventOdds] = []
        cursor = start_date
        while cursor <= end_date:
            results.extend(self._scrape_single_date(cursor))
            cursor += timedelta(days=1)
        return results

    def _scrape_single_date(self, target_date: date) -> List[EventOdds]:
        scoreboard_html = self._http_get(self._scoreboard_url(target_date))
        initial_state = self._parse_window_json(scoreboard_html, "__INITIAL_STATE__")
        config_state = self._parse_window_json(scoreboard_html, "__config")

        events = self._extract_events(initial_state)
        if not events:
            return []

        event_ids = [meta.eid for meta in events]
        api_headers, endpoint = self._build_api_headers(config_state)
        lines = self._fetch_current_lines(event_ids, api_headers, endpoint)
        return self._merge_lines_with_events(events, lines)

    def _scoreboard_url(self, day: date) -> str:
        return (
            f"{ODDSTRADER_BASE_URL}{NBA_SCORES_PATH}"
            f"?date={day.strftime('%Y%m%d')}&eid&g=game&m=merged"
        )

    def _http_get(self, url: str) -> str:
        headers = {"User-Agent": self.user_agent}
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read().decode("utf-8")
        except (HTTPError, URLError) as exc:  # pragma: no cover - network errors
            raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc

    def _http_json(self, url: str, headers: Dict[str, str]) -> Dict[str, Any]:
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:  # pragma: no cover - network errors
            raise RuntimeError(f"Failed request {url}: {exc}") from exc

    def _parse_window_json(self, html: str, var_name: str) -> Dict[str, Any]:
        marker = f"window.{var_name}"
        start = html.find(marker)
        if start == -1:
            raise ValueError(f"{var_name} not found in HTML payload")

        idx = start + len(marker)
        while idx < len(html) and html[idx].isspace():
            idx += 1
        if idx >= len(html) or html[idx] != "=":
            # attempt to find the next '=' even if other characters intervene
            eq_pos = html.find("=", idx)
            if eq_pos == -1:
                raise ValueError(f"{var_name} assignment not found")
            idx = eq_pos

        start = idx + 1
        while start < len(html) and html[start].isspace():
            start += 1
        opening = html[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escaped = False

        for idx in range(start, len(html)):
            char = html[idx]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = in_string
            elif char == '"':
                in_string = not in_string

            if in_string:
                continue

            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    snippet = html[start : idx + 1]
                    return json.loads(snippet)
        raise ValueError(f"Could not parse {var_name}")

    def _extract_events(self, state: Dict[str, Any]) -> List[EventMetadata]:
        events_blob = (
            state.get("events", {})
            .get("events", {})
        )
        events: List[EventMetadata] = []

        for event in events_blob.values():
            participants = [
                self._build_participant(part)
                for part in event.get("participants", {}).values()
            ]
            home = next((p for p in participants if p.is_home), None)
            away = next((p for p in participants if not p.is_home), None)
            if not home or not away:
                continue

            scheduled = datetime.fromtimestamp(
                event["dt"] / 1000, tz=timezone.utc
            )
            events.append(
                EventMetadata(
                    eid=event["eid"],
                    scheduled=scheduled,
                    home=home,
                    away=away,
                )
            )
        return events

    def _build_participant(self, payload: Dict[str, Any]) -> ParticipantInfo:
        source = payload.get("source", {})
        name = source.get("nn") or source.get("nam") or "Unknown"
        return ParticipantInfo(
            partid=payload["partid"],
            name=name,
            abbreviation=source.get("abbr") or "",
            is_home=bool(payload.get("ih")),
        )

    def _build_api_headers(self, config_state: Dict[str, Any]) -> Tuple[Dict[str, str], str]:
        apigateway = config_state.get("apigateway", {})
        headers = {
            "User-Agent": self.user_agent,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "clientuniqueid": str(uuid.uuid4()),
            "site-domain": "ot-com",
            "Origin": ODDSTRADER_BASE_URL,
            "Referer": f"{ODDSTRADER_BASE_URL}/",
        }
        headers.update(apigateway.get("headers", {}))

        endpoints = config_state.get("endpoints", {})
        endpoint_hint = endpoints.get("oddsv2Service")

        parsed = urlparse(endpoint_hint) if endpoint_hint else None
        proto = apigateway.get("proto") or (parsed.scheme if parsed and parsed.scheme else "https")
        domain = apigateway.get("domain") or (parsed.hostname if parsed else None)
        port = apigateway.get("port") or (parsed.port if parsed else None)
        if not domain:
            raise ValueError("Oddstrader API endpoint not present in config")

        if not port or (proto == "https" and port == 443) or (proto == "http" and port == 80):
            port_part = ""
        else:
            port_part = f":{port}"

        base_path = parsed.path if parsed and parsed.path else "/odds-v2"
        api_url = f"{proto}://{domain}{port_part}{base_path.rstrip('/')}/odds-v2-service"
        return headers, api_url

    def _fetch_current_lines(
        self,
        event_ids: List[int],
        headers: Dict[str, str],
        endpoint: str,
    ) -> List[Dict[str, Any]]:
        provider_ids = ",".join(str(pid) for pid in BOOK_PRIORITY)
        event_list = ",".join(str(eid) for eid in event_ids)
        market_ids = ",".join(str(mid) for mid in MARKET_TYPES.keys())
        query = (
            "{A_CL:currentLines(paid:["
            f"{provider_ids}"
            "],eid:["
            f"{event_list}"
            "],mtid:["
            f"{market_ids}"
            "])}"
        )
        url = f"{endpoint}?query={quote(query, safe=':,[]{}()')}"
        response = self._http_json(url, headers)
        data = response.get("data", {})
        return data.get("A_CL", [])

    def _merge_lines_with_events(
        self, events: Iterable[EventMetadata], line_rows: List[Dict[str, Any]]
    ) -> List[EventOdds]:
        grouped: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}

        for row in line_rows:
            key = (row["eid"], row["mtid"], row["partid"])
            grouped.setdefault(key, []).append(row)

        results: List[EventOdds] = []
        for event in events:
            markets: Dict[str, Dict[str, MarketLine]] = {}

            for (eid, mtid, partid), candidates in grouped.items():
                if eid != event.eid:
                    continue
                selected = self._pick_preferred_line(candidates)
                if not selected:
                    continue
                market_key = MARKET_TYPES.get(mtid)
                if not market_key:
                    continue
                selection_label = self._derive_selection(event, mtid, partid)
                if not selection_label:
                    continue
                markets.setdefault(market_key, {})
                markets[market_key][selection_label] = self._build_market_line(
                    market_key, selection_label, selected
                )

            results.append(
                EventOdds(
                    event_id=event.eid,
                    scheduled=event.scheduled,
                    home_team=event.home.name,
                    away_team=event.away.name,
                    markets=markets,
                )
            )
        return results

    def _pick_preferred_line(
        self, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any] | None:
        for provider in BOOK_PRIORITY:
            for row in candidates:
                if row.get("paid") == provider:
                    return row
        return None

    def _derive_selection(
        self, event: EventMetadata, mtid: int, partid: int
    ) -> str | None:
        if mtid in TEAM_MARKET_TYPE_IDS:
            if partid == event.home.partid:
                return "home"
            if partid == event.away.partid:
                return "away"
            return None
        if mtid == 402:
            return TOTAL_PARTICIPANT_LABELS.get(partid)
        return str(partid)

    def _build_market_line(
        self, market: str, selection: str, line: Dict[str, Any]
    ) -> MarketLine:
        updated_at = datetime.fromtimestamp(
            float(line.get("tim", "0")) / 1000, tz=timezone.utc
        )
        price = int(line.get("ap"))
        point_value = float(line["adj"]) if line.get("adj") is not None else None
        provider = line.get("paid")
        return MarketLine(
            market=market,
            selection=selection,
            price=price,
            point=point_value,
            book=BOOK_LABELS.get(provider, str(provider)),
            updated_at=updated_at,
        )
