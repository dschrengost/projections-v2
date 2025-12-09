from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
import gzip
import random
import time
from typing import Any, Deque, Dict, Iterable, List
from http.cookiejar import CookieJar
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener

BREF_BASE_URL = "https://www.basketball-reference.com"
SCOREBOARD_PATH = "/boxscores/"
_BOX_LINK_PATTERN = re.compile(r'href="(/boxscores/\d{9}[A-Z]{3}\.html)"')


@dataclass(frozen=True)
class PlayerBoxScore:
    """Box score line for a single player."""

    player_id: str | None
    player: str
    starter: bool
    reason: str | None
    basic: Dict[str, Any] = field(default_factory=dict)
    advanced: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TeamBoxScore:
    """Aggregated stats for one team within a game."""

    team_name: str
    team_abbr: str
    is_home: bool
    score: int
    record: str | None
    players: List[PlayerBoxScore] = field(default_factory=list)
    totals: Dict[str, Any] = field(default_factory=dict)
    advanced_totals: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GameBoxScore:
    """Full box score for a single NBA game."""

    game_id: str
    event_time: datetime | None
    venue: str | None
    attendance: int | None
    officials: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    away: TeamBoxScore | None = None
    home: TeamBoxScore | None = None


class BasketballReferenceScraper:
    """Scrape NBA box scores from basketball-reference.com."""

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        user_agent: str | None = None,
        delay: float = 1.5,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
        rate_limit_max_requests: int = 30,
        rate_limit_window_seconds: float = 50.0,
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
        self.delay = max(delay, 0.0)
        self.max_retries = max(1, int(max_retries))
        self.backoff_factor = backoff_factor if backoff_factor > 1 else 1.0
        self.rate_limit_max_requests = max(0, int(rate_limit_max_requests))
        self.rate_limit_window = max(rate_limit_window_seconds, 0.0)
        self._request_timestamps: Deque[float] = deque()
        self.cookie_jar = CookieJar()
        self.opener = build_opener(HTTPCookieProcessor(self.cookie_jar))
        self.base_headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }

    def fetch_daily_box_scores(self, target_date: date | None = None) -> List[GameBoxScore]:
        """Return every completed game for the provided date."""

        day = target_date or datetime.now(timezone.utc).date()
        return self._scrape_single_date(day)

    def fetch_range(self, start_date: date, end_date: date) -> List[GameBoxScore]:
        """Scrape an inclusive date range."""

        if end_date < start_date:
            raise ValueError("end_date must be greater than or equal to start_date")

        results: List[GameBoxScore] = []
        cursor = start_date
        while cursor <= end_date:
            results.extend(self._scrape_single_date(cursor))
            cursor += timedelta(days=1)
        return results

    def _scrape_single_date(self, day: date) -> List[GameBoxScore]:
        scoreboard_html = self._http_get(self._scoreboard_url(day))
        boxscore_paths = self._extract_boxscore_paths(scoreboard_html)
        games: List[GameBoxScore] = []
        for path in boxscore_paths:
            boxscore_html = self._http_get(f"{BREF_BASE_URL}{path}")
            games.append(self._parse_boxscore_page(boxscore_html, path))
        return games

    def _scoreboard_url(self, day: date) -> str:
        return f"{BREF_BASE_URL}{SCOREBOARD_PATH}?month={day.month}&day={day.day}&year={day.year}"

    def _http_get(self, url: str) -> str:
        attempt = 0
        while True:
            self._respect_rate_limit()
            request = Request(url, headers=self._build_headers(url))
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    payload = self._decode_body(response)
            except HTTPError as exc:
                self._record_request_timestamp()
                retry_after = self._retry_after_header(exc)
                if not self._should_retry(exc.code, attempt):
                    raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
                self._sleep_backoff(attempt, retry_after)
                attempt += 1
                continue
            except URLError as exc:
                self._record_request_timestamp()
                if attempt >= self.max_retries - 1:
                    raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
                self._sleep_backoff(attempt, None)
                attempt += 1
                continue
            self._record_request_timestamp()
            self._sleep_delay()
            return payload

    def _extract_boxscore_paths(self, html: str) -> List[str]:
        seen: List[str] = []
        for match in _BOX_LINK_PATTERN.finditer(html):
            href = match.group(1)
            if href not in seen:
                seen.append(href)
        return seen

    def _parse_boxscore_page(self, html: str, path: str) -> GameBoxScore:
        document = _parse_document(html)
        scorebox = document.find_first(tag="div", class_contains="scorebox")
        if scorebox is None:
            raise ValueError("scorebox not found in box score payload")

        team_nodes = self._find_team_nodes(scorebox)
        if len(team_nodes) < 2:
            raise ValueError("Could not locate both teams in scorebox")

        away_info = self._parse_scorebox_team(team_nodes[0], is_home=False)
        home_info = self._parse_scorebox_team(team_nodes[1], is_home=True)

        meta_node = scorebox.find_first(tag="div", class_contains="scorebox_meta")
        event_time, venue, attendance, officials, notes = self._parse_scorebox_meta(meta_node)

        away_team = self._build_team_boxscore(html, away_info)
        home_team = self._build_team_boxscore(html, home_info)

        game_id = path.rsplit("/", 1)[-1].removesuffix(".html")
        return GameBoxScore(
            game_id=game_id,
            event_time=event_time,
            venue=venue,
            attendance=attendance,
            officials=officials,
            notes=notes,
            away=away_team,
            home=home_team,
        )

    def _parse_scorebox_team(self, node: HTMLNode, *, is_home: bool) -> Dict[str, Any]:
        name_link = node.find_first(tag="a")
        team_name = name_link.get_text().strip() if name_link else node.get_text().strip()
        href = name_link.get("href") if name_link else ""
        team_abbr = self._team_abbreviation_from_href(href)

        score_node = node.find_first(tag="div", class_contains="scores")
        score = _safe_int(score_node.get_text()) if score_node else 0
        record_node = node.find_first(tag="div", class_contains="record")
        record = record_node.get_text().strip() if record_node else None

        return {
            "team_name": team_name,
            "team_abbr": team_abbr,
            "is_home": is_home,
            "score": score,
            "record": record,
        }

    def _team_abbreviation_from_href(self, href: str | None) -> str:
        if not href:
            return ""
        parts = [segment for segment in href.split("/") if segment]
        return parts[1] if len(parts) > 1 else ""

    def _parse_scorebox_meta(
        self, meta: HTMLNode | None
    ) -> tuple[datetime | None, str | None, int | None, List[str], List[str]]:
        if meta is None:
            return None, None, None, [], []

        event_time: datetime | None = None
        venue: str | None = None
        attendance: int | None = None
        officials: List[str] = []
        notes: List[str] = []
        seen_time = False

        for div in meta.iter_children(tag="div"):
            text = div.get_text().strip()
            if not text:
                continue
            if text.startswith("Attendance:"):
                match = re.search(r"Attendance:\s*([\d,]+)", text)
                if match:
                    attendance = int(match.group(1).replace(",", ""))
            elif text.startswith("Officials:"):
                crew = text.split(":", 1)[1]
                officials = [name.strip() for name in crew.split(",") if name.strip()]
            elif _looks_like_time_string(text):
                event_time = _parse_event_time(text)
                seen_time = True
            elif venue is None and seen_time:
                venue = text
            else:
                notes.append(text)

        return event_time, venue, attendance, officials, notes

    def _find_team_nodes(self, scorebox: HTMLNode) -> List[HTMLNode]:
        nodes: List[HTMLNode] = []
        seen: set[int] = set()

        for strong in scorebox.iter_subtree(include_self=False):
            if strong.tag != "strong":
                continue
            link = next(
                (
                    child
                    for child in strong.iter_subtree(include_self=False)
                    if child.tag == "a"
                ),
                None,
            )
            if link is None:
                continue
            href = link.get("href") or ""
            if not href.startswith("/teams/"):
                continue
            container = self._ascend_to_team_container(strong, scorebox)
            if container and id(container) not in seen:
                nodes.append(container)
                seen.add(id(container))
            if len(nodes) == 2:
                break
        return nodes

    def _ascend_to_team_container(
        self, node: HTMLNode, stop: HTMLNode
    ) -> HTMLNode | None:
        current = node
        while current is not None and current is not stop:
            if (
                current.tag == "div"
                and not current.has_class("scorebox_meta")
                and self._has_direct_child_with_class(current, "scores")
            ):
                return current
            current = current.parent
        return None

    def _has_direct_child_with_class(self, node: HTMLNode, class_name: str) -> bool:
        for child in node.children:
            if (
                isinstance(child, HTMLNode)
                and child.tag == "div"
                and child.has_class(class_name)
            ):
                return True
        return False

    def _decode_body(self, response) -> str:
        data = response.read()
        encoding = (response.headers.get("Content-Encoding") or "").lower()
        if encoding == "gzip":
            data = gzip.decompress(data)
        return data.decode("utf-8")

    def _build_headers(self, url: str) -> Dict[str, str]:
        headers = dict(self.base_headers)
        if "/boxscores/" in url:
            headers["Referer"] = f"{BREF_BASE_URL}{SCOREBOARD_PATH}"
        else:
            headers["Referer"] = BREF_BASE_URL
        return headers

    def _should_retry(self, status: int | None, attempt: int) -> bool:
        if attempt >= self.max_retries - 1:
            return False
        return status in {429, 500, 502, 503, 504}

    def _sleep_delay(self) -> None:
        if self.delay > 0:
            jitter = random.uniform(0, 0.3)
            time.sleep(self.delay + jitter)

    def _sleep_backoff(self, attempt: int, retry_after: float | None) -> None:
        wait = self.backoff_factor**attempt
        total = self.delay + wait + random.uniform(0, 0.5)
        if retry_after is not None:
            total = max(total, retry_after)
        time.sleep(total)

    def _respect_rate_limit(self) -> None:
        if self.rate_limit_window <= 0 or self.rate_limit_max_requests <= 0:
            return
        window = self.rate_limit_window
        timestamps = self._request_timestamps
        while True:
            now = time.monotonic()
            while timestamps and now - timestamps[0] >= window:
                timestamps.popleft()
            if len(timestamps) < self.rate_limit_max_requests:
                return
            sleep_for = timestamps[0] + window - now
            if sleep_for <= 0:
                timestamps.popleft()
                continue
            time.sleep(sleep_for)

    def _record_request_timestamp(self) -> None:
        if self.rate_limit_window <= 0 or self.rate_limit_max_requests <= 0:
            return
        now = time.monotonic()
        self._request_timestamps.append(now)
        window = self.rate_limit_window
        while self._request_timestamps and now - self._request_timestamps[0] >= window:
            self._request_timestamps.popleft()

    def _retry_after_header(self, exc: HTTPError | None) -> float | None:
        if exc is None:
            return None
        retry_after = exc.headers.get("Retry-After")
        if not retry_after:
            return None
        retry_after = retry_after.strip()
        if retry_after.isdigit():
            return float(retry_after)
        try:
            parsed = parsedate_to_datetime(retry_after)
            now = datetime.now(parsed.tzinfo or timezone.utc)
            seconds = (parsed - now).total_seconds()
            return max(seconds, 0.0)
        except ValueError:
            return None

    def _build_team_boxscore(self, html: str, info: Dict[str, Any]) -> TeamBoxScore:
        team_abbr = info["team_abbr"]
        basic_table_html = _extract_table_html(html, f"box-{team_abbr}-game-basic")
        advanced_table_html = _extract_table_html(html, f"box-{team_abbr}-game-advanced")

        basic_rows = self._parse_player_rows(basic_table_html)
        advanced_rows = self._parse_player_rows(advanced_table_html)
        advanced_lookup = {row["key"]: row for row in advanced_rows}

        players: List[PlayerBoxScore] = []
        seen_keys: set[str] = set()
        for row in basic_rows:
            adv = advanced_lookup.get(row["key"])
            players.append(
                PlayerBoxScore(
                    player_id=row["slug"],
                    player=row["name"],
                    starter=row["starter"],
                    reason=row["reason"],
                    basic=row["stats"],
                    advanced=adv["stats"] if adv else {},
                )
            )
            seen_keys.add(row["key"])

        for row in advanced_rows:
            if row["key"] in seen_keys:
                continue
            players.append(
                PlayerBoxScore(
                    player_id=row["slug"],
                    player=row["name"],
                    starter=row["starter"],
                    reason=row["reason"],
                    basic={},
                    advanced=row["stats"],
                )
            )

        return TeamBoxScore(
            team_name=info["team_name"],
            team_abbr=team_abbr,
            is_home=info["is_home"],
            score=info["score"],
            record=info["record"],
            players=players,
            totals=self._parse_totals(basic_table_html),
            advanced_totals=self._parse_totals(advanced_table_html),
        )

    def _parse_player_rows(self, table_html: str | None) -> List[Dict[str, Any]]:
        if table_html is None:
            return []

        table_doc = _parse_fragment(table_html)
        table = table_doc.find_first(tag="table")
        if table is None:
            return []

        body = table.find_first(tag="tbody")
        if body is None:
            return []

        rows: List[Dict[str, Any]] = []
        bench_section = False
        for tr in body.iter_children(tag="tr"):
            if tr.has_class("thead"):
                bench_section = True
                continue

            player_cell = next(
                (child for child in tr.iter_children() if child.tag == "th" and child.get("data-stat") == "player"),
                None,
            )
            if player_cell is None:
                continue

            name = player_cell.get_text().strip()
            if not name:
                continue

            slug = player_cell.get("data-append-csv")
            key = slug or name
            reason_cell = next(
                (child for child in tr.iter_children() if child.tag == "td" and child.get("data-stat") == "reason"),
                None,
            )
            reason = reason_cell.get_text().strip() if reason_cell else None

            stats: Dict[str, Any] = {}
            for td in tr.iter_children(tag="td"):
                stat_code = td.get("data-stat")
                if not stat_code or stat_code == "reason":
                    continue
                stats[stat_code] = _coerce_stat_value(td.get_text())

            rows.append(
                {
                    "key": key,
                    "slug": slug,
                    "name": name,
                    "starter": not bench_section,
                    "reason": reason,
                    "stats": stats,
                }
            )
        return rows

    def _parse_totals(self, table_html: str | None) -> Dict[str, Any]:
        if table_html is None:
            return {}

        table_doc = _parse_fragment(table_html)
        table = table_doc.find_first(tag="table")
        if table is None:
            return {}

        tfoot = table.find_first(tag="tfoot")
        if tfoot is None:
            return {}

        totals: Dict[str, Any] = {}
        for row in tfoot.iter_children(tag="tr"):
            for cell in row.iter_children():
                if cell.tag not in {"td", "th"}:
                    continue
                stat_code = cell.get("data-stat")
                if not stat_code or stat_code == "player":
                    continue
                totals[stat_code] = _coerce_stat_value(cell.get_text())
        return totals


@dataclass
class HTMLNode:
    tag: str
    attrs: Dict[str, str]
    contents: List[Any] = field(default_factory=list)
    parent: HTMLNode | None = None

    def __post_init__(self) -> None:
        self.children: List[HTMLNode] = []

    def append_child(self, node: HTMLNode) -> None:
        node.parent = self
        self.contents.append(node)
        self.children.append(node)

    def append_text(self, text: str) -> None:
        if text:
            self.contents.append(text)

    def get(self, attr: str, default: str | None = None) -> str | None:
        return self.attrs.get(attr, default)

    def get_text(self, strip: bool = True) -> str:
        parts: List[str] = []
        for item in self.contents:
            if isinstance(item, HTMLNode):
                parts.append(item.get_text(strip=False))
            else:
                parts.append(item)
        combined = "".join(parts)
        return combined.strip() if strip else combined

    def iter_children(self, tag: str | None = None) -> Iterable[HTMLNode]:
        for child in self.children:
            if tag is None or child.tag == tag:
                yield child

    def iter_subtree(self, include_self: bool = True) -> Iterable[HTMLNode]:
        if include_self:
            yield self
        for child in self.children:
            yield from child.iter_subtree(include_self=True)

    def find_first(
        self,
        *,
        tag: str | None = None,
        id_value: str | None = None,
        class_contains: str | None = None,
        attrs: Dict[str, str] | None = None,
        include_self: bool = False,
    ) -> HTMLNode | None:
        for node in self.iter_subtree(include_self=include_self):
            if node is self and not include_self:
                continue
            if _matches(node, tag=tag, id_value=id_value, class_contains=class_contains, attrs=attrs):
                return node
        return None

    def has_class(self, target: str) -> bool:
        return target in self.class_list()

    def class_list(self) -> List[str]:
        value = self.attrs.get("class", "")
        return [cls for cls in value.split() if cls]


@dataclass
class HTMLDocument:
    root: HTMLNode
    comments: List[str]

    def find_first(
        self,
        *,
        tag: str | None = None,
        id_value: str | None = None,
        class_contains: str | None = None,
        attrs: Dict[str, str] | None = None,
    ) -> HTMLNode | None:
        for node in self.root.iter_subtree(include_self=False):
            if _matches(node, tag=tag, id_value=id_value, class_contains=class_contains, attrs=attrs):
                return node
        return None


class _MiniHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = HTMLNode("__root__", {})
        self.stack: List[HTMLNode] = [self.root]
        self.comments: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str | None]]) -> None:
        node = HTMLNode(tag, {key: value or "" for key, value in attrs})
        self.stack[-1].append_child(node)
        self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: List[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if len(self.stack) > 1:
            self.stack.pop()

    def handle_data(self, data: str) -> None:
        self.stack[-1].append_text(data)

    def handle_comment(self, data: str) -> None:
        self.comments.append(data)


def _parse_document(html: str) -> HTMLDocument:
    parser = _MiniHTMLParser()
    parser.feed(html)
    parser.close()
    return HTMLDocument(parser.root, parser.comments)


def _parse_fragment(html: str) -> HTMLDocument:
    return _parse_document(html)


def _matches(
    node: HTMLNode,
    *,
    tag: str | None,
    id_value: str | None,
    class_contains: str | None,
    attrs: Dict[str, str] | None,
) -> bool:
    if tag and node.tag != tag:
        return False
    if id_value and node.get("id") != id_value:
        return False
    if class_contains and not node.has_class(class_contains):
        return False
    if attrs:
        for key, value in attrs.items():
            if node.get(key) != value:
                return False
    return True


def _extract_table_html(html: str, element_id: str) -> str | None:
    pattern = re.compile(
        rf"<table[^>]+id=[\"']{re.escape(element_id)}[\"'][^>]*>.*?</table>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(html)
    return match.group(0) if match else None


_TIME_PATTERN = re.compile(r"\d{1,2}:\d{2}\s*(?:AM|PM)", flags=re.IGNORECASE)


def _looks_like_time_string(value: str) -> bool:
    return bool(_TIME_PATTERN.search(value))


def _parse_event_time(value: str) -> datetime | None:
    for fmt in ("%I:%M %p, %B %d, %Y", "%I:%M %p, %b %d, %Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _safe_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        digits = "".join(ch for ch in value if ch.isdigit())
        return int(digits) if digits else 0


def _coerce_stat_value(raw: str | None) -> Any:
    if raw is None:
        return None
    text = raw.strip()
    if not text or text == "\xa0":
        return None
    normalized = text.replace(",", "")
    try:
        return int(normalized)
    except ValueError:
        pass
    try:
        return float(normalized)
    except ValueError:
        pass
    return text
