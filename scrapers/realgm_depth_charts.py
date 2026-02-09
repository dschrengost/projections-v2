"""Scraper for RealGM NBA depth charts.

This scraper fetches the RealGM NBA depth charts page and extracts
depth chart data for all 30 NBA teams. It requires Playwright to
bypass Cloudflare protection.

Usage:
    from scrapers.realgm_depth_charts import scrape_realgm_depth_charts

    df = scrape_realgm_depth_charts()
    # Returns DataFrame with columns:
    #   team_name, player_name, realgm_player_id, position, depth_role,
    #   depth_order, recent_stats, movement, scraped_at
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore

logger = logging.getLogger(__name__)

REALGM_DEPTH_CHARTS_URL = "https://basketball.realgm.com/nba/depth-charts"

# Map CSS class to depth role
DEPTH_ROLE_MAP = {
    "depth_starters": "starter",
    "depth_rotation": "rotation",
    "depth_limpt": "limited",
}

# Standard positions
POSITIONS = ["PG", "SG", "SF", "PF", "C"]


def _season_start_year_for_day(day: date) -> int:
    return int(day.year) if int(day.month) >= 8 else int(day.year) - 1


def _run_slug(ts: pd.Timestamp | datetime | None = None) -> str:
    if ts is None:
        dt = pd.Timestamp.now(tz="UTC")
    else:
        dt = pd.Timestamp(ts)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{_run_slug()}.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _check_playwright() -> bool:
    """Check if Playwright is available."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
        return True
    except ImportError:
        return False


def _check_playwright_chromium() -> tuple[bool, str | None]:
    """Check whether Playwright Chromium is installed and launchable."""
    if not _check_playwright():
        return False, "playwright_import_missing"
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return False, "playwright_import_missing"

    try:
        with sync_playwright() as p:
            exe_path = p.chromium.executable_path
            if not exe_path:
                return False, "chromium_executable_missing"
            if not Path(exe_path).exists():
                return False, "chromium_not_installed"
        return True, None
    except Exception:
        return False, "chromium_not_installed"


def realgm_dependency_report() -> dict[str, object]:
    """Return dependency readiness report for RealGM scraping."""
    missing: list[str] = []
    details: list[str] = []

    if BeautifulSoup is None:
        missing.append("beautifulsoup4")
    try:
        import lxml  # noqa: F401
    except Exception:
        missing.append("lxml")
    if not _check_playwright():
        missing.append("playwright")
    else:
        chromium_ok, chromium_reason = _check_playwright_chromium()
        if not chromium_ok:
            missing.append("chromium")
            if chromium_reason:
                details.append(str(chromium_reason))

    available = len(missing) == 0
    return {
        "available": available,
        "missing": missing,
        "details": details,
    }


def realgm_dependencies_available() -> bool:
    """Check if all required dependencies are available."""
    report = realgm_dependency_report()
    return bool(report.get("available", False))


@dataclass(frozen=True)
class DepthChartEntry:
    """A single player's depth chart entry."""

    team_name: str
    player_name: str
    realgm_player_id: Optional[int]
    position: str
    depth_role: str
    depth_order: int
    recent_stats: Optional[str]
    movement: Optional[str]  # "up", "down", or None
    scraped_at: pd.Timestamp


class RealGMDepthChartsScraper:
    """Scrape NBA depth charts from RealGM."""

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        headless: bool = True,
        url: str = REALGM_DEPTH_CHARTS_URL,
    ) -> None:
        if BeautifulSoup is None:
            raise ImportError(
                "BeautifulSoup4 is required for RealGM scraping. "
                "Install with: pip install beautifulsoup4 lxml"
            )
        if not _check_playwright():
            raise ImportError(
                "Playwright is required for RealGM scraping (Cloudflare protection). "
                "Install with: pip install playwright && playwright install chromium"
            )
        self.timeout = timeout
        self.headless = headless
        self.url = url

    def fetch_html(self) -> str:
        """Fetch the raw HTML from RealGM depth charts page using Playwright."""
        from playwright.sync_api import sync_playwright

        logger.info(f"[realgm] Fetching {self.url} with Playwright...")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            )
            page = context.new_page()

            try:
                # Use domcontentloaded to avoid waiting for all resources
                page.goto(
                    self.url,
                    wait_until="domcontentloaded",
                    timeout=int(self.timeout * 1000),
                )
                # Wait for Cloudflare challenge to resolve
                page.wait_for_timeout(5000)

                # Verify we got past Cloudflare
                title = page.title()
                if "Just a moment" in title:
                    # Still on Cloudflare challenge page, wait longer
                    logger.warning("[realgm] Still on Cloudflare challenge, waiting...")
                    page.wait_for_timeout(10000)

                html = page.content()
                logger.info(f"[realgm] Fetched {len(html)} bytes")
                return html

            finally:
                browser.close()

    def parse_depth_charts(self, html: str) -> list[DepthChartEntry]:
        """Parse the HTML and extract depth chart data for all teams."""
        soup = BeautifulSoup(html, "lxml")
        entries: list[DepthChartEntry] = []
        scraped_at = pd.Timestamp.now(tz="UTC")

        # Find all team sections - each starts with an h2 containing "Depth Chart"
        h2_tags = soup.find_all("h2", class_="clearfix")

        for h2 in h2_tags:
            h2_text = h2.get_text(strip=True)
            if "Depth Chart" not in h2_text:
                continue

            # Extract team name from "2025-2026 Atlanta Hawks Depth Chart"
            match = re.match(r"\d{4}-\d{4}\s+(.+?)\s+Depth Chart", h2_text)
            if not match:
                continue
            team_name = match.group(1)

            # Find the table following this h2
            # The actual depth chart table has data-toggle="table" attribute
            # Navigate: h2 -> next sibling div.large-column-left -> find table with data-toggle
            container = h2.find_next_sibling("div", class_="large-column-left")
            if not container:
                continue

            # Find the depth chart table (has data-toggle="table" attribute)
            table = container.find("table", {"data-toggle": "table"})
            if not table:
                # Fallback: look for table with depth-chart-cell inside
                table = container.find("table", class_="table")
                if not table or not table.find("td", class_="depth-chart-cell"):
                    continue

            # Parse the table
            team_entries = self._parse_team_table(table, team_name, scraped_at)
            entries.extend(team_entries)

        logger.info(f"[realgm] Parsed {len(entries)} depth chart entries")
        return entries

    def _parse_team_table(
        self,
        table,
        team_name: str,
        scraped_at: pd.Timestamp,
    ) -> list[DepthChartEntry]:
        """Parse a single team's depth chart table."""
        entries: list[DepthChartEntry] = []

        # Track depth order per role
        role_order: dict[str, int] = {}

        tbody = table.find("tbody")
        if not tbody:
            return entries

        for row in tbody.find_all("tr"):
            # Determine the depth role from row class
            row_classes = row.get("class", [])
            depth_role = None
            for cls in row_classes:
                if cls in DEPTH_ROLE_MAP:
                    depth_role = DEPTH_ROLE_MAP[cls]
                    break

            if not depth_role:
                continue

            # Get or initialize depth order for this role
            if depth_role not in role_order:
                role_order[depth_role] = 0
            current_order = role_order[depth_role]
            role_order[depth_role] += 1

            # Parse each cell in the row
            cells = row.find_all("td", class_="depth-chart-cell")
            for cell in cells:
                entry = self._parse_player_cell(
                    cell, team_name, depth_role, current_order, scraped_at
                )
                if entry:
                    entries.append(entry)

        return entries

    def _parse_player_cell(
        self,
        cell,
        team_name: str,
        depth_role: str,
        depth_order: int,
        scraped_at: pd.Timestamp,
    ) -> Optional[DepthChartEntry]:
        """Parse a single player cell."""
        # Get position from data-th attribute
        position = cell.get("data-th", "")
        if position not in POSITIONS:
            return None

        # Find player link
        player_link = cell.find("a")
        if not player_link:
            return None

        player_name = player_link.get_text(strip=True)
        if not player_name or player_name == "&nbsp;":
            return None

        # Extract RealGM player ID from href
        # Format: /player/Player-Name/Summary/12345
        href = player_link.get("href", "")
        realgm_player_id = None
        id_match = re.search(r"/Summary/(\d+)", href)
        if id_match:
            realgm_player_id = int(id_match.group(1))

        # Extract recent stats (text after the link)
        # The cell typically has: <a>Name</a><br>19p 3r 6a
        cell_text = cell.get_text(separator="\n", strip=True)
        lines = cell_text.split("\n")
        recent_stats = None
        if len(lines) > 1:
            stats_line = lines[1].strip()
            # Validate it looks like stats (contains p/r/a patterns)
            if re.search(r"\d+[pra]", stats_line):
                recent_stats = stats_line

        # Check for movement indicators (up/down arrows in background style)
        movement = None
        style = cell.get("style", "")
        if "arrow-up" in style:
            movement = "up"
        elif "arrow-dwn" in style:
            movement = "down"

        return DepthChartEntry(
            team_name=team_name,
            player_name=player_name,
            realgm_player_id=realgm_player_id,
            position=position,
            depth_role=depth_role,
            depth_order=depth_order,
            recent_stats=recent_stats,
            movement=movement,
            scraped_at=scraped_at,
        )

    def scrape(self) -> list[DepthChartEntry]:
        """Fetch and parse RealGM depth charts."""
        html = self.fetch_html()
        return self.parse_depth_charts(html)


def scrape_realgm_depth_charts(
    *,
    headless: bool = True,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """
    Fetch and normalize RealGM NBA depth charts into a DataFrame.

    Args:
        headless: Run browser in headless mode (default True)
        timeout: Page load timeout in seconds (default 60)

    Returns a DataFrame with columns:
        - team_name: str (e.g., "Atlanta Hawks")
        - player_name: str
        - realgm_player_id: int | None
        - position: str (PG, SG, SF, PF, C)
        - depth_role: str (starter, rotation, limited)
        - depth_order: int (0-based order within role)
        - recent_stats: str | None (e.g., "19p 3r 6a")
        - movement: str | None (up, down)
        - scraped_at: Timestamp
    """
    scraper = RealGMDepthChartsScraper(headless=headless, timeout=timeout)
    entries = scraper.scrape()

    if not entries:
        return pd.DataFrame(columns=[
            "team_name",
            "player_name",
            "realgm_player_id",
            "position",
            "depth_role",
            "depth_order",
            "recent_stats",
            "movement",
            "scraped_at",
        ])

    records = [
        {
            "team_name": e.team_name,
            "player_name": e.player_name,
            "realgm_player_id": e.realgm_player_id,
            "position": e.position,
            "depth_role": e.depth_role,
            "depth_order": e.depth_order,
            "recent_stats": e.recent_stats,
            "movement": e.movement,
            "scraped_at": e.scraped_at,
        }
        for e in entries
    ]

    df = pd.DataFrame.from_records(records)

    # Sort by team, depth role order, depth order, position
    role_order = {"starter": 0, "rotation": 1, "limited": 2}
    pos_order = {pos: i for i, pos in enumerate(POSITIONS)}

    df["_role_sort"] = df["depth_role"].map(role_order)
    df["_pos_sort"] = df["position"].map(pos_order)
    df = df.sort_values(
        ["team_name", "_role_sort", "depth_order", "_pos_sort"]
    ).drop(columns=["_role_sort", "_pos_sort"])

    return df.reset_index(drop=True)


def get_starters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter depth chart DataFrame to only include starters.

    Returns one row per team per position (5 starters per team).
    """
    if df.empty:
        return df

    starters = df[df["depth_role"] == "starter"].copy()

    # Keep only the first starter at each position per team
    starters = (
        starters.sort_values(["team_name", "position", "depth_order"])
        .groupby(["team_name", "position"], as_index=False)
        .first()
    )

    return starters


def save_realgm_depth_charts_bronze(
    df: pd.DataFrame,
    *,
    game_date: date,
    data_root: Path | str | None = None,
    run_ts: datetime | pd.Timestamp | None = None,
) -> dict[str, Path]:
    """Persist RealGM depth chart snapshot to bronze history + latest pointers.

    Layout:
      - bronze/realgm/depth_charts/season=YYYY/date=YYYY-MM-DD/run_ts=<slug>/depth_charts.parquet
      - bronze/realgm/depth_charts/season=YYYY/date=YYYY-MM-DD/depth_charts.parquet  (day latest)
      - bronze/realgm/depth_charts_latest.parquet                                     (global latest)
      - bronze/realgm/depth_charts.parquet                                            (back-compat alias)
    """
    if data_root is None:
        data_root = Path(os.environ.get("PROJECTIONS_DATA_ROOT", "data"))
    else:
        data_root = Path(data_root)

    out = df.copy()
    if "scraped_at" not in out.columns:
        out["scraped_at"] = pd.Timestamp.now(tz="UTC")
    out["scraped_at"] = pd.to_datetime(out["scraped_at"], utc=True, errors="coerce")
    out = out.dropna(subset=["scraped_at"]).copy()
    if out.empty:
        raise ValueError("RealGM depth chart frame is empty after scraped_at normalization.")

    season = _season_start_year_for_day(game_date)
    slug = _run_slug(run_ts)
    bronze_root = data_root / "bronze" / "realgm"
    day_dir = bronze_root / "depth_charts" / f"season={season}" / f"date={game_date.isoformat()}"

    history_path = day_dir / f"run_ts={slug}" / "depth_charts.parquet"
    day_latest_path = day_dir / "depth_charts.parquet"
    global_latest_path = bronze_root / "depth_charts_latest.parquet"
    global_alias_path = bronze_root / "depth_charts.parquet"

    _atomic_write_parquet(out, history_path)
    _atomic_write_parquet(out, day_latest_path)
    _atomic_write_parquet(out, global_latest_path)
    _atomic_write_parquet(out, global_alias_path)
    return {
        "history_path": history_path,
        "day_latest_path": day_latest_path,
        "global_latest_path": global_latest_path,
        "global_alias_path": global_alias_path,
    }


__all__ = [
    "DepthChartEntry",
    "RealGMDepthChartsScraper",
    "scrape_realgm_depth_charts",
    "get_starters",
    "save_realgm_depth_charts_bronze",
    "realgm_dependencies_available",
    "realgm_dependency_report",
]


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Scrape RealGM NBA depth charts")
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser in visible mode (for debugging)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Page load timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--starters-only",
        action="store_true",
        help="Only show starters",
    )
    parser.add_argument(
        "--team",
        type=str,
        default=None,
        help="Filter to specific team (partial match)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path",
    )
    args = parser.parse_args()

    if not realgm_dependencies_available():
        print("Error: Missing dependencies. Install with:")
        print("  pip install beautifulsoup4 lxml playwright")
        print("  playwright install chromium")
        sys.exit(1)

    print("Fetching RealGM depth charts...")
    df = scrape_realgm_depth_charts(
        headless=not args.no_headless,
        timeout=args.timeout,
    )

    if df.empty:
        print("No data found!")
        sys.exit(1)

    if args.starters_only:
        df = get_starters(df)

    if args.team:
        df = df[df["team_name"].str.contains(args.team, case=False)]

    print(f"\nFound {len(df)} entries across {df['team_name'].nunique()} teams")

    # Summary by team
    print("\nTeam summary:")
    team_summary = df.groupby("team_name").agg(
        starters=("depth_role", lambda x: (x == "starter").sum()),
        rotation=("depth_role", lambda x: (x == "rotation").sum()),
        limited=("depth_role", lambda x: (x == "limited").sum()),
        total=("player_name", "count"),
    )
    print(team_summary.to_string())

    # Show sample data
    print("\nSample data (first 20 rows):")
    display_cols = [
        "team_name",
        "player_name",
        "position",
        "depth_role",
        "depth_order",
        "recent_stats",
        "movement",
    ]
    print(df[display_cols].head(20).to_string(index=False))

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")
