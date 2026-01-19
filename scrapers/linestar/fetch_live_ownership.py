"""
Fetch live projected ownership from LineStar for today's NBA slates.

This module discovers today's PID, fetches projected ownership data via the
GetSalariesV5 API, and returns a normalized DataFrame compatible with the
ownership pipeline.

Usage:
    from scrapers.linestar.fetch_live_ownership import fetch_linestar_ownership

    df = fetch_linestar_ownership(site="dk", sport="nba")
    # Returns DataFrame with: player_name, team, pos, salary, proj_own_pct, proj_fpts, ...

Requirements:
    - Valid storage_state.json from login_password.py
    - Playwright installed
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Lazy import playwright to avoid import errors when not installed
_playwright_available = None


def _check_playwright():
    global _playwright_available
    if _playwright_available is None:
        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
            _playwright_available = True
        except ImportError:
            _playwright_available = False
    return _playwright_available


BASE = Path(__file__).parent
STATE_PATH = BASE / "storage_state.json"
if not STATE_PATH.exists():
    alt = BASE.parent / "storage_state.json"
    if alt.exists():
        STATE_PATH = alt

API = "https://www.linestarapp.com/DesktopModules/DailyFantasyApi/API/Fantasy/GetSalariesV5"
SITE_MAP = {"dk": 1, "draftkings": 1, "fd": 2, "fanduel": 2}
SPORT_MAP = {"nba": 2}

# Default credentials - can be overridden via environment variables
import os
LS_EMAIL = os.getenv("LS_EMAIL", "dschrengost@gmail.com")
LS_PASSWORD = os.getenv("LS_PASSWORD", "@Raven23!")


def refresh_session(
    email: Optional[str] = None,
    password: Optional[str] = None,
    headless: bool = True,
) -> bool:
    """
    Perform headless login to refresh the session with premium credentials.

    Args:
        email: LineStar account email (defaults to LS_EMAIL env or hardcoded)
        password: LineStar account password (defaults to LS_PASSWORD env or hardcoded)
        headless: Run browser in headless mode (default True for server use)

    Returns:
        True if login succeeded, False otherwise
    """
    if not _check_playwright():
        raise ImportError("playwright is required: uv add playwright && playwright install firefox")

    from playwright.sync_api import sync_playwright

    email = email or LS_EMAIL
    password = password or LS_PASSWORD

    if not email or not password:
        raise ValueError("LineStar credentials required (set LS_EMAIL and LS_PASSWORD env vars)")

    print(f"[linestar] Refreshing session for {email}...")

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()

        try:
            page.goto("https://linestarapp.com/Projections/Sport/NBA/Site/DraftKings", wait_until="load")
            page.wait_for_timeout(3000)

            # Check if login modal is already open (sometimes auto-opens)
            modal = page.locator("#myModal_587")
            if not modal.is_visible():
                # Click login link
                page.locator("a.LoginLink").click(force=True)
                page.wait_for_timeout(2000)

            # Fill credentials
            page.locator("#dnn_ctr587_Login_Login_DNN_txtUsername").fill(email)
            page.locator("#dnn_ctr587_Login_Login_DNN_txtPassword").fill(password)

            # Click login button
            page.locator('#myModal_587 a.dnnPrimaryAction:has-text("Login")').click()

            # Wait for navigation
            page.wait_for_timeout(5000)

            # Save session state
            context.storage_state(path=str(STATE_PATH))
            print(f"[linestar] Session saved to {STATE_PATH}")

            browser.close()
            return True

        except Exception as e:
            print(f"[linestar] Login failed: {e}")
            browser.close()
            return False


def _parse_ls_date(label: str) -> Optional[date]:
    """Parse LineStar date label like 'Jan 18, 2026' to date."""
    try:
        return datetime.strptime(label.strip(), "%b %d, %Y").date()
    except ValueError:
        return None


def _parse_dotnet_ticks(ticks_str: str) -> Optional[datetime]:
    """Parse .NET /Date(...)/ format to datetime."""
    if not ticks_str or not isinstance(ticks_str, str):
        return None
    m = re.search(r"/Date\((\d+)(?:[+-]\d+)?\)/", ticks_str)
    if m:
        ts_ms = int(m.group(1))
        return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return None


def discover_today_pid(
    site: str = "dk",
    sport: str = "nba",
    target_date: Optional[date] = None,
) -> Optional[int]:
    """
    Discover the LineStar PID for today's (or target) date.

    Calls GetSalariesV5 without periodId to get the Periods list.
    """
    if not _check_playwright():
        raise ImportError("playwright is required: uv add playwright && playwright install chromium")

    from playwright.sync_api import sync_playwright

    site_id = SITE_MAP.get(site.lower())
    sport_id = SPORT_MAP.get(sport.lower())
    if site_id is None or sport_id is None:
        raise ValueError(f"Unknown site={site} or sport={sport}")

    if target_date is None:
        target_date = date.today()

    with sync_playwright() as p:
        rc = p.request.new_context(
            storage_state=str(STATE_PATH) if STATE_PATH.exists() else None
        )
        r = rc.get(API, params={"site": site_id, "sport": sport_id})
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {r.text()[:200]}")

        try:
            data = r.json()
        except Exception:
            data = json.loads(r.text())

    periods = data.get("Periods") or []
    if not periods:
        return None

    for p in periods:
        pid = p.get("Id") or p.get("PeriodId")
        label = p.get("Name") or p.get("Label") or ""
        p_date = _parse_ls_date(label)
        if p_date == target_date and pid is not None:
            return int(pid)

    return None


def fetch_salaries_payload(
    pid: int,
    site: str = "dk",
    sport: str = "nba",
) -> dict:
    """Fetch the raw GetSalariesV5 payload for a specific PID."""
    if not _check_playwright():
        raise ImportError("playwright is required: uv add playwright && playwright install chromium")

    from playwright.sync_api import sync_playwright

    site_id = SITE_MAP.get(site.lower())
    sport_id = SPORT_MAP.get(sport.lower())
    if site_id is None or sport_id is None:
        raise ValueError(f"Unknown site={site} or sport={sport}")

    with sync_playwright() as p:
        rc = p.request.new_context(
            storage_state=str(STATE_PATH) if STATE_PATH.exists() else None
        )
        r = rc.get(API, params={"periodId": pid, "site": site_id, "sport": sport_id})
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {r.text()[:200]}")

        try:
            return r.json()
        except Exception:
            return json.loads(r.text())


def _raw_to_float(v) -> Optional[float]:
    """Parse a numeric value from LineStar payload fields."""
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        m = re.search(r"-?\d+(?:\.\d+)?", str(v))
        return float(m.group(0)) if m else None


def _ownership_scale(values: list[float]) -> float:
    """Infer ownership scale from observed values (percent vs fraction)."""
    if not values:
        return 1.0
    max_val = max(values)
    # If all values are <= 1.0, treat them as fractions and scale to percent.
    return 100.0 if max_val <= 1.0 else 1.0


def _money_to_int(v) -> Optional[int]:
    """Parse salary/money value to int."""
    if v is None:
        return None
    m = re.search(r"(\d[\d,]*)", str(v))
    return int(m.group(1).replace(",", "")) if m else None


def _pick_proj_fpts(r: dict) -> Optional[float]:
    """Extract projected FPTS from salary row."""
    for k in ("AggProj", "Proj", "PP", "Projection", "ProjPoints", "FPTS", "FP", "PS"):
        try:
            v = float(r.get(k))
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass

    # Fallback: average of floor and ceiling
    try:
        f = float(r.get("Floor"))
        c = float(r.get("Ceil"))
        if f is not None and c is not None:
            return (f + c) / 2.0
    except (TypeError, ValueError):
        pass

    # Final fallback: PPG
    try:
        return float(r.get("PPG"))
    except (TypeError, ValueError):
        return None


def normalize_salary_row(r: dict) -> dict:
    """Normalize a LineStar salary row to standard fields."""
    out = {}
    out["salary_id"] = r.get("SalaryId") or r.get("Id")
    out["ls_player_id"] = r.get("PlayerId") or r.get("PID")
    out["player_name"] = (
        r.get("Name") or r.get("PlayerName") or r.get("Player") or r.get("FullName")
    )
    out["pos"] = r.get("POS") or r.get("Position") or r.get("Positions")
    out["team"] = r.get("PTEAM") or r.get("Team") or r.get("TeamAbbrev")

    sal = r.get("SAL") or r.get("Salary") or r.get("DKSalary") or r.get("FDSalary") or r.get("S")
    out["salary"] = _money_to_int(sal)

    out["ls_proj_fpts"] = _pick_proj_fpts(r)
    out["ls_floor_fpts"] = r.get("Floor")
    out["ls_ceil_fpts"] = r.get("Ceil")
    out["ls_conf"] = r.get("Conf")
    out["ls_ppg"] = r.get("PPG")
    out["ls_opp_rank"] = r.get("OppRank")

    out["game_id"] = r.get("GID")
    out["home_team"] = r.get("HTEAM")
    out["away_team"] = r.get("OTEAM")
    out["matchup"] = r.get("GI")
    out["game_ticks"] = r.get("GT")

    # Parse game time
    if out["game_ticks"]:
        out["game_time"] = _parse_dotnet_ticks(out["game_ticks"])
    else:
        out["game_time"] = None

    # Tidy numeric fields
    for c in ["ls_proj_fpts", "ls_floor_fpts", "ls_ceil_fpts", "ls_conf", "ls_ppg", "ls_opp_rank"]:
        try:
            out[c] = float(out[c]) if out[c] is not None else None
        except (TypeError, ValueError):
            out[c] = None

    # Strip strings
    for k in ("player_name", "team", "pos", "matchup", "home_team", "away_team"):
        if out.get(k) is not None:
            out[k] = str(out[k]).strip()

    return out


def parse_ownership_payload(
    payload: dict,
    *,
    site: str = "dk",
    sport: str = "nba",
    pid: int,
) -> pd.DataFrame:
    """
    Parse GetSalariesV5 payload into a DataFrame of projected ownership.

    Returns DataFrame with columns:
        player_name, team, pos, salary, proj_own_pct, ls_proj_fpts,
        ls_floor_fpts, ls_ceil_fpts, ls_conf, ls_player_id, salary_id,
        game_id, matchup, game_time, scraped_at
    """
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected payload type: {type(payload)}")

    # Parse salaries container
    salaries_map = {}
    container_raw = payload.get("SalaryContainerJson")
    if isinstance(container_raw, str):
        try:
            container = json.loads(container_raw)
            salaries = container.get("Salaries") or []
            for row in salaries:
                norm = normalize_salary_row(row)
                sid = norm.get("salary_id")
                if sid is not None:
                    salaries_map[str(sid)] = norm
        except Exception:
            pass

    # Parse projected ownership
    ownership = payload.get("Ownership") or {}
    projected = ownership.get("Projected") if isinstance(ownership, dict) else None

    rows = []
    if isinstance(projected, dict) and projected:
        # Iterate over all slates
        for slate_id, slate_entries in projected.items():
            if not isinstance(slate_entries, list):
                continue
            owned_raw = []
            for item in slate_entries:
                val = _raw_to_float(item.get("Owned"))
                if val is not None:
                    owned_raw.append(val)
            scale = _ownership_scale(owned_raw)

            for item in slate_entries:
                sid = str(item.get("SalaryId") or "")
                srow = salaries_map.get(sid, {})
                owned_val = _raw_to_float(item.get("Owned"))

                rows.append({
                    "player_name": srow.get("player_name"),
                    "team": srow.get("team"),
                    "pos": srow.get("pos"),
                    "salary": srow.get("salary"),
                    "proj_own_pct": owned_val * scale if owned_val is not None else None,
                    "ls_proj_fpts": srow.get("ls_proj_fpts"),
                    "ls_floor_fpts": srow.get("ls_floor_fpts"),
                    "ls_ceil_fpts": srow.get("ls_ceil_fpts"),
                    "ls_conf": srow.get("ls_conf"),
                    "ls_ppg": srow.get("ls_ppg"),
                    "ls_opp_rank": srow.get("ls_opp_rank"),
                    "ls_player_id": srow.get("ls_player_id") or item.get("PlayerId"),
                    "salary_id": item.get("SalaryId"),
                    "game_id": srow.get("game_id"),
                    "matchup": srow.get("matchup"),
                    "game_time": srow.get("game_time"),
                    "home_team": srow.get("home_team"),
                    "away_team": srow.get("away_team"),
                    "ls_slate_id": slate_id,
                    "ls_period_id": pid,
                    "site": "DraftKings" if SITE_MAP.get(site.lower()) == 1 else "FanDuel",
                    "sport": sport.upper(),
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                })

    df = pd.DataFrame(rows)

    # Ensure expected columns exist
    expected_cols = [
        "player_name", "team", "pos", "salary", "proj_own_pct",
        "ls_proj_fpts", "ls_floor_fpts", "ls_ceil_fpts", "ls_conf",
        "ls_player_id", "salary_id", "game_id", "matchup", "game_time",
        "ls_slate_id", "ls_period_id", "site", "sport", "scraped_at",
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df


def _is_truncated_response(payload: dict) -> bool:
    """Check if the API response is truncated (non-premium)."""
    return payload.get("IsTruncated", False)


def fetch_linestar_ownership(
    site: str = "dk",
    sport: str = "nba",
    target_date: Optional[date] = None,
    pid: Optional[int] = None,
    auto_refresh: bool = True,
) -> pd.DataFrame:
    """
    Fetch projected ownership from LineStar for today (or target date).

    Args:
        site: "dk" or "fd"
        sport: "nba"
        target_date: Date to fetch (default: today)
        pid: Optional explicit PID (will auto-discover if not provided)
        auto_refresh: If True, auto-refresh session when data is truncated

    Returns:
        DataFrame with projected ownership data, including:
        - player_name, team, pos, salary
        - proj_own_pct (LineStar's projected ownership %)
        - ls_proj_fpts, ls_floor_fpts, ls_ceil_fpts (LineStar's FPTS projections)
        - ls_slate_id (LineStar's internal slate ID)
        - and more...

    Raises:
        ImportError: If playwright not installed
        RuntimeError: If API call fails
        ValueError: If no PID found for date
    """
    if target_date is None:
        target_date = date.today()

    if pid is None:
        pid = discover_today_pid(site=site, sport=sport, target_date=target_date)
        if pid is None:
            raise ValueError(f"No LineStar PID found for {target_date}")

    payload = fetch_salaries_payload(pid=pid, site=site, sport=sport)

    # Check if we got truncated data (session expired or non-premium)
    if auto_refresh and _is_truncated_response(payload):
        print("[linestar] Got truncated response - refreshing session...")
        if refresh_session():
            # Retry after refresh
            payload = fetch_salaries_payload(pid=pid, site=site, sport=sport)
            if _is_truncated_response(payload):
                print("[linestar] Warning: Still getting truncated data after refresh")
        else:
            print("[linestar] Warning: Session refresh failed")

    df = parse_ownership_payload(payload, site=site, sport=sport, pid=pid)

    return df


def get_main_slate_ownership(
    site: str = "dk",
    sport: str = "nba",
    target_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Convenience function to get ownership for the main slate only.

    The main slate is identified as the one with the most players.
    """
    df = fetch_linestar_ownership(site=site, sport=sport, target_date=target_date)

    if df.empty:
        return df

    # Find the slate with the most players
    slate_counts = df.groupby("ls_slate_id").size()
    main_slate = slate_counts.idxmax()

    return df[df["ls_slate_id"] == main_slate].copy()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fetch LineStar ownership projections")
    ap.add_argument("--site", default="dk", help="dk or fd")
    ap.add_argument("--sport", default="nba", help="nba")
    ap.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: today)")
    ap.add_argument("--pid", type=int, default=None, help="Explicit PID (auto-discovers if not set)")
    ap.add_argument("--main-slate", action="store_true", help="Only show main slate")
    ap.add_argument("--out", default=None, help="Output CSV path")
    ap.add_argument("--refresh", action="store_true", help="Force session refresh before fetching")
    args = ap.parse_args()

    # Force refresh if requested
    if args.refresh:
        print("Forcing session refresh...")
        if refresh_session():
            print("Session refreshed successfully")
        else:
            print("Session refresh failed")
            exit(1)

    target = date.fromisoformat(args.date) if args.date else None

    print(f"Fetching LineStar ownership for {target or 'today'}...")

    if args.main_slate:
        df = get_main_slate_ownership(site=args.site, sport=args.sport, target_date=target)
    else:
        df = fetch_linestar_ownership(
            site=args.site, sport=args.sport, target_date=target, pid=args.pid
        )

    print(f"\nFetched {len(df)} rows across {df['ls_slate_id'].nunique()} slates")

    if not df.empty:
        print("\nSlate breakdown:")
        for slate_id, group in df.groupby("ls_slate_id"):
            print(f"  Slate {slate_id}: {len(group)} players")

        print("\nTop 10 by projected ownership:")
        top = df.nlargest(10, "proj_own_pct")[
            ["player_name", "team", "pos", "salary", "proj_own_pct", "ls_proj_fpts"]
        ]
        print(top.to_string(index=False))

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")
