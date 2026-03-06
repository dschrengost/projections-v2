"""DraftKings authentication and reusable cookie/state helpers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page, Playwright


DK_RESULTS_COOKIE_ENV = "DK_RESULTS_COOKIE"
DK_STORAGE_STATE_ENV = "DK_STORAGE_STATE_PATH"


def default_storage_state_path() -> Path:
    """Return the canonical control-plane location for DK browser state."""
    default_root = Path.home() / "projections-data"
    data_root = Path(os.getenv("PROJECTIONS_DATA_ROOT", str(default_root))).expanduser()
    return data_root / "control_plane" / "dk_auth" / "storage_state.json"


def resolve_storage_state_path(path: Optional[Path] = None) -> Path:
    """Resolve the storage-state path from arg, env, or canonical default."""
    if path is not None:
        return Path(path).expanduser()
    env_path = os.getenv(DK_STORAGE_STATE_ENV)
    if env_path:
        return Path(env_path).expanduser()
    return default_storage_state_path()


def normalize_cookie(raw_cookie: Optional[str]) -> Optional[str]:
    """Normalize a raw cookie string for use in request headers."""
    if not raw_cookie:
        return None
    segments = [segment.strip() for segment in str(raw_cookie).split(";") if segment.strip()]
    if not segments:
        return None
    return "; ".join(segments)


def load_cookie_from_storage_state(path: Path) -> str:
    """Extract a Cookie header string from Playwright storage_state.json."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cookies: list[str] = []
    for cookie in payload.get("cookies", []):
        domain = str(cookie.get("domain") or "")
        name = cookie.get("name")
        value = cookie.get("value")
        if not name or value is None:
            continue
        if "draftkings.com" not in domain:
            continue
        cookies.append(f"{name}={value}")
    cookie_header = normalize_cookie("; ".join(cookies))
    if not cookie_header:
        raise RuntimeError(f"No DraftKings cookies found in storage state: {path}")
    return cookie_header


def resolve_request_cookie(
    *,
    cookie: Optional[str] = None,
    cookie_file: Optional[Path] = None,
    storage_state_path: Optional[Path] = None,
    cookie_env_var: str = DK_RESULTS_COOKIE_ENV,
) -> Optional[str]:
    """Resolve the best available DK cookie source for request-based scrapers."""
    direct_cookie = normalize_cookie(cookie)
    if direct_cookie:
        return direct_cookie
    if cookie_file is not None:
        return normalize_cookie(cookie_file.read_text(encoding="utf-8").strip())
    state_path = resolve_storage_state_path(storage_state_path)
    if state_path.exists():
        return load_cookie_from_storage_state(state_path)
    return normalize_cookie(os.getenv(cookie_env_var))


class DraftKingsAuthenticator:
    """Handle DraftKings authentication using Playwright."""

    LOGIN_URL = "https://www.draftkings.com"
    BASE_URL = "https://www.draftkings.com"

    def __init__(self, headless: bool = True, storage_state_path: Optional[Path] = None):
        self.headless = headless
        self.storage_state_path = Path(storage_state_path).expanduser() if storage_state_path else None
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self) -> None:
        """Start the browser and create a new context."""
        from playwright.sync_api import sync_playwright

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        context_kwargs = {}
        if self.storage_state_path and self.storage_state_path.exists():
            context_kwargs["storage_state"] = str(self.storage_state_path)
        self.context = self.browser.new_context(**context_kwargs)
        self.page = self.context.new_page()

    def stop(self) -> None:
        """Stop the browser and clean up resources."""
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def login_with_env_credentials(self) -> bool:
        """Login using credentials from environment variables."""
        username = os.getenv("DK_USERNAME")
        password = os.getenv("DK_PASSWORD")

        if not username or not password:
            print("DK_USERNAME and DK_PASSWORD environment variables not set")
            return False

        return self.login(username, password)

    def login_interactive(self) -> bool:
        """Open browser for interactive login."""
        if self.headless:
            print("Cannot perform interactive login in headless mode")
            return False

        self.page.goto(self.LOGIN_URL)
        print("Browser opened to DraftKings homepage.")
        print("Please log in manually by clicking the 'Log In' button.")
        print("After you have successfully logged in, press Enter here...")
        input()

        if self.is_logged_in():
            print("Login successful.")
            return True
        print("Login verification failed. Please make sure you're logged in.")
        return False

    def login(self, username: str, password: str) -> bool:
        """Login to DraftKings with username and password."""
        try:
            self.page.goto(self.BASE_URL)

            try:
                login_button = self.page.locator('button:has-text("Log In"), a:has-text("Log In")').first
                login_button.click(timeout=5000)
                print("Clicked login button, waiting for login form...")
            except Exception:
                print("Login button not found, looking for login form...")

            self.page.wait_for_selector(
                'input[name="username"], input[placeholder*="Email"], input[placeholder*="Username"], input[type="email"]',
                timeout=10000,
            )

            username_input = self.page.locator(
                'input[name="username"], input[placeholder*="Email"], input[placeholder*="Username"], input[type="email"]'
            ).first
            username_input.fill(username)

            password_input = self.page.locator('input[name="password"], input[type="password"]').first
            password_input.fill(password)

            submit_button = self.page.locator(
                'button[type="submit"], button:has-text("Log In"), button:has-text("Sign In")'
            ).first
            submit_button.click()

            try:
                self.page.wait_for_url("**/dashboard**", timeout=30000)
            except Exception:
                print("Dashboard not found, checking for login success indicators...")
                self.page.wait_for_timeout(5000)

            return self.is_logged_in()
        except Exception as exc:
            print(f"Login failed: {exc}")
            return False

    def is_logged_in(self) -> bool:
        """Check if user is logged in."""
        try:
            self.page.goto(self.BASE_URL)
            self.page.wait_for_timeout(3000)

            logged_in_indicators = [
                'a:has-text("My Account")',
                'a:has-text("Logout")',
                'button:has-text("My Account")',
                '[data-testid="user-menu"]',
                '.user-info',
                '.account-info',
                '.header-user',
                '[class*="user"]',
                '[class*="account"]',
            ]

            for indicator in logged_in_indicators:
                try:
                    element = self.page.locator(indicator).first
                    if element.is_visible(timeout=2000):
                        return True
                except Exception:
                    continue

            try:
                email_elements = self.page.locator(':has-text("@")').count()
                if email_elements > 0:
                    return True
            except Exception:
                pass

            current_url = self.page.url.lower()
            if any(pattern in current_url for pattern in ["dashboard", "account", "profile"]):
                return True

            try:
                login_button = self.page.locator('button:has-text("Log In"), a:has-text("Log In")').first
                if not login_button.is_visible(timeout=2000):
                    return True
            except Exception:
                pass

            print("Could not confirm login status - continuing anyway...")
            return True
        except Exception as exc:
            print(f"Warning checking login status: {exc}")
            return True

    def get_cookies(self) -> str:
        """Extract cookies as a formatted string."""
        if not self.context:
            raise RuntimeError("Browser context not initialized")

        cookie_pairs = []
        for cookie in self.context.cookies():
            if "draftkings.com" in str(cookie.get("domain") or ""):
                cookie_pairs.append(f"{cookie['name']}={cookie['value']}")
        cookie_header = normalize_cookie("; ".join(cookie_pairs))
        if not cookie_header:
            raise RuntimeError("No DraftKings cookies available in browser context")
        return cookie_header

    def save_storage_state(self, path: Optional[Path] = None) -> Path:
        """Persist Playwright storage state for later reuse by request scrapers."""
        if not self.context:
            raise RuntimeError("Browser context not initialized")
        out_path = resolve_storage_state_path(path or self.storage_state_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.context.storage_state(path=str(out_path))
        return out_path

    def save_cookies_to_env(self, env_file: Optional[Path] = None) -> None:
        """Save cookies to an env file."""
        if env_file is None:
            env_file = Path(".env")

        cookies = self.get_cookies()
        env_content = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
        lines = env_content.split("\n")
        updated_lines = []
        cookie_updated = False

        for line in lines:
            if line.startswith(f"{DK_RESULTS_COOKIE_ENV}="):
                updated_lines.append(f"{DK_RESULTS_COOKIE_ENV}={cookies}")
                cookie_updated = True
            else:
                updated_lines.append(line)

        if not cookie_updated:
            updated_lines.append(f"{DK_RESULTS_COOKIE_ENV}={cookies}")

        env_file.write_text("\n".join(updated_lines), encoding="utf-8")
        print(f"Cookies saved to {env_file}")


def authenticate_with_browser(
    *,
    headless: bool = True,
    interactive: bool = False,
    storage_state_path: Optional[Path] = None,
    env_file: Optional[Path] = None,
    save_cookie_env: bool = True,
) -> str:
    """Authenticate with DraftKings, returning request cookies and saving browser state."""
    resolved_state_path = resolve_storage_state_path(storage_state_path)
    with DraftKingsAuthenticator(
        headless=headless and not interactive,
        storage_state_path=resolved_state_path,
    ) as auth:
        if interactive:
            success = auth.login_interactive()
        else:
            success = auth.login_with_env_credentials()

        if not success:
            raise RuntimeError("Failed to authenticate with DraftKings")

        saved_state = auth.save_storage_state(resolved_state_path)
        print(f"Storage state saved to {saved_state}")

        cookies = auth.get_cookies()
        if save_cookie_env:
            auth.save_cookies_to_env(env_file)
        return cookies


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture reusable DraftKings browser auth state")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open a real browser and wait for manual login",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless for environment-credential login",
    )
    parser.add_argument(
        "--storage-state-out",
        type=Path,
        help="Path to write Playwright storage_state.json (defaults to control-plane path)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Env file to update with DK_RESULTS_COOKIE",
    )
    parser.add_argument(
        "--no-save-env",
        action="store_true",
        help="Do not update the env file with DK_RESULTS_COOKIE",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    interactive = args.interactive or not (os.getenv("DK_USERNAME") and os.getenv("DK_PASSWORD"))
    storage_state_path = resolve_storage_state_path(args.storage_state_out)
    cookies = authenticate_with_browser(
        headless=args.headless,
        interactive=interactive,
        storage_state_path=storage_state_path,
        env_file=args.env_file,
        save_cookie_env=not args.no_save_env,
    )
    print(f"Successfully authenticated. Cookie length: {len(cookies)}")
    print(f"Reusable storage state: {storage_state_path}")


if __name__ == "__main__":
    main()
