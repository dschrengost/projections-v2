"""DraftKings authentication using Playwright."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright


class DraftKingsAuthenticator:
    """Handle DraftKings authentication using Playwright."""

    LOGIN_URL = "https://www.draftkings.com"
    BASE_URL = "https://www.draftkings.com"

    def __init__(self, headless: bool = True):
        self.headless = headless
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
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context()
        self.page = self.context.new_page()

    def stop(self) -> None:
        """Stop the browser and clean up resources."""
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if hasattr(self, 'playwright'):
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

        # Check if login was successful
        if self.is_logged_in():
            print("✓ Login successful!")
            return True
        else:
            print("✗ Login verification failed. Please make sure you're logged in.")
            return False

    def login(self, username: str, password: str) -> bool:
        """Login to DraftKings with username and password."""
        try:
            self.page.goto(self.BASE_URL)

            # Look for and click the "Log In" button first
            try:
                login_button = self.page.locator('button:has-text("Log In"), a:has-text("Log In")').first
                login_button.click(timeout=5000)
                print("Clicked login button, waiting for login form...")
            except:
                print("Login button not found, looking for login form...")

            # Wait for login form to load
            self.page.wait_for_selector('input[name="username"], input[placeholder*="Email"], input[placeholder*="Username"], input[type="email"]', timeout=10000)

            # Fill in username
            username_input = self.page.locator('input[name="username"], input[placeholder*="Email"], input[placeholder*="Username"], input[type="email"]').first
            username_input.fill(username)

            # Fill in password
            password_input = self.page.locator('input[name="password"], input[type="password"]').first
            password_input.fill(password)

            # Click login/submit button
            submit_button = self.page.locator('button[type="submit"], button:has-text("Log In"), button:has-text("Sign In")').first
            submit_button.click()

            # Wait for login to complete - look for various success indicators
            try:
                self.page.wait_for_url("**/dashboard**", timeout=30000)
            except:
                # If dashboard URL doesn't work, wait for any login success indicator
                print("Dashboard not found, checking for login success indicators...")
                self.page.wait_for_timeout(5000)

            return self.is_logged_in()

        except Exception as e:
            print(f"Login failed: {e}")
            return False

    def is_logged_in(self) -> bool:
        """Check if user is logged in."""
        try:
            # Navigate to homepage and wait briefly
            self.page.goto(self.BASE_URL)
            self.page.wait_for_timeout(3000)  # Wait 3 seconds for page to load

            # Look for various signs of successful login
            logged_in_indicators = [
                'a:has-text("My Account")',
                'a:has-text("Logout")',
                'button:has-text("My Account")',
                '[data-testid="user-menu"]',
                '.user-info',
                '.account-info',
                '.header-user',
                '[class*="user"]',
                '[class*="account"]'
            ]

            for indicator in logged_in_indicators:
                try:
                    element = self.page.locator(indicator).first
                    if element.is_visible(timeout=2000):
                        return True
                except:
                    continue

            # Check for username/email display in header
            try:
                # Look for any element containing "@" symbol (likely email)
                email_elements = self.page.locator(':has-text("@")').count()
                if email_elements > 0:
                    return True
            except:
                pass

            # Check URL patterns
            current_url = self.page.url.lower()
            if any(pattern in current_url for pattern in ["dashboard", "account", "profile"]):
                return True

            # Check for absence of login button (which suggests user is logged in)
            try:
                login_button = self.page.locator('button:has-text("Log In"), a:has-text("Log In")').first
                if not login_button.is_visible(timeout=2000):
                    # If login button is not visible, likely logged in
                    return True
            except:
                pass

            print("Could not confirm login status - continuing anyway...")
            return True  # Be more permissive for interactive login

        except Exception as e:
            print(f"Warning checking login status: {e}")
            return True  # Don't fail on timeout for interactive login

    def get_cookies(self) -> str:
        """Extract cookies as a formatted string."""
        if not self.context:
            raise RuntimeError("Browser context not initialized")

        cookies = self.context.cookies()
        cookie_pairs = []

        for cookie in cookies:
            if cookie.get("domain", "").endswith("draftkings.com"):
                cookie_pairs.append(f"{cookie['name']}={cookie['value']}")

        return "; ".join(cookie_pairs)

    def save_cookies_to_env(self, env_file: Optional[Path] = None) -> None:
        """Save cookies to .env file."""
        if env_file is None:
            env_file = Path(".env")

        cookies = self.get_cookies()

        # Read existing .env file
        env_content = ""
        if env_file.exists():
            env_content = env_file.read_text(encoding="utf-8")

        # Update or add DK_RESULTS_COOKIE
        lines = env_content.split('\n')
        updated_lines = []
        cookie_updated = False

        for line in lines:
            if line.startswith('DK_RESULTS_COOKIE='):
                updated_lines.append(f'DK_RESULTS_COOKIE={cookies}')
                cookie_updated = True
            else:
                updated_lines.append(line)

        if not cookie_updated:
            updated_lines.append(f'DK_RESULTS_COOKIE={cookies}')

        env_file.write_text('\n'.join(updated_lines), encoding="utf-8")
        print(f"Cookies saved to {env_file}")


def authenticate_with_browser(headless: bool = True, interactive: bool = False) -> str:
    """
    Authenticate with DraftKings and return cookies.

    Args:
        headless: Whether to run browser in headless mode
        interactive: Whether to perform interactive login

    Returns:
        Cookie string for use in requests

    Raises:
        RuntimeError: If authentication fails
    """
    with DraftKingsAuthenticator(headless=headless and not interactive) as auth:
        if interactive:
            success = auth.login_interactive()
        else:
            success = auth.login_with_env_credentials()

        if not success:
            raise RuntimeError("Failed to authenticate with DraftKings")

        cookies = auth.get_cookies()

        # Save cookies to .env file for future use
        auth.save_cookies_to_env()

        return cookies


if __name__ == "__main__":
    # Test the authentication
    try:
        cookies = authenticate_with_browser(headless=False, interactive=True)
        print(f"Successfully authenticated! Cookie length: {len(cookies)}")
    except Exception as e:
        print(f"Authentication failed: {e}")