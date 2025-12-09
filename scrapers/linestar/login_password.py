import os, time, pathlib
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

BASE = pathlib.Path(__file__).parent
PROFILE_DIR = BASE / ".pw_linestar_profile"   # same as capture script
START_URL = "https://linestarapp.com/Ownership/Sport/NBA/Site/DraftKings"

load_dotenv()
EMAIL = os.getenv("LS_EMAIL") or ""
PASSWORD = os.getenv("LS_PASSWORD") or ""

def main():
    if not EMAIL or not PASSWORD:
        raise SystemExit("Set LS_EMAIL and LS_PASSWORD in a .env file.")

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            channel="chrome",
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = ctx.new_page()
        page.goto(START_URL, wait_until="load")

        # Open the login modal (header Sign In)
        page.get_by_text("Sign In", exact=False).first.click()

        # Fill in the email/password fields
        page.get_by_label("Email", exact=False).fill(EMAIL)
        page.get_by_label("Password", exact=False).fill(PASSWORD)

        # Submit
        page.get_by_role("button", name=lambda n: "Login" in n or "Sign In" in n).click()

        # Wait for table to appear as proof of success
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("table thead", timeout=20000)

        # Persist cookies for both styles of usage
        ctx.storage_state(path=str(BASE / "storage_state.json"))
        print("✅ Logged in and saved storage_state.json")

        # Optional: small settle + screenshot
        time.sleep(1)
        page.screenshot(path=str(BASE / "captures_login_ok.png"), full_page=True)

        ctx.close()

if __name__ == "__main__":
    main()