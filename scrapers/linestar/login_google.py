# scrapers/linestar/login_google.py
import pathlib
from playwright.sync_api import sync_playwright

BASE = pathlib.Path(__file__).parent
PROFILE_DIR = BASE / ".pw_linestar_profile"   # persists your Google/LineStar session

def main():
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            channel="chrome",           # <-- use your system Chrome
            headless=False,
            args=[
                # a little stealth help (not strictly required)
                "--disable-blink-features=AutomationControlled",
            ],
        )
        page = ctx.new_page()
        page.goto("https://www.linestarapp.com/Ownership/Sport/NBA/Site/DraftKings")
        print("\n1) Click Sign In ➜ Continue with Google, complete 2FA.")
        print("2) Confirm you’re back on LineStar and see the table.")
        input("3) Press ENTER here to save the session… ")
        ctx.storage_state(path=str(BASE / "storage_state.json"))
        ctx.close()

if __name__ == "__main__":
    main()