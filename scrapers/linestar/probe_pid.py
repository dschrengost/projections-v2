# scrapers/linestar/probe_pid.py
import json, pathlib, sys
from playwright.sync_api import sync_playwright

PID = int(sys.argv[1]) if len(sys.argv) > 1 else 2395
API = "https://www.linestarapp.com/DesktopModules/DailyFantasyApi/API/Fantasy/GetSalariesV5"
PARAMS = {"periodId": PID, "site": 1, "sport": 2}  # DK=1, NBA=2

BASE = pathlib.Path(__file__).parent
STATE_PATH = BASE / "storage_state.json"
if not STATE_PATH.exists():
    # fall back to top-level scrapers/storage_state.json if present
    alt = BASE.parent / "storage_state.json"
    if alt.exists():
        STATE_PATH = alt

def main():
    with sync_playwright() as p:
        rc = p.request.new_context(storage_state=str(STATE_PATH) if STATE_PATH.exists() else None)
        r = rc.get(API, params=PARAMS)
        print("HTTP status:", r.status)
        txt = r.text()
        try:
            data = r.json()
        except Exception:
            data = json.loads(txt)

    if not isinstance(data, dict):
        print("Top-level is not a dict; type:", type(data))
        print("First 300 chars:", str(data)[:300])
        return

    print("Top-level keys:", list(data.keys()))

    if "Ownership" in data and isinstance(data["Ownership"], dict):
        own = data["Ownership"]
        print("Ownership keys:", list(own.keys()))
        proj = own.get("Projected")
        if isinstance(proj, dict):
            print("Projected slate count:", len(proj))
            if proj:
                sid = next(iter(proj.keys()))
                print("Example projected slate id:", sid, "entries:", len(proj[sid]))
                if proj[sid]:
                    print("Projected entry keys:", list(proj[sid][0].keys()))
        cr = own.get("ContestResults")
        if isinstance(cr, list):
            print("ContestResults count:", len(cr))

    scj = data.get("SalaryContainerJson")
    if isinstance(scj, str):
        print("SalaryContainerJson length:", len(scj))
        try:
            container = json.loads(scj)
            print("Container keys:", list(container.keys()))
            for k in ("Slates", "Salaries", "SalariesMap", "Games"):
                if k in container:
                    v = container[k]
                    print(f"{k} count:", len(v) if hasattr(v, "__len__") else type(v))
            ssm = container.get("SalariesMap", {}).get("SlateSalaryMap", {})
            print("SlateSalaryMap keys (first 5):", list(ssm.keys())[:5])
        except Exception as e:
            print("Failed to parse SalaryContainerJson:", e)

if __name__ == "__main__":
    main()