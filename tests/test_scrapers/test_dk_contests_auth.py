from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTH_PATH = ROOT / "scrapers" / "dk_contests" / "auth.py"
SPEC = importlib.util.spec_from_file_location("dk_contests_auth", AUTH_PATH)
assert SPEC is not None and SPEC.loader is not None
dk_auth = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dk_auth)


def test_load_cookie_from_storage_state_filters_to_draftkings(tmp_path: Path) -> None:
    state_path = tmp_path / "storage_state.json"
    state_path.write_text(
        """
        {
          "cookies": [
            {"name": "sessionA", "value": "abc", "domain": ".draftkings.com"},
            {"name": "sessionB", "value": "xyz", "domain": "www.draftkings.com"},
            {"name": "ignore_me", "value": "nope", "domain": ".example.com"}
          ],
          "origins": []
        }
        """.strip(),
        encoding="utf-8",
    )

    cookie_header = dk_auth.load_cookie_from_storage_state(state_path)

    assert cookie_header == "sessionA=abc; sessionB=xyz"


def test_resolve_request_cookie_prefers_storage_state_over_env(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "storage_state.json"
    state_path.write_text(
        """
        {
          "cookies": [
            {"name": "from_state", "value": "123", "domain": ".draftkings.com"}
          ],
          "origins": []
        }
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DK_RESULTS_COOKIE", "from_env=456")

    cookie_header = dk_auth.resolve_request_cookie(
        storage_state_path=state_path,
        cookie_env_var="DK_RESULTS_COOKIE",
    )

    assert cookie_header == "from_state=123"


def test_resolve_request_cookie_uses_explicit_cookie_first(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "storage_state.json"
    state_path.write_text(
        """
        {
          "cookies": [
            {"name": "from_state", "value": "123", "domain": ".draftkings.com"}
          ],
          "origins": []
        }
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DK_RESULTS_COOKIE", "from_env=456")

    cookie_header = dk_auth.resolve_request_cookie(
        cookie="explicit=789",
        storage_state_path=state_path,
        cookie_env_var="DK_RESULTS_COOKIE",
    )

    assert cookie_header == "explicit=789"
