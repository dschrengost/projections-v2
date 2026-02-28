from __future__ import annotations

import sys
import types
from datetime import date
from pathlib import Path

import pandas as pd

from projections.cli import score_ownership_linestar


def _dk_slate(rows: list[dict], draft_group_id: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["draft_group_id"] = draft_group_id
    df["_name_norm"] = df["player_name"].apply(score_ownership_linestar._normalize_name)  # noqa: SLF001
    return df


def test_fetch_and_match_ownership_returns_multiple_slates(monkeypatch, tmp_path: Path) -> None:
    dk_slates = {
        "111111": _dk_slate(
            [
                {"player_id": 1, "player_name": "Alpha Guard", "team": "AAA", "pos": "PG", "salary": 8000},
                {"player_id": 2, "player_name": "Beta Wing", "team": "BBB", "pos": "SG", "salary": 7000},
                {"player_id": 3, "player_name": "Gamma Big", "team": "AAA", "pos": "C", "salary": 6500},
                {"player_id": 4, "player_name": "Delta Six", "team": "BBB", "pos": "SF", "salary": 6200},
            ],
            "111111",
        ),
        "222222": _dk_slate(
            [
                {"player_id": 5, "player_name": "Epsilon Guard", "team": "CCC", "pos": "PG", "salary": 6000},
                {"player_id": 6, "player_name": "Zeta Wing", "team": "DDD", "pos": "SG", "salary": 5600},
                {"player_id": 7, "player_name": "Eta Big", "team": "CCC", "pos": "C", "salary": 5400},
                {"player_id": 8, "player_name": "Theta Six", "team": "DDD", "pos": "SF", "salary": 5000},
            ],
            "222222",
        ),
    }
    monkeypatch.setattr(score_ownership_linestar, "_load_dk_salaries", lambda *_args, **_kwargs: dk_slates)

    ls_df = pd.DataFrame(
        [
            {"player_name": "Alpha Guard", "team": "AAA", "salary": 8000, "proj_own_pct": 30.0, "ls_slate_id": "LS_MAIN"},
            {"player_name": "Beta Wing", "team": "BBB", "salary": 7000, "proj_own_pct": 24.0, "ls_slate_id": "LS_MAIN"},
            {"player_name": "Gamma Big", "team": "AAA", "salary": 6500, "proj_own_pct": 18.0, "ls_slate_id": "LS_MAIN"},
            {"player_name": "Delta Six", "team": "BBB", "salary": 6200, "proj_own_pct": 12.0, "ls_slate_id": "LS_MAIN"},
            {"player_name": "Epsilon Guard", "team": "CCC", "salary": 6000, "proj_own_pct": 26.0, "ls_slate_id": "LS_ALT"},
            {"player_name": "Zeta Wing", "team": "DDD", "salary": 5600, "proj_own_pct": 20.0, "ls_slate_id": "LS_ALT"},
            {"player_name": "Eta Big", "team": "CCC", "salary": 5400, "proj_own_pct": 16.0, "ls_slate_id": "LS_ALT"},
            {"player_name": "Theta Six", "team": "DDD", "salary": 5000, "proj_own_pct": 11.0, "ls_slate_id": "LS_ALT"},
        ]
    )
    ls_df["ls_proj_fpts"] = None
    ls_df["ls_floor_fpts"] = None
    ls_df["ls_ceil_fpts"] = None

    fake_pkg = types.ModuleType("linestar")
    fake_mod = types.ModuleType("linestar.fetch_live_ownership")
    fake_mod.fetch_linestar_ownership = lambda **_kwargs: ls_df
    monkeypatch.setitem(sys.modules, "linestar", fake_pkg)
    monkeypatch.setitem(sys.modules, "linestar.fetch_live_ownership", fake_mod)

    results = score_ownership_linestar.fetch_and_match_ownership(
        date(2026, 2, 28),
        tmp_path,
        run_id="20260228T180002Z",
    )

    assert set(results) == {"111111", "222222"}
    assert set(results["111111"]["ls_slate_id"].astype(str)) == {"LS_MAIN"}
    assert set(results["222222"]["ls_slate_id"].astype(str)) == {"LS_ALT"}
    assert results["111111"]["pred_own_pct"].sum() == 84.0
    assert results["222222"]["pred_own_pct"].sum() == 73.0
