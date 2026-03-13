"""Tests for the in-house ownership training base builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "ownership" / "build_ownership_inhouse_base.py"
    spec = importlib.util.spec_from_file_location("build_ownership_inhouse_base", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_match_dk_slates_to_salary_slates_prefers_best_overlap():
    mod = _load_module()

    dk = pd.DataFrame(
        {
            "slate_id": ["dk_main"] * 4 + ["dk_late"] * 3,
            "game_date": ["2025-12-01"] * 7,
            "player_name_norm": ["a", "b", "c", "d", "x", "y", "z"],
        }
    )
    salaries = pd.DataFrame(
        {
            "salary_slate_id": ["111"] * 5 + ["222"] * 4,
            "salary_game_date": ["2025-12-01"] * 9,
            "player_name_norm": ["a", "b", "c", "d", "e", "x", "y", "z", "w"],
        }
    )

    out, matches = mod.match_dk_slates_to_salary_slates(
        dk,
        salaries,
        max_day_offset=0,
        min_overlap_coeff=0.80,
        min_intersection=2,
    )

    mapping = {match.dk_slate_id: match.salary_slate_id for match in matches}
    assert mapping["dk_main"] == "111"
    assert mapping["dk_late"] == "222"

    out_map = out.groupby("slate_id")["salary_slate_id"].first().to_dict()
    assert out_map["dk_main"] == "111"
    assert out_map["dk_late"] == "222"


def test_match_dk_slates_to_salary_slates_rejects_tiny_subset_slate():
    mod = _load_module()

    dk = pd.DataFrame(
        {
            "slate_id": ["dk_main"] * 6,
            "game_date": ["2025-12-01"] * 6,
            "player_name_norm": ["a", "b", "c", "d", "e", "f"],
        }
    )
    salaries = pd.DataFrame(
        {
            "salary_slate_id": ["tiny"] * 3 + ["full"] * 7,
            "salary_game_date": ["2025-12-01"] * 10,
            "player_name_norm": ["a", "b", "c", "a", "b", "c", "d", "e", "f", "g"],
        }
    )

    out, matches = mod.match_dk_slates_to_salary_slates(
        dk,
        salaries,
        max_day_offset=0,
        min_overlap_coeff=0.80,
        min_intersection=2,
        min_recall_dk=0.80,
    )

    assert len(matches) == 1
    assert matches[0].salary_slate_id == "full"
    assert out["salary_slate_id"].iloc[0] == "full"


def test_build_training_rows_for_slate_zero_fills_missing_labels():
    mod = _load_module()

    salary_slate = pd.DataFrame(
        {
            "salary_slate_id": ["111", "111"],
            "salary_game_date": ["2025-12-01", "2025-12-01"],
            "dk_player_id": pd.Series([10, 11], dtype="Int64"),
            "player_name": ["Alpha", "Bravo"],
            "player_name_norm": ["alpha", "bravo"],
            "team": ["AAA", "BBB"],
            "pos": ["PG", "C"],
            "salary": [8000, 4200],
        }
    )
    dk_labels = pd.DataFrame(
        {
            "player_name_norm": ["alpha"],
            "own_pct": [37.5],
        }
    )
    minutes_context = pd.DataFrame(
        {
            "player_name_norm": ["alpha", "bravo"],
            "team": ["AAA", "BBB"],
            "nba_player_id": pd.Series([1001, 1002], dtype="Int64"),
            "player_is_out": [0, 0],
            "player_is_questionable": [1, 0],
            "team_outs_count": [2, 0],
            "total_close": [232.5, 228.0],
            "spread_close": [-3.5, 4.0],
            "team_implied_total": [118.0, 112.0],
        }
    )
    projections = pd.DataFrame(
        {
            "nba_player_id": pd.Series([1001, 1002], dtype="Int64"),
            "proj_fpts": [44.0, 21.0],
            "play_prob_eff": [0.99, 0.85],
        }
    )

    result = mod.build_training_rows_for_slate(
        salary_slate=salary_slate,
        dk_labels=dk_labels,
        minutes_context=minutes_context,
        projection_context=projections,
        dk_slate_id="2025-12-01_0",
        game_date_str="2025-12-01",
    )

    assert len(result) == 2
    own_map = dict(zip(result["player_name"], result["actual_own_pct"]))
    assert own_map["Alpha"] == 37.5
    assert own_map["Bravo"] == 0.0

    bravo = result[result["player_name"] == "Bravo"].iloc[0]
    assert bravo["proj_fpts"] == 21.0
    assert bravo["player_id"] == "1002"
    assert bravo["team_outs_count"] == 0
    assert bravo["slate_id"] == "2025-12-01_0"


def test_select_main_draft_group_ids_from_meta_prefers_largest_classic_gpp():
    mod = _load_module()

    meta = pd.DataFrame(
        {
            "draft_group_id": [111, 111, 222, 333],
            "contest_name": [
                "NBA $200K Fadeaway [$50K to 1st]",
                "NBA $15K Mini Max",
                "NBA Showdown $40K And-One",
                "NBA $25K Night Owl",
            ],
            "game_type": ["Classic", "Classic", "Classic", "Classic"],
            "prize_pool": [200000, 15000, 40000, 25000],
            "current_entries": [13000, 4000, 5000, 6000],
            "max_entries": [150, 20, 20, 150],
        }
    )

    assert mod._select_main_draft_group_ids_from_meta(meta) == {"111"}
