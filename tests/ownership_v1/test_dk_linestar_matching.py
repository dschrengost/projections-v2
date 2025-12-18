"""Tests for slate-aware DK↔LineStar matching in build_ownership_dk_base."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "ownership" / "build_ownership_dk_base.py"
    spec = importlib.util.spec_from_file_location("build_ownership_dk_base", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_match_handles_date_drift_plus_one_day():
    mod = _load_module()

    dk = pd.DataFrame(
        {
            "slate_id": ["2025-01-01_0"] * 4,
            "game_date": ["2025-01-01"] * 4,
            "player_name_norm": ["a", "b", "c", "d"],
        }
    )
    linestar = pd.DataFrame(
        {
            "slate_id": ["ls_2025-01-02_main"] * 5,
            "game_date": ["2025-01-02"] * 5,
            "player_name_norm": ["a", "b", "c", "d", "e"],
        }
    )

    out, matches = mod.match_dk_slates_to_linestar(
        dk,
        linestar,
        max_day_offset=1,
        min_overlap_coeff=0.90,
        min_intersection=3,
    )

    assert len(matches) == 1
    assert matches[0].linestar_slate_id == "ls_2025-01-02_main"
    assert matches[0].date_offset_days == 1
    assert out["linestar_slate_id"].nunique() == 1
    assert out["linestar_slate_id"].iloc[0] == "ls_2025-01-02_main"


def test_match_disambiguates_multi_slate_dates_by_overlap():
    mod = _load_module()

    dk = pd.DataFrame(
        {
            "slate_id": ["dk_a"] * 4 + ["dk_b"] * 3,
            "game_date": ["2025-01-01"] * 7,
            "player_name_norm": ["a", "b", "c", "d", "x", "y", "z"],
        }
    )
    linestar = pd.DataFrame(
        {
            "slate_id": ["ls_a"] * 5 + ["ls_b"] * 4,
            "game_date": ["2025-01-01"] * 9,
            "player_name_norm": ["a", "b", "c", "d", "e", "x", "y", "z", "w"],
        }
    )

    out, matches = mod.match_dk_slates_to_linestar(
        dk,
        linestar,
        max_day_offset=0,
        min_overlap_coeff=0.80,
        min_intersection=2,
    )
    mapping = {m.dk_slate_id: m.linestar_slate_id for m in matches}
    assert mapping["dk_a"] == "ls_a"
    assert mapping["dk_b"] == "ls_b"

    out_map = out.groupby("slate_id")["linestar_slate_id"].first().to_dict()
    assert out_map["dk_a"] == "ls_a"
    assert out_map["dk_b"] == "ls_b"


def test_match_respects_min_overlap_threshold():
    mod = _load_module()

    dk = pd.DataFrame(
        {
            "slate_id": ["dk"] * 5,
            "game_date": ["2025-01-01"] * 5,
            "player_name_norm": ["a", "b", "c", "d", "e"],
        }
    )
    linestar = pd.DataFrame(
        {
            "slate_id": ["ls"] * 5,
            "game_date": ["2025-01-01"] * 5,
            "player_name_norm": ["a", "b", "x", "y", "z"],
        }
    )

    out, matches = mod.match_dk_slates_to_linestar(
        dk,
        linestar,
        max_day_offset=0,
        min_overlap_coeff=0.80,  # requires >= 80% overlap on the smaller set; we only have 2/5
        min_intersection=1,
    )
    assert len(matches) == 1
    assert matches[0].linestar_slate_id is None
    assert out["linestar_slate_id"].isna().all()
