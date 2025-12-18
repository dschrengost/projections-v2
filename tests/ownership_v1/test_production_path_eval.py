"""Tests for production-path ownership evaluator helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "ownership" / "evaluate_ownership_production_path.py"
    spec = importlib.util.spec_from_file_location("evaluate_ownership_production_path", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_normalize_name_strips_suffix_tokens():
    mod = _load_module()
    assert mod.normalize_name("Gary Trent Jr.", strip_suffix=False) != mod.normalize_name("Gary Trent Jr.", strip_suffix=True)
    assert mod.normalize_name("Gary Trent Jr.", strip_suffix=True) == mod.normalize_name("Gary Trent", strip_suffix=True)


def test_match_slates_by_overlap_maps_correct_draft_group():
    mod = _load_module()

    actual = pd.DataFrame(
        {
            "slate_id": ["s0"] * 4 + ["s1"] * 3,
            "game_date": ["2025-01-01"] * 7,
            "player_name_norm": ["a", "b", "c", "d", "x", "y", "z"],
        }
    )
    preds = pd.DataFrame(
        {
            "draft_group_id": ["dg0"] * 5 + ["dg1"] * 4,
            "game_date": ["2025-01-01"] * 9,
            "player_name_norm": ["a", "b", "c", "d", "e", "x", "y", "z", "w"],
        }
    )

    out, matches = mod.match_slates_by_player_overlap(
        actual,
        preds,
        source_slate_id_col="slate_id",
        target_slate_id_col="draft_group_id",
        max_day_offset=0,
        min_overlap_coeff=0.80,
        min_intersection=2,
    )

    mapping = out.groupby("slate_id", sort=False)["draft_group_id"].first().to_dict()
    assert mapping["s0"] == "dg0"
    assert mapping["s1"] == "dg1"
    assert len(matches) == 2


def test_match_slates_by_overlap_enforces_one_to_one():
    mod = _load_module()

    # Both source slates overlap the same target; s_good should win by intersection.
    actual = pd.DataFrame(
        {
            "slate_id": ["s_good"] * 5 + ["s_bad"] * 3,
            "game_date": ["2025-01-01"] * 8,
            "player_name_norm": ["a", "b", "c", "d", "e", "a", "b", "c"],
        }
    )
    preds = pd.DataFrame(
        {
            "draft_group_id": ["dg0"] * 5,
            "game_date": ["2025-01-01"] * 5,
            "player_name_norm": ["a", "b", "c", "d", "e"],
        }
    )

    out, _ = mod.match_slates_by_player_overlap(
        actual,
        preds,
        source_slate_id_col="slate_id",
        target_slate_id_col="draft_group_id",
        max_day_offset=0,
        min_overlap_coeff=0.60,
        min_intersection=2,
    )

    mapping = out.groupby("slate_id", sort=False)["draft_group_id"].first().to_dict()
    assert mapping["s_good"] == "dg0"
    assert pd.isna(mapping["s_bad"])

