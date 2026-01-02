from __future__ import annotations

import json
from pathlib import Path

import pytest

from projections.sim_v2.config import load_sim_v2_profile


def test_load_sim_v2_profile_parses_contest_sim_rank_mode(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(
        json.dumps(
            {
                "profiles": {
                    "p": {
                        "mean_source": "rates",
                        "contest_sim": {"rank_mode": "combined_sort"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    profile_cfg = load_sim_v2_profile(profile="p", profiles_path=profiles_path)
    assert profile_cfg.contest_sim_rank_mode == "combined_sort"


def test_load_sim_v2_profile_contest_sim_rank_mode_defaults_to_none(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(
        json.dumps({"profiles": {"p": {"mean_source": "rates"}}}),
        encoding="utf-8",
    )

    profile_cfg = load_sim_v2_profile(profile="p", profiles_path=profiles_path)
    assert profile_cfg.contest_sim_rank_mode is None


def test_load_sim_v2_profile_contest_sim_rank_mode_rejects_invalid(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(
        json.dumps(
            {
                "profiles": {
                    "p": {
                        "mean_source": "rates",
                        "contest_sim": {"rank_mode": "nope"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contest_sim\\.rank_mode"):
        _ = load_sim_v2_profile(profile="p", profiles_path=profiles_path)
