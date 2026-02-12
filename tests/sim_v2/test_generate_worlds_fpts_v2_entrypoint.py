from __future__ import annotations

import inspect

import pytest

from scripts.sim_v2 import generate_worlds_fpts_v2 as module


def test_generate_worlds_main_accepts_game_id_option() -> None:
    assert "game_id" in inspect.signature(module.main).parameters


def test_generate_worlds_main_unwraps_typer_option_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _fake_load_sim_v2_profile(*, profile: str, profiles_path: object):
        seen["profile"] = profile
        seen["profiles_path"] = profiles_path
        raise RuntimeError("stop_after_profile_load")

    monkeypatch.setattr(module, "load_sim_v2_profile", _fake_load_sim_v2_profile)

    with pytest.raises(RuntimeError, match="stop_after_profile_load"):
        module.main(start_date="2026-02-12", end_date="2026-02-12")

    assert seen["profile"] == "baseline"
    assert seen["profiles_path"] is None
