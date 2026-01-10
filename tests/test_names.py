from __future__ import annotations

from projections.names import normalize_player_name


def test_normalize_player_name_handles_initials_and_apostrophes() -> None:
    assert normalize_player_name("R.J. Barrett") == "rj barrett"
    assert normalize_player_name("D'Angelo Russell") == "dangelo russell"


def test_normalize_player_name_aliases_gg_jackson_ii() -> None:
    assert normalize_player_name("GG Jackson II") == "gg jackson"
    assert normalize_player_name("G.G. Jackson II") == "gg jackson"
    assert normalize_player_name("Gregory Jackson II") == "gg jackson"
