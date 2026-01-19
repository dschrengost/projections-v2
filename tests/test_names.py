from __future__ import annotations

from projections.names import normalize_player_name


def test_normalize_player_name_handles_initials_and_apostrophes() -> None:
    assert normalize_player_name("R.J. Barrett") == "rj barrett"
    assert normalize_player_name("D'Angelo Russell") == "dangelo russell"


def test_normalize_player_name_aliases_gg_jackson_ii() -> None:
    assert normalize_player_name("GG Jackson II") == "gg jackson"
    assert normalize_player_name("G.G. Jackson II") == "gg jackson"
    assert normalize_player_name("Gregory Jackson II") == "gg jackson"


def test_normalize_player_name_strips_suffixes() -> None:
    """Generational and honorific suffixes should be stripped for matching."""
    # III suffix
    assert normalize_player_name("Jimmy Butler III") == "jimmy butler"
    assert normalize_player_name("Jimmy Butler") == "jimmy butler"
    # Jr suffix
    assert normalize_player_name("Tim Hardaway Jr") == "tim hardaway"
    assert normalize_player_name("Tim Hardaway Jr.") == "tim hardaway"
    assert normalize_player_name("Jaren Jackson Jr") == "jaren jackson"
    # II suffix
    assert normalize_player_name("Gary Payton II") == "gary payton"
    # IV suffix
    assert normalize_player_name("Wendell Carter IV") == "wendell carter"
    # Sr suffix (rare but possible)
    assert normalize_player_name("John Smith Sr") == "john smith"


def test_normalize_player_name_both_sources_match() -> None:
    """Internal naming with suffix should match external naming without suffix."""
    internal = normalize_player_name("Jimmy Butler III")
    external = normalize_player_name("Jimmy Butler")
    assert internal == external == "jimmy butler"

    internal = normalize_player_name("Gary Payton II")
    external = normalize_player_name("Gary Payton")
    assert internal == external == "gary payton"


def test_normalize_player_name_handles_accented_characters() -> None:
    """European player names with diacritics should match ASCII equivalents."""
    # Serbian/Croatian ć
    assert normalize_player_name("Nikola Jokić") == "nikola jokic"
    assert normalize_player_name("Nikola Jokić") == normalize_player_name("Nikola Jokic")

    # Slovenian č and ć
    assert normalize_player_name("Luka Dončić") == "luka doncic"
    assert normalize_player_name("Luka Dončić") == normalize_player_name("Luka Doncic")

    # Lithuanian ū and č
    assert normalize_player_name("Jonas Valančiūnas") == "jonas valanciunas"

    # Latvian ņ and ģ
    assert normalize_player_name("Kristaps Porziņģis") == "kristaps porzingis"

    # Serbian ć
    assert normalize_player_name("Bogdan Bogdanović") == "bogdan bogdanovic"
