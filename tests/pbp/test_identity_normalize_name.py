from __future__ import annotations

from projections.pbp.identity import normalize_name


def test_normalize_name_basic() -> None:
    assert normalize_name("Karl-Anthony Towns") == "karl anthony towns"
    assert normalize_name("  Jayson   Tatum  ") == "jayson tatum"


def test_normalize_name_accents_and_punct() -> None:
    assert normalize_name("Nikola Jokić") == "nikola jokic"
    assert normalize_name("O.G. Anunoby") == "o g anunoby"


def test_normalize_name_suffixes_removed() -> None:
    assert normalize_name("Gary Payton II") == "gary payton"
    assert normalize_name("Kenyon Martin Jr.") == "kenyon martin"

