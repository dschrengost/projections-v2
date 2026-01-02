import pandas as pd

from projections.api.optimizer_service import _normalize_name, _normalize_team


def test_normalize_name_strips_punctuation_and_accents() -> None:
    assert _normalize_name("R.J. Barrett") == "rj barrett"
    assert _normalize_name("D'Angelo Russell") == "dangelo russell"
    assert _normalize_name("Nikola Jokić") == "nikola jokic"


def test_normalize_team_maps_common_aliases() -> None:
    assert _normalize_team("PHO") == "PHX"
    assert _normalize_team(" GS ") == "GSW"
    assert _normalize_team("NY") == "NYK"
    assert _normalize_team("BRK") == "BKN"


def test_join_keys_match_between_projections_and_salaries() -> None:
    proj_df = pd.DataFrame(
        [
            {
                "player_id": 1631117,
                "player_name": "R.J. Barrett",
                "team_tricode": "NYK",
            }
        ]
    )
    sal_df = pd.DataFrame(
        [
            {
                "dk_player_id": 999,
                "display_name": "RJ Barrett",
                "team_abbrev": "NY",
            }
        ]
    )

    proj_df["__join_name"] = proj_df["player_name"].apply(_normalize_name)
    proj_df["__join_team"] = proj_df["team_tricode"].apply(_normalize_team)
    sal_df["__join_name"] = sal_df["display_name"].apply(_normalize_name)
    sal_df["__join_team"] = sal_df["team_abbrev"].apply(_normalize_team)

    merged = proj_df.merge(sal_df, on=["__join_name", "__join_team"], how="inner")
    assert len(merged) == 1

