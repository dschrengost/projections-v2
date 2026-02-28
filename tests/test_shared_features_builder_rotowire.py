from datetime import date

import pandas as pd

from projections.builders.features_builder import FeatureBuildConfig, SharedFeaturesBuilder


def test_rotowire_starters_override_lineup_signals(tmp_path) -> None:
    target_day = date(2025, 1, 1)
    rotowire_path = (
        tmp_path / "silver" / "rotowire_lineups" / f"date={target_day}" / "lineups.parquet"
    )
    rotowire_path.parent.mkdir(parents=True, exist_ok=True)
    rotowire_df = pd.DataFrame(
        {
            "player_name": ["Player B", "Player C"],
            "lineup_role": ["confirmed_starter", "projected_starter"],
            "ingested_ts": [pd.Timestamp("2025-01-01", tz="UTC")] * 2,
        }
    )
    rotowire_df.to_parquet(rotowire_path, index=False)

    roster = pd.DataFrame(
        {
            "player_name": ["Player A", "Player B", "Player C"],
            "active_flag": [True, True, True],
            "lineup_role": ["projected_starter"] * 3,
            "is_projected_starter": [True, True, True],
            "is_confirmed_starter": [False, True, False],
        }
    )

    cfg = FeatureBuildConfig(
        data_root=tmp_path,
        season=2025,
        as_of_ts=pd.Timestamp("2025-01-01", tz="UTC"),
        target_day=target_day,
    )
    builder = SharedFeaturesBuilder(cfg)
    merged = builder._load_and_merge_rotowire_starters(roster, warnings=[])

    by_name = merged.set_index("player_name")
    assert not by_name.loc["Player A", "is_projected_starter"]
    assert not by_name.loc["Player A", "is_confirmed_starter"]
    assert pd.isna(by_name.loc["Player A", "lineup_role"])
    assert by_name.loc["Player B", "is_projected_starter"]
    assert by_name.loc["Player B", "is_confirmed_starter"]
    assert by_name.loc["Player B", "lineup_role"] == "confirmed_starter"
    assert by_name.loc["Player C", "is_projected_starter"]
    assert not by_name.loc["Player C", "is_confirmed_starter"]
    assert by_name.loc["Player C", "lineup_role"] == "projected_starter"


def test_rotowire_out_rows_clear_starter_flags(tmp_path) -> None:
    target_day = date(2025, 1, 1)
    rotowire_path = (
        tmp_path / "silver" / "rotowire_lineups" / f"date={target_day}" / "lineups.parquet"
    )
    rotowire_path.parent.mkdir(parents=True, exist_ok=True)
    rotowire_df = pd.DataFrame(
        {
            "player_name": ["Player A", "Player B"],
            "lineup_role": ["out", "confirmed_starter"],
            "ingested_ts": [pd.Timestamp("2025-01-01", tz="UTC")] * 2,
        }
    )
    rotowire_df.to_parquet(rotowire_path, index=False)

    roster = pd.DataFrame(
        {
            "player_name": ["Player A", "Player B"],
            "active_flag": [True, True],
            "lineup_role": ["projected_starter", pd.NA],
            "is_projected_starter": [True, False],
            "is_confirmed_starter": [True, False],
        }
    )

    cfg = FeatureBuildConfig(
        data_root=tmp_path,
        season=2025,
        as_of_ts=pd.Timestamp("2025-01-01", tz="UTC"),
        target_day=target_day,
    )
    builder = SharedFeaturesBuilder(cfg)
    merged = builder._load_and_merge_rotowire_starters(roster, warnings=[])

    by_name = merged.set_index("player_name")
    assert not by_name.loc["Player A", "is_projected_starter"]
    assert not by_name.loc["Player A", "is_confirmed_starter"]
    assert by_name.loc["Player A", "lineup_role"] == "out"
    assert by_name.loc["Player B", "is_projected_starter"]
    assert by_name.loc["Player B", "is_confirmed_starter"]
    assert by_name.loc["Player B", "lineup_role"] == "confirmed_starter"
