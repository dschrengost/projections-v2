from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.rotation.live_features_v1 import (
    _refresh_latest_player_priors_from_history,
    _refresh_latest_team_priors_from_history,
    _update_player_priors_with_latest_labels,
    load_latest_rotation_priors_by_entity,
)


def test_load_latest_rotation_priors_skips_suspicious_large_partitions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    season = 2025
    team_root = (
        tmp_path
        / "silver"
        / "rotation_priors_v1"
        / "team_game_priors"
        / f"season={season}"
    )
    player_root = (
        tmp_path
        / "silver"
        / "rotation_priors_v1"
        / "player_game_priors"
        / f"season={season}"
    )
    team_root.mkdir(parents=True, exist_ok=True)
    player_root.mkdir(parents=True, exist_ok=True)

    # Baseline valid partitions.
    pd.DataFrame(
        {
            "game_date": ["2026-03-01"],
            "team_id": [1610612760],
            "marker": [1],
        }
    ).to_parquet(team_root / "game_id=0022500001.parquet", index=False)
    pd.DataFrame(
        {
            "game_date": ["2026-03-01"],
            "person_id": [1642349],
            "marker": [1],
            "game_id": ["0022500001"],
        }
    ).to_parquet(player_root / "game_id=0022500001.parquet", index=False)

    # Extra partition that will be monkeypatched to an absurdly large row count.
    pd.DataFrame({"dummy": [1]}).to_parquet(team_root / "game_id=0022500002.parquet", index=False)
    pd.DataFrame({"dummy": [1]}).to_parquet(player_root / "game_id=0022500002.parquet", index=False)

    original_read_parquet = pd.read_parquet

    def _fake_read_parquet(path, *args, **kwargs):  # noqa: ANN001
        text = str(path)
        name = Path(path).name
        if name == "game_id=0022500002.parquet" and "team_game_priors" in text:
            return pd.DataFrame(
                {
                    "game_date": ["2026-03-04"] * 6001,
                    "team_id": [1610612760] * 6001,
                    "marker": [999] * 6001,
                }
            )
        if name == "game_id=0022500002.parquet" and "player_game_priors" in text:
            return pd.DataFrame(
                {
                    "game_date": ["2026-03-04"] * 6001,
                    "person_id": [1642349] * 6001,
                    "marker": [999] * 6001,
                    "game_id": ["0022500002"] * 6001,
                }
            )
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _fake_read_parquet)

    team_priors, player_priors = load_latest_rotation_priors_by_entity(
        tmp_path,
        season=season,
        team_ids=[1610612760],
        player_ids=[1642349],
    )

    assert len(team_priors) == 1
    assert len(player_priors) == 1
    # If suspicious partitions are not skipped, marker would be 999 from newer date.
    assert int(team_priors.iloc[0]["marker"]) == 1
    assert int(player_priors.iloc[0]["marker"]) == 1


def test_update_player_priors_with_latest_labels_updates_windows_and_counts(
    tmp_path: Path,
) -> None:
    season = 2025
    labels_path = tmp_path / "labels" / f"season={season}" / "boxscore_labels.parquet"
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"player_id": 999, "game_id": 22501001, "game_date": "2026-03-10", "minutes": 20.0, "starter_flag_label": 0},
            {"player_id": 999, "game_id": 22501002, "game_date": "2026-03-11", "minutes": 30.0, "starter_flag_label": 0},
            {"player_id": 999, "game_id": 22501003, "game_date": "2026-03-12", "minutes": 40.0, "starter_flag_label": 1},
            {"player_id": 999, "game_id": 22501004, "game_date": "2026-03-13", "minutes": 50.0, "starter_flag_label": 1},
            {"player_id": 999, "game_id": 22501005, "game_date": "2026-03-14", "minutes": 60.0, "starter_flag_label": 1},
            {"player_id": 999, "game_id": 22501006, "game_date": "2026-03-15", "minutes": 70.0, "starter_flag_label": 1},
        ]
    ).to_parquet(labels_path, index=False)

    player_priors = pd.DataFrame(
        [
            {
                "person_id": 999,
                "game_id": "0022501043",
                "started_proxy_rate_prior_5": 0.0,
                "started_proxy_rate_prior_10": 0.0,
                "started_proxy_rate_prior_20": 0.0,
                "minutes_from_stints_prior_5": 0.0,
                "minutes_from_stints_prior_10": 0.0,
                "minutes_from_stints_prior_20": 0.0,
                "minutes_from_stints_std_prior_5": 0.0,
                "minutes_from_stints_std_prior_10": 0.0,
                "minutes_from_stints_std_prior_20": 0.0,
                "player_prior_n_games_5": 0,
                "player_prior_n_games_10": 0,
                "player_prior_n_games_20": 0,
                "player_prior_source_max_game_date_5": pd.NaT,
                "player_prior_source_max_game_date_10": pd.NaT,
                "player_prior_source_max_game_date_20": pd.NaT,
            }
        ]
    )

    updated = _update_player_priors_with_latest_labels(player_priors, tmp_path, season)
    row = updated.iloc[0]

    # Latest 5 games minutes are [70, 60, 50, 40, 30]
    assert float(row["minutes_from_stints_prior_5"]) == 50.0
    assert round(float(row["minutes_from_stints_std_prior_5"]), 6) == round(
        pd.Series([70.0, 60.0, 50.0, 40.0, 30.0]).std(ddof=0), 6
    )
    # Windows larger than available labels should use all available games.
    assert float(row["minutes_from_stints_prior_10"]) == 45.0
    assert float(row["minutes_from_stints_prior_20"]) == 45.0

    # Starter labels latest 5 are [1, 1, 1, 1, 0] => 0.8
    assert round(float(row["started_proxy_rate_prior_5"]), 6) == 0.8
    # Across all 6 games starter mean is 4/6.
    assert round(float(row["started_proxy_rate_prior_10"]), 6) == round(4.0 / 6.0, 6)
    assert round(float(row["started_proxy_rate_prior_20"]), 6) == round(4.0 / 6.0, 6)

    assert int(row["player_prior_n_games_5"]) == 5
    assert int(row["player_prior_n_games_10"]) == 6
    assert int(row["player_prior_n_games_20"]) == 6

    expected_latest = pd.Timestamp("2026-03-15")
    assert pd.Timestamp(row["player_prior_source_max_game_date_5"]) == expected_latest
    assert pd.Timestamp(row["player_prior_source_max_game_date_10"]) == expected_latest
    assert pd.Timestamp(row["player_prior_source_max_game_date_20"]) == expected_latest


def test_refresh_latest_player_priors_from_history_updates_context_buckets() -> None:
    player_history = pd.DataFrame(
        [
            {
                "person_id": 999,
                "game_id": "0022501001",
                "game_date": "2026-03-10",
                "minutes_from_stints": 12.0,
                "started_proxy": 0,
                "ctx_same_pos_bucket": "deep",
                "fg3_pct": 0.25,
                "three_pa_share": 0.20,
            },
            {
                "person_id": 999,
                "game_id": "0022501002",
                "game_date": "2026-03-11",
                "minutes_from_stints": 30.0,
                "started_proxy": 1,
                "ctx_same_pos_bucket": "thin",
                "fg3_pct": 0.40,
                "three_pa_share": 0.45,
            },
            {
                "person_id": 999,
                "game_id": "0022501003",
                "game_date": "2026-03-12",
                "minutes_from_stints": 28.0,
                "started_proxy": 1,
                "ctx_same_pos_bucket": "thin",
                "fg3_pct": 0.50,
                "three_pa_share": 0.50,
            },
        ]
    )

    refreshed = _refresh_latest_player_priors_from_history(player_history)
    row = refreshed.iloc[0]

    assert float(row["minutes_from_stints_prior_5"]) == 70.0 / 3.0
    assert round(float(row["started_proxy_rate_prior_5"]), 6) == round(2.0 / 3.0, 6)
    assert int(row["player_prior_n_games_5"]) == 3

    assert int(row["ctx_same_pos_thin_prior_n_games_5"]) == 2
    assert float(row["minutes_from_stints_ctx_same_pos_thin_prior_5"]) == 29.0
    assert float(row["started_proxy_rate_ctx_same_pos_thin_prior_5"]) == 1.0
    assert int(row["ctx_same_pos_deep_prior_n_games_5"]) == 1
    assert float(row["minutes_from_stints_ctx_same_pos_deep_prior_5"]) == 12.0
    assert round(float(row["fg3_pct_prior_5"]), 6) == round((0.25 + 0.40 + 0.50) / 3.0, 6)
    assert round(float(row["three_pa_share_prior_5"]), 6) == round((0.20 + 0.45 + 0.50) / 3.0, 6)


def test_refresh_latest_team_priors_from_history_updates_allowed_priors() -> None:
    team_history = pd.DataFrame(
        [
            {
                "team_id": 10,
                "game_id": "0022501001",
                "game_date": "2026-03-10",
                "depth_6": 110.0,
                "depth_gap_10_6": 15.0,
                "fg2_pct_allowed": 0.48,
                "fg3_pct_allowed": 0.36,
                "fta_rate_allowed": 0.20,
                "efg_pct_allowed": 0.55,
                "three_pa_share_allowed": 0.42,
                "team_ot_flag": 0,
            },
            {
                "team_id": 10,
                "game_id": "0022501002",
                "game_date": "2026-03-11",
                "depth_6": 112.0,
                "depth_gap_10_6": 16.0,
                "fg2_pct_allowed": 0.50,
                "fg3_pct_allowed": 0.34,
                "fta_rate_allowed": 0.22,
                "efg_pct_allowed": 0.54,
                "three_pa_share_allowed": 0.40,
                "team_ot_flag": 1,
            },
        ]
    )

    refreshed = _refresh_latest_team_priors_from_history(team_history)
    row = refreshed.iloc[0]

    assert int(row["team_prior_n_games_5"]) == 2
    assert round(float(row["fg2_pct_allowed_prior_5"]), 6) == 0.49
    assert round(float(row["fg3_pct_allowed_prior_5"]), 6) == 0.35
    assert round(float(row["fta_rate_allowed_prior_5"]), 6) == 0.21
    assert round(float(row["three_pa_share_allowed_prior_5"]), 6) == 0.41
    assert round(float(row["team_ot_rate_prior_5"]), 6) == 0.5
