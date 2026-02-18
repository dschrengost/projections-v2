from __future__ import annotations

import json

import pandas as pd

from projections.features.action_props import (
    ACTION_MARKET_FEATURE_COLUMNS,
    attach_action_props_features,
    build_action_props_feature_snapshots,
    load_action_props_long_from_bronze,
)


def test_load_action_props_long_from_bronze_parses_points_market(tmp_path) -> None:
    day = "2025-01-01"
    payload = {
        "game_id": 123456,
        "teams": ["NY", "BOS"],
        "away_team_id": 1,
        "home_team_id": 2,
        "fetched_at": "2025-01-01T20:00:00Z",
        "props": {
            "players": {
                "10": {
                    "full_name": "Jalen Brunson",
                    "display_text": "NY - PG",
                    "team_id": 1,
                }
            },
            "player_props": {
                "points": [
                    {
                        "player_id": "10",
                        "custom_pick_type_name": "Points",
                        "lines": {
                            "15": [
                                {"period": "event", "side": "over", "odds": -110, "value": 25.5},
                                {"period": "event", "side": "under", "odds": -110, "value": 25.5},
                            ]
                        },
                    }
                ]
            },
        },
    }
    (tmp_path / f"{day}_123456_NY_BOS.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    out = load_action_props_long_from_bronze(
        props_dir=tmp_path,
        game_date=pd.Timestamp(day),
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert row["team_tricode"] == "NYK"
    assert row["player_name_norm"] == "jalen brunson"
    assert row["prop_key"] == "pts"
    assert float(row["line"]) == 25.5
    assert abs(float(row["p_over"]) - 0.5) < 1e-6


def test_load_action_props_long_from_bronze_clamps_stale_fetched_at(tmp_path) -> None:
    day = "2025-01-01"
    payload = {
        "game_id": 123456,
        "teams": ["NY", "BOS"],
        "away_team_id": 1,
        "home_team_id": 2,
        "fetched_at": "2026-02-18T20:00:00Z",
        "props": {
            "players": {
                "10": {
                    "full_name": "Jalen Brunson",
                    "display_text": "NY - PG",
                    "team_id": 1,
                }
            },
            "player_props": {
                "points": [
                    {
                        "player_id": "10",
                        "custom_pick_type_name": "Points",
                        "lines": {
                            "15": [
                                {"period": "event", "side": "over", "odds": -110, "value": 25.5},
                                {"period": "event", "side": "under", "odds": -110, "value": 25.5},
                            ]
                        },
                    }
                ]
            },
        },
    }
    (tmp_path / f"{day}_123456_NY_BOS.json").write_text(json.dumps(payload), encoding="utf-8")

    out = load_action_props_long_from_bronze(
        props_dir=tmp_path,
        game_date=pd.Timestamp(day),
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert pd.Timestamp(row["action_props_as_of_ts"]) == pd.Timestamp("2025-01-01T00:00:00Z")


def test_build_action_props_feature_snapshots_pivots_markets() -> None:
    long_df = pd.DataFrame(
        {
            "game_date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
            "team_tricode": ["NYK", "NYK"],
            "player_name": ["Jalen Brunson", "Jalen Brunson"],
            "player_name_norm": ["jalen brunson", "jalen brunson"],
            "prop_key": ["pts", "ast"],
            "line": [25.5, 7.5],
            "p_over": [0.51, 0.49],
            "line_std": [0.3, 0.2],
            "books": [4.0, 3.0],
            "action_props_as_of_ts": [
                pd.Timestamp("2025-01-01T20:00:00Z"),
                pd.Timestamp("2025-01-01T20:00:00Z"),
            ],
            "action_game_id": [123456, 123456],
            "source_file": ["a.json", "a.json"],
        }
    )

    out = build_action_props_feature_snapshots(long_df)
    assert len(out) == 1
    row = out.iloc[0]
    assert int(row["an_has_any_props"]) == 1
    assert int(row["an_has_pts"]) == 1
    assert int(row["an_has_ast"]) == 1
    assert float(row["an_pts_line"]) == 25.5
    assert float(row["an_ast_line"]) == 7.5
    assert float(row["an_props_market_count"]) == 2
    assert "an_reb_line" in out.columns
    assert float(row["an_reb_line"]) == 0.0


def test_attach_action_props_features_uses_latest_valid_asof() -> None:
    base = pd.DataFrame(
        {
            "game_date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
            "team_tricode": ["NYK", "BOS"],
            "player_name": ["Jalen Brunson", "Jaylen Brown"],
            "tip_ts": [
                pd.Timestamp("2025-01-01T22:00:00Z"),
                pd.Timestamp("2025-01-01T22:00:00Z"),
            ],
            "feature_as_of_ts": [
                pd.Timestamp("2025-01-01T21:00:00Z"),
                pd.Timestamp("2025-01-01T21:00:00Z"),
            ],
        }
    )

    long_df = pd.DataFrame(
        {
            "game_date": [
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-01-01"),
            ],
            "team_tricode": ["NYK", "NYK"],
            "player_name": ["Jalen Brunson", "Jalen Brunson"],
            "player_name_norm": ["jalen brunson", "jalen brunson"],
            "prop_key": ["pts", "pts"],
            "line": [24.5, 26.5],
            "p_over": [0.52, 0.5],
            "line_std": [0.1, 0.1],
            "books": [3.0, 5.0],
            "action_props_as_of_ts": [
                pd.Timestamp("2025-01-01T20:00:00Z"),  # valid
                pd.Timestamp("2025-01-01T21:30:00Z"),  # after feature_as_of_ts, should be dropped
            ],
            "action_game_id": [123456, 123456],
            "source_file": ["a.json", "b.json"],
        }
    )
    snapshots = build_action_props_feature_snapshots(long_df)

    out = attach_action_props_features(base, snapshots, strict_asof=True)
    assert len(out) == 2
    first = out.iloc[0]
    second = out.iloc[1]

    assert int(first["an_has_any_props"]) == 1
    assert float(first["an_pts_line"]) == 24.5
    assert pd.Timestamp(first["action_props_as_of_ts"]) == pd.Timestamp("2025-01-01T20:00:00Z")

    assert int(second["an_has_any_props"]) == 0
    for col in ACTION_MARKET_FEATURE_COLUMNS:
        assert col in out.columns


def test_attach_action_props_features_supports_next_day_snapshot_with_strict_asof() -> None:
    base = pd.DataFrame(
        {
            "game_date": [pd.Timestamp("2025-01-01")],
            "team_tricode": ["NYK"],
            "player_name": ["Jalen Brunson"],
            "tip_ts": [pd.Timestamp("2025-01-02T02:00:00Z")],
            "feature_as_of_ts": [pd.Timestamp("2025-01-02T01:00:00Z")],
        }
    )

    long_df = pd.DataFrame(
        {
            "game_date": [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-02")],
            "team_tricode": ["NYK", "NYK"],
            "player_name": ["Jalen Brunson", "Jalen Brunson"],
            "player_name_norm": ["jalen brunson", "jalen brunson"],
            "prop_key": ["pts", "pts"],
            "line": [24.5, 25.5],
            "p_over": [0.52, 0.49],
            "line_std": [0.1, 0.1],
            "books": [3.0, 5.0],
            "action_props_as_of_ts": [
                pd.Timestamp("2025-01-02T00:30:00Z"),  # valid (<= feature_as_of_ts)
                pd.Timestamp("2025-01-02T01:30:00Z"),  # invalid (> feature_as_of_ts)
            ],
            "action_game_id": [123456, 123456],
            "source_file": ["a.json", "b.json"],
        }
    )
    snapshots = build_action_props_feature_snapshots(long_df)

    out = attach_action_props_features(
        base,
        snapshots,
        strict_asof=True,
        game_date_offsets=(0, -1),
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert int(row["an_has_any_props"]) == 1
    assert float(row["an_pts_line"]) == 24.5
    assert pd.Timestamp(row["action_props_as_of_ts"]) == pd.Timestamp("2025-01-02T00:30:00Z")


def test_attach_action_props_features_can_clamp_late_asof_to_game_date() -> None:
    base = pd.DataFrame(
        {
            "game_date": [pd.Timestamp("2025-01-01")],
            "team_tricode": ["NYK"],
            "player_name": ["Jalen Brunson"],
            "tip_ts": [pd.Timestamp("2025-01-01T22:00:00Z")],
            "feature_as_of_ts": [pd.Timestamp("2025-01-01T21:00:00Z")],
        }
    )
    long_df = pd.DataFrame(
        {
            "game_date": [pd.Timestamp("2025-01-01")],
            "team_tricode": ["NYK"],
            "player_name": ["Jalen Brunson"],
            "player_name_norm": ["jalen brunson"],
            "prop_key": ["pts"],
            "line": [24.5],
            "p_over": [0.52],
            "line_std": [0.1],
            "books": [3.0],
            "action_props_as_of_ts": [pd.Timestamp("2026-02-18T20:00:00Z")],
            "action_game_id": [123456],
            "source_file": ["a.json"],
        }
    )
    snapshots = build_action_props_feature_snapshots(long_df)

    out = attach_action_props_features(
        base,
        snapshots,
        strict_asof=True,
        clamp_late_asof_to_game_date=True,
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert int(row["an_has_any_props"]) == 1
    assert pd.Timestamp(row["action_props_as_of_ts"]) == pd.Timestamp("2025-01-01T00:00:00Z")


def test_attach_action_props_features_overwrites_existing_columns() -> None:
    base = pd.DataFrame(
        {
            "game_date": [pd.Timestamp("2025-01-01")],
            "team_tricode": ["NYK"],
            "player_name": ["Jalen Brunson"],
            "tip_ts": [pd.Timestamp("2025-01-01T22:00:00Z")],
            "feature_as_of_ts": [pd.Timestamp("2025-01-01T21:00:00Z")],
            "an_has_any_props": [0],
            "an_pts_line": [0.0],
        }
    )
    long_df = pd.DataFrame(
        {
            "game_date": [pd.Timestamp("2025-01-01")],
            "team_tricode": ["NYK"],
            "player_name": ["Jalen Brunson"],
            "player_name_norm": ["jalen brunson"],
            "prop_key": ["pts"],
            "line": [24.5],
            "p_over": [0.52],
            "line_std": [0.1],
            "books": [3.0],
            "action_props_as_of_ts": [pd.Timestamp("2025-01-01T20:00:00Z")],
            "action_game_id": [123456],
            "source_file": ["a.json"],
        }
    )
    snapshots = build_action_props_feature_snapshots(long_df)
    out = attach_action_props_features(base, snapshots, strict_asof=True)

    row = out.iloc[0]
    assert int(row["an_has_any_props"]) == 1
    assert float(row["an_pts_line"]) == 24.5
