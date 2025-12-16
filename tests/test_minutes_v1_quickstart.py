"""Quick-start unit tests covering snapshots and label freezing."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from projections.features import availability as availability_features
from projections.minutes_v1 import ensure_as_of_column, freeze_boxscore_labels, latest_pre_tip_snapshot
from projections.minutes_v1.constants import AvailabilityStatus, STATUS_PRIORS
from projections.minutes_v1.features import MinutesFeatureBuilder


def test_ensure_as_of_column_adds_missing_timestamp():
    df = pd.DataFrame({"game_id": [1]})
    ts = datetime(2024, 10, 1, tzinfo=timezone.utc)
    enriched = ensure_as_of_column(df, timestamp=ts)
    assert "as_of_ts" in enriched.columns
    assert enriched.loc[0, "as_of_ts"] == pd.Timestamp(ts)


def test_latest_pre_tip_snapshot_selects_latest_before_tip():
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "player_id": [10, 10, 10],
            "as_of_ts": [
                "2024-10-21T12:00:00Z",
                "2024-10-21T15:00:00Z",
                "2024-10-21T20:00:00Z",
            ],
            "tip_ts": ["2024-10-21T19:00:00Z"] * 3,
            "status": ["Q", "PROB", "OUT"],
        }
    )
    result = latest_pre_tip_snapshot(df, group_cols=["game_id", "player_id"], tip_ts_col="tip_ts")
    assert len(result) == 1
    assert result.iloc[0]["status"] == "PROB"


def test_freeze_boxscore_labels_writes_season_partitions(tmp_path: Path):
    df = pd.DataFrame(
        {
            "game_id": [1] * 5 + [2] * 5,
            "player_id": list(range(10)),
            "player_name": [f"Player {i}" for i in range(10)],
            "minutes": [30.0, 29.5, 29.0, 28.5, 28.0, 27.5, 27.0, 26.5, 26.0, 25.5],
            "starter_flag": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "team_id": [100] * 5 + [101] * 5,
            "season": ["2024-25"] * 10,
            "game_date": ["2024-10-21"] * 5 + ["2024-10-22"] * 5,
            "source": ["test"] * 10,
        }
    )
    written = freeze_boxscore_labels(df, tmp_path)
    assert "2024-25" in written
    assert written["2024-25"].exists()
    stored = pd.read_parquet(written["2024-25"])
    assert "starter_flag_label" in stored.columns
    starter_counts = (
        stored.groupby(["game_id", "team_id"])["starter_flag_label"].sum().unique().tolist()
    )
    assert starter_counts == [5]
    with pytest.raises(FileExistsError):
        freeze_boxscore_labels(df, tmp_path)


def test_minutes_feature_builder_produces_core_columns():
    schedule = pd.DataFrame(
        {
            "game_id": [101, 102],
            "season": ["2024-25", "2024-25"],
            "game_date": ["2024-10-21", "2024-10-23"],
            "tip_ts": ["2024-10-21T23:00:00Z", "2024-10-23T00:00:00Z"],
            "home_team_id": [1, 2],
            "away_team_id": [2, 1],
        }
    )
    injuries = pd.DataFrame(
        {
            "game_id": [101, 102],
            "player_id": [900, 900],
            "status": ["Probable", "Available"],
            "restriction_flag": [False, False],
            "ramp_flag": [False, False],
            "games_since_return": [3, 4],
            "days_since_return": [7, 9],
            "as_of_ts": ["2024-10-21T20:00:00Z", "2024-10-22T20:00:00Z"],
        }
    )
    odds = pd.DataFrame(
        {
            "game_id": [101, 102],
            "home_line": [-5.5, 4.0],
            "total": [225.5, 219.0],
            "as_of_ts": ["2024-10-21T21:00:00Z", "2024-10-22T21:00:00Z"],
        }
    )
    roster = pd.DataFrame(
        {
            "team_id": [1, 1, 1, 1],
            "game_date": ["2024-10-21", "2024-10-21", "2024-10-23", "2024-10-23"],
            "player_id": [900, 901, 900, 902],
            "active_flag": [True, True, True, True],
            "listed_pos": ["PG", "SG", "PG", "PF"],
            "as_of_ts": [
                "2024-10-20T12:00:00Z",
                "2024-10-20T12:00:00Z",
                "2024-10-22T12:00:00Z",
                "2024-10-22T12:00:00Z",
            ],
        }
    )
    coach = pd.DataFrame(
        {
            "coach_id": [1],
            "coach_name": ["Test Coach"],
            "team_id": [1],
            "start_date": ["2024-07-01"],
            "end_date": [None],
        }
    )
    labels = pd.DataFrame(
        {
            "game_id": [101, 102],
            "player_id": [900, 900],
            "team_id": [1, 1],
            "season": ["2024-25", "2024-25"],
            "game_date": ["2024-10-21", "2024-10-23"],
            "minutes": [32.0, 28.0],
            "starter_flag": [1, 1],
        }
    )

    builder = MinutesFeatureBuilder(
        schedule=schedule,
        injuries_snapshot=injuries,
        odds_snapshot=odds,
        roster_nightly=roster,
        coach_tenure=coach,
    )
    features = builder.build(labels)

    assert {"prior_play_prob", "recent_start_pct_10", "available_G", "feature_as_of_ts"}.issubset(features.columns)
    assert features["feature_as_of_ts"].le(features["tip_ts"]).all()
    expected_prior = STATUS_PRIORS[AvailabilityStatus.PROBABLE]
    assert features.loc[features["game_id"] == 101, "prior_play_prob"].iloc[0] == expected_prior
    assert features.loc[features["game_id"] == 101, "home_flag"].iloc[0] == 1
    assert features.loc[features["game_id"] == 102, "home_flag"].iloc[0] == 0
    assert features.loc[features["game_id"] == 101, "available_G"].iloc[0] == 2
    # Second game should have a recent start pct of 1.0 after two consecutive starts.
    assert features.loc[features["game_id"] == 102, "recent_start_pct_10"].iloc[0] == pytest.approx(1.0)
    assert "starter_prev_game_asof" in features.columns


def test_attach_availability_features_dedupes_duplicate_snapshots() -> None:
    base = pd.DataFrame(
        {
            "game_id": [101],
            "player_id": [900],
            "tip_ts": ["2024-10-21T23:00:00Z"],
        }
    )
    injuries = pd.DataFrame(
        {
            "game_id": [101, 101],
            "player_id": [900, 900],
            "status": ["Q", "OUT"],
            "restriction_flag": [False, False],
            "ramp_flag": [False, False],
            "games_since_return": [pd.NA, pd.NA],
            "days_since_return": [pd.NA, pd.NA],
            "as_of_ts": ["2024-10-21T20:00:00Z", "2024-10-21T22:30:00Z"],
        }
    )

    enriched = availability_features.attach_availability_features(base, injuries_snapshot=injuries)
    assert len(enriched) == len(base)
    assert enriched.loc[0, "status"] == AvailabilityStatus.OUT
    assert enriched.loc[0, "prior_play_prob"] == STATUS_PRIORS[AvailabilityStatus.OUT]
    assert enriched.loc[0, "injury_as_of_ts"] == pd.Timestamp("2024-10-21T22:30:00Z")


def test_minutes_feature_builder_history_not_corrupted_by_duplicate_odds() -> None:
    schedule = pd.DataFrame(
        {
            "game_id": [101, 102],
            "season": ["2024-25", "2024-25"],
            "game_date": ["2024-10-21", "2024-10-23"],
            "tip_ts": ["2024-10-21T23:00:00Z", "2024-10-23T00:00:00Z"],
            "home_team_id": [1, 2],
            "away_team_id": [2, 1],
        }
    )
    injuries = pd.DataFrame(
        {
            "game_id": [101, 102],
            "player_id": [900, 900],
            "status": ["Probable", "Available"],
            "restriction_flag": [False, False],
            "ramp_flag": [False, False],
            "games_since_return": [3, 4],
            "days_since_return": [7, 9],
            "as_of_ts": ["2024-10-21T20:00:00Z", "2024-10-22T20:00:00Z"],
        }
    )
    odds = pd.DataFrame(
        {
            "game_id": [101, 101, 102, 102],
            "home_line": [-5.5, -5.5, 4.0, 4.0],
            "total": [225.5, 225.5, 219.0, 219.0],
            "as_of_ts": [
                "2024-10-21T21:00:00Z",
                "2024-10-21T21:00:00Z",
                "2024-10-22T21:00:00Z",
                "2024-10-22T21:00:00Z",
            ],
        }
    )
    roster = pd.DataFrame(
        {
            "team_id": [1, 1, 1, 1],
            "game_date": ["2024-10-21", "2024-10-21", "2024-10-23", "2024-10-23"],
            "player_id": [900, 901, 900, 902],
            "active_flag": [True, True, True, True],
            "listed_pos": ["PG", "SG", "PG", "PF"],
            "as_of_ts": [
                "2024-10-20T12:00:00Z",
                "2024-10-20T12:00:00Z",
                "2024-10-22T12:00:00Z",
                "2024-10-22T12:00:00Z",
            ],
        }
    )
    labels = pd.DataFrame(
        {
            "game_id": [101, 102],
            "player_id": [900, 900],
            "team_id": [1, 1],
            "season": ["2024-25", "2024-25"],
            "game_date": ["2024-10-21", "2024-10-23"],
            "minutes": [32.0, 28.0],
            "starter_flag": [1, 1],
        }
    )

    builder = MinutesFeatureBuilder(
        schedule=schedule,
        injuries_snapshot=injuries,
        odds_snapshot=odds,
        roster_nightly=roster,
        coach_tenure=pd.DataFrame(),
    )
    features = builder.build(labels)

    assert features.loc[features["game_id"] == 102, "min_last1"].iloc[0] == pytest.approx(32.0)
