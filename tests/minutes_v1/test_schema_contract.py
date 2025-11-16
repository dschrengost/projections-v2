from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest

from projections.minutes_v1.schemas import (
    ARENA_TIMEZONE_SCHEMA,
    BOX_SCORE_LABELS_SCHEMA,
    COACH_TENURE_SCHEMA,
    FEATURES_MINUTES_V1_SCHEMA,
    INJURIES_RAW_SCHEMA,
    INJURIES_SNAPSHOT_SCHEMA,
    ODDS_RAW_SCHEMA,
    ODDS_SNAPSHOT_SCHEMA,
    ROSTER_NIGHTLY_RAW_SCHEMA,
    ROSTER_NIGHTLY_SCHEMA,
    SCHEDULE_SCHEMA,
    SCHEDULE_STATIC_SCHEMA,
    SchemaError,
    TableSchema,
    STRING_DTYPE,
    enforce_schema,
    validate_with_pandera,
)


def _dtype_matches(actual: str, expected: str) -> bool:
    if expected == STRING_DTYPE and actual == "string":
        return True
    return actual == expected


def _assert_schema(df: pd.DataFrame, schema: TableSchema, **kwargs) -> pd.DataFrame:
    """Run enforcement and assert columns/dtypes/PK uniqueness."""

    result = enforce_schema(df, schema, **kwargs)
    validate_with_pandera(result, schema)
    assert list(result.columns) == list(schema.columns)
    for column, expected_dtype in schema.pandas_dtypes.items():
        actual_dtype = str(result[column].dtype)
        assert _dtype_matches(actual_dtype, expected_dtype)
    if schema.primary_key and schema.enforce_primary_key:
        assert result.duplicated(list(schema.primary_key)).sum() == 0
    return result


def _injuries_raw_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_id": [1610612737, 1610612737],
            "report_date": ["2024-10-21", "2024-10-22"],
            "as_of_ts": ["2024-10-21T15:00:00Z", "2024-10-22T15:30:00Z"],
            "player_name": ["Player A", "Player B"],
            "player_id": [1001, None],
            "status_raw": ["Out", "Questionable"],
            "notes_raw": ["ankle restriction", ""],
            "game_id": [22001001, 22001002],
            "ingested_ts": ["2024-10-21T16:00:00Z", "2024-10-22T16:00:00Z"],
            "source": ["nba.com/a", "nba.com/b"],
            "source_row_id": ["row-1", "row-2"],
            "status": ["OUT", "Q"],
            "restriction_flag": [True, False],
            "ramp_flag": [False, True],
            "games_since_return": [None, 2],
            "days_since_return": [None, 5],
        }
    )


def _injuries_snapshot_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [22001001, 22001002],
            "player_id": [1001, 1002],
            "as_of_ts": ["2024-10-21T18:00:00Z", "2024-10-22T18:30:00Z"],
            "status": ["OUT", "AVAIL"],
            "restriction_flag": [True, False],
            "ramp_flag": [False, False],
            "games_since_return": [1, None],
            "days_since_return": [3, None],
            "ingested_ts": ["2024-10-21T18:01:00Z", "2024-10-22T18:31:00Z"],
            "source": ["injuries-feed", "injuries-feed"],
            "selection_rule": ["latest_leq_tip", "no_pre_tip_snapshot"],
            "snapshot_missing": [0, 1],
        }
    )


def _odds_raw_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "book": ["demo-book", "demo-book"],
            "game_id": [22001001, 22001002],
            "market": ["spread_total", "spread_total"],
            "home_team_id": [1610612737, 1610612738],
            "away_team_id": [1610612745, 1610612747],
            "spread_home": [-5.5, 3.0],
            "total": [225.5, 219.0],
            "as_of_ts": ["2024-10-21T12:00:00Z", "2024-10-22T12:30:00Z"],
            "ingested_ts": ["2024-10-21T12:05:00Z", "2024-10-22T12:35:00Z"],
            "source": ["oddstrader", "oddstrader"],
        }
    )


def _odds_snapshot_partial() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [22001001],
            "as_of_ts": ["2024-10-21T12:00:00Z"],
            "spread_home": [-5.5],
            "total": [225.5],
            "book": ["demo-book"],
            "ingested_ts": ["2024-10-21T12:05:00Z"],
            "source": ["oddstrader"],
        }
    )


def _roster_raw_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_id": [1610612737, 1610612737],
            "game_id": [22001001, 22001001],
            "game_date": ["2024-10-21", "2024-10-21"],
            "player_id": [900001, 900002],
            "player_name": ["Guard One", "Guard Two"],
            "active_flag": [1, 0],
            "starter_flag": [1, 0],
            "listed_pos": ["PG", "SG"],
            "ingested_ts": ["2024-10-21T15:00:00Z", "2024-10-21T15:00:00Z"],
            "source": ["nba.com", "nba.com"],
            "as_of_ts": ["2024-10-21T14:00:00Z", "2024-10-21T14:00:00Z"],
        }
    )


def _roster_snapshot_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [22001001, 22001001],
            "team_id": [1610612737, 1610612737],
            "player_id": [900001, 900002],
            "player_name": ["Guard One", "Guard Two"],
            "game_date": ["2024-10-21", "2024-10-21"],
            "as_of_ts": ["2024-10-21T22:00:00Z", "2024-10-21T22:00:00Z"],
            "active_flag": [True, False],
            "starter_flag": [True, False],
            "lineup_role": ["confirmed_starter", pd.NA],
            "lineup_status": ["Confirmed", pd.NA],
            "lineup_roster_status": ["Active", "Inactive"],
            "lineup_timestamp": ["2024-10-21T21:00:00Z", None],
            "is_projected_starter": [True, False],
            "is_confirmed_starter": [True, False],
            "listed_pos": ["PG", "SG"],
            "height_in": [76, 78],
            "weight_lb": [200, 215],
            "age": [27.5, 24.0],
            "ingested_ts": ["2024-10-21T20:00:00Z", "2024-10-21T20:00:00Z"],
            "source": ["nba.com", "nba.com"],
        }
    )


def _schedule_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [22001001, 22001002],
            "game_code": ["20241021ATL", "20241022BOS"],
            "season": ["2025", "2025"],
            "game_date": ["2024-10-21", "2024-10-22"],
            "tip_day": ["2024-10-21", "2024-10-22"],
            "tip_ts": ["2024-10-21T23:00:00Z", "2024-10-22T23:00:00Z"],
            "home_team_id": [1610612737, 1610612738],
            "home_team_name": ["Hawks", "Celtics"],
            "home_team_city": ["Atlanta", "Boston"],
            "home_team_tricode": ["ATL", "BOS"],
            "away_team_id": [1610612745, 1610612755],
            "away_team_name": ["Magic", "Heat"],
            "away_team_city": ["Orlando", "Miami"],
            "away_team_tricode": ["ORL", "MIA"],
            "arena_id": ["ATL01", "BOS01"],
            "arena_name": ["State Farm Arena", "TD Garden"],
            "arena_city": ["Atlanta", "Boston"],
            "arena_state": ["GA", "MA"],
        }
    )


def _boxscore_labels_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [22001001, 22001001],
            "player_id": [900001, 900002],
            "team_id": [1610612737, 1610612737],
            "player_name": ["Guard One", "Guard Two"],
            "season": ["2024-25", "2024-25"],
            "game_date": ["2024-10-21", "2024-10-21"],
            "minutes": [31.5, 24.0],
            "starter_flag": [1, 0],
            "starter_flag_label": [1, 0],
            "source": ["nba.com/boxscore", "nba.com/boxscore"],
            "label_frozen_ts": ["2024-10-22T00:00:00Z", "2024-10-22T00:00:00Z"],
        }
    )


def _features_sample() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dtype_map = FEATURES_MINUTES_V1_SCHEMA.pandas_dtypes
    for idx in range(2):
        record: dict[str, object] = {}
        for column in FEATURES_MINUTES_V1_SCHEMA.columns:
            dtype = dtype_map.get(column, "object")
            if dtype == "Int64":
                record[column] = idx + 1
            elif dtype == "Float64":
                record[column] = 10.0 + idx
            elif dtype == "boolean":
                record[column] = bool(idx % 2)
            elif dtype == "string[pyarrow]":
                record[column] = f"{column}-{idx}"
            elif dtype == "datetime64[ns, UTC]":
                record[column] = f"2024-10-2{idx}T00:00:00Z"
            elif dtype == "datetime64[ns]":
                record[column] = f"2024-10-2{idx}"
            else:
                record[column] = f"value-{idx}"
        rows.append(record)
    return pd.DataFrame(rows)


def _coach_tenure_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "coach_id": [1],
            "coach_name": ["Test Coach"],
            "team_id": [1610612737],
            "start_date": ["2024-07-01"],
            "end_date": [None],
        }
    )


def _arena_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "arena_id": ["ATL01"],
            "arena_name": ["State Farm Arena"],
            "tz": ["US/Eastern"],
            "city": ["Atlanta"],
            "state": ["GA"],
        }
    )


def _schedule_static_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [22001001],
            "season": ["2025"],
            "game_date": ["2024-10-21"],
            "tip_ts": ["2024-10-21T23:00:00Z"],
            "home_team_id": [1610612737],
            "away_team_id": [1610612745],
            "arena_id": ["ATL01"],
            "home_city": ["Atlanta"],
            "away_city": ["Orlando"],
        }
    )


def test_injuries_raw_schema_contract() -> None:
    df = _injuries_raw_sample()
    result = _assert_schema(df, INJURIES_RAW_SCHEMA)
    assert result["games_since_return"].isna().iloc[0]


def test_injuries_snapshot_schema_contract() -> None:
    df = _injuries_snapshot_sample()
    _assert_schema(df, INJURIES_SNAPSHOT_SCHEMA)


def test_odds_raw_schema_contract() -> None:
    df = _odds_raw_sample()
    _assert_schema(df, ODDS_RAW_SCHEMA)


def test_odds_snapshot_adds_optional_book_pref() -> None:
    df = _odds_snapshot_partial()
    with pytest.raises(SchemaError):
        enforce_schema(df, ODDS_SNAPSHOT_SCHEMA)
    result = enforce_schema(df, ODDS_SNAPSHOT_SCHEMA, allow_missing_optional=True)
    assert "book_pref" in result.columns


def test_roster_raw_schema_contract() -> None:
    df = _roster_raw_sample()
    result = _assert_schema(df, ROSTER_NIGHTLY_RAW_SCHEMA)
    assert result["active_flag"].dtype == "boolean"


def test_roster_snapshot_schema_contract() -> None:
    df = _roster_snapshot_sample()
    _assert_schema(df, ROSTER_NIGHTLY_SCHEMA)


def test_schedule_schema_contract() -> None:
    df = _schedule_sample()
    _assert_schema(df, SCHEDULE_SCHEMA, allow_missing_optional=True)


def test_schedule_static_schema_contract() -> None:
    df = _schedule_static_sample()
    _assert_schema(df, SCHEDULE_STATIC_SCHEMA, allow_missing_optional=True)


def test_boxscore_labels_schema_contract() -> None:
    df = _boxscore_labels_sample()
    _assert_schema(df, BOX_SCORE_LABELS_SCHEMA)


def test_features_minutes_schema_contract() -> None:
    df = _features_sample()
    _assert_schema(df, FEATURES_MINUTES_V1_SCHEMA)


def test_coach_tenure_schema_contract() -> None:
    df = _coach_tenure_sample()
    _assert_schema(df, COACH_TENURE_SCHEMA, allow_missing_optional=True)


def test_arena_tz_schema_contract() -> None:
    df = _arena_sample()
    _assert_schema(df, ARENA_TIMEZONE_SCHEMA)


def test_missing_required_column_raises() -> None:
    df = _injuries_raw_sample().drop(columns=["status_raw"])
    with pytest.raises(SchemaError):
        enforce_schema(df, INJURIES_RAW_SCHEMA)


def test_pandera_detects_primary_key_duplicates() -> None:
    df = _injuries_raw_sample()
    duplicate_cols = ["report_date", "as_of_ts", "team_id", "player_name", "source_row_id"]
    df.loc[1, duplicate_cols] = df.loc[0, duplicate_cols].values
    enforced = enforce_schema(df, INJURIES_RAW_SCHEMA)
    with pytest.raises(pa.errors.SchemaErrors):
        validate_with_pandera(enforced, INJURIES_RAW_SCHEMA)
