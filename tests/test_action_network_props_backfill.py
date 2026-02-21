import json

from scrapers.action_network.props_backfill import (
    _date_set_with_utc_buffer,
    _filter_games_by_date,
    _should_refresh_existing_file,
)


def test_date_set_with_utc_buffer_includes_next_day() -> None:
    dates = _date_set_with_utc_buffer("2026-02-21", "2026-02-21", utc_buffer_days=1)
    assert dates == {"2026-02-21", "2026-02-22"}


def test_filter_games_by_date_respects_utc_buffer() -> None:
    games = {
        "2026-02-20": [{"game_id": 1}],
        "2026-02-21": [{"game_id": 2}],
        "2026-02-22": [{"game_id": 3}],
    }
    filtered = _filter_games_by_date(
        games,
        start_date="2026-02-21",
        end_date="2026-02-21",
        utc_buffer_days=1,
    )
    assert set(filtered.keys()) == {"2026-02-21", "2026-02-22"}


def test_filter_games_by_date_without_buffer_is_strict() -> None:
    games = {
        "2026-02-20": [{"game_id": 1}],
        "2026-02-21": [{"game_id": 2}],
        "2026-02-22": [{"game_id": 3}],
    }
    filtered = _filter_games_by_date(
        games,
        start_date="2026-02-21",
        end_date="2026-02-21",
        utc_buffer_days=0,
    )
    assert set(filtered.keys()) == {"2026-02-21"}


def test_should_refresh_existing_file_true_when_stale(tmp_path) -> None:
    p = tmp_path / "props.json"
    p.write_text(
        json.dumps({"fetched_at": "2026-02-21T00:00:00Z", "props": {"player_props": {}}}),
        encoding="utf-8",
    )
    assert _should_refresh_existing_file(p, refresh_older_than_minutes=1) is True


def test_should_refresh_existing_file_false_when_disabled(tmp_path) -> None:
    p = tmp_path / "props.json"
    p.write_text(
        json.dumps({"fetched_at": "2026-02-21T00:00:00Z", "props": {"player_props": {}}}),
        encoding="utf-8",
    )
    assert _should_refresh_existing_file(p, refresh_older_than_minutes=0) is False
