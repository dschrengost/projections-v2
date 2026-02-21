from scrapers.action_network.props_backfill import (
    _date_set_with_utc_buffer,
    _filter_games_by_date,
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
