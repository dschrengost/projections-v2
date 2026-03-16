from __future__ import annotations

from projections.fd.slates import list_fixture_lists_for_date


def test_list_fixture_lists_for_date_filters_and_infers_types() -> None:
    payload = {
        "fixture_lists": [
            {
                "id": 1001,
                "sport": "nba",
                "start_date": "2026-01-10T23:30:00Z",
                "label": "NBA Main",
                "contests": 12,
                "fixtures": 6,
            },
            {
                "id": 1002,
                "sport": "nba",
                "start_date": "2026-01-10T05:00:00Z",
                "label": "NBA Single Game Showdown",
                "contests": 4,
                "fixtures": 1,
            },
            {
                "id": 2001,
                "sport": "nfl",
                "start_date": "2026-01-10T18:00:00Z",
                "label": "NFL Main",
                "contests": 20,
                "fixtures": 10,
            },
        ]
    }

    all_slates = list_fixture_lists_for_date(
        "2026-01-10",
        slate_type="all",
        fixture_lists_payload=payload,
    )
    assert len(all_slates) == 2
    assert set(all_slates["draft_group_id"].tolist()) == {1001, 1002}

    showdown = list_fixture_lists_for_date(
        "2026-01-10",
        slate_type="showdown",
        fixture_lists_payload=payload,
    )
    assert len(showdown) == 1
    row = showdown.iloc[0]
    assert row["draft_group_id"] == 1002
    assert row["slate_type"] == "showdown"
