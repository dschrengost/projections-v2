import pandas as pd

from projections.api import optimizer_service


def test_get_slates_for_date_merges_disk_when_api_partial(monkeypatch):
    api_df = pd.DataFrame(
        [
            {
                "game_date": "2025-12-28",
                "slate_type": "night",
                "draft_group_id": 222,
                "n_contests": 10,
                "earliest_start": "2025-12-29T01:00:00+00:00",
                "latest_start": "2025-12-29T03:30:00+00:00",
                "example_contest_name": "NBA Night",
            }
        ]
    )

    disk_slates = [
        {
            "game_date": "2025-12-28",
            "slate_type": "main",
            "draft_group_id": 111,
            "n_contests": 0,
            "earliest_start": "2025-12-28T23:00:00+00:00",
            "latest_start": "2025-12-29T02:00:00+00:00",
            "example_contest_name": "NBA Main",
            "games": [{"matchup": "AAA@BBB"}],
        },
        {
            "game_date": "2025-12-28",
            "slate_type": "night",
            "draft_group_id": 222,
            "n_contests": 0,
            "earliest_start": "2025-12-29T01:00:00+00:00",
            "latest_start": "2025-12-29T03:30:00+00:00",
            "example_contest_name": "NBA Night (disk)",
            "games": [{"matchup": "CCC@DDD"}],
        },
    ]

    monkeypatch.setattr(optimizer_service, "list_draft_groups_for_date", lambda *_, **__: api_df)
    monkeypatch.setattr(optimizer_service, "_discover_slates_from_disk", lambda *_: disk_slates)
    monkeypatch.setattr(optimizer_service, "_discover_slates_from_bronze_draftables", lambda *_: [])

    slates = optimizer_service.get_slates_for_date("2025-12-28", slate_type="all")
    by_dg = {int(s["draft_group_id"]): s for s in slates}

    assert set(by_dg) == {111, 222}
    assert by_dg[222]["n_contests"] == 10
    assert by_dg[222]["games"] == [{"matchup": "CCC@DDD"}]


def test_get_slates_for_date_uses_disk_when_api_empty(monkeypatch):
    api_df = pd.DataFrame(
        columns=[
            "game_date",
            "slate_type",
            "draft_group_id",
            "n_contests",
            "earliest_start",
            "latest_start",
            "example_contest_name",
        ]
    )
    disk_slates = [
        {
            "game_date": "2025-12-28",
            "slate_type": "main",
            "draft_group_id": 111,
            "n_contests": 0,
            "earliest_start": None,
            "latest_start": None,
            "example_contest_name": "NBA Main",
        }
    ]

    monkeypatch.setattr(optimizer_service, "list_draft_groups_for_date", lambda *_, **__: api_df)
    monkeypatch.setattr(optimizer_service, "_discover_slates_from_disk", lambda *_: disk_slates)
    monkeypatch.setattr(optimizer_service, "_discover_slates_from_bronze_draftables", lambda *_: [])

    assert optimizer_service.get_slates_for_date("2025-12-28", slate_type="all") == disk_slates


def test_get_slates_for_date_refines_showdown_from_draftables(monkeypatch, tmp_path):
    api_df = pd.DataFrame(
        [
            {
                "game_date": "2025-12-28",
                "slate_type": "main",  # misclassified by name heuristic
                "draft_group_id": 999,
                "n_contests": 1,
                "earliest_start": "2025-12-28T23:00:00+00:00",
                "latest_start": "2025-12-28T23:00:00+00:00",
                "example_contest_name": "NBA Something",
            }
        ]
    )

    bronze = tmp_path / "bronze" / "dk" / "draftables"
    bronze.mkdir(parents=True, exist_ok=True)
    (bronze / "draftables_raw_999.json").write_text(
        """
        {
          "competitions": [
            {
              "competitionId": 1,
              "awayTeam": {"abbreviation": "AAA"},
              "homeTeam": {"abbreviation": "BBB"},
              "startTime": "2025-12-28T23:00:00.0000000Z"
            }
          ]
        }
        """.strip()
    )

    monkeypatch.setattr(optimizer_service, "list_draft_groups_for_date", lambda *_, **__: api_df)
    monkeypatch.setattr(optimizer_service, "_discover_slates_from_disk", lambda *_: [])
    monkeypatch.setattr(optimizer_service, "get_data_root", lambda: tmp_path)

    slates = optimizer_service.get_slates_for_date("2025-12-28", slate_type="all")
    assert len(slates) == 1
    assert slates[0]["draft_group_id"] == 999
    assert slates[0]["slate_type"] == "showdown"


def test_get_slates_for_date_uses_bronze_when_gold_missing(monkeypatch, tmp_path):
    api_df = pd.DataFrame(
        columns=[
            "game_date",
            "slate_type",
            "draft_group_id",
            "n_contests",
            "earliest_start",
            "latest_start",
            "example_contest_name",
        ]
    )

    bronze = tmp_path / "bronze" / "dk" / "draftables"
    bronze.mkdir(parents=True, exist_ok=True)
    (bronze / "draftables_raw_123.json").write_text(
        """
        {
          "competitions": [
            {
              "competitionId": 1,
              "awayTeam": {"abbreviation": "AAA"},
              "homeTeam": {"abbreviation": "BBB"},
              "startTime": "2025-12-28T23:00:00.0000000Z"
            },
            {
              "competitionId": 2,
              "awayTeam": {"abbreviation": "CCC"},
              "homeTeam": {"abbreviation": "DDD"},
              "startTime": "2025-12-29T00:00:00.0000000Z"
            }
          ],
          "Contests": [{"n": "NBA Main $10K Something"}]
        }
        """.strip()
    )

    monkeypatch.setattr(optimizer_service, "list_draft_groups_for_date", lambda *_, **__: api_df)
    monkeypatch.setattr(optimizer_service, "get_data_root", lambda: tmp_path)

    slates = optimizer_service.get_slates_for_date("2025-12-28", slate_type="all")
    assert any(int(s["draft_group_id"]) == 123 for s in slates)
