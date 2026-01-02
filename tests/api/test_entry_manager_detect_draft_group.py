from pathlib import Path

from projections import paths
from projections.api.entry_manager_api import (
    EntryFileState,
    _detect_draft_group_candidates,
    _sample_entry_draftable_ids,
)


def test_detect_draft_group_candidates_matches_by_draftable_ids(monkeypatch, tmp_path: Path):
    data_root = tmp_path
    bronze = data_root / "bronze" / "dk" / "draftables"
    bronze.mkdir(parents=True, exist_ok=True)

    (bronze / "draftables_raw_111.json").write_text(
        """
        {"draftables":[{"draftableId": 123},{"draftableId": 456},{"draftableId": 789}],"competitions":[{"competitionId": 1}]}
        """.strip()
    )
    (bronze / "draftables_raw_222.json").write_text(
        """
        {"draftables":[{"draftableId": 999},{"draftableId": 888}],"competitions":[{"competitionId": 1},{"competitionId": 2}]}
        """.strip()
    )

    monkeypatch.setattr(paths, "data_path", lambda: data_root)

    entry_state = EntryFileState(
        game_date="2025-12-28",
        draft_group_id=0,
        contest_id="c",
        contest_name="n",
        entry_fee="1",
        created_at="t",
        updated_at="t",
        client_revision=1,
        header=[],
        entries=[
            {
                "entry_id": "1",
                "entry_key": "1",
                "contest_id": "c",
                "contest_name": "n",
                "entry_fee": "1",
                "PG": "A (123)",
                "SG": "B (456)",
                "SF": "",
                "PF": "",
                "C": "",
                "G": "",
                "F": "",
                "UTIL": "",
            }
        ],
    )

    sample_ids = _sample_entry_draftable_ids(entry_state)
    assert set(sample_ids) == {123, 456}

    candidates = _detect_draft_group_candidates(sample_ids, max_files=10, min_match_count=1)
    assert candidates
    assert candidates[0].draft_group_id == 111
    assert candidates[0].match_count == 2
    assert candidates[0].slate_type == "showdown"


def test_detect_draft_group_candidates_filters_by_game_date(monkeypatch, tmp_path: Path):
    data_root = tmp_path
    bronze = data_root / "bronze" / "dk" / "draftables"
    bronze.mkdir(parents=True, exist_ok=True)

    (bronze / "draftables_raw_111.json").write_text(
        """
        {
          "competitions": [{"competitionId": 1, "startTime": "2025-12-28T23:00:00.0000000Z"}],
          "draftables": [{"draftableId": 123}, {"draftableId": 456}]
        }
        """.strip()
    )
    (bronze / "draftables_raw_222.json").write_text(
        """
        {
          "competitions": [{"competitionId": 1, "startTime": "2025-12-29T23:00:00.0000000Z"}],
          "draftables": [{"draftableId": 123}, {"draftableId": 456}]
        }
        """.strip()
    )

    monkeypatch.setattr(paths, "data_path", lambda: data_root)

    candidates = _detect_draft_group_candidates([123, 456], game_date="2025-12-28", max_files=10, min_match_count=1)
    assert candidates
    assert candidates[0].draft_group_id == 111
