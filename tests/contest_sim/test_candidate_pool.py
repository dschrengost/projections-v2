from __future__ import annotations

from pathlib import Path

from projections.contest_sim.field_candidate_pool import (
    CandidatePool,
    load_candidate_pool,
    merge_candidates,
    save_candidate_pool,
)


def test_merge_candidates_dedupes_and_caps() -> None:
    existing = [["2", "1"], ["3"]]
    new = [["1", "2"], ["4"], ["5"]]
    merged = merge_candidates(existing, new, max_size=3)
    assert merged == [["1", "2"], ["3"], ["4"]]


def test_candidate_pool_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "candidate_pool_v0.jsonl.gz"
    pool = CandidatePool(
        lineups=[["2", "1"], ["3"]],
        meta={"game_date": "2099-01-01", "draft_group_id": 123, "version": "v0"},
    )
    save_candidate_pool(pool, path)
    loaded = load_candidate_pool(path)
    assert loaded.lineups == [["1", "2"], ["3"]]
    assert loaded.meta["game_date"] == "2099-01-01"
    assert loaded.meta["draft_group_id"] == 123

