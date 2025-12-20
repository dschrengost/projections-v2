from __future__ import annotations

from projections.contest_sim.field_library_builder import build_field_library_from_candidates


def test_build_field_library_from_candidates_dedupes_and_weights_sum_to_candidates() -> None:
    candidates = [
        ["1", "2", "3", "4", "5", "6", "7", "8"],
        ["1", "2", "3", "4", "5", "6", "7", "8"],  # duplicate
        ["9", "10", "11", "12", "13", "14", "15", "16"],
    ]

    player_pool = [
        {"player_id": str(i), "salary": 6000, "own_proj": 10.0, "team": "A"}
        for i in range(1, 17)
    ]

    lib = build_field_library_from_candidates(
        candidates,
        k=10,
        player_pool=player_pool,
        method="test",
    )

    assert len(lib.lineups) == len(lib.weights)
    assert sum(lib.weights) == len(candidates)
    assert lib.meta["method"] == "test"


def test_seven_of_eight_guard_rejects_near_duplicates() -> None:
    # Two lineups share 7 of 8 players; guard should prevent selecting both.
    lu1 = ["1", "2", "3", "4", "5", "6", "7", "8"]
    lu2 = ["1", "2", "3", "4", "5", "6", "7", "9"]
    lu3 = ["20", "21", "22", "23", "24", "25", "26", "27"]

    candidates = [lu1] * 5 + [lu2] * 4 + [lu3]

    player_pool = [
        {"player_id": str(i), "salary": 6000, "own_proj": 10.0, "team": "A"}
        for i in [*range(1, 10), *range(20, 28)]
    ]

    lib = build_field_library_from_candidates(
        candidates,
        k=2,
        player_pool=player_pool,
        method="test",
    )

    assert len(lib.lineups) == 2
    assert sum(lib.weights) == len(candidates)

    sets = [set(lu) for lu in lib.lineups]
    has_lu1 = set(lu1) in sets
    has_lu2 = set(lu2) in sets
    assert not (has_lu1 and has_lu2)
