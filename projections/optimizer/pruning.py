from typing import List, Dict, Set, Any, Optional

BASES = ("PG", "SG", "SF", "PF", "C")


def prune_safely(
    players_in: List[Dict[str, Any]],
    *,
    locks: Optional[List[str]] = None,
    proj_floor: Optional[float] = None,
    k_per_pos: int = 24,
    k_global: int = 48,
    keep_value_per_pos: int = 4,
) -> List[Dict[str, Any]]:
    locks = set(locks or [])
    # 0) projection floor (locks bypass floor)
    pool = [
        p
        for p in players_in
        if proj_floor is None or p["proj"] >= proj_floor or p["player_id"] in locks
    ]

    # 1) top-K per base
    per_pos_keep: Set[str] = set()
    for b in BASES:
        bucket = [p for p in pool if b in p["positions"]]
        bucket.sort(key=lambda x: x["proj"], reverse=True)
        per_pos_keep.update(p["player_id"] for p in bucket[:k_per_pos])

    # 2) top-K global
    global_keep = {
        p["player_id"]
        for p in sorted(pool, key=lambda x: x["proj"], reverse=True)[:k_global]
    }

    # 3) value safety: keep a few cheapest per base
    value_keep: Set[str] = set()
    for b in BASES:
        bucket = [p for p in pool if b in p["positions"]]
        bucket.sort(key=lambda x: x["salary"])
        value_keep.update(p["player_id"] for p in bucket[:keep_value_per_pos])

    keep_ids = locks | per_pos_keep | global_keep | value_keep
    pruned = [p for p in pool if p["player_id"] in keep_ids]

    # guardrail: ensure at least one candidate for each base
    assert all(any(b in p["positions"] for p in pruned) for b in BASES), (
        "Pruning removed a base position."
    )
    return pruned
