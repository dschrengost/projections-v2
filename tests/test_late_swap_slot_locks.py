from __future__ import annotations

from datetime import datetime, timezone

import pytest

from projections.api.entry_manager_api import _parse_game_start
from projections.optimizer.cpsat_solver import solve_cpsat_iterative_counts
from projections.optimizer.optimizer_types import Constraints, OwnershipPenaltySettings


def test_parse_game_start_handles_dk_fractional_seconds() -> None:
    dt = _parse_game_start("2025-12-20T00:30:00.0000000Z")
    assert dt == datetime(2025, 12, 20, 0, 30, tzinfo=timezone.utc)


def test_parse_game_start_assumes_utc_when_missing_tz() -> None:
    dt = _parse_game_start("2025-12-20T00:30:00")
    assert dt == datetime(2025, 12, 20, 0, 30, tzinfo=timezone.utc)


def test_late_swap_slot_locks_route_to_per_slot_solver() -> None:
    # Construct a minimal DK pool that is feasible exactly at $50k.
    # We intentionally lock PG to player "B" even though the greedy DK assignment
    # (used in counts-only) would put "A" in PG due to pid ordering.
    players = [
        {"player_id": "A", "name": "A", "team": "T1", "positions": ["PG", "SG"], "salary": 6000, "proj": 10.0},
        {"player_id": "B", "name": "B", "team": "T1", "positions": ["PG", "SG"], "salary": 6000, "proj": 20.0},
        {"player_id": "C", "name": "C", "team": "T1", "positions": ["PG", "SG"], "salary": 6000, "proj": 30.0},
        {"player_id": "D", "name": "D", "team": "T2", "positions": ["SF", "PF"], "salary": 6000, "proj": 10.0},
        {"player_id": "E", "name": "E", "team": "T2", "positions": ["SF", "PF"], "salary": 6000, "proj": 10.0},
        {"player_id": "F", "name": "F", "team": "T2", "positions": ["SF", "PF"], "salary": 6000, "proj": 10.0},
        {"player_id": "G", "name": "G", "team": "T3", "positions": ["C"], "salary": 7000, "proj": 10.0},
        {"player_id": "H", "name": "H", "team": "T4", "positions": ["C"], "salary": 7000, "proj": 10.0},
    ]

    constraints = Constraints(N_lineups=1, unique_players=1, min_salary=0, max_salary=50000)
    constraints.lock_ids = ["B"]
    constraints.lock_slots = {"PG": "B"}
    constraints.ban_ids = []
    constraints.ownership_penalty = OwnershipPenaltySettings(enabled=False)

    lineups, _diag = solve_cpsat_iterative_counts(players, constraints, seed=0, site="dk")
    assert lineups
    lineup = lineups[0]
    assert any(p.player_id == "B" and p.pos == "PG" for p in lineup.players)


def test_lock_slots_reject_duplicate_player_assignment() -> None:
    players = [
        {"player_id": "A", "name": "A", "team": "T1", "positions": ["PG", "SG"], "salary": 1000, "proj": 1.0},
        {"player_id": "B", "name": "B", "team": "T2", "positions": ["SF", "PF"], "salary": 1000, "proj": 2.0},
        {"player_id": "C", "name": "C", "team": "T3", "positions": ["C"], "salary": 1000, "proj": 3.0},
        {"player_id": "D", "name": "D", "team": "T4", "positions": ["PG", "SG"], "salary": 1000, "proj": 4.0},
        {"player_id": "E", "name": "E", "team": "T5", "positions": ["SF", "PF"], "salary": 1000, "proj": 5.0},
        {"player_id": "F", "name": "F", "team": "T6", "positions": ["PG", "SG"], "salary": 1000, "proj": 6.0},
        {"player_id": "G", "name": "G", "team": "T7", "positions": ["SF", "PF"], "salary": 1000, "proj": 7.0},
        {"player_id": "H", "name": "H", "team": "T8", "positions": ["C"], "salary": 1000, "proj": 8.0},
    ]

    constraints = Constraints(N_lineups=1, unique_players=1, min_salary=0, max_salary=50000)
    constraints.lock_slots = {"PG": "A", "SG": "A"}  # invalid: same player for two slots
    constraints.lock_ids = ["A"]
    constraints.ban_ids = []
    constraints.ownership_penalty = OwnershipPenaltySettings(enabled=False)

    with pytest.raises(ValueError, match="multiple slots"):
        solve_cpsat_iterative_counts(players, constraints, seed=0, site="dk")
