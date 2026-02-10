from __future__ import annotations

from datetime import datetime, timezone

import pytest

from projections.api.entry_manager_api import _parse_game_start
from projections.optimizer.cpsat_solver import solve_cpsat_iterative, solve_cpsat_iterative_counts
from projections.optimizer.objective import LateSwapBonusConfig, set_active_late_swap_bonus
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


def test_randomness_pct_changes_per_slot_objective() -> None:
    players = [
        {"player_id": "A", "name": "A", "team": "T1", "positions": ["PG"], "salary": 9000, "proj": 20.0, "stddev": 1.0},
        {"player_id": "B", "name": "B", "team": "T2", "positions": ["PG"], "salary": 9000, "proj": 19.5, "stddev": 1.0},
        {"player_id": "C", "name": "C", "team": "T3", "positions": ["SG"], "salary": 5800, "proj": 10.0, "stddev": 0.0},
        {"player_id": "D", "name": "D", "team": "T4", "positions": ["SF"], "salary": 5800, "proj": 10.0, "stddev": 0.0},
        {"player_id": "E", "name": "E", "team": "T5", "positions": ["PF"], "salary": 5800, "proj": 10.0, "stddev": 0.0},
        {"player_id": "F", "name": "F", "team": "T6", "positions": ["C"], "salary": 5800, "proj": 10.0, "stddev": 0.0},
        {"player_id": "G", "name": "G", "team": "T7", "positions": ["PG", "SG"], "salary": 5800, "proj": 10.0, "stddev": 0.0},
        {"player_id": "H", "name": "H", "team": "T8", "positions": ["SF", "PF"], "salary": 5800, "proj": 10.0, "stddev": 0.0},
        {"player_id": "I", "name": "I", "team": "T9", "positions": ["C"], "salary": 5800, "proj": 10.0, "stddev": 0.0},
    ]

    no_rand_constraints = Constraints(N_lineups=1, unique_players=1, min_salary=0, max_salary=50000)
    no_rand_constraints.ban_ids = []
    no_rand_constraints.lock_ids = []
    no_rand_constraints.randomness_pct = 0.0
    no_rand_constraints.ownership_penalty = OwnershipPenaltySettings(enabled=False)

    rand_constraints = Constraints(N_lineups=1, unique_players=1, min_salary=0, max_salary=50000)
    rand_constraints.ban_ids = []
    rand_constraints.lock_ids = []
    rand_constraints.randomness_pct = 100.0
    rand_constraints.ownership_penalty = OwnershipPenaltySettings(enabled=False)

    lineups_no_rand, _ = solve_cpsat_iterative(players, no_rand_constraints, seed=0, site="dk")
    assert lineups_no_rand
    ids_no_rand = {p.player_id for p in lineups_no_rand[0].players}
    assert "A" in ids_no_rand
    assert "B" not in ids_no_rand

    lineups_rand, _ = solve_cpsat_iterative(players, rand_constraints, seed=0, site="dk")
    assert lineups_rand
    ids_rand = {p.player_id for p in lineups_rand[0].players}
    assert "B" in ids_rand
    assert "A" not in ids_rand


def test_late_swap_bonus_uses_game_start_for_per_slot_solver() -> None:
    players = [
        {"player_id": "A", "name": "A", "team": "T1", "positions": ["PG"], "salary": 9000, "proj": 20.0, "stddev": 0.0, "game_start_utc": "2026-01-10T00:00:00Z"},
        {"player_id": "B", "name": "B", "team": "T2", "positions": ["PG"], "salary": 9000, "proj": 20.0, "stddev": 0.0, "game_start_utc": "2026-01-10T03:00:00Z"},
        {"player_id": "C", "name": "C", "team": "T3", "positions": ["SG"], "salary": 5800, "proj": 10.0, "stddev": 0.0},
        {"player_id": "D", "name": "D", "team": "T4", "positions": ["SF"], "salary": 5800, "proj": 10.0, "stddev": 0.0},
        {"player_id": "E", "name": "E", "team": "T5", "positions": ["PF"], "salary": 5800, "proj": 10.0, "stddev": 0.0},
        {"player_id": "F", "name": "F", "team": "T6", "positions": ["C"], "salary": 5800, "proj": 10.0, "stddev": 0.0},
        {"player_id": "G", "name": "G", "team": "T7", "positions": ["PG", "SG"], "salary": 5800, "proj": 10.0, "stddev": 0.0},
        {"player_id": "H", "name": "H", "team": "T8", "positions": ["SF", "PF"], "salary": 5800, "proj": 10.0, "stddev": 0.0},
        {"player_id": "I", "name": "I", "team": "T9", "positions": ["C"], "salary": 5800, "proj": 10.0, "stddev": 0.0},
    ]

    constraints = Constraints(N_lineups=1, unique_players=1, min_salary=0, max_salary=50000)
    constraints.lock_ids = []
    constraints.ban_ids = []
    constraints.ownership_penalty = OwnershipPenaltySettings(enabled=False)

    set_active_late_swap_bonus(None)
    try:
        set_active_late_swap_bonus(LateSwapBonusConfig(enabled=True, bonus_per_hour=1.0, max_bonus=5.0))
        lineups, _ = solve_cpsat_iterative(players, constraints, seed=0, site="dk")
    finally:
        set_active_late_swap_bonus(None)

    assert lineups
    ids = {p.player_id for p in lineups[0].players}
    assert "B" in ids
    assert "A" not in ids
