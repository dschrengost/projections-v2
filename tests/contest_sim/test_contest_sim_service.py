from __future__ import annotations

import numpy as np

from projections.contest_sim import contest_sim_service


def test_self_play_entry_accounting_totals_to_field_size(monkeypatch) -> None:
    # 1 world, 2 players
    worlds = np.array([[10.0, 0.0]], dtype=np.float64)  # (W, P)
    player_index = {"1": 0, "2": 1}

    def _fake_load_worlds_matrix(game_date: str, data_root=None):  # type: ignore[no-untyped-def]
        return worlds, player_index

    monkeypatch.setattr(contest_sim_service, "load_worlds_matrix", _fake_load_worlds_matrix)

    result = contest_sim_service.run_contest_simulation(
        user_lineups=[["1"], ["2"]],
        game_date="2099-01-01",
        field_size_override=10,
        entry_fee=1.0,
        archetype="GPP Standard (20% paid)",
    )

    assert result.stats.debug["user_total_entries"] == 2
    assert result.stats.debug["field_total_entries"] == 8
    assert result.stats.debug["total_entries"] == 10


def test_dupe_penalty_adjusts_expected_payout(monkeypatch) -> None:
    worlds = np.array([[10.0]], dtype=np.float64)
    player_index = {"1": 0}

    def _fake_load_worlds_matrix(game_date: str, data_root=None):  # type: ignore[no-untyped-def]
        return worlds, player_index

    monkeypatch.setattr(contest_sim_service, "load_worlds_matrix", _fake_load_worlds_matrix)
    monkeypatch.setattr(contest_sim_service, "compute_batch_dupe_penalties", lambda **_: [0.5])

    result = contest_sim_service.run_contest_simulation(
        user_lineups=[["1"]],
        game_date="2099-01-01",
        field_size_override=2,
        entry_fee=1.0,
        archetype="GPP Standard (20% paid)",
        player_ownership={"1": 50.0},
    )

    r0 = result.results[0]
    assert r0.dupe_penalty == 0.5
    assert r0.unadjusted_expected_payout is not None
    assert r0.expected_payout == r0.adjusted_expected_payout
    assert r0.expected_payout == r0.unadjusted_expected_payout * 0.5
    assert r0.expected_value == r0.expected_payout - 1.0


def test_dupe_penalty_disabled_when_lineup_present_in_field(monkeypatch) -> None:
    worlds = np.array([[10.0]], dtype=np.float64)
    player_index = {"1": 0}

    def _fake_load_worlds_matrix(game_date: str, data_root=None):  # type: ignore[no-untyped-def]
        return worlds, player_index

    monkeypatch.setattr(contest_sim_service, "load_worlds_matrix", _fake_load_worlds_matrix)
    monkeypatch.setattr(contest_sim_service, "compute_batch_dupe_penalties", lambda **_: [0.5])

    result = contest_sim_service.run_contest_simulation(
        user_lineups=[["1"]],
        game_date="2099-01-01",
        field_size_override=2,
        entry_fee=1.0,
        archetype="GPP Standard (20% paid)",
        player_ownership={"1": 50.0},
        field_lineups=[["1"]],
        field_weights=[1],
    )

    r0 = result.results[0]
    assert r0.dupe_penalty == 1.0
    assert r0.expected_payout == r0.unadjusted_expected_payout
    assert result.stats.debug["dupe_penalty_disabled_for_field_matches"] == 1


def test_dupe_penalty_applies_when_lineup_not_in_field(monkeypatch) -> None:
    worlds = np.array([[10.0, 0.0]], dtype=np.float64)
    player_index = {"1": 0, "2": 1}

    def _fake_load_worlds_matrix(game_date: str, data_root=None):  # type: ignore[no-untyped-def]
        return worlds, player_index

    monkeypatch.setattr(contest_sim_service, "load_worlds_matrix", _fake_load_worlds_matrix)
    monkeypatch.setattr(contest_sim_service, "compute_batch_dupe_penalties", lambda **_: [0.5])

    result = contest_sim_service.run_contest_simulation(
        user_lineups=[["1"]],
        game_date="2099-01-01",
        field_size_override=2,
        entry_fee=1.0,
        archetype="GPP Standard (20% paid)",
        player_ownership={"1": 50.0},
        field_lineups=[["2"]],
        field_weights=[1],
    )

    r0 = result.results[0]
    assert r0.dupe_penalty == 0.5
    assert r0.expected_payout == r0.unadjusted_expected_payout * 0.5
    assert result.stats.debug["dupe_penalty_disabled_for_field_matches"] == 0
