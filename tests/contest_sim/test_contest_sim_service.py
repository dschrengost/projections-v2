from __future__ import annotations

import numpy as np

from projections.contest_sim import contest_sim_service


def test_self_play_entry_accounting_totals_to_field_size(monkeypatch) -> None:
    # 1 world, 2 players
    worlds = np.array([[10.0, 0.0]], dtype=np.float64)  # (W, P)
    player_index = {"1": 0, "2": 1}

    def _fake_load_worlds_matrix(game_date: str, data_root=None, run_id=None):  # type: ignore[no-untyped-def]
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

    def _fake_load_worlds_matrix(game_date: str, data_root=None, run_id=None):  # type: ignore[no-untyped-def]
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

    def _fake_load_worlds_matrix(game_date: str, data_root=None, run_id=None):  # type: ignore[no-untyped-def]
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

    def _fake_load_worlds_matrix(game_date: str, data_root=None, run_id=None):  # type: ignore[no-untyped-def]
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


def test_ucvar90_computed_correctly() -> None:
    """Test that UCVaR90 is computed as mean of top 10% of scores."""
    # Create synthetic scores where we can compute UCVaR90 exactly
    # 10 worlds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # p90 = 9th value in sorted = 8.1 (using np.percentile linear interpolation)
    # For 10 values, top 10% = the single value >= p90, which is 9
    # UCVaR90 = mean([9]) = 9
    scores = np.arange(10, dtype=np.float64)  # [0, 1, 2, ..., 9]

    p90 = np.percentile(scores, 90)
    tail_mask = scores >= p90
    ucv90 = scores[tail_mask].mean()

    # With 10 values and p90~8.1, values >= 8.1 are [9]
    assert p90 == 8.1
    assert ucv90 == 9.0


def test_ucvar90_and_tail_score_in_results(monkeypatch) -> None:
    """Test that ucv90, tail_score, select_score are populated in results."""
    # 100 worlds, 1 player: uniform scores from 0-99
    worlds = np.arange(100, dtype=np.float64).reshape(-1, 1)  # (W=100, P=1)
    player_index = {"1": 0}

    def _fake_load_worlds_matrix(game_date: str, data_root=None, run_id=None):
        return worlds, player_index

    monkeypatch.setattr(contest_sim_service, "load_worlds_matrix", _fake_load_worlds_matrix)

    result = contest_sim_service.run_contest_simulation(
        user_lineups=[["1"]],
        game_date="2099-01-01",
        field_size_override=2,
        entry_fee=1.0,
        archetype="GPP Standard (20% paid)",
    )

    r0 = result.results[0]

    # With scores [0, 1, 2, ..., 99]:
    # p90 = 89.1 (linear interpolation)
    # UCVaR90 = mean of scores >= 89.1 = mean([90, 91, ..., 99]) = 94.5
    assert r0.p90 is not None
    assert r0.ucv90 is not None
    assert r0.tail_score is not None
    assert r0.select_score is not None

    # Verify p90 is around 89
    assert 88 < r0.p90 < 91

    # Verify UCVaR90 is higher than p90 (it's the mean of the top 10%)
    assert r0.ucv90 > r0.p90

    # Verify tail_score = 0.6*p90 + 0.4*ucv90
    expected_tail = 0.6 * r0.p90 + 0.4 * r0.ucv90
    assert abs(r0.tail_score - expected_tail) < 0.01

    # Without dupe penalty (dupe_penalty=1.0), select_score should equal tail_score
    assert r0.dupe_penalty == 1.0
    assert abs(r0.select_score - r0.tail_score) < 0.01


def test_select_score_applies_dupe_penalty(monkeypatch) -> None:
    """Test that select_score correctly subtracts dupe penalty impact."""
    # 100 worlds, 1 player
    worlds = np.arange(100, dtype=np.float64).reshape(-1, 1)
    player_index = {"1": 0}

    def _fake_load_worlds_matrix(game_date: str, data_root=None, run_id=None):
        return worlds, player_index

    monkeypatch.setattr(contest_sim_service, "load_worlds_matrix", _fake_load_worlds_matrix)
    monkeypatch.setattr(contest_sim_service, "compute_batch_dupe_penalties", lambda **_: [0.8])

    result = contest_sim_service.run_contest_simulation(
        user_lineups=[["1"]],
        game_date="2099-01-01",
        field_size_override=2,
        entry_fee=1.0,
        archetype="GPP Standard (20% paid)",
        player_ownership={"1": 50.0},
    )

    r0 = result.results[0]
    assert r0.dupe_penalty == 0.8

    # penalty_impact = (1 - 0.8) * mean = 0.2 * mean
    # select_score = tail_score - penalty_impact
    penalty_impact = (1.0 - r0.dupe_penalty) * r0.mean
    expected_select = r0.tail_score - penalty_impact

    assert abs(r0.select_score - expected_select) < 0.01
    assert r0.select_score < r0.tail_score  # penalty reduces score


def test_robust_floor_metrics_computed(monkeypatch) -> None:
    """score_lcb95/score_cvar10/robust_floor should be populated and coherent."""
    worlds = np.arange(100, dtype=np.float64).reshape(-1, 1)
    player_index = {"1": 0}

    def _fake_load_worlds_matrix(game_date: str, data_root=None, run_id=None):
        return worlds, player_index

    monkeypatch.setattr(contest_sim_service, "load_worlds_matrix", _fake_load_worlds_matrix)

    result = contest_sim_service.run_contest_simulation(
        user_lineups=[["1"]],
        game_date="2099-01-01",
        field_size_override=2,
        entry_fee=1.0,
        archetype="GPP Standard (20% paid)",
    )

    r0 = result.results[0]
    assert r0.score_lcb95 is not None
    assert r0.score_cvar10 is not None
    assert r0.robust_floor is not None
    assert r0.robust_floor == min(r0.score_lcb95, r0.score_cvar10)
