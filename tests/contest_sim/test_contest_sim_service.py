from __future__ import annotations

import numpy as np

from projections.contest_sim import contest_sim_service


def _player_worlds(
    worlds: np.ndarray,
    player_index: dict[str, int],
    *,
    minutes: np.ndarray | None = None,
) -> contest_sim_service.PlayerWorlds:
    return contest_sim_service.PlayerWorlds(
        fpts_matrix=worlds,
        player_index=player_index,
        minutes_matrix=minutes,
    )


def test_self_play_entry_accounting_totals_to_field_size(monkeypatch) -> None:
    worlds = np.array([[10.0, 0.0]], dtype=np.float64)
    player_index = {"1": 0, "2": 1}

    monkeypatch.setattr(
        contest_sim_service,
        "load_player_worlds",
        lambda *args, **kwargs: _player_worlds(worlds, player_index),
    )

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

    monkeypatch.setattr(
        contest_sim_service,
        "load_player_worlds",
        lambda *args, **kwargs: _player_worlds(worlds, player_index),
    )
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

    monkeypatch.setattr(
        contest_sim_service,
        "load_player_worlds",
        lambda *args, **kwargs: _player_worlds(worlds, player_index),
    )
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

    monkeypatch.setattr(
        contest_sim_service,
        "load_player_worlds",
        lambda *args, **kwargs: _player_worlds(worlds, player_index),
    )
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
    scores = np.arange(10, dtype=np.float64)

    p90 = np.percentile(scores, 90)
    tail_mask = scores >= p90
    ucv90 = scores[tail_mask].mean()

    assert p90 == 8.1
    assert ucv90 == 9.0


def test_ucvar90_and_tail_score_in_results(monkeypatch) -> None:
    worlds = np.arange(100, dtype=np.float64).reshape(-1, 1)
    player_index = {"1": 0}

    monkeypatch.setattr(
        contest_sim_service,
        "load_player_worlds",
        lambda *args, **kwargs: _player_worlds(worlds, player_index),
    )

    result = contest_sim_service.run_contest_simulation(
        user_lineups=[["1"]],
        game_date="2099-01-01",
        field_size_override=2,
        entry_fee=1.0,
        archetype="GPP Standard (20% paid)",
    )

    r0 = result.results[0]
    assert r0.p90 is not None
    assert r0.ucv90 is not None
    assert r0.tail_score is not None
    assert r0.select_score is not None
    assert 88 < r0.p90 < 91
    assert r0.ucv90 > r0.p90
    expected_tail = 0.6 * r0.p90 + 0.4 * r0.ucv90
    assert abs(r0.tail_score - expected_tail) < 0.01
    assert r0.dupe_penalty == 1.0
    assert abs(r0.select_score - r0.tail_score) < 0.01


def test_select_score_applies_dupe_penalty(monkeypatch) -> None:
    worlds = np.arange(100, dtype=np.float64).reshape(-1, 1)
    player_index = {"1": 0}

    monkeypatch.setattr(
        contest_sim_service,
        "load_player_worlds",
        lambda *args, **kwargs: _player_worlds(worlds, player_index),
    )
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
    penalty_impact = (1.0 - r0.dupe_penalty) * r0.mean
    expected_select = r0.tail_score - penalty_impact
    assert abs(r0.select_score - expected_select) < 0.01
    assert r0.select_score < r0.tail_score


def test_robust_floor_metrics_computed(monkeypatch) -> None:
    worlds = np.arange(100, dtype=np.float64).reshape(-1, 1)
    player_index = {"1": 0}

    monkeypatch.setattr(
        contest_sim_service,
        "load_player_worlds",
        lambda *args, **kwargs: _player_worlds(worlds, player_index),
    )

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


def test_strategy_overrides_adjust_contest_sim_worlds(monkeypatch) -> None:
    worlds = np.array([[10.0], [20.0], [0.0]], dtype=np.float64)
    minutes = np.array([[10.0], [20.0], [0.0]], dtype=np.float64)
    player_index = {"1": 0}

    monkeypatch.setattr(
        contest_sim_service,
        "load_player_worlds",
        lambda *args, **kwargs: _player_worlds(worlds, player_index, minutes=minutes),
    )
    monkeypatch.setattr(
        contest_sim_service,
        "load_unified_projections_df",
        lambda *args, **kwargs: None,
        raising=False,
    )

    from projections.api.strategy_overrides import PlayerStrategyOverride, SlateStrategyOverrides

    monkeypatch.setattr(
        "projections.api.strategy_overrides.load_slate_strategy_overrides",
        lambda game_date, draft_group_id: SlateStrategyOverrides(
            game_date=game_date,
            draft_group_id=draft_group_id,
            overrides={"1": PlayerStrategyOverride(player_id="1", minutes_delta=10.0)},
        ),
    )

    monkeypatch.setattr(
        contest_sim_service,
        "compute_expected_user_payouts_vectorized",
        lambda **kwargs: type(
            "PayoutResult",
            (),
            {
                "expected_payouts": np.array([0.0]),
                "win_rates": np.array([0.0]),
                "top_1pct_rates": np.array([0.0]),
                "top_5pct_rates": np.array([0.0]),
                "top_10pct_rates": np.array([0.0]),
                "cash_rates": np.array([0.0]),
            },
        )(),
    )

    result = contest_sim_service.run_contest_simulation(
        user_lineups=[["1"]],
        game_date="2099-01-01",
        draft_group_id=123,
        field_size_override=2,
        entry_fee=1.0,
        archetype="GPP Standard (20% paid)",
        use_strategy_overrides=True,
    )

    assert result.results[0].mean > float(np.mean(worlds[:, 0]))
    assert result.stats.debug["strategy_overrides_enabled"] is True
    assert result.stats.debug["strategy_overrides_applied"] is True
