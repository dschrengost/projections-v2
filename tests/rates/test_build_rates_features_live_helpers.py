import pytest
import pandas as pd

from projections.cli.build_rates_features_live import (
    _compute_player_priors,
    _compute_team_context,
    _compute_vacancy_features,
    _status_to_out_probability,
)


def test_status_to_out_probability_masks_avail_and_unknown() -> None:
    status = pd.Series(["OUT", "Q", "PROB", "AVAIL", "available", "UNK", None, "QUESTIONABLE", "INACTIVE", "O"])
    out_prob = _status_to_out_probability(status).tolist()

    assert out_prob[0] == pytest.approx(1.0)
    assert out_prob[1] == pytest.approx(0.45)  # 1 - 0.55
    assert out_prob[2] == pytest.approx(0.22)  # 1 - 0.78
    assert out_prob[3] == pytest.approx(0.0)
    assert out_prob[4] == pytest.approx(0.0)
    assert out_prob[5] == pytest.approx(0.0)
    assert out_prob[6] == pytest.approx(0.0)
    assert out_prob[7] == pytest.approx(0.45)
    assert out_prob[8] == pytest.approx(1.0)
    assert out_prob[9] == pytest.approx(1.0)


def test_compute_player_priors_season_and_recency_rates() -> None:
    history = pd.DataFrame(
        [
            {
                "player_id": 100,
                "tip_ts": pd.Timestamp("2025-01-01T00:00:00Z"),
                "minutes_played": 10.0,
                "fga": 10.0,
                "fgm": 5.0,
                "three_pa": 4.0,
                "three_pm": 2.0,
                "fta": 2.0,
                "ftm": 2.0,
                "assists": 6.0,
                "turnovers": 2.0,
                "oreb": 1.0,
                "dreb": 4.0,
                "steals": 1.0,
                "blocks": 0.0,
            },
            {
                "player_id": 100,
                "tip_ts": pd.Timestamp("2025-01-03T00:00:00Z"),
                "minutes_played": 20.0,
                "fga": 12.0,
                "fgm": 6.0,
                "three_pa": 2.0,
                "three_pm": 1.0,
                "fta": 4.0,
                "ftm": 3.0,
                "assists": 4.0,
                "turnovers": 3.0,
                "oreb": 0.0,
                "dreb": 6.0,
                "steals": 0.0,
                "blocks": 1.0,
            },
        ]
    )

    priors = _compute_player_priors(history, player_ids={100})
    assert len(priors) == 1
    row = priors.iloc[0].to_dict()

    # Season: minutes=30, fga2=(6+10)=16 => 0.533333...
    assert row["season_fga2_per_min"] == pytest.approx(16.0 / 30.0)
    assert row["season_3pa_per_min"] == pytest.approx(6.0 / 30.0)
    assert row["season_fta_per_min"] == pytest.approx(6.0 / 30.0)
    assert row["season_ast_per_min"] == pytest.approx(10.0 / 30.0)

    # Last1 window is game 2 only: minutes=20, fga2=10, fga3=2
    assert row["last1_minutes_sum"] == pytest.approx(20.0)
    assert row["last1_fga2_per_min"] == pytest.approx(10.0 / 20.0)
    assert row["last1_fga3_per_min"] == pytest.approx(2.0 / 20.0)

    # With only 2 games in history, last10 should match season.
    assert row["last10_minutes_sum"] == pytest.approx(30.0)
    assert row["last10_fga2_per_min"] == pytest.approx(row["season_fga2_per_min"])
    assert row["last10_fga3_per_min"] == pytest.approx(row["season_3pa_per_min"])


def test_compute_player_priors_clips_efficiency_rates() -> None:
    history = pd.DataFrame(
        [
            {
                "player_id": 200,
                "tip_ts": pd.Timestamp("2025-01-01T00:00:00Z"),
                "minutes_played": 12.0,
                "fga": 10.0,
                "fgm": 10.0,
                "three_pa": 5.0,
                "three_pm": 5.0,
                "fta": 10.0,
                "ftm": 0.0,
                "assists": 0.0,
                "turnovers": 0.0,
                "oreb": 0.0,
                "dreb": 0.0,
                "steals": 0.0,
                "blocks": 0.0,
            }
        ]
    )

    priors = _compute_player_priors(history, player_ids={200})
    row = priors.iloc[0].to_dict()

    assert row["season_fg2_pct"] == pytest.approx(0.75)
    assert row["season_fg3_pct"] == pytest.approx(0.55)
    assert row["season_ft_pct"] == pytest.approx(0.5)


def test_compute_team_context_aggregates_possessions() -> None:
    team_history = pd.DataFrame(
        [
            {"game_id": 1, "team_id": 10, "points_for": 100, "points_against": 90, "fga": 80, "fta": 20, "turnovers": 10},
            {"game_id": 2, "team_id": 10, "points_for": 110, "points_against": 105, "fga": 82, "fta": 18, "turnovers": 12},
            {"game_id": 1, "team_id": 11, "points_for": 95, "points_against": 100, "fga": 78, "fta": 22, "turnovers": 9},
        ]
    )

    ctx = _compute_team_context(team_history, team_ids={10, 11})
    assert set(ctx["team_id"].tolist()) == {10, 11}

    row10 = ctx.set_index("team_id").loc[10].to_dict()
    poss10 = (80 + 0.44 * 20 + 10) + (82 + 0.44 * 18 + 12)
    assert row10["team_pace_szn"] == pytest.approx(poss10 / 2)
    assert row10["team_off_rtg_szn"] == pytest.approx(100.0 * (210 / poss10))
    assert row10["team_def_rtg_szn"] == pytest.approx(100.0 * (195 / poss10))


def test_compute_vacancy_features_weights_season_totals_by_out_prob() -> None:
    player_history = pd.DataFrame(
        [
            {
                "player_id": 10,
                "tip_ts": pd.Timestamp("2025-10-21T23:00:00Z"),
                "minutes_played": 30.0,
                "fga": 12.0,
                "three_pa": 4.0,
                "fta": 2.0,
                "assists": 6.0,
            }
        ]
    )

    minutes_preds = pd.DataFrame(
        [
            {
                "season": 2025,
                "game_id": 2,
                "team_id": 100,
                "player_id": 10,
                "tip_ts": pd.Timestamp("2025-10-23T23:00:00Z"),
                "status": "OUT",
                "pos_bucket": "G",
            }
        ]
    )

    vac = _compute_vacancy_features(player_history, minutes_preds)
    assert len(vac) == 1
    row = vac.iloc[0].to_dict()

    assert row["game_id"] == 2
    assert row["team_id"] == 100
    assert row["vac_min_szn"] == pytest.approx(30.0)
    assert row["vac_fga_szn"] == pytest.approx(12.0)
    assert row["vac_ast_szn"] == pytest.approx(6.0)
    assert row["vac_min_guard_szn"] == pytest.approx(30.0)

