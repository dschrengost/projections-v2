from __future__ import annotations

import pandas as pd

from projections.rotation.rotation_set_minutes_features_v1 import join_rotation_priors


def test_join_rotation_priors_merges_opponent_team_prior_columns_with_prefix() -> None:
    df = pd.DataFrame(
        [
            {
                "game_id": "0022501001",
                "team_id": 10,
                "player_id": 101,
                "opponent_team_id": 20,
            }
        ]
    )
    team_priors = pd.DataFrame(
        [
            {
                "game_id": "0022501001",
                "team_id": 10,
                "fg2_pct_allowed_prior_5": 0.48,
            },
            {
                "game_id": "0022501001",
                "team_id": 20,
                "fg2_pct_allowed_prior_5": 0.52,
                "team_prior_n_games_5": 5,
            },
        ]
    )

    out = join_rotation_priors(df, team_priors=team_priors, player_priors=pd.DataFrame())

    assert float(out.loc[0, "fg2_pct_allowed_prior_5"]) == 0.48
    assert float(out.loc[0, "opp_fg2_pct_allowed_prior_5"]) == 0.52
    assert int(out.loc[0, "opp_team_prior_n_games_5"]) == 5
