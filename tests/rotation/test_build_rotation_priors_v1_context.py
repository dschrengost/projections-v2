from __future__ import annotations

import pandas as pd

from scripts.rotation.build_rotation_priors_v1 import _compute_context_bucket_priors


def test_compute_context_bucket_priors_uses_trailing_window_overall_games() -> None:
    df = pd.DataFrame(
        [
            {
                "person_id": 1,
                "team_id": 100,
                "game_id_norm": "0022501001",
                "game_date": "2026-03-10",
                "ctx_same_pos_bucket": "deep",
                "minutes_from_stints": 12.0,
                "started_proxy": 0.0,
            },
            {
                "person_id": 1,
                "team_id": 100,
                "game_id_norm": "0022501002",
                "game_date": "2026-03-11",
                "ctx_same_pos_bucket": "thin",
                "minutes_from_stints": 30.0,
                "started_proxy": 1.0,
            },
            {
                "person_id": 1,
                "team_id": 100,
                "game_id_norm": "0022501003",
                "game_date": "2026-03-12",
                "ctx_same_pos_bucket": "thin",
                "minutes_from_stints": 28.0,
                "started_proxy": 1.0,
            },
            {
                "person_id": 1,
                "team_id": 100,
                "game_id_norm": "0022501004",
                "game_date": "2026-03-13",
                "ctx_same_pos_bucket": "normal",
                "minutes_from_stints": 18.0,
                "started_proxy": 0.0,
            },
        ]
    )

    out = _compute_context_bucket_priors(
        df,
        group_cols=["person_id"],
        date_col="game_date",
        windows=[2],
        value_cols=["minutes_from_stints", "started_proxy"],
        bucket_col="ctx_same_pos_bucket",
        bucket_values=("thin", "normal", "deep"),
    )

    row3 = out.loc[out["game_id_norm"] == "0022501003"].iloc[0]
    assert int(row3["ctx_same_pos_thin_prior_n_games_2"]) == 1
    assert float(row3["minutes_from_stints_ctx_same_pos_thin_prior_2"]) == 30.0

    row4 = out.loc[out["game_id_norm"] == "0022501004"].iloc[0]
    assert int(row4["ctx_same_pos_thin_prior_n_games_2"]) == 2
    assert float(row4["minutes_from_stints_ctx_same_pos_thin_prior_2"]) == 29.0
    assert float(row4["started_proxy_ctx_same_pos_thin_prior_2"]) == 1.0
