from __future__ import annotations

from datetime import date

import pandas as pd

from projections.ops.overrides import apply_overrides_to_rates_df, upsert_overrides


def test_apply_overrides_to_rates_df_matches_float_game_id(tmp_path) -> None:
    game_date = date(2026, 1, 2)

    upsert_overrides(
        game_date,
        updates=[
            {
                "game_id": "123",
                "player_id": "456",
                "pred_ast_per_min": 0.42,
            }
        ],
        data_root=tmp_path,
    )

    rates = pd.DataFrame(
        {
            "game_id": [123.0],
            "team_id": [1],
            "player_id": [456.0],
            "pred_ast_per_min": [0.1],
        }
    )
    out = apply_overrides_to_rates_df(rates, game_date=game_date, data_root=tmp_path)
    assert float(out.loc[0, "pred_ast_per_min"]) == 0.42

