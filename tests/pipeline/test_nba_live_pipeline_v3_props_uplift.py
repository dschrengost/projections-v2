from __future__ import annotations

import pandas as pd

from prefect_flows.live_nba_pipeline_v3 import _apply_props_uplift_calibration_to_worlds


def test_props_uplift_handles_existing_direction_suffix_columns() -> None:
    worlds_df = pd.DataFrame(
        {
            "world_idx": [0, 1],
            "game_id": [1, 1],
            "team_id": [10, 10],
            "player_id": [100, 100],
            "minutes": [30.0, 30.0],
            "pts": [25.0, 25.0],
            "reb": [6.0, 6.0],
            "ast": [4.0, 4.0],
            "stl": [1.0, 1.0],
            "blk": [0.0, 0.0],
            "tov": [2.0, 2.0],
            "dk_fpts": [40.0, 40.0],
            "direction_x": ["legacy", "legacy"],
        }
    )
    features_df = pd.DataFrame(
        {
            "game_id": [1],
            "team_id": [10],
            "player_id": [100],
            "player_name": ["Test Player"],
            "an_pts_line": [31.0],
            "an_has_pts": [1.0],
            "an_reb_line": [9.0],
            "an_has_reb": [1.0],
            "an_ast_line": [7.0],
            "an_has_ast": [1.0],
        }
    )

    out, report = _apply_props_uplift_calibration_to_worlds(
        worlds_df,
        features_df=features_df,
    )

    assert report["applied"] is True
    assert report["stats"]["pts"]["applied_player_count"] == 1
    assert report["stats"]["reb"]["applied_player_count"] == 1
    assert report["stats"]["ast"]["applied_player_count"] == 1
    assert "direction_x" in out.columns
    assert out["direction_x"].eq("legacy").all()
    assert "direction_y" not in out.columns
