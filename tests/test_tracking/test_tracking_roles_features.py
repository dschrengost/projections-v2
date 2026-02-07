import pandas as pd

from projections.tracking.roles import compute_cumulative_tracking


def test_compute_cumulative_tracking_builds_3pa_profile_features() -> None:
    df = pd.DataFrame(
        [
            {
                "season": 2025,
                "game_date": "2026-01-01",
                "game_id": 1,
                "team_id": 10,
                "player_id": 100,
                "minutes_tracking": 20.0,
                "touches": 40.0,
                "time_of_poss": 4.0,
                "potential_ast_raw": 8.0,
                "passes_made": 30.0,
                "drives": 6.0,
                "drive_fta": 2.0,
                "drive_pf": 1.0,
                "paint_touches": 10.0,
                "catch_shoot_fg3a": 4.0,
                "pull_up_fg3a": 2.0,
            },
            {
                "season": 2025,
                "game_date": "2026-01-03",
                "game_id": 2,
                "team_id": 10,
                "player_id": 100,
                "minutes_tracking": 22.0,
                "touches": 45.0,
                "time_of_poss": 4.5,
                "potential_ast_raw": 7.0,
                "passes_made": 31.0,
                "drives": 7.0,
                "drive_fta": 1.0,
                "drive_pf": 2.0,
                "paint_touches": 11.0,
                "catch_shoot_fg3a": 3.0,
                "pull_up_fg3a": 4.0,
            },
        ]
    )
    df["game_date"] = pd.to_datetime(df["game_date"])

    out = compute_cumulative_tracking(df).sort_values(["game_date", "game_id"]).reset_index(drop=True)

    assert pd.isna(out.loc[0, "track_catch_shoot_fg3a_per_min_szn"])
    assert pd.isna(out.loc[0, "track_pull_up_fg3a_per_min_szn"])
    assert pd.isna(out.loc[0, "track_pull_up_3pa_share_szn"])
    assert out.loc[1, "track_catch_shoot_fg3a_per_min_szn"] == 0.2
    assert out.loc[1, "track_pull_up_fg3a_per_min_szn"] == 0.1
    assert out.loc[1, "track_pull_up_3pa_share_szn"] == (2.0 / 6.0)
