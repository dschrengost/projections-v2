import pandas as pd

from projections.rotation.guardrails import apply_rotation_minutes_guardrails


def test_rotation_overlay_forces_out_rows_to_zero_and_avoids_tail_clamp() -> None:
    df = pd.DataFrame(
        [
            # OUT row: rotation emits huge minutes, baseline is 0.
            {
                "game_id": 1,
                "team_id": 10,
                "player_id": 100,
                "rotation_minutes_p50": 30.0,
                "minutes_p50": 0.0,
                "is_out": 1,
                "status": "OUT",
                "minutes_features_row_missing": 0,
                "injury_snapshot_missing": 0.0,
            },
            # Active rows: small rotation minutes that will be scaled up to hit 240.
            {
                "game_id": 1,
                "team_id": 10,
                "player_id": 101,
                "rotation_minutes_p50": 20.0,
                "minutes_p50": 20.0,
                "is_out": 0,
                "status": "AVAIL",
                "minutes_features_row_missing": 0,
                "injury_snapshot_missing": 0.0,
            },
            {
                "game_id": 1,
                "team_id": 10,
                "player_id": 102,
                "rotation_minutes_p50": 20.0,
                "minutes_p50": 20.0,
                "is_out": 0,
                "status": "AVAIL",
                "minutes_features_row_missing": 0,
                "injury_snapshot_missing": 0.0,
            },
        ]
    )

    result = apply_rotation_minutes_guardrails(
        df,
        rotation_p50_col="rotation_minutes_p50",
        baseline_p50_col="minutes_p50",
        blend_weight=1.0,
        dnp_tail_minutes_threshold=8.0,
        team_target_minutes=240.0,
    )

    out_row = result.minutes_p50.loc[df["player_id"] == 100].iloc[0]
    assert out_row == 0.0
    # With OUT forced to 0 pre-scale, tail clamp should not trigger.
    assert int(result.summary.get("tail_clamped_team_games", 0)) == 0
