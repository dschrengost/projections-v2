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


def test_espn_out_name_normalization_matches_rotation_overlay() -> None:
    """Verify ESPN OUT name normalization correctly matches player names.
    
    This tests the same normalization logic used in score_minutes_rotation_set_v1
    to apply ESPN OUT overrides before guardrails.
    """
    from projections.cli.score_minutes_v1 import _normalize_name_for_matching

    # ESPN uses ASCII, NBA uses Unicode
    espn_out_players = {"kawhi leonard", "luka doncic", "bradley beal"}

    df = pd.DataFrame([
        {"player_id": 1, "player_name": "Kawhi Leonard"},    # Should match
        {"player_id": 2, "player_name": "Luka Dončić"},      # Unicode - should match
        {"player_id": 3, "player_name": "James Harden"},     # Should NOT match
        {"player_id": 4, "player_name": "BRADLEY BEAL"},     # Case insensitive - should match
    ])

    normalized = df["player_name"].astype(str).map(_normalize_name_for_matching)
    espn_mask = normalized.isin(espn_out_players)

    assert espn_mask.sum() == 3
    assert espn_mask.iloc[0] == True   # Kawhi
    assert espn_mask.iloc[1] == True   # Luka (Unicode normalized)
    assert espn_mask.iloc[2] == False  # Harden
    assert espn_mask.iloc[3] == True   # Beal (case insensitive)
