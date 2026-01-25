import pandas as pd
import pytest


def test_reconcile_team_minutes_with_rotation_cap_limits_players() -> None:
    from projections.cli.score_minutes_rmh_v1 import _reconcile_team_minutes_with_rotation_cap

    # 15-man team, 5 starters. RMH can over-allocate non-trivial minutes to too many players,
    # so we cap to 10 and reconcile to 240 among the kept set.
    df = pd.DataFrame(
        {
            "game_id": [1] * 15,
            "team_id": [100] * 15,
            "player_id": list(range(1, 16)),
            "player_name": [f"p{i}" for i in range(1, 16)],
            "status": ["Ava"] * 15,
            "starter_flag": [1] * 5 + [0] * 10,
            # Descending minutes signal; tail is still >0 to stress the cap logic.
            "minutes_p50": [34, 33, 32, 31, 30, 22, 18, 15, 12, 10, 8, 7, 6, 5, 4],
        }
    )

    eff = _reconcile_team_minutes_with_rotation_cap(
        df,
        target_minutes=240.0,
        minutes_col="minutes_p50",
        in_rotation_threshold_min=5.0,
        max_rotation_players=10,
    )

    assert float(eff.sum()) == pytest.approx(240.0, abs=1e-6)
    assert int((eff > 0).sum()) <= 10
    assert int((eff.loc[df["starter_flag"].astype(bool)] > 0).sum()) == 5
    assert int((eff.loc[~df["starter_flag"].astype(bool)] > 0).sum()) <= 5


def test_reconcile_team_minutes_with_rotation_cap_excludes_out_players() -> None:
    from projections.cli.score_minutes_rmh_v1 import _reconcile_team_minutes_with_rotation_cap

    df = pd.DataFrame(
        {
            "game_id": [1] * 6,
            "team_id": [100] * 6,
            "player_id": list(range(1, 7)),
            "status": ["Ava", "Ava", "OUT", "Ava", "OUT", "Ava"],
            "starter_flag": [1, 1, 1, 0, 0, 0],
            "minutes_p50": [35, 33, 31, 20, 10, 5],
            # Ensure OUT rows would otherwise be eligible by minutes.
            "play_prob": [1, 1, 1, 1, 1, 1],
        }
    )

    eff = _reconcile_team_minutes_with_rotation_cap(
        df,
        target_minutes=240.0,
        minutes_col="minutes_p50",
        in_rotation_threshold_min=5.0,
        max_rotation_players=5,
    )

    out_mask = df["status"].astype("string").str.upper().eq("OUT").fillna(False)
    assert float(eff.loc[out_mask].sum()) == 0.0
    assert float(eff.sum()) == pytest.approx(240.0, abs=1e-6)
