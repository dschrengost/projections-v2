from __future__ import annotations

import pandas as pd

from projections.projections_bundle import add_canonical_projection_fields


def test_add_canonical_projection_fields_maps_uncond_and_play_prob() -> None:
    df = pd.DataFrame(
        [
            {
                "player_id": 1,
                "play_prob": 0.25,
                "sim_p_active": 0.10,
                "minutes_p50_cond": 20.0,
                "minutes_sim_mean_uncond": 2.0,
                "minutes_sim_p50_uncond": 0.0,
                "dk_fpts_mean_uncond": 3.5,
            }
        ]
    )

    out = add_canonical_projection_fields(df)

    assert out.loc[0, "p_play_raw"] == 0.25
    # When sim_p_active exists, treat it as effective play prob.
    assert out.loc[0, "minutes_sim_p_active"] == 0.10
    assert out.loc[0, "p_play_eff"] == 0.10

    assert out.loc[0, "minutes_cond_p50"] == 20.0
    assert out.loc[0, "minutes_sim_uncond_mean"] == 2.0
    assert out.loc[0, "minutes_sim_uncond_p50"] == 0.0
    assert out.loc[0, "fpts_sim_uncond_mean"] == 3.5


def test_add_canonical_projection_fields_falls_back_when_sim_p_active_missing() -> None:
    df = pd.DataFrame(
        [
            {
                "player_id": 1,
                "play_prob": 1.5,  # should clip
                "play_prob_eff": 0.9,
                "dk_fpts_mean": 10.0,
            }
        ]
    )

    out = add_canonical_projection_fields(df)

    assert out.loc[0, "p_play_raw"] == 1.0
    assert out.loc[0, "p_play_eff"] == 0.9
    assert out.loc[0, "minutes_sim_p_active"] == 0.9

    # If unconditional mean is missing, approximate using p_play_eff.
    assert out.loc[0, "fpts_sim_cond_mean"] == 10.0
    assert out.loc[0, "fpts_sim_uncond_mean"] == 9.0


def test_add_canonical_projection_fields_maps_sim_prefixed_legacy_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "player_id": 1,
                "play_prob": 0.5,
                "sim_p_active": 0.4,
                # Legacy minutes API path prefixes sim columns with `sim_`.
                "sim_minutes_sim_mean_uncond": 12.0,
                "sim_minutes_sim_p50_uncond": 0.0,
                "sim_dk_fpts_mean_uncond": 10.0,
            }
        ]
    )

    out = add_canonical_projection_fields(df)

    assert out.loc[0, "minutes_sim_p_active"] == 0.4
    assert out.loc[0, "minutes_sim_uncond_mean"] == 12.0
    assert out.loc[0, "minutes_sim_uncond_p50"] == 0.0
    assert out.loc[0, "fpts_sim_uncond_mean"] == 10.0
