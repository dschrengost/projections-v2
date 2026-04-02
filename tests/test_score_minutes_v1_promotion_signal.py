from __future__ import annotations

import pandas as pd

from projections.cli.score_minutes_v1 import _attach_promotion_signal_columns


def test_attach_promotion_signal_columns_flags_sparse_propless_candidate() -> None:
    df = pd.DataFrame(
        {
            "player_id": [1, 2],
            "prior_play_prob": [0.10, 0.90],
            "recent_start_pct_10": [0.0, 0.60],
            "an_implied_minutes": [22.0, 8.0],
            "minutes_p50": [0.0, 10.0],
            "minutes_p90": [24.0, 12.0],
            "an_has_any_props": [0.0, 1.0],
            "an_props_market_count": [0.0, 3.0],
        }
    )

    out = _attach_promotion_signal_columns(df)

    assert {
        "promotion_signal_score",
        "promotion_signal_flag",
        "promotion_signal_sparse_prior",
        "promotion_signal_propless",
    }.issubset(out.columns)

    sparse_row = out.loc[out["player_id"] == 1].iloc[0]
    stable_row = out.loc[out["player_id"] == 2].iloc[0]

    assert int(sparse_row["promotion_signal_flag"]) == 1
    assert int(sparse_row["promotion_signal_sparse_prior"]) == 1
    assert int(sparse_row["promotion_signal_propless"]) == 1
    assert float(sparse_row["promotion_signal_score"]) > float(stable_row["promotion_signal_score"])
    assert int(stable_row["promotion_signal_flag"]) == 0


def test_attach_promotion_signal_columns_handles_missing_optional_inputs() -> None:
    df = pd.DataFrame(
        {
            "player_id": [11, 12],
            "minutes_p50": [12.0, 18.0],
            "minutes_p90": [18.0, 30.0],
        }
    )

    out = _attach_promotion_signal_columns(df)
    assert {"promotion_signal_score", "promotion_signal_flag"}.issubset(out.columns)
    assert out["promotion_signal_score"].notna().all()
    assert out["promotion_signal_flag"].isin([0, 1]).all()


def test_attach_promotion_signal_columns_uses_play_prob_sparse_fallback() -> None:
    df = pd.DataFrame(
        {
            "player_id": [21, 22],
            "prior_play_prob": [0.97, 0.97],
            "play_prob": [0.20, 0.90],
            "minutes_p50": [10.0, 10.0],
            "minutes_p90": [24.0, 24.0],
            "an_implied_minutes": [18.0, 18.0],
            "an_has_any_props": [0.0, 0.0],
        }
    )

    out = _attach_promotion_signal_columns(df)

    sparse = out.set_index("player_id")["promotion_signal_sparse_prior"]
    assert int(sparse.loc[21]) == 1
    assert int(sparse.loc[22]) == 0


def test_attach_promotion_signal_columns_fallback_flag_when_implied_and_p90_missing() -> None:
    df = pd.DataFrame(
        {
            "player_id": [31, 32],
            "prior_play_prob": [0.97, 0.97],
            "play_prob": [0.00, 0.20],
            "recent_start_pct_10": [0.0, 0.0],
            "an_implied_minutes": [0.0, 0.0],
            "minutes_p50": [0.0, 0.0],
            "minutes_p90": [12.0, 12.0],
            "vac_min_szn": [320.0, 60.0],
            "sum_min_7d": [24.0, 0.0],
            "an_has_any_props": [0.0, 0.0],
            "an_props_market_count": [0.0, 0.0],
        }
    )

    out = _attach_promotion_signal_columns(df)
    by_id = out.set_index("player_id")

    # Row 31 satisfies fallback gate: sparse+propless+very low play_prob plus
    # vacancy/recency pressure and score threshold.
    assert int(by_id.loc[31, "promotion_signal_flag"]) == 1
    # Row 32 does not satisfy fallback play_prob condition and should remain unflagged.
    assert int(by_id.loc[32, "promotion_signal_flag"]) == 0
    assert float(by_id.loc[31, "promotion_signal_score"]) > float(by_id.loc[32, "promotion_signal_score"])


def test_attach_promotion_signal_columns_vacancy_and_recency_raise_score() -> None:
    df = pd.DataFrame(
        {
            "player_id": [41, 42],
            "prior_play_prob": [0.97, 0.97],
            "play_prob": [0.20, 0.20],
            "recent_start_pct_10": [0.0, 0.0],
            "an_implied_minutes": [10.0, 10.0],
            "minutes_p50": [10.0, 10.0],
            "minutes_p90": [18.0, 18.0],
            "an_has_any_props": [0.0, 0.0],
            "vac_min_szn": [360.0, 20.0],
            "sum_min_7d": [30.0, 0.0],
        }
    )

    out = _attach_promotion_signal_columns(df).set_index("player_id")
    assert float(out.loc[41, "promotion_signal_score"]) > float(out.loc[42, "promotion_signal_score"])
