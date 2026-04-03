from __future__ import annotations

import pandas as pd
import pytest
from pandas.core.groupby.generic import DataFrameGroupBy

from prefect_flows.live_nba_pipeline_v3 import (
    _apply_team_implied_points_reconcile_to_worlds,
    _apply_propless_tail_calibration_to_worlds,
    _apply_props_uplift_calibration_to_worlds,
)


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


def test_props_uplift_avoids_world_scale_merge_on_large_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worlds_df = pd.DataFrame(
        {
            "world_idx": [0, 1, 0, 1],
            "game_id": [1, 1, 1, 1],
            "team_id": [10, 10, 20, 20],
            "player_id": [100, 100, 200, 200],
            "minutes": [30.0, 29.0, 28.0, 27.0],
            "pts": [25.0, 24.0, 14.0, 13.0],
            "reb": [6.0, 5.0, 8.0, 7.0],
            "ast": [4.0, 4.0, 5.0, 4.0],
            "stl": [1.0, 1.0, 1.0, 1.0],
            "blk": [0.0, 0.0, 1.0, 1.0],
            "tov": [2.0, 2.0, 1.0, 1.0],
            "dk_fpts": [40.0, 38.0, 33.0, 31.0],
        }
    )
    features_df = pd.DataFrame(
        {
            "game_id": [1, 1],
            "team_id": [10, 20],
            "player_id": [100, 200],
            "player_name": ["A", "B"],
            "an_pts_line": [31.0, 13.0],
            "an_has_pts": [1.0, 1.0],
            "an_reb_line": [9.0, 6.0],
            "an_has_reb": [1.0, 1.0],
            "an_ast_line": [7.0, 4.0],
            "an_has_ast": [1.0, 1.0],
        }
    )

    original_merge = pd.DataFrame.merge

    def _guarded_merge(self: pd.DataFrame, *args, **kwargs):  # type: ignore[no-untyped-def]
        other = args[0] if args else kwargs.get("right")
        if isinstance(other, pd.DataFrame):
            if "world_idx" in self.columns and {"mu", "sf_mean", "sf_var"}.issubset(
                other.columns
            ):
                raise AssertionError("unexpected worlds-scale merge path used")
        return original_merge(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "merge", _guarded_merge)

    out, report = _apply_props_uplift_calibration_to_worlds(
        worlds_df,
        features_df=features_df,
    )

    assert report["applied"] is True
    assert len(out) == len(worlds_df)


def test_props_uplift_avoids_pandas_groupby_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worlds_df = pd.DataFrame(
        {
            "world_idx": [0, 1, 0, 1],
            "game_id": [1, 1, 1, 1],
            "team_id": [10, 10, 20, 20],
            "player_id": [100, 100, 200, 200],
            "minutes": [30.0, 29.0, 28.0, 27.0],
            "pts": [25.0, 24.0, 14.0, 13.0],
            "reb": [6.0, 5.0, 8.0, 7.0],
            "ast": [4.0, 4.0, 5.0, 4.0],
            "stl": [1.0, 1.0, 1.0, 1.0],
            "blk": [0.0, 0.0, 1.0, 1.0],
            "tov": [2.0, 2.0, 1.0, 1.0],
            "dk_fpts": [40.0, 38.0, 33.0, 31.0],
        }
    )
    features_df = pd.DataFrame(
        {
            "game_id": [1, 1],
            "team_id": [10, 20],
            "player_id": [100, 200],
            "player_name": ["A", "B"],
            "an_pts_line": [31.0, 13.0],
            "an_has_pts": [1.0, 1.0],
            "an_reb_line": [9.0, 6.0],
            "an_has_reb": [1.0, 1.0],
            "an_ast_line": [7.0, 4.0],
            "an_has_ast": [1.0, 1.0],
        }
    )

    def _fail_mean(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("unexpected pandas DataFrameGroupBy.mean path used")

    monkeypatch.setattr(DataFrameGroupBy, "mean", _fail_mean)

    out, report = _apply_props_uplift_calibration_to_worlds(
        worlds_df,
        features_df=features_df,
    )

    assert report["applied"] is True
    assert len(out) == len(worlds_df)


def test_team_implied_points_reconcile_preserves_covered_players_and_cuts_uncovered() -> None:
    worlds_df = pd.DataFrame(
        {
            "world_idx": [0, 1, 0, 1, 0, 1],
            "game_id": [1, 1, 1, 1, 2, 2],
            "team_id": [10, 10, 10, 10, 20, 20],
            "player_id": [100, 100, 101, 101, 200, 200],
            "minutes": [32.0, 32.0, 28.0, 28.0, 30.0, 30.0],
            "pts": [30.0, 32.0, 9.0, 9.0, 18.0, 18.0],
            "reb": [5.0, 5.0, 4.0, 4.0, 6.0, 6.0],
            "ast": [4.0, 4.0, 2.0, 2.0, 5.0, 5.0],
            "stl": [1.0, 1.0, 0.7, 0.7, 1.1, 1.1],
            "blk": [0.5, 0.5, 0.3, 0.3, 0.6, 0.6],
            "tov": [2.0, 2.0, 1.0, 1.0, 2.0, 2.0],
            "dk_fpts": [0.0] * 6,
        }
    )
    features_df = pd.DataFrame(
        {
            "game_id": [1, 1, 2],
            "team_id": [10, 10, 20],
            "player_id": [100, 101, 200],
            "player_name": ["Prop A", "Bench B", "Other Team"],
            "team_implied_total": [34.0, 34.0, 18.0],
            "an_has_pts": [1.0, 0.0, 1.0],
            "an_pts_line": [30.5, pd.NA, 18.0],
            "an_props_market_count": [5.0, 0.0, 5.0],
        }
    )

    out, report = _apply_team_implied_points_reconcile_to_worlds(
        worlds_df,
        features_df=features_df,
        pre_calibration_pts_anchor=None,
        enabled=True,
        alpha=1.0,
        deadband_points=0.0,
    )

    assert report["applied"] is True
    assert report["team_count_adjusted"] == 1
    assert report["player_count_adjusted"] == 1
    team_10_total = out.loc[out["team_id"] == 10].groupby("player_id")["pts"].mean().sum()
    assert team_10_total == pytest.approx(34.0)
    prop_pts = out.loc[out["player_id"] == 100, "pts"].reset_index(drop=True)
    pd.testing.assert_series_equal(
        prop_pts,
        worlds_df.loc[worlds_df["player_id"] == 100, "pts"].reset_index(drop=True),
        check_names=False,
    )
    assert out.loc[out["player_id"] == 101, "pts"].mean() == pytest.approx(3.0)


def test_team_implied_points_reconcile_allocates_positive_residual_to_best_uncovered() -> None:
    worlds_df = pd.DataFrame(
        {
            "world_idx": [0, 1, 0, 1, 0, 1],
            "game_id": [1, 1, 1, 1, 1, 1],
            "team_id": [10, 10, 10, 10, 10, 10],
            "player_id": [100, 100, 101, 101, 102, 102],
            "minutes": [34.0, 34.0, 26.0, 26.0, 10.0, 10.0],
            "pts": [30.0, 30.0, 4.0, 4.0, 2.0, 2.0],
            "reb": [5.0, 5.0, 4.0, 4.0, 2.0, 2.0],
            "ast": [4.0, 4.0, 2.0, 2.0, 1.0, 1.0],
            "stl": [1.0, 1.0, 0.7, 0.7, 0.3, 0.3],
            "blk": [0.5, 0.5, 0.3, 0.3, 0.1, 0.1],
            "tov": [2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            "dk_fpts": [0.0] * 6,
        }
    )
    features_df = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "team_id": [10, 10, 10],
            "player_id": [100, 101, 102],
            "player_name": ["Prop A", "Upside B", "Low C"],
            "team_implied_total": [40.0, 40.0, 40.0],
            "an_has_pts": [1.0, 0.0, 0.0],
            "an_pts_line": [30.0, pd.NA, pd.NA],
            "an_props_market_count": [5.0, 0.0, 0.0],
            "prior_play_prob": [1.0, 0.9, 0.2],
        }
    )

    out, report = _apply_team_implied_points_reconcile_to_worlds(
        worlds_df,
        features_df=features_df,
        pre_calibration_pts_anchor=None,
        enabled=True,
        alpha=1.0,
        deadband_points=0.0,
    )

    assert report["applied"] is True
    assert out.loc[out["player_id"] == 100, "pts"].mean() == pytest.approx(30.0)
    assert out.loc[out["player_id"] == 101, "pts"].mean() == pytest.approx(8.0)
    assert out.loc[out["player_id"] == 102, "pts"].mean() == pytest.approx(2.0)
    assert report["total_unresolved_team_gap_mean"] == pytest.approx(0.0)


def test_props_uplift_adjusts_stocks_bidirectionally() -> None:
    worlds_df = pd.DataFrame(
        {
            "world_idx": [0, 1, 0, 1],
            "game_id": [1, 1, 1, 1],
            "team_id": [10, 10, 20, 20],
            "player_id": [100, 100, 200, 200],
            "minutes": [32.0, 32.0, 28.0, 28.0],
            "pts": [20.0, 20.0, 16.0, 16.0],
            "reb": [8.0, 8.0, 6.0, 6.0],
            "ast": [4.0, 4.0, 3.0, 3.0],
            "stl": [0.8, 0.8, 1.8, 1.8],
            "blk": [2.0, 2.0, 0.9, 0.9],
            "tov": [2.0, 2.0, 1.0, 1.0],
            "dk_fpts": [38.6, 38.6, 31.6, 31.6],
        }
    )
    features_df = pd.DataFrame(
        {
            "game_id": [1, 1],
            "team_id": [10, 20],
            "player_id": [100, 200],
            "player_name": ["Victor Wembanyama", "Dyson Daniels"],
            "an_stl_line": [1.4, 1.1],
            "an_has_stl": [1.0, 1.0],
            "an_stl_books": [5.0, 4.0],
            "an_blk_line": [3.2, 0.5],
            "an_has_blk": [1.0, 1.0],
            "an_blk_books": [5.0, 4.0],
            "an_props_market_count": [6.0, 5.0],
        }
    )

    out, report = _apply_props_uplift_calibration_to_worlds(
        worlds_df,
        features_df=features_df,
        scope="all_players",
        confidence_weighted=True,
    )

    assert report["applied"] is True
    assert report["stats"]["stl"]["applied_player_count_up"] == 1
    assert report["stats"]["stl"]["applied_player_count_down"] == 1
    assert report["stats"]["blk"]["applied_player_count_up"] == 1
    assert report["stats"]["blk"]["applied_player_count_down"] == 1

    wemby = out[out["player_id"] == 100]
    dyson = out[out["player_id"] == 200]
    assert wemby["stl"].mean() > 0.8
    assert wemby["blk"].mean() > 2.0
    assert dyson["stl"].mean() < 1.8
    assert dyson["blk"].mean() < 0.9


def test_propless_tail_calibration_only_adjusts_propless_players() -> None:
    worlds_df = pd.DataFrame(
        {
            "world_idx": [0, 1, 2, 3, 0, 1, 2, 3],
            "game_id": [1, 1, 1, 1, 1, 1, 1, 1],
            "team_id": [10, 10, 10, 10, 20, 20, 20, 20],
            "player_id": [100, 100, 100, 100, 200, 200, 200, 200],
            "minutes": [24.0, 24.0, 24.0, 24.0, 32.0, 32.0, 32.0, 32.0],
            "pts": [10.0, 12.0, 14.0, 22.0, 24.0, 25.0, 26.0, 27.0],
            "reb": [4.0, 5.0, 6.0, 8.0, 8.0, 8.0, 9.0, 9.0],
            "ast": [3.0, 3.0, 4.0, 6.0, 5.0, 5.0, 6.0, 6.0],
            "stl": [0.5, 0.5, 0.8, 1.2, 1.0, 1.0, 1.1, 1.1],
            "blk": [0.2, 0.3, 0.3, 0.8, 0.5, 0.5, 0.6, 0.6],
            "tov": [2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
            "oreb": [1.2, 1.4, 1.8, 2.3, 2.5, 2.5, 2.7, 2.7],
            "dreb": [2.8, 3.6, 4.2, 5.7, 5.5, 5.5, 6.3, 6.3],
            "dk_fpts": [0.0] * 8,
        }
    )
    features_df = pd.DataFrame(
        {
            "game_id": [1, 1],
            "team_id": [10, 20],
            "player_id": [100, 200],
            "an_props_market_count": [0.0, 5.0],
            "an_has_pts": [0.0, 1.0],
            "an_pts_line": [pd.NA, 25.5],
        }
    )

    pre = worlds_df.copy()
    out, report = _apply_propless_tail_calibration_to_worlds(
        worlds_df,
        features_df=features_df,
        enabled=True,
        min_minutes_mean=18.0,
        min_dk_mean=0.0,
        tail_boost=0.2,
        max_tail_scale=1.3,
    )

    assert report["applied"] is True
    assert report["eligible_player_count"] == 1
    pre_propless = pre.loc[pre["player_id"] == 100, "pts"].max()
    post_propless = out.loc[out["player_id"] == 100, "pts"].max()
    assert post_propless > pre_propless
    pd.testing.assert_series_equal(
        out.loc[out["player_id"] == 200, "pts"].reset_index(drop=True),
        pre.loc[pre["player_id"] == 200, "pts"].reset_index(drop=True),
        check_names=False,
    )


def test_propless_tail_calibration_uses_has_signals_not_default_line_fill() -> None:
    worlds_df = pd.DataFrame(
        {
            "world_idx": [0, 1, 2, 3, 0, 1, 2, 3],
            "game_id": [1, 1, 1, 1, 1, 1, 1, 1],
            "team_id": [10, 10, 10, 10, 20, 20, 20, 20],
            "player_id": [100, 100, 100, 100, 200, 200, 200, 200],
            "minutes": [26.0, 26.0, 26.0, 26.0, 30.0, 30.0, 30.0, 30.0],
            "pts": [11.0, 12.0, 14.0, 24.0, 21.0, 22.0, 23.0, 31.0],
            "reb": [4.0, 5.0, 6.0, 8.0, 7.0, 7.0, 8.0, 9.0],
            "ast": [3.0, 3.0, 4.0, 6.0, 4.0, 4.0, 5.0, 6.0],
            "stl": [0.5, 0.5, 0.8, 1.2, 1.0, 1.0, 1.1, 1.2],
            "blk": [0.2, 0.3, 0.3, 0.8, 0.4, 0.5, 0.6, 0.7],
            "tov": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "oreb": [1.0, 1.2, 1.4, 2.0, 2.0, 2.1, 2.2, 2.4],
            "dreb": [3.0, 3.8, 4.6, 6.0, 5.0, 4.9, 5.8, 6.6],
            "dk_fpts": [0.0] * 8,
        }
    )
    # Lines are default-filled at 0.0 for both players; only has/market flags
    # should drive props-vs-propless detection.
    features_df = pd.DataFrame(
        {
            "game_id": [1, 1],
            "team_id": [10, 20],
            "player_id": [100, 200],
            "an_has_any_props": [0.0, 1.0],
            "an_props_market_count": [0.0, 4.0],
            "an_has_pts": [0.0, 1.0],
            "an_pts_line": [0.0, 0.0],
            "an_reb_line": [0.0, 0.0],
            "an_ast_line": [0.0, 0.0],
        }
    )

    pre = worlds_df.copy()
    out, report = _apply_propless_tail_calibration_to_worlds(
        worlds_df,
        features_df=features_df,
        enabled=True,
        min_minutes_mean=18.0,
        min_dk_mean=0.0,
        tail_boost=0.2,
        max_tail_scale=1.3,
    )

    assert report["applied"] is True
    assert report["eligible_player_count"] == 1
    assert out.loc[out["player_id"] == 100, "pts"].max() > pre.loc[pre["player_id"] == 100, "pts"].max()
    pd.testing.assert_series_equal(
        out.loc[out["player_id"] == 200, "pts"].reset_index(drop=True),
        pre.loc[pre["player_id"] == 200, "pts"].reset_index(drop=True),
        check_names=False,
    )
