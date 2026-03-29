from __future__ import annotations

import pandas as pd
import pytest

from scripts.rotation.run_gtv2_promotion_alignment import _apply_tree_rate_mean_override


def test_apply_tree_rate_mean_override_rescales_stats_and_recomputes_dk(tmp_path) -> None:
    worlds = pd.DataFrame(
        {
            "game_date": ["2026-03-01", "2026-03-01"],
            "game_id": [1, 1],
            "team_id": [10, 10],
            "player_id": [100, 100],
            "world_idx": [0, 1],
            "minutes": [30.0, 30.0],
            "pts": [20.0, 20.0],
            "oreb": [1.0, 1.0],
            "dreb": [2.0, 2.0],
            "ast": [2.0, 2.0],
            "stl": [1.0, 1.0],
            "blk": [0.0, 0.0],
            "tov": [2.0, 2.0],
            "reb": [3.0, 3.0],
            "dk_fpts": [0.0, 0.0],
        }
    )
    pred_csv = tmp_path / "tree_preds.csv"
    pd.DataFrame(
        {
            "game_date": ["2026-03-01"],
            "game_id": [1],
            "team_id": [10],
            "player_id": [100],
            "pred_ast_per_min": [0.2],
            "pred_oreb_per_min": [0.1],
            "pred_dreb_per_min": [0.2],
        }
    ).to_csv(pred_csv, index=False)

    out, report = _apply_tree_rate_mean_override(
        worlds,
        predictions_csv=pred_csv,
        blend_alpha=1.0,
    )

    assert report["applied"] is True
    assert report["player_count_with_predictions"] == 1
    assert out["ast"].mean() == pytest.approx(6.0)
    assert out["oreb"].mean() == pytest.approx(3.0)
    assert out["dreb"].mean() == pytest.approx(2.0)
    assert out["reb"].mean() == pytest.approx(5.0)
    assert out["dk_fpts"].mean() == pytest.approx(36.25)


def test_apply_tree_rate_mean_override_preserves_team_world_dreb_budget(tmp_path) -> None:
    worlds = pd.DataFrame(
        {
            "game_date": ["2026-03-01"] * 4,
            "game_id": [1] * 4,
            "team_id": [10] * 4,
            "player_id": [100, 101, 100, 101],
            "world_idx": [0, 0, 1, 1],
            "minutes": [30.0, 30.0, 30.0, 30.0],
            "pts": [20.0, 10.0, 20.0, 10.0],
            "oreb": [1.0, 1.0, 1.0, 1.0],
            "dreb": [8.0, 2.0, 8.0, 2.0],
            "ast": [2.0, 1.0, 2.0, 1.0],
            "stl": [1.0, 0.0, 1.0, 0.0],
            "blk": [0.0, 0.0, 0.0, 0.0],
            "tov": [2.0, 1.0, 2.0, 1.0],
            "reb": [9.0, 3.0, 9.0, 3.0],
            "dk_fpts": [0.0, 0.0, 0.0, 0.0],
        }
    )
    pred_csv = tmp_path / "tree_preds.csv"
    pd.DataFrame(
        {
            "game_date": ["2026-03-01", "2026-03-01"],
            "game_id": [1, 1],
            "team_id": [10, 10],
            "player_id": [100, 101],
            "pred_ast_per_min": [0.0, 0.0],
            "pred_dreb_per_min": [1.0 / 30.0, 19.0 / 30.0],
        }
    ).to_csv(pred_csv, index=False)

    out, report = _apply_tree_rate_mean_override(
        worlds,
        predictions_csv=pred_csv,
        blend_alpha=0.5,
        share_cap_mult=1.5,
        share_cap_add=0.05,
    )

    before_team = worlds.groupby(["game_date", "game_id", "team_id", "world_idx"])["dreb"].sum().reset_index(drop=True)
    after_team = out.groupby(["game_date", "game_id", "team_id", "world_idx"])["dreb"].sum().reset_index(drop=True)
    assert after_team.tolist() == pytest.approx(before_team.tolist())
    post_means = out.groupby("player_id")["dreb"].mean().to_dict()
    assert post_means[100] == pytest.approx(4.5)
    assert post_means[101] == pytest.approx(5.5)
    dreb_report = next(item for item in report["stat_reports"] if item.get("stat") == "dreb")
    assert dreb_report["mode"] == "team_budget_share_override"
    assert dreb_report["share_cap_applied"] is True
    assert dreb_report["clipped_group_count"] == 2


def test_apply_tree_rate_mean_override_bucket_hierarchy_preserves_bucket_totals(tmp_path) -> None:
    worlds = pd.DataFrame(
        {
            "game_date": ["2026-03-01"] * 6,
            "game_id": [1] * 6,
            "team_id": [10] * 6,
            "player_id": [100, 101, 102, 100, 101, 102],
            "world_idx": [0, 0, 0, 1, 1, 1],
            "minutes": [30.0] * 6,
            "pts": [20.0, 10.0, 8.0, 20.0, 10.0, 8.0],
            "oreb": [1.0] * 6,
            "dreb": [5.0, 4.0, 1.0, 5.0, 4.0, 1.0],
            "ast": [2.0, 1.0, 1.0, 2.0, 1.0, 1.0],
            "stl": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "blk": [0.0] * 6,
            "tov": [2.0, 1.0, 1.0, 2.0, 1.0, 1.0],
            "reb": [6.0, 5.0, 2.0, 6.0, 5.0, 2.0],
            "dk_fpts": [0.0] * 6,
        }
    )
    pred_csv = tmp_path / "tree_preds.csv"
    pd.DataFrame(
        {
            "game_date": ["2026-03-01", "2026-03-01", "2026-03-01"],
            "game_id": [1, 1, 1],
            "team_id": [10, 10, 10],
            "player_id": [100, 101, 102],
            "pred_dreb_per_min": [1.0 / 30.0, 19.0 / 30.0, 0.0],
        }
    ).to_csv(pred_csv, index=False)
    role_bucket_df = pd.DataFrame(
        {
            "game_date": ["2026-03-01", "2026-03-01", "2026-03-01"],
            "game_id": [1, 1, 1],
            "team_id": [10, 10, 10],
            "player_id": [100, 101, 102],
            "pos_bucket": ["B", "W", "W"],
        }
    )

    out, report = _apply_tree_rate_mean_override(
        worlds,
        predictions_csv=pred_csv,
        blend_alpha=1.0,
        role_bucket_df=role_bucket_df,
        dreb_bucket_hierarchy_enabled=True,
    )

    bucket_before = (
        worlds.merge(role_bucket_df, on=["game_date", "game_id", "team_id", "player_id"], how="left")
        .groupby(["game_date", "game_id", "team_id", "world_idx", "pos_bucket"])["dreb"]
        .sum()
        .sort_index()
    )
    bucket_after = (
        out.merge(role_bucket_df, on=["game_date", "game_id", "team_id", "player_id"], how="left")
        .groupby(["game_date", "game_id", "team_id", "world_idx", "pos_bucket"])["dreb"]
        .sum()
        .sort_index()
    )
    assert bucket_after.tolist() == pytest.approx(bucket_before.tolist())

    post_means = out.groupby("player_id")["dreb"].mean().to_dict()
    assert post_means[100] == pytest.approx(5.0)
    assert post_means[101] == pytest.approx(5.0)
    assert post_means[102] == pytest.approx(0.0)
    dreb_report = next(item for item in report["stat_reports"] if item.get("stat") == "dreb")
    assert dreb_report["bucket_hierarchy_enabled"] is True


def test_apply_tree_rate_mean_override_preserves_team_world_oreb_budget(tmp_path) -> None:
    worlds = pd.DataFrame(
        {
            "game_date": ["2026-03-01"] * 4,
            "game_id": [1] * 4,
            "team_id": [10] * 4,
            "player_id": [100, 101, 100, 101],
            "world_idx": [0, 0, 1, 1],
            "minutes": [30.0, 30.0, 30.0, 30.0],
            "pts": [20.0, 10.0, 20.0, 10.0],
            "oreb": [8.0, 2.0, 8.0, 2.0],
            "dreb": [1.0, 1.0, 1.0, 1.0],
            "ast": [2.0, 1.0, 2.0, 1.0],
            "stl": [1.0, 0.0, 1.0, 0.0],
            "blk": [0.0, 0.0, 0.0, 0.0],
            "tov": [2.0, 1.0, 2.0, 1.0],
            "reb": [9.0, 3.0, 9.0, 3.0],
            "dk_fpts": [0.0, 0.0, 0.0, 0.0],
        }
    )
    pred_csv = tmp_path / "tree_preds.csv"
    pd.DataFrame(
        {
            "game_date": ["2026-03-01", "2026-03-01"],
            "game_id": [1, 1],
            "team_id": [10, 10],
            "player_id": [100, 101],
            "pred_oreb_per_min": [1.0 / 30.0, 19.0 / 30.0],
        }
    ).to_csv(pred_csv, index=False)

    out, report = _apply_tree_rate_mean_override(
        worlds,
        predictions_csv=pred_csv,
        blend_alpha=0.5,
        oreb_share_override_enabled=True,
    )

    before_team = worlds.groupby(["game_date", "game_id", "team_id", "world_idx"])["oreb"].sum().reset_index(drop=True)
    after_team = out.groupby(["game_date", "game_id", "team_id", "world_idx"])["oreb"].sum().reset_index(drop=True)
    assert after_team.tolist() == pytest.approx(before_team.tolist())
    post_means = out.groupby("player_id")["oreb"].mean().to_dict()
    assert post_means[100] == pytest.approx(5.5)
    assert post_means[101] == pytest.approx(4.5)
    oreb_report = next(item for item in report["stat_reports"] if item.get("stat") == "oreb")
    assert oreb_report["mode"] == "team_budget_share_override"


def test_apply_tree_rate_mean_override_does_not_allocate_rebounds_to_zero_minute_rows(tmp_path) -> None:
    worlds = pd.DataFrame(
        {
            "game_date": ["2026-03-01"] * 4,
            "game_id": [1] * 4,
            "team_id": [10] * 4,
            "player_id": [100, 101, 100, 101],
            "world_idx": [0, 0, 1, 1],
            "minutes": [30.0, 0.0, 30.0, 0.0],
            "pts": [20.0, 0.0, 20.0, 0.0],
            "oreb": [8.0, 0.0, 8.0, 0.0],
            "dreb": [4.0, 0.0, 4.0, 0.0],
            "ast": [2.0, 0.0, 2.0, 0.0],
            "stl": [1.0, 0.0, 1.0, 0.0],
            "blk": [0.0, 0.0, 0.0, 0.0],
            "tov": [2.0, 0.0, 2.0, 0.0],
            "reb": [12.0, 0.0, 12.0, 0.0],
            "dk_fpts": [0.0, 0.0, 0.0, 0.0],
        }
    )
    pred_csv = tmp_path / "tree_preds.csv"
    pd.DataFrame(
        {
            "game_date": ["2026-03-01", "2026-03-01"],
            "game_id": [1, 1],
            "team_id": [10, 10],
            "player_id": [100, 101],
            "pred_oreb_per_min": [1.0 / 30.0, 19.0 / 30.0],
            "pred_dreb_per_min": [1.0 / 30.0, 19.0 / 30.0],
        }
    ).to_csv(pred_csv, index=False)

    out, _ = _apply_tree_rate_mean_override(
        worlds,
        predictions_csv=pred_csv,
        blend_alpha=0.75,
        oreb_share_override_enabled=True,
    )

    zero_min_rows = out["minutes"] <= 1e-9
    assert out.loc[zero_min_rows, "oreb"].abs().max() == pytest.approx(0.0)
    assert out.loc[zero_min_rows, "dreb"].abs().max() == pytest.approx(0.0)


def test_apply_tree_rate_mean_override_no_matching_players_is_noop(tmp_path) -> None:
    worlds = pd.DataFrame(
        {
            "game_date": ["2026-03-01", "2026-03-01"],
            "game_id": [1, 1],
            "team_id": [10, 10],
            "player_id": [100, 101],
            "world_idx": [0, 0],
            "minutes": [30.0, 28.0],
            "pts": [20.0, 15.0],
            "oreb": [2.0, 1.0],
            "dreb": [6.0, 5.0],
            "ast": [7.0, 4.0],
            "stl": [1.0, 0.0],
            "blk": [0.0, 1.0],
            "tov": [3.0, 2.0],
            "reb": [8.0, 6.0],
            "dk_fpts": [40.5, 31.0],
        }
    )
    pred_csv = tmp_path / "tree_preds.csv"
    pd.DataFrame(
        {
            "game_date": ["2026-03-02"],
            "game_id": [2],
            "team_id": [99],
            "player_id": [999],
            "pred_ast_per_min": [0.4],
        }
    ).to_csv(pred_csv, index=False)

    out, report = _apply_tree_rate_mean_override(
        worlds,
        predictions_csv=pred_csv,
        blend_alpha=0.75,
    )

    assert report["applied"] is False
    assert report["player_count_with_predictions"] == 0
    assert report["skip_reason"] == "no_matching_players"
    pd.testing.assert_frame_equal(out, worlds)
