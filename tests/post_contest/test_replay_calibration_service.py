from __future__ import annotations

import pandas as pd
from pathlib import Path

from projections.post_contest import replay_calibration_service


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_build_replay_calibration_artifacts_aggregates_outputs(tmp_path: Path) -> None:
    analytics_dir = (
        tmp_path
        / "analytics"
        / "contest_flashback"
        / "date=2099-01-01"
        / "contest_id=123"
        / "user=daniel"
        / "analytics"
    )
    analytics_dir.mkdir(parents=True)
    (analytics_dir / "summary.json").write_text("{}")

    _write_parquet(
        analytics_dir / "player_calibration.parquet",
        [
            {
                "game_date": "2099-01-01",
                "contest_id": "123",
                "draft_group_id": 999,
                "player_id": "1",
                "proj_fpts": 30.0,
                "proj_ownership_pct": 20.0,
                "actual_player_fpts": 35.0,
                "sim_mean_fpts": 30.0,
                "sim_p10_fpts": 20.0,
                "sim_p90_fpts": 40.0,
                "actual_fpts_sim_percentile": 0.75,
                "actual_minutes": 32.0,
                "sim_mean_minutes": 30.0,
                "sim_p10_minutes": 24.0,
                "sim_p90_minutes": 36.0,
                "actual_minutes_sim_percentile": 0.65,
                "actual_contest_own_pct": 25.0,
                "actual_opponent_own_pct": 20.0,
                "modeled_field_own_pct": 18.0,
            },
            {
                "game_date": "2099-01-01",
                "contest_id": "123",
                "draft_group_id": 999,
                "player_id": "2",
                "proj_fpts": 32.0,
                "proj_ownership_pct": 22.0,
                "actual_player_fpts": 15.0,
                "sim_mean_fpts": 28.0,
                "sim_p10_fpts": 18.0,
                "sim_p90_fpts": 38.0,
                "actual_fpts_sim_percentile": 0.05,
                "actual_minutes": 20.0,
                "sim_mean_minutes": 28.0,
                "sim_p10_minutes": 22.0,
                "sim_p90_minutes": 34.0,
                "actual_minutes_sim_percentile": 0.10,
                "actual_contest_own_pct": 18.0,
                "actual_opponent_own_pct": 16.0,
                "modeled_field_own_pct": 14.0,
            },
        ],
    )
    _write_parquet(
        analytics_dir / "lineup_calibration.parquet",
        [
            {
                "game_date": "2099-01-01",
                "contest_id": "123",
                "draft_group_id": 999,
                "lineup_key": "1|2|3|4|5|6|7|8",
                "lineup_source": "entered",
                "sim_roi": 0.2,
                "sim_cash_rate": 0.3,
                "realized_rank": 10,
                "realized_prize": 5.0,
            },
            {
                "game_date": "2099-01-01",
                "contest_id": "123",
                "draft_group_id": 999,
                "lineup_key": "9|10|11|12|13|14|15|16",
                "lineup_source": "candidate",
                "sim_roi": 0.5,
                "sim_cash_rate": 0.4,
            },
        ],
    )
    _write_parquet(
        analytics_dir / "field_calibration.parquet",
        [
            {
                "game_date": "2099-01-01",
                "contest_id": "123",
                "draft_group_id": 999,
                "actual_field_size": 1000,
                "actual_dupe_rate": 0.20,
                "modeled_dupe_rate": 0.15,
                "actual_salary_left_mean": 500.0,
                "modeled_salary_left_mean": 700.0,
                "actual_projected_own_sum_mean": 180.0,
                "modeled_projected_own_sum_mean": 195.0,
                "player_ownership_mae_pct": 5.0,
                "top20_player_ownership_mae_pct": 8.0,
                "salary_left_hist_l1": 0.3,
                "projected_own_sum_hist_l1": 0.2,
                "dupe_hist_l1": 0.1,
            }
        ],
    )
    _write_parquet(
        analytics_dir / "regret_summary.parquet",
        [
            {
                "game_date": "2099-01-01",
                "contest_id": "123",
                "draft_group_id": 999,
                "candidate_pool_available": True,
                "selection_regret_roi": 0.3,
                "selection_regret_cash_rate": 0.1,
            }
        ],
    )

    bundle = replay_calibration_service.build_replay_calibration_artifacts(
        data_root=tmp_path,
        output_dir=tmp_path / "gold" / "replay_calibration",
    )

    assert bundle.player_fpts_calibration_path.exists()
    assert bundle.player_minutes_calibration_path.exists()
    assert bundle.ownership_recalibration_path.exists()
    assert bundle.field_model_calibration_path.exists()
    assert bundle.optimizer_regret_by_contest_path.exists()
    assert bundle.optimizer_regret_by_bucket_path.exists()
    assert bundle.optimizer_regret_examples_path.exists()
    assert bundle.summary_path.exists()


def test_optimizer_regret_frames_dedupes_field_join_rows() -> None:
    regret_df = pd.DataFrame(
        [
            {
                "game_date": "2099-01-01",
                "contest_id": "123",
                "draft_group_id": 999,
                "candidate_pool_available": True,
                "selection_regret_roi": 0.1,
                "selection_regret_cash_rate": 0.02,
            }
        ]
    )
    lineup_df = pd.DataFrame()
    field_df = pd.DataFrame(
        [
            {"game_date": "2099-01-01", "contest_id": "123", "draft_group_id": 999, "actual_field_size": 1000},
            {"game_date": "2099-01-01", "contest_id": "123", "draft_group_id": 999, "actual_field_size": 1000},
        ]
    )

    by_contest, by_bucket, _ = replay_calibration_service._optimizer_regret_frames(regret_df, lineup_df, field_df)

    assert len(by_contest) == 1
    assert len(by_bucket) == 1
    assert by_bucket.iloc[0]["contest_count"] == 1
