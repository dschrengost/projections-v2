"""Unit tests for injury-regime evaluation helpers."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from projections.cli.eval_minutes_injury_regime import (
    _build_eval_slices,
    _build_injury_regime_table,
    _compress_to_top_k,
    _compute_metrics,
    _load_labels,
)


def test_compress_to_top_k_scales_to_240_and_keeps_k():
    df = pd.DataFrame(
        {
            "game_id": [1] * 10,
            "team_id": [100] * 10,
            "_pred_minutes": [40, 30, 20, 10, 5, 4, 3, 2, 1, 0],
        }
    )
    compressed = _compress_to_top_k(df, pred_col="_pred_minutes", k=8, cap=48.0)
    assert compressed.sum() == pytest.approx(240.0)
    assert int((compressed > 0).sum()) == 8


def test_build_injury_regime_table_counts_prev_starters_out():
    # Two games for one team, ordered by tip_ts.
    labels = pd.DataFrame(
        {
            "game_date": [pd.Timestamp("2025-11-01"), pd.Timestamp("2025-11-04")],
            "game_id": [10, 20],
            "team_id": [1, 1],
            "player_id": [101, 101],
            "minutes": [30.0, 0.0],
            "starter_flag_label": [1, 0],
        }
    )
    # Add remaining starters for game 10.
    starters = pd.DataFrame(
        {
            "game_date": [pd.Timestamp("2025-11-01")] * 4,
            "game_id": [10] * 4,
            "team_id": [1] * 4,
            "player_id": [102, 103, 104, 105],
            "minutes": [30.0, 30.0, 30.0, 30.0],
            "starter_flag_label": [1, 1, 1, 1],
        }
    )
    labels = pd.concat([labels, starters], ignore_index=True)

    features = pd.DataFrame(
        {
            "game_date": [pd.Timestamp("2025-11-01")] * 6 + [pd.Timestamp("2025-11-04")] * 6,
            "game_id": [10] * 6 + [20] * 6,
            "team_id": [1] * 12,
            "player_id": [101, 102, 103, 104, 105, 200, 101, 102, 103, 104, 105, 200],
            "tip_ts": [pd.Timestamp("2025-11-01T00:00:00Z")] * 6 + [pd.Timestamp("2025-11-04T00:00:00Z")] * 6,
            "_out_flag": [False] * 6 + [True, False, False, False, False, False],
        }
    )

    table = _build_injury_regime_table(labels=labels, features=features, min_starters_out=1, min_team_out=99)
    row = table[(table["game_id"] == 20) & (table["team_id"] == 1)].iloc[0]
    assert int(row["starter_out_count"]) == 1
    assert bool(row["injury_regime"]) is True


def test_build_eval_slices_enforces_strict_non_injury():
    eval_frame = pd.DataFrame(
        {
            "game_id": [1, 1, 2],
            "team_id": [10, 10, 20],
            "player_id": [101, 102, 201],
            "injury_regime": [True, False, False],
            "starter_out_count": [1, 0, 0],
            "team_out_count": [2, 0, 1],
        }
    )

    slices = _build_eval_slices(eval_frame)
    assert len(slices["all_games"]) == len(eval_frame)
    assert len(slices["non_injury"]) == 1
    assert slices["non_injury"]["starter_out_count"].max() == 0
    assert slices["non_injury"]["team_out_count"].max() == 0
    assert len(slices["all_games"]) > len(slices["non_injury"])


def test_load_labels_prefers_legacy_boxscore_labels(tmp_path):
    legacy_root = tmp_path / "labels" / "season=2025"
    legacy_root.mkdir(parents=True)
    gold_root = tmp_path / "gold" / "labels_minutes_v1" / "season=2025" / "game_date=2025-11-01"
    gold_root.mkdir(parents=True)

    legacy = pd.DataFrame(
        {
            "game_id": [1] * 12,
            "team_id": [100] * 6 + [200] * 6,
            "player_id": list(range(1000, 1006)) + list(range(2000, 2006)),
            "game_date": [pd.Timestamp("2025-11-01")] * 12,
            "minutes": [40, 38, 35, 32, 30, 5, 41, 39, 36, 31, 28, 4],
        }
    )
    gold = pd.DataFrame(
        {
            "game_id": [999] * 10,
            "team_id": [100] * 5 + [200] * 5,
            "player_id": list(range(3000, 3005)) + list(range(4000, 4005)),
            "game_date": [pd.Timestamp("2025-11-01")] * 10,
            "minutes": [48] * 10,
            "starter_flag_label": [1] * 10,
        }
    )

    legacy.to_parquet(legacy_root / "boxscore_labels.parquet", index=False)
    gold.to_parquet(gold_root / "labels.parquet", index=False)

    loaded = _load_labels(data_root=tmp_path, start=date(2025, 11, 1), end=date(2025, 11, 1))
    assert loaded["game_id"].nunique() == 1
    assert int(loaded["game_id"].iloc[0]) == 1

    starter_sums = loaded.groupby(["game_id", "team_id"], sort=False)["starter_flag_label"].sum()
    assert (starter_sums == 5).all()


def test_load_labels_handles_mixed_minutes_formats_across_seasons(tmp_path):
    legacy_2024 = tmp_path / "labels" / "season=2024"
    legacy_2025 = tmp_path / "labels" / "season=2025"
    legacy_2024.mkdir(parents=True)
    legacy_2025.mkdir(parents=True)

    # season=2024 uses ISO8601 duration strings; season=2025 uses numeric floats.
    season_2024 = pd.DataFrame(
        {
            "game_id": [1] * 10,
            "team_id": [100] * 5 + [200] * 5,
            "player_id": list(range(1000, 1005)) + list(range(2000, 2005)),
            "game_date": [pd.Timestamp("2025-04-10")] * 10,
            "minutes": ["PT38M0S", "PT30M0S", "PT20M0S", "PT10M0S", "PT5M0S"] * 2,
        }
    )
    season_2025 = pd.DataFrame(
        {
            "game_id": [2] * 10,
            "team_id": [100] * 5 + [200] * 5,
            "player_id": list(range(1100, 1105)) + list(range(2100, 2105)),
            "game_date": [pd.Timestamp("2025-10-22")] * 10,
            "minutes": [40.0, 35.0, 30.0, 25.0, 10.0] * 2,
        }
    )

    season_2024.to_parquet(legacy_2024 / "boxscore_labels.parquet", index=False)
    season_2025.to_parquet(legacy_2025 / "boxscore_labels.parquet", index=False)

    loaded = _load_labels(data_root=tmp_path, start=date(2025, 4, 1), end=date(2025, 10, 31))
    # Ensure the mixed formats are preserved (not coerced to 0.0 for the numeric season).
    assert float(loaded.loc[loaded["game_id"] == 2, "minutes"].max()) == pytest.approx(40.0)
    assert float(loaded.loc[loaded["game_id"] == 1, "minutes"].max()) == pytest.approx(38.0)

    starter_sums = loaded.groupby(["game_id", "team_id"], sort=False)["starter_flag_label"].sum()
    assert (starter_sums == 5).all()


def test_compute_metrics_top9_and_tail_minutes_are_defined_on_actual_top9():
    df = pd.DataFrame(
        {
            "game_id": [1] * 12,
            "team_id": [10] * 12,
            "player_id": list(range(100, 112)),
            "minutes": [36, 34, 32, 30, 28, 20, 18, 16, 14, 6, 4, 2],
            "starter_flag_label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "starter_out_count": [0] * 12,
        }
    )
    df["_pred_minutes"] = [34, 33, 31, 29, 27, 18, 14, 12, 10, 8, 7, 7]

    metrics = _compute_metrics(df, pred_col="_pred_minutes")

    assert metrics.top9_sum_actual_mean == pytest.approx(228.0)
    assert metrics.top9_sum_pred_mean == pytest.approx(208.0)
    assert metrics.top9_sum_bias == pytest.approx(-20.0)
    assert metrics.tail_minutes_actual_mean == pytest.approx(12.0)
    assert metrics.tail_minutes_pred_mean == pytest.approx(32.0)
    assert metrics.tail_minutes_bias == pytest.approx(20.0)
    assert metrics.top9_player_mae == pytest.approx(20.0 / 9.0)

    # Team-240 scaling should reduce top9 error magnitude when total != 240.
    assert metrics.top9_sum_mae_team240 < metrics.top9_sum_mae
    assert metrics.top9_player_mae_team240 < metrics.top9_player_mae
