from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from projections.contest_sim import contest_sim_service


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_load_worlds_matrix_falls_back_to_gtv2_worlds(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / "artifacts"
        / "gtv2_worlds"
        / "game_date=2026-02-28"
        / "run=20260228T204500Z"
    )
    worlds_df = pd.DataFrame(
        [
            {"world_idx": 0, "player_id": 201950, "dk_fpts": 30.0},
            {"world_idx": 0, "player_id": 203924, "dk_fpts": 20.0},
            {"world_idx": 1, "player_id": 201950, "dk_fpts": 32.0},
            {"world_idx": 1, "player_id": 203924, "dk_fpts": 18.0},
        ]
    )
    _write_parquet(run_dir / "worlds.parquet", worlds_df)

    worlds_matrix, player_index = contest_sim_service.load_worlds_matrix(
        "2026-02-28",
        data_root=tmp_path,
        run_id="20260228T204500Z",
    )

    assert player_index == {"201950": 0, "203924": 1}
    assert worlds_matrix.shape == (2, 2)
    assert np.allclose(worlds_matrix, np.array([[30.0, 20.0], [32.0, 18.0]]))


def test_load_worlds_matrix_gtv2_drops_invalid_rows_without_stringifying_full_column(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "artifacts"
        / "gtv2_worlds"
        / "game_date=2026-03-01"
        / "run=20260301T130002Z"
    )
    worlds_df = pd.DataFrame(
        [
            {"world_idx": 0, "player_id": 101, "dk_fpts": 11.0},
            {"world_idx": 0, "player_id": 202, "dk_fpts": 22.0},
            {"world_idx": 1, "player_id": 101, "dk_fpts": 33.0},
            {"world_idx": None, "player_id": 999, "dk_fpts": 99.0},
            {"world_idx": 1, "player_id": None, "dk_fpts": 44.0},
        ]
    )
    _write_parquet(run_dir / "worlds.parquet", worlds_df)

    worlds_matrix, player_index = contest_sim_service.load_worlds_matrix(
        "2026-03-01",
        data_root=tmp_path,
        run_id="20260301T130002Z",
    )

    assert player_index == {"101": 0, "202": 1}
    assert worlds_matrix.shape == (2, 2)
    assert np.allclose(worlds_matrix, np.array([[11.0, 22.0], [33.0, 0.0]]))


def test_load_worlds_matrix_gtv2_handles_non_numeric_player_ids(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / "artifacts"
        / "gtv2_worlds"
        / "game_date=2026-03-02"
        / "run=20260302T130002Z"
    )
    worlds_df = pd.DataFrame(
        [
            {"world_idx": 0, "player_id": "alpha", "dk_fpts": 9.0},
            {"world_idx": 0, "player_id": "beta", "dk_fpts": 12.0},
            {"world_idx": 1, "player_id": "alpha", "dk_fpts": 15.0},
            {"world_idx": 1, "player_id": "beta", "dk_fpts": 18.0},
        ]
    )
    _write_parquet(run_dir / "worlds.parquet", worlds_df)

    worlds_matrix, player_index = contest_sim_service.load_worlds_matrix(
        "2026-03-02",
        data_root=tmp_path,
        run_id="20260302T130002Z",
    )

    assert player_index == {"alpha": 0, "beta": 1}
    assert worlds_matrix.shape == (2, 2)
    assert np.allclose(worlds_matrix, np.array([[9.0, 12.0], [15.0, 18.0]]))
