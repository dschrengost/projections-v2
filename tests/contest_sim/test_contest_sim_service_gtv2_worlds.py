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
