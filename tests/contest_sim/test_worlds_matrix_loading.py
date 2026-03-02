from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from projections.contest_sim.contest_sim_service import load_worlds_matrix


def _write_world(
    path: Path, *, world_id: int, player_ids: list[int], fpts: list[float], fpts_col: str = "dk_fpts_world"
) -> None:
    df = pd.DataFrame(
        {
            "world_id": [world_id] * len(player_ids),
            "player_id": player_ids,
            fpts_col: fpts,
        }
    )
    df.to_parquet(path, index=False)


def test_load_worlds_matrix_aggregates_sparse_world_files(tmp_path: Path) -> None:
    base = tmp_path / "artifacts" / "sim_v2" / "worlds_fpts_v2" / "game_date=2099-01-01"
    base.mkdir(parents=True, exist_ok=True)

    _write_world(base / "world=0000.parquet", world_id=0, player_ids=[1, 2], fpts=[10.0, 20.0])
    _write_world(base / "world=0001.parquet", world_id=1, player_ids=[2, 3], fpts=[30.0, 40.0])

    worlds, player_index = load_worlds_matrix(
        "2099-01-01",
        data_root=tmp_path,
        worlds_source="sim_v2",
    )
    assert worlds.shape == (2, 3)
    assert set(player_index.keys()) == {"1", "2", "3"}
    assert np.isclose(worlds[0, player_index["1"]], 10.0)
    assert np.isclose(worlds[0, player_index["2"]], 20.0)
    assert np.isclose(worlds[0, player_index["3"]], 0.0)  # missing => 0
    assert np.isclose(worlds[1, player_index["1"]], 0.0)  # missing => 0
    assert np.isclose(worlds[1, player_index["2"]], 30.0)
    assert np.isclose(worlds[1, player_index["3"]], 40.0)


def test_load_worlds_matrix_synthetic_respects_play_prob(tmp_path: Path) -> None:
    base = tmp_path / "artifacts" / "sim_v2" / "worlds_fpts_v2" / "game_date=2099-01-02"
    base.mkdir(parents=True, exist_ok=True)

    proj = pd.DataFrame(
        {
            "player_id": ["1", "2"],
            "dk_fpts_mean": [20.0, 20.0],
            "dk_fpts_std": [0.1, 0.1],
            "play_prob": [0.0, 1.0],
        }
    )
    proj.to_parquet(base / "projections.parquet", index=False)

    worlds, player_index = load_worlds_matrix(
        "2099-01-02",
        data_root=tmp_path,
        worlds_source="sim_v2",
        n_synthetic_worlds=256,
        seed=123,
    )
    assert worlds.shape == (256, 2)

    col_out = player_index["1"]
    col_in = player_index["2"]
    assert np.all(worlds[:, col_out] == 0.0)
    assert float(np.max(worlds[:, col_in])) > 0.0


def test_load_worlds_matrix_defaults_to_gtv2_and_fails_loud_on_missing(tmp_path: Path) -> None:
    base = tmp_path / "artifacts" / "sim_v2" / "worlds_fpts_v2" / "game_date=2099-01-03"
    base.mkdir(parents=True, exist_ok=True)
    _write_world(base / "world=0000.parquet", world_id=0, player_ids=[1], fpts=[10.0])

    try:
        load_worlds_matrix("2099-01-03", data_root=tmp_path)
    except FileNotFoundError as exc:
        assert "No gtv2 worlds data" in str(exc)
    else:
        raise AssertionError("Expected missing gtv2 worlds to fail for default live loader")
