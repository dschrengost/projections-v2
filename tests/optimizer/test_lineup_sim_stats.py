from __future__ import annotations

import numpy as np
import pandas as pd

from projections.optimizer.lineup_sim_stats import (
    compute_lineup_distribution_stats,
    load_world_fpts_matrix,
)


def test_compute_lineup_distribution_stats_matches_direct_percentiles() -> None:
    rng = np.random.default_rng(123)
    fpts_by_world = rng.normal(loc=0.0, scale=1.0, size=(200, 3)).astype(np.float32)
    world_player_ids = np.array([101, 202, 303], dtype=int)
    lineups = [
        ("101", "202"),
        ("202", "303"),
        ("101", "303"),
    ]

    stats = compute_lineup_distribution_stats(
        lineups=lineups,
        world_player_ids=world_player_ids,
        fpts_by_world=fpts_by_world,
    )
    assert len(stats) == len(lineups)

    pid_to_col = {pid: idx for idx, pid in enumerate(world_player_ids)}
    for lu, out in zip(lineups, stats, strict=True):
        cols = [pid_to_col[int(pid)] for pid in lu]
        totals = fpts_by_world[:, cols].sum(axis=1)
        expected = np.percentile(totals, [10, 50, 75, 90]).astype(float)
        assert out.mean == float(np.mean(totals))
        assert out.stdev == float(np.std(totals, ddof=0))
        assert out.p10 == float(expected[0])
        assert out.p50 == float(expected[1])
        assert out.p75 == float(expected[2])
        assert out.p90 == float(expected[3])
        assert out.ceiling_upside == float(expected[3] - np.mean(totals))


def test_lineup_p90_is_not_sum_of_player_p90s() -> None:
    # Construct a case where player-level p90s are small (interpolated),
    # but lineup p90 is large because in 20% of worlds exactly one player spikes.
    n_worlds = 100
    player_a = np.zeros(n_worlds, dtype=np.float32)
    player_b = np.zeros(n_worlds, dtype=np.float32)
    player_a[90:] = 10.0  # spikes in last 10%
    player_b[80:90] = 10.0  # spikes in a different 10%

    fpts_by_world = np.stack([player_a, player_b], axis=1)  # (W, P)
    world_player_ids = np.array([1, 2], dtype=int)
    lineups = [("1", "2")]

    player_p90_sum = float(np.percentile(player_a, 90) + np.percentile(player_b, 90))
    stats = compute_lineup_distribution_stats(
        lineups=lineups,
        world_player_ids=world_player_ids,
        fpts_by_world=fpts_by_world,
    )[0]

    totals = player_a + player_b
    expected_lineup_p90 = float(np.percentile(totals, 90))

    assert stats.p90 == expected_lineup_p90
    assert stats.p90 != player_p90_sum


def test_load_world_fpts_matrix_roundtrips_world_files(tmp_path) -> None:
    worlds_dir = tmp_path / "game_date=2099-01-01"
    worlds_dir.mkdir(parents=True, exist_ok=True)

    df0 = pd.DataFrame(
        {
            "world_id": [0, 0],
            "player_id": [11, 22],
            "dk_fpts_world": [1.5, 2.5],
        }
    )
    df1 = pd.DataFrame(
        {
            "world_id": [1, 1],
            "player_id": [11, 22],
            "dk_fpts_world": [3.0, 4.0],
        }
    )

    df0.to_parquet(worlds_dir / "world=0000.parquet", index=False)
    df1.to_parquet(worlds_dir / "world=0001.parquet", index=False)

    world_ids, player_ids, fpts = load_world_fpts_matrix(
        worlds_dir=worlds_dir,
        player_ids=[11, 22],
    )

    assert world_ids.tolist() == [0, 1]
    assert player_ids.tolist() == [11, 22]
    assert fpts.shape == (2, 2)
    assert fpts[0, 0] == 1.5
    assert fpts[0, 1] == 2.5
    assert fpts[1, 0] == 3.0
    assert fpts[1, 1] == 4.0

