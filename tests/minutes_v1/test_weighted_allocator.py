import numpy as np
import pandas as pd

from projections.cli import score_minutes_v1 as score_cli
from projections.minutes_v1.reconcile import ReconcileConfig, reconcile_minutes_p50_all


def _team_frame(
    *,
    game_id: int,
    team_id: int,
    minutes_p50: list[float],
    starters: list[int],
    play_prob: float = 1.0,
    status: str = "Ava",
) -> pd.DataFrame:
    if len(minutes_p50) != len(starters):
        raise ValueError("minutes_p50 and starters must have the same length")
    n = len(minutes_p50)
    return pd.DataFrame(
        {
            "game_id": [game_id] * n,
            "team_id": [team_id] * n,
            "player_id": list(range(1000, 1000 + n)),
            "minutes_p50": minutes_p50,
            "minutes_p10": [0.0] * n,
            "minutes_p90": [60.0] * n,
            "starter_flag": starters,
            "is_projected_starter": starters,
            "play_prob": [play_prob] * n,
            "status": [status] * n,
        }
    )


def test_weighted_allocator_non_uniform_when_weights_vary() -> None:
    df = _team_frame(
        game_id=1,
        team_id=10,
        minutes_p50=[34, 32, 30, 28, 26, 20, 18, 14, 10, 6, 5, 4],
        starters=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    )
    cfg = ReconcileConfig(method="weighted", min_rotation_size=8, max_rotation_size=13, rotation_mass_cutoff=1.0)
    out = reconcile_minutes_p50_all(df, cfg)

    assert np.isclose(out["minutes_p50"].sum(), 240.0, rtol=0, atol=1e-9)

    starters = out[(out["starter_flag"] > 0) & (out["minutes_p50"] > 0.01)]["minutes_p50"].to_numpy()
    bench = out[(out["starter_flag"] == 0) & (out["minutes_p50"] > 0.01)]["minutes_p50"].to_numpy()
    assert starters.size == 5
    assert bench.size >= 3
    assert float(np.std(starters)) > 0.25
    assert float(np.std(bench)) > 0.25


def test_weighted_allocator_depth_is_not_forced_to_10() -> None:
    df_a = _team_frame(
        game_id=1,
        team_id=100,
        minutes_p50=[28] * 5 + [10] * 8,  # 13 candidates
        starters=[1, 1, 1, 1, 1] + [0] * 8,
    )
    df_b = _team_frame(
        game_id=1,
        team_id=200,
        minutes_p50=[28] * 5 + [10] * 6,  # 11 candidates
        starters=[1, 1, 1, 1, 1] + [0] * 6,
    )
    df = pd.concat([df_a, df_b], ignore_index=True)
    cfg = ReconcileConfig(method="weighted", min_rotation_size=8, max_rotation_size=13, rotation_mass_cutoff=1.0)
    out = reconcile_minutes_p50_all(df, cfg)

    eps = 0.01
    nz = out.groupby(["game_id", "team_id"])["minutes_p50"].apply(lambda s: int((s > eps).sum()))
    assert nz.loc[(1, 100)] == 13
    assert nz.loc[(1, 200)] == 11


def test_equal_weights_allow_uniform_but_flagged() -> None:
    df = _team_frame(
        game_id=2,
        team_id=20,
        minutes_p50=[20.0] * 10,
        starters=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    )
    # No rank columns present, so the allocator should keep a uniform split.
    cfg = ReconcileConfig(method="weighted", min_rotation_size=8, max_rotation_size=10, rotation_mass_cutoff=1.0)
    out = reconcile_minutes_p50_all(df, cfg)

    nz = out[out["minutes_p50"] > 0.01]
    starters = nz[nz["starter_flag"] > 0]["minutes_p50"].to_numpy()
    bench = nz[nz["starter_flag"] == 0]["minutes_p50"].to_numpy()
    assert float(np.std(starters)) == 0.0
    assert float(np.std(bench)) == 0.0

    diag = score_cli._compute_allocation_diagnostics(out, eps=0.01, std_threshold=0.25)
    assert len(diag["per_team_game"]) == 1
    entry = diag["per_team_game"][0]
    assert entry["uniform_split_starter"] is True
    assert entry["uniform_split_bench"] is True


def test_out_players_are_zeroed() -> None:
    df = _team_frame(
        game_id=3,
        team_id=30,
        minutes_p50=[30, 28, 26, 24, 22, 18, 14, 10],
        starters=[1, 1, 1, 1, 1, 0, 0, 0],
    )
    df.loc[7, "status"] = "OUT"
    df.loc[7, "play_prob"] = 0.0
    cfg = ReconcileConfig(method="weighted", min_rotation_size=8, max_rotation_size=13, rotation_mass_cutoff=1.0)
    out = reconcile_minutes_p50_all(df, cfg)

    assert float(out.loc[7, "minutes_p50"]) == 0.0


def test_weighted_allocator_is_deterministic() -> None:
    df = _team_frame(
        game_id=4,
        team_id=40,
        minutes_p50=[34, 32, 30, 28, 26, 20, 18, 14, 10, 6],
        starters=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    )
    cfg = ReconcileConfig(method="weighted", min_rotation_size=8, max_rotation_size=13, rotation_mass_cutoff=0.995)
    out1 = reconcile_minutes_p50_all(df, cfg)
    out2 = reconcile_minutes_p50_all(df, cfg)
    assert np.allclose(out1["minutes_p50"].to_numpy(), out2["minutes_p50"].to_numpy(), rtol=0, atol=0.0)

