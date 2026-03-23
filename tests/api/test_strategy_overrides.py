from __future__ import annotations

import numpy as np
import pandas as pd

from projections.api import optimizer_service, strategy_overrides
from projections.contest_sim.contest_sim_service import PlayerWorlds


def test_apply_strategy_overrides_to_worlds_preserves_zero_worlds_and_scales_active_minutes() -> None:
    overrides = strategy_overrides.SlateStrategyOverrides(
        game_date="2026-03-01",
        draft_group_id=123,
        overrides={
            "1": strategy_overrides.PlayerStrategyOverride(
                player_id="1",
                minutes_delta=5.0,
            )
        },
    )

    adjusted_fpts, adjusted_minutes, diagnostics = strategy_overrides.apply_strategy_overrides_to_worlds(
        fpts_matrix=np.array([[10.0], [20.0], [0.0]], dtype=np.float64),
        minutes_matrix=np.array([[10.0], [20.0], [0.0]], dtype=np.float64),
        player_index={"1": 0},
        overrides=overrides,
        model_minutes_by_player={"1": 15.0},
        model_fpts_by_player={"1": 10.0},
    )

    np.testing.assert_allclose(adjusted_fpts[:, 0], np.array([15.0, 25.0, 0.0]))
    assert adjusted_minutes is not None
    np.testing.assert_allclose(adjusted_minutes[:, 0], np.array([15.0, 25.0, 0.0]))
    assert diagnostics["matched_override_count"] == 1
    assert diagnostics["applied_minutes_delta_count"] == 1
    assert diagnostics["applied_fpts_delta_count"] == 0


def test_apply_strategy_overrides_to_worlds_composes_minutes_then_fpts_delta() -> None:
    overrides = strategy_overrides.SlateStrategyOverrides(
        game_date="2026-03-01",
        draft_group_id=123,
        overrides={
            "1": strategy_overrides.PlayerStrategyOverride(
                player_id="1",
                minutes_delta=10.0,
                fpts_delta=5.0,
            )
        },
    )

    adjusted_fpts, adjusted_minutes, diagnostics = strategy_overrides.apply_strategy_overrides_to_worlds(
        fpts_matrix=np.array([[10.0], [20.0], [0.0]], dtype=np.float64),
        minutes_matrix=np.array([[10.0], [20.0], [0.0]], dtype=np.float64),
        player_index={"1": 0},
        overrides=overrides,
        model_minutes_by_player={"1": 15.0},
        model_fpts_by_player={"1": 10.0},
    )

    np.testing.assert_allclose(adjusted_fpts[:, 0], np.array([24.0, 36.0, 0.0]))
    assert adjusted_minutes is not None
    np.testing.assert_allclose(adjusted_minutes[:, 0], np.array([20.0, 30.0, 0.0]))
    assert diagnostics["applied_minutes_delta_count"] == 1
    assert diagnostics["applied_fpts_delta_count"] == 1


def test_build_player_pool_uses_adjusted_world_summaries_when_strategy_overrides_enabled(monkeypatch) -> None:
    proj_df = pd.DataFrame(
        [
            {
                "player_id": "1",
                "player_name": "Alpha Guard",
                "team_tricode": "AAA",
                "proj_fpts": 10.0,
                "minutes": 10.0,
            }
        ]
    )
    sal_df = pd.DataFrame(
        [
            {
                "dk_player_id": 101,
                "display_name": "Alpha Guard",
                "positions": ["PG"],
                "salary": 7000,
                "team_abbrev": "AAA",
                "status": None,
                "is_disabled": False,
            }
        ]
    )

    monkeypatch.setattr(optimizer_service, "load_projections_for_date", lambda *args, **kwargs: proj_df)
    monkeypatch.setattr(optimizer_service, "load_salaries_for_date", lambda *args, **kwargs: sal_df)
    monkeypatch.setattr(optimizer_service, "load_ownership_for_date", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "projections.api.strategy_overrides.load_slate_strategy_overrides",
        lambda game_date, draft_group_id: strategy_overrides.SlateStrategyOverrides(
            game_date=game_date,
            draft_group_id=draft_group_id,
            overrides={
                "1": strategy_overrides.PlayerStrategyOverride(
                    player_id="1",
                    minutes_delta=10.0,
                )
            },
        ),
    )
    monkeypatch.setattr(
        "projections.contest_sim.contest_sim_service.load_player_worlds",
        lambda *args, **kwargs: PlayerWorlds(
            fpts_matrix=np.array([[10.0], [20.0], [0.0]], dtype=np.float64),
            player_index={"1": 0},
            minutes_matrix=np.array([[10.0], [20.0], [0.0]], dtype=np.float64),
        ),
    )

    pool = optimizer_service.build_player_pool(
        game_date="2026-03-01",
        draft_group_id=123,
        site="dk",
        use_user_overrides=True,
    )

    assert len(pool) == 1
    player = pool[0]
    assert player["model_proj"] == 10.0
    assert player["override_minutes_delta"] == 10.0
    assert player["has_override"] is True
    assert abs(player["effective_minutes"] - ((20.0 + 30.0 + 0.0) / 3.0)) < 1e-6
    assert abs(player["effective_proj"] - ((20.0 + 30.0 + 0.0) / 3.0)) < 1e-6
    assert player["proj"] == player["effective_proj"]


def test_apply_strategy_overrides_matches_float_like_player_ids() -> None:
    overrides = strategy_overrides.SlateStrategyOverrides(
        game_date="2026-03-01",
        draft_group_id=123,
        overrides={
            "1626171": strategy_overrides.PlayerStrategyOverride(
                player_id="1626171",
                fpts_delta=15.0,
            )
        },
    )
    df = pd.DataFrame(
        [
            {
                "player_id": 1626171.0,
                "player_name": "Bobby Portis",
                "proj_fpts": 20.0,
                "minutes": 20.0,
            }
        ]
    )
    out = strategy_overrides.apply_strategy_overrides(df, overrides)
    row = out.iloc[0]
    assert bool(row["has_override"])
    assert float(row["override_fpts_delta"]) == 15.0
    assert float(row["effective_fpts"]) > float(row["model_fpts"])


def test_apply_strategy_overrides_to_worlds_matches_float_like_player_index() -> None:
    overrides = strategy_overrides.SlateStrategyOverrides(
        game_date="2026-03-01",
        draft_group_id=123,
        overrides={
            "1626171": strategy_overrides.PlayerStrategyOverride(
                player_id="1626171",
                minutes_delta=5.0,
            )
        },
    )
    adjusted_fpts, adjusted_minutes, diagnostics = strategy_overrides.apply_strategy_overrides_to_worlds(
        fpts_matrix=np.array([[10.0], [20.0], [0.0]], dtype=np.float64),
        minutes_matrix=np.array([[10.0], [20.0], [0.0]], dtype=np.float64),
        player_index={"1626171.0": 0},
        overrides=overrides,
        model_minutes_by_player={"1626171": 15.0},
        model_fpts_by_player={"1626171": 10.0},
    )
    np.testing.assert_allclose(adjusted_fpts[:, 0], np.array([15.0, 25.0, 0.0]))
    assert adjusted_minutes is not None
    np.testing.assert_allclose(adjusted_minutes[:, 0], np.array([15.0, 25.0, 0.0]))
    assert diagnostics["matched_override_count"] == 1


def test_apply_strategy_overrides_rebalances_ownership_when_renormalize() -> None:
    overrides = strategy_overrides.SlateStrategyOverrides(
        game_date="2026-03-01",
        draft_group_id=123,
        overrides={
            "1": strategy_overrides.PlayerStrategyOverride(
                player_id="1",
                fpts_delta=12.0,
            )
        },
    )
    df = pd.DataFrame(
        [
            {"player_id": "1", "proj_fpts": 12.0, "minutes": 24.0, "pred_own_pct": 25.0},
            {"player_id": "2", "proj_fpts": 24.0, "minutes": 32.0, "pred_own_pct": 30.0},
            {"player_id": "3", "proj_fpts": 18.0, "minutes": 28.0, "pred_own_pct": 35.0},
        ]
    )

    out = strategy_overrides.apply_strategy_overrides(df, overrides, ownership_mode="renormalize")

    own_before = float(out["model_own"].sum())
    own_after = float(out["effective_own"].sum())
    assert abs(own_after - own_before) < 1e-9
    assert float(out.loc[out["player_id"] == "1", "effective_own"].iloc[0]) > 25.0
    assert float(out.loc[out["player_id"] == "2", "effective_own"].iloc[0]) < 30.0
    assert float(out.loc[out["player_id"] == "3", "effective_own"].iloc[0]) < 35.0


def test_apply_strategy_overrides_keeps_ownership_raw_mode() -> None:
    overrides = strategy_overrides.SlateStrategyOverrides(
        game_date="2026-03-01",
        draft_group_id=123,
        overrides={
            "1": strategy_overrides.PlayerStrategyOverride(
                player_id="1",
                fpts_delta=12.0,
            )
        },
    )
    df = pd.DataFrame(
        [
            {"player_id": "1", "proj_fpts": 12.0, "minutes": 24.0, "pred_own_pct": 25.0},
            {"player_id": "2", "proj_fpts": 24.0, "minutes": 32.0, "pred_own_pct": 30.0},
            {"player_id": "3", "proj_fpts": 18.0, "minutes": 28.0, "pred_own_pct": 35.0},
        ]
    )

    out = strategy_overrides.apply_strategy_overrides(df, overrides, ownership_mode="raw")
    np.testing.assert_allclose(
        out["effective_own"].to_numpy(dtype=float),
        out["model_own"].to_numpy(dtype=float),
    )
