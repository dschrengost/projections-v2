from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from projections.minutes.reconcile import (
    IN_ROTATION_THRESHOLD_MIN,
    MINUTES_CONTRACT_VERSION,
    reconcile_team_minutes,
    reconcile_team_minutes_matrix,
)
from projections.ops.overrides import apply_overrides_to_minutes_df, upsert_overrides


def test_reconcile_team_minutes_hits_target_and_preserves_locked_player() -> None:
    # Base: 245 total, one locked player at 40.
    df = pd.DataFrame(
        {
            "game_id": [1] * 8,
            "team_id": [10] * 8,
            "player_id": list(range(1, 9)),
            "status": ["available"] * 8,
            "minutes_p50": [40.0, 34.0, 33.0, 32.0, 31.0, 25.0, 25.0, 25.0],
        }
    )
    locked = pd.Series([True] + [False] * 7, index=df.index)
    out, diag = reconcile_team_minutes(
        df,
        240.0,
        minutes_col="minutes_p50",
        locked_mask=locked,
        default_cap=48.0,
        in_rotation_threshold_min=float(IN_ROTATION_THRESHOLD_MIN),
    )
    assert float(out.sum()) == pytest.approx(240.0, abs=1e-6)
    assert float(out.iloc[0]) == pytest.approx(40.0, abs=1e-9)
    assert diag.locked_infeasible is False


def test_reconcile_team_minutes_relaxes_caps_when_infeasible() -> None:
    # 5 players with cap=41 cannot reach 240 => caps are relaxed.
    df = pd.DataFrame(
        {
            "game_id": [1] * 5,
            "team_id": [10] * 5,
            "player_id": list(range(100, 105)),
            "status": ["available"] * 5,
            "minutes_p50": [40.0] * 5,
            "cap": [41.0] * 5,
        }
    )
    out, diag = reconcile_team_minutes(
        df,
        240.0,
        minutes_col="minutes_p50",
        cap_col="cap",
        locked_mask=None,
        default_cap=41.0,
    )
    assert float(out.sum()) == pytest.approx(240.0, abs=1e-6)
    assert float(out.max()) > 41.0
    assert diag.cap_infeasible is True


def test_reconcile_team_minutes_forces_out_to_zero() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1, 1],
            "team_id": [10, 10],
            "player_id": [1, 2],
            "status": ["out", "available"],
            "minutes_p50": [20.0, 220.0],
        }
    )
    out, diag = reconcile_team_minutes(df, 240.0, minutes_col="minutes_p50", default_cap=240.0)
    assert float(out.iloc[0]) == 0.0
    assert float(out.sum()) == pytest.approx(240.0, abs=1e-6)
    assert diag.n_out_or_dnp == 1


def test_reconcile_team_minutes_matrix_projects_each_row_to_target() -> None:
    m0 = np.full((2, 5), 40.0, dtype=float)
    active = np.ones_like(m0, dtype=bool)
    out, stats = reconcile_team_minutes_matrix(
        m0,
        active,
        target_minutes=240.0,
        cap_minutes=41.0,
        weights=None,
        tiers=None,
    )
    np.testing.assert_allclose(out.sum(axis=1), 240.0, rtol=0.0, atol=1e-6)
    assert float(out.max()) > 41.0
    assert int(stats["n_cap_infeasible_rows"]) == 2


def test_apply_overrides_minutes_delta_before_reconcile(tmp_path: Path) -> None:
    slate_day = date(2026, 1, 18)
    # Player 100 gets +5 minutes_delta; team is then reconciled back to 240 by moving other players.
    upsert_overrides(
        slate_day,
        [
            {
                "game_id": "1",
                "player_id": "100",
                "fields": {"minutes_delta": 5.0},
            }
        ],
        data_root=tmp_path,
    )

    base = pd.DataFrame(
        [
            {"game_id": 1, "team_id": 10, "player_id": 100, "status": "available", "minutes_p50": 35.0},
            {"game_id": 1, "team_id": 10, "player_id": 101, "status": "available", "minutes_p50": 34.0},
            {"game_id": 1, "team_id": 10, "player_id": 102, "status": "available", "minutes_p50": 33.0},
            {"game_id": 1, "team_id": 10, "player_id": 103, "status": "available", "minutes_p50": 32.0},
            {"game_id": 1, "team_id": 10, "player_id": 104, "status": "available", "minutes_p50": 31.0},
            {"game_id": 1, "team_id": 10, "player_id": 105, "status": "available", "minutes_p50": 25.0},
            {"game_id": 1, "team_id": 10, "player_id": 106, "status": "available", "minutes_p50": 25.0},
            {"game_id": 1, "team_id": 10, "player_id": 107, "status": "available", "minutes_p50": 25.0},
        ]
    )
    base["minutes_p50_cond"] = base["minutes_p50"]

    out = apply_overrides_to_minutes_df(
        base,
        game_date=slate_day,
        data_root=tmp_path,
        force_reconcile=True,
    )
    assert {"minutes_p50_model", "minutes_final", "minutes_delta", "minutes_delta_applied", "ops_override_applied"}.issubset(
        out.columns
    )
    assert {"minutes_contract_version", "minutes_contract_hash"}.issubset(out.columns)
    assert int(out["minutes_contract_version"].iloc[0]) == int(MINUTES_CONTRACT_VERSION)

    team = out[(out["game_id"] == 1) & (out["team_id"] == 10)].copy()
    team["minutes_final"] = pd.to_numeric(team["minutes_final"], errors="coerce").fillna(0.0)
    assert float(team["minutes_final"].sum()) == pytest.approx(240.0, abs=1e-3)

    p100 = team[team["player_id"].astype(int) == 100].iloc[0]
    assert float(p100["minutes_p50_model"]) == pytest.approx(35.0, abs=1e-9)
    assert float(p100["minutes_p50"]) == pytest.approx(40.0, abs=1e-6)
    assert bool(p100["minutes_delta_applied"]) is True
    assert bool(p100["ops_override_applied"]) is True


def test_apply_overrides_ops_depth_role_out_zeroes_minutes(tmp_path: Path) -> None:
    slate_day = date(2026, 1, 18)
    upsert_overrides(
        slate_day,
        [
            {
                "game_id": "1",
                "player_id": "100",
                "fields": {"ops_depth_role": "out"},
            }
        ],
        data_root=tmp_path,
    )

    base = pd.DataFrame(
        [
            {"game_id": 1, "team_id": 10, "player_id": 100, "status": "available", "play_prob": 0.8, "minutes_p50": 35.0},
            {"game_id": 1, "team_id": 10, "player_id": 101, "status": "available", "play_prob": 0.9, "minutes_p50": 34.0},
            {"game_id": 1, "team_id": 10, "player_id": 102, "status": "available", "play_prob": 0.9, "minutes_p50": 33.0},
            {"game_id": 1, "team_id": 10, "player_id": 103, "status": "available", "play_prob": 0.9, "minutes_p50": 32.0},
            {"game_id": 1, "team_id": 10, "player_id": 104, "status": "available", "play_prob": 0.9, "minutes_p50": 31.0},
            {"game_id": 1, "team_id": 10, "player_id": 105, "status": "available", "play_prob": 0.9, "minutes_p50": 25.0},
            {"game_id": 1, "team_id": 10, "player_id": 106, "status": "available", "play_prob": 0.9, "minutes_p50": 25.0},
            {"game_id": 1, "team_id": 10, "player_id": 107, "status": "available", "play_prob": 0.9, "minutes_p50": 25.0},
        ]
    )
    base["minutes_p50_cond"] = base["minutes_p50"]

    out = apply_overrides_to_minutes_df(base, game_date=slate_day, data_root=tmp_path, force_reconcile=True)
    team = out[(out["game_id"] == 1) & (out["team_id"] == 10)].copy()
    team["minutes_final"] = pd.to_numeric(team["minutes_final"], errors="coerce").fillna(0.0)
    assert float(team["minutes_final"].sum()) == pytest.approx(240.0, abs=1e-3)

    p100 = team[team["player_id"].astype(int) == 100].iloc[0]
    assert str(p100["status"]).strip().lower() == "out"
    assert float(p100["play_prob"]) == pytest.approx(0.0, abs=1e-9)
    assert float(p100["minutes_p50"]) == pytest.approx(0.0, abs=1e-9)
    assert float(p100["minutes_p50_cond"]) == pytest.approx(0.0, abs=1e-9)
    assert str(p100.get("ops_depth_role") or "").strip().lower() == "out"
    assert bool(p100["ops_override_applied"]) is True


def test_apply_overrides_ops_depth_role_starter_promotes_signals_and_minutes(tmp_path: Path) -> None:
    slate_day = date(2026, 1, 18)
    upsert_overrides(
        slate_day,
        [
            {
                "game_id": "1",
                "player_id": "100",
                "fields": {"ops_depth_role": "starter"},
            }
        ],
        data_root=tmp_path,
    )

    base = pd.DataFrame(
        [
            {
                "game_id": 1,
                "team_id": 10,
                "player_id": 100,
                "status": "available",
                "play_prob": 0.0,
                "rotation_prob": 0.82,
                "is_projected_starter": 0,
                "starter_flag": 0,
                "is_starter": 0,
                "minutes_p10": 0.0,
                "minutes_p50": 0.0,
                "minutes_p90": 12.0,
            },
            {"game_id": 1, "team_id": 10, "player_id": 101, "status": "available", "play_prob": 0.95, "rotation_prob": 0.95, "is_projected_starter": 1, "starter_flag": 1, "is_starter": 1, "minutes_p10": 28.0, "minutes_p50": 34.0, "minutes_p90": 40.0},
            {"game_id": 1, "team_id": 10, "player_id": 102, "status": "available", "play_prob": 0.95, "rotation_prob": 0.95, "is_projected_starter": 1, "starter_flag": 1, "is_starter": 1, "minutes_p10": 27.0, "minutes_p50": 33.0, "minutes_p90": 39.0},
            {"game_id": 1, "team_id": 10, "player_id": 103, "status": "available", "play_prob": 0.95, "rotation_prob": 0.95, "is_projected_starter": 1, "starter_flag": 1, "is_starter": 1, "minutes_p10": 26.0, "minutes_p50": 32.0, "minutes_p90": 38.0},
            {"game_id": 1, "team_id": 10, "player_id": 104, "status": "available", "play_prob": 0.90, "rotation_prob": 0.90, "is_projected_starter": 1, "starter_flag": 1, "is_starter": 1, "minutes_p10": 25.0, "minutes_p50": 31.0, "minutes_p90": 37.0},
            {"game_id": 1, "team_id": 10, "player_id": 105, "status": "available", "play_prob": 0.80, "rotation_prob": 0.80, "is_projected_starter": 0, "starter_flag": 0, "is_starter": 0, "minutes_p10": 8.0, "minutes_p50": 25.0, "minutes_p90": 31.0},
            {"game_id": 1, "team_id": 10, "player_id": 106, "status": "available", "play_prob": 0.80, "rotation_prob": 0.80, "is_projected_starter": 0, "starter_flag": 0, "is_starter": 0, "minutes_p10": 8.0, "minutes_p50": 25.0, "minutes_p90": 31.0},
            {"game_id": 1, "team_id": 10, "player_id": 107, "status": "available", "play_prob": 0.80, "rotation_prob": 0.80, "is_projected_starter": 0, "starter_flag": 0, "is_starter": 0, "minutes_p10": 8.0, "minutes_p50": 25.0, "minutes_p90": 31.0},
        ]
    )
    for q in ("minutes_p10", "minutes_p50", "minutes_p90"):
        base[f"{q}_cond"] = base[q]

    out = apply_overrides_to_minutes_df(base, game_date=slate_day, data_root=tmp_path, force_reconcile=True)
    team = out[(out["game_id"] == 1) & (out["team_id"] == 10)].copy()
    team["minutes_final"] = pd.to_numeric(team["minutes_final"], errors="coerce").fillna(0.0)
    assert float(team["minutes_final"].sum()) == pytest.approx(240.0, abs=1e-3)

    p100 = team[team["player_id"].astype(int) == 100].iloc[0]
    assert str(p100.get("ops_depth_role") or "").strip().lower() == "starter"
    assert int(pd.to_numeric(p100["is_projected_starter"], errors="coerce") or 0) == 1
    assert int(pd.to_numeric(p100["starter_flag"], errors="coerce") or 0) == 1
    assert int(pd.to_numeric(p100["is_starter"], errors="coerce") or 0) == 1
    assert float(pd.to_numeric(p100["play_prob"], errors="coerce") or 0.0) >= 0.55
    assert float(pd.to_numeric(p100["minutes_p50"], errors="coerce") or 0.0) >= 18.0
    assert float(pd.to_numeric(p100["minutes_p90"], errors="coerce") or 0.0) >= float(
        pd.to_numeric(p100["minutes_p50"], errors="coerce") or 0.0
    )


def test_apply_overrides_promotes_zero_minute_bench_in_ops_injury_regime(tmp_path: Path) -> None:
    slate_day = date(2026, 1, 18)
    upsert_overrides(
        slate_day,
        [
            {"game_id": "1", "player_id": "100", "fields": {"ops_depth_role": "out"}},
            {"game_id": "1", "player_id": "101", "fields": {"ops_depth_role": "out"}},
        ],
        data_root=tmp_path,
    )

    base = pd.DataFrame(
        [
            {"game_id": 1, "team_id": 10, "player_id": 100, "status": "available", "play_prob": 0.90, "rotation_prob": 0.90, "minutes_p10": 20.0, "minutes_p50": 30.0, "minutes_p90": 35.0},
            {"game_id": 1, "team_id": 10, "player_id": 101, "status": "available", "play_prob": 0.90, "rotation_prob": 0.90, "minutes_p10": 20.0, "minutes_p50": 30.0, "minutes_p90": 35.0},
            {"game_id": 1, "team_id": 10, "player_id": 102, "status": "available", "play_prob": 0.95, "rotation_prob": 0.95, "minutes_p10": 24.0, "minutes_p50": 34.0, "minutes_p90": 40.0},
            {"game_id": 1, "team_id": 10, "player_id": 103, "status": "available", "play_prob": 0.95, "rotation_prob": 0.95, "minutes_p10": 23.0, "minutes_p50": 33.0, "minutes_p90": 39.0},
            {"game_id": 1, "team_id": 10, "player_id": 104, "status": "available", "play_prob": 0.90, "rotation_prob": 0.90, "minutes_p10": 22.0, "minutes_p50": 32.0, "minutes_p90": 38.0},
            {"game_id": 1, "team_id": 10, "player_id": 105, "status": "available", "play_prob": 0.85, "rotation_prob": 0.85, "minutes_p10": 21.0, "minutes_p50": 31.0, "minutes_p90": 37.0},
            {"game_id": 1, "team_id": 10, "player_id": 106, "status": "available", "play_prob": 0.75, "rotation_prob": 0.75, "minutes_p10": 8.0, "minutes_p50": 20.0, "minutes_p90": 26.0},
            {"game_id": 1, "team_id": 10, "player_id": 107, "status": "available", "play_prob": 0.00, "rotation_prob": 0.60, "minutes_p10": 0.0, "minutes_p50": 0.0, "minutes_p90": 14.0},
            {"game_id": 1, "team_id": 10, "player_id": 108, "status": "available", "play_prob": 0.10, "rotation_prob": 0.20, "minutes_p10": 0.0, "minutes_p50": 0.0, "minutes_p90": 5.0},
        ]
    )
    for q in ("minutes_p10", "minutes_p50", "minutes_p90"):
        base[f"{q}_cond"] = base[q]

    out = apply_overrides_to_minutes_df(base, game_date=slate_day, data_root=tmp_path, force_reconcile=True)
    team = out[(out["game_id"] == 1) & (out["team_id"] == 10)].copy()
    team["minutes_final"] = pd.to_numeric(team["minutes_final"], errors="coerce").fillna(0.0)
    assert float(team["minutes_final"].sum()) == pytest.approx(240.0, abs=1e-3)

    p107 = team[team["player_id"].astype(int) == 107].iloc[0]
    assert float(pd.to_numeric(p107["minutes_p50"], errors="coerce") or 0.0) >= 6.0
    assert float(pd.to_numeric(p107["play_prob"], errors="coerce") or 0.0) >= 0.15

def test_minutes_api_prefers_effective_minutes(tmp_path: Path) -> None:
    from projections.api import minutes_api
    from projections.pipeline.effective_inputs import EFFECTIVE_MINUTES_FILENAME

    run_dir = tmp_path / "run=test"
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.DataFrame([{"player_id": 1, "minutes_p50": 10.0}])
    effective = pd.DataFrame([{"player_id": 1, "minutes_p50": 20.0}])
    baseline.to_parquet(run_dir / "minutes.parquet", index=False)
    effective.to_parquet(run_dir / EFFECTIVE_MINUTES_FILENAME, index=False)

    df = minutes_api._load_minutes(run_dir)
    assert float(pd.to_numeric(df["minutes_p50"], errors="coerce").iloc[0]) == 20.0


def test_apply_overrides_minutes_final_prefers_effective_minutes(tmp_path: Path) -> None:
    slate_day = date(2026, 1, 18)

    base = pd.DataFrame(
        [
            {
                "game_id": 1,
                "team_id": 10,
                "player_id": 100,
                "status": "available",
                "minutes_p50": 12.0,
                "minutes_p50_cond": 12.0,
                "effective_minutes": 200.0,
            },
            {
                "game_id": 1,
                "team_id": 10,
                "player_id": 101,
                "status": "available",
                "minutes_p50": 12.0,
                "minutes_p50_cond": 12.0,
                "effective_minutes": 40.0,
            },
        ]
    )

    out = apply_overrides_to_minutes_df(
        base,
        game_date=slate_day,
        data_root=tmp_path,
        force_reconcile=True,
    )

    assert {"minutes_final", "minutes_contract_version", "minutes_contract_hash"}.issubset(out.columns)
    out["minutes_final"] = pd.to_numeric(out["minutes_final"], errors="coerce").fillna(0.0)
    out["effective_minutes"] = pd.to_numeric(out["effective_minutes"], errors="coerce").fillna(0.0)
    assert float(out["minutes_final"].sum()) == pytest.approx(240.0, abs=1e-6)
    pd.testing.assert_series_equal(out["minutes_final"], out["effective_minutes"], check_names=False)
    # Ensure we don't force-reconcile the raw model quantiles when effective minutes exist.
    assert float(pd.to_numeric(out["minutes_p50"], errors="coerce").fillna(0.0).sum()) == pytest.approx(
        24.0, abs=1e-6
    )
