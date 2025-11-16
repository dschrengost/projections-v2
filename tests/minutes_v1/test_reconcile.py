from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from projections.minutes_v1.reconcile import (
    ReconcileConfig,
    TeamMinutesConfig,
    load_reconcile_config,
    reconcile_minutes_p50_all,
)


def _make_team_frame(minutes: list[float], starters: list[int], team_id: int = 1) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "game_id": 100,
            "team_id": team_id,
            "player_id": range(len(minutes)),
            "minutes_p50": minutes,
            "minutes_p10": np.maximum(np.array(minutes) - 4, 0.0),
            "minutes_p90": np.array(minutes) + 6,
            "p_play": 0.8,
            "is_projected_starter": starters,
        }
    )
    return base


def test_reconcile_enforces_team_total() -> None:
    minutes = [32.0, 30.0, 28.0, 26.0, 20.0, 18.0, 12.0, 10.0]
    starters = [1, 1, 1, 1, 0, 0, 0, 0]
    df = _make_team_frame(minutes, starters=starters)
    config = ReconcileConfig(team_minutes=TeamMinutesConfig(target=200.0, tolerance=0.0))

    reconciled = reconcile_minutes_p50_all(df, config)
    total = reconciled["minutes_p50"].sum()
    assert pytest.approx(total, rel=1e-4) == 200.0

    deltas = (reconciled["minutes_p50"] - df["minutes_p50"]).abs()
    starter_delta = deltas[df["is_projected_starter"] == 1].mean()
    rotation_delta = deltas[df["is_projected_starter"] == 0].mean()
    assert starter_delta < rotation_delta


def test_reconcile_respects_bounds() -> None:
    minutes = [32.0, 30.0, 28.0, 26.0, 20.0, 18.0, 12.0, 10.0]
    df = _make_team_frame(minutes, starters=[1, 1, 1, 1, 0, 0, 0, 0])
    df.loc[0, "minutes_cap"] = 24.0
    df.loc[1, "minutes_cap"] = 20.0
    config = ReconcileConfig(team_minutes=TeamMinutesConfig(target=200.0, tolerance=0.0))

    reconciled = reconcile_minutes_p50_all(df, config)

    assert pytest.approx(reconciled["minutes_p50"].sum(), rel=1e-4) == 200.0
    assert reconciled.loc[0, "minutes_p50"] <= 24.0 + 1e-6
    assert reconciled.loc[1, "minutes_p50"] >= min(18.0, config.bounds.starter_floor) - 1e-6


def test_rotation_mask_excludes_low_probability_players() -> None:
    minutes = [30.0, 28.0, 22.0, 18.0, 16.0, 12.0, 0.0, 0.0]
    df = _make_team_frame(minutes, starters=[1, 1, 1, 0, 0, 0, 0, 0])
    df.loc[6:, "p_play"] = 0.01
    df.loc[6:, "minutes_p50"] = 0.0
    config = ReconcileConfig(team_minutes=TeamMinutesConfig(target=150.0, tolerance=0.0))

    reconciled = reconcile_minutes_p50_all(df, config)
    low_prob = reconciled.loc[df["player_id"] >= 6, "minutes_p50"]
    assert (low_prob == 0.0).all()
    assert pytest.approx(reconciled["minutes_p50"].sum(), rel=1e-4) == 150.0


def test_load_reconcile_config_parses_weights(tmp_path: Path) -> None:
    payload = {
        "l2_reconcile": {
            "weights": {
                "starter_penalty": 2.0,
                "rotation_penalty": 1.0,
                "deep_penalty": 0.2,
                "spread_epsilon": 0.25,
                "scale_with_spread": False,
            }
        }
    }
    config_path = tmp_path / "reconcile.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_reconcile_config(config_path)
    weights = config.weights
    assert weights.starter_penalty == 2.0
    assert weights.rotation_penalty == 1.0
    assert weights.deep_penalty == 0.2
    assert weights.spread_epsilon == 0.25
    assert weights.scale_with_spread is False
