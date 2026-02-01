from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from projections.cli.build_minutes_prior_internal import build_minutes_prior_internal_df


def _data_root() -> Path:
    return Path(os.environ.get("PROJECTIONS_DATA_ROOT", "/home/daniel/projections-data")).expanduser().resolve()


def _require_season_2024_inputs(root: Path) -> None:
    if not (root / "silver" / "nba_daily_lineups" / "season=2024").exists():
        pytest.skip("Missing silver/nba_daily_lineups season=2024 (requires local projections-data)")
    if not (root / "gold" / "minutes_for_rates_reconciled" / "season=2024").exists():
        pytest.skip("Missing gold/minutes_for_rates_reconciled season=2024 (requires local projections-data)")
    if not (root / "artifacts" / "pbp_v1" / "LATEST_PUBLISHED").exists():
        pytest.skip("Missing artifacts/pbp_v1/LATEST_PUBLISHED (requires local projections-data)")


def test_minutes_prior_internal_smoke_is_stable_and_in_internal_id_space() -> None:
    root = _data_root()
    _require_season_2024_inputs(root)

    df1, unmapped1 = build_minutes_prior_internal_df(
        season_start_year=2024,
        data_root=root,
        limit_paths=10,  # keep test runtime reasonable
        emit_diagnostics=False,
    )
    df2, unmapped2 = build_minutes_prior_internal_df(
        season_start_year=2024,
        data_root=root,
        limit_paths=10,
        emit_diagnostics=False,
    )

    assert df1.equals(df2)
    assert unmapped1.equals(unmapped2)

    required_cols = ["game_id", "team_id", "player_id", "minutes_prior", "minutes_p10", "minutes_p90", "play_prob"]
    assert list(df1.columns) == required_cols

    assert df1["game_id"].astype("string").str.len().eq(10).all()
    assert df1["team_id"].dtype == "int64"
    assert df1["player_id"].dtype == "int64"
    assert df1["minutes_prior"].dtype == "float64"
    assert df1["minutes_p10"].dtype == "float64"
    assert df1["minutes_p90"].dtype == "float64"
    assert df1["play_prob"].dtype == "float64"

    assert int(df1["player_id"].max()) <= 2000

    # Deterministic content hash
    h1 = int(pd.util.hash_pandas_object(df1, index=False).sum())
    h2 = int(pd.util.hash_pandas_object(df2, index=False).sum())
    assert h1 == h2

