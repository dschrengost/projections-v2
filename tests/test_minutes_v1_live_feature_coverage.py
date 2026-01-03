from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from projections.builders.features_builder import FeatureBuildConfig, SharedFeaturesBuilder


def _nan_rate(df: pd.DataFrame, cols: list[str]) -> float:
    present = [c for c in cols if c in df.columns]
    if not present or df.empty:
        return 0.0
    return float(df[present].isna().mean().mean())


def test_live_minutes_context_and_dispersion_priors_non_null(tmp_path: Path) -> None:
    """Regression: avoid feature deserts for live minutes_v1 scoring."""

    cfg = FeatureBuildConfig(
        data_root=tmp_path,
        season=2025,
        as_of_ts=pd.Timestamp("2025-12-15T18:30:00Z"),
        target_day=date(2025, 12, 15),
    )
    builder = SharedFeaturesBuilder(cfg)

    # ------------------------------------------------------------------
    # Arrange: minimal rates_training_base partitions for team context.
    # Only one team appears per day to reproduce the historical NaN issue
    # (latest partition doesn't cover all slate teams).
    # ------------------------------------------------------------------
    rates_root = tmp_path / "gold" / "rates_training_base" / "season=2025"
    day_a = rates_root / "game_date=2025-12-13"
    day_b = rates_root / "game_date=2025-12-14"
    day_a.mkdir(parents=True, exist_ok=True)
    day_b.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "team_id": [1, 1],
            "team_pace_szn": [97.0, 97.0],
            "team_off_rtg_szn": [111.0, 111.0],
            "team_def_rtg_szn": [108.0, 108.0],
        }
    ).to_parquet(day_a / "rates_training_base.parquet", index=False)

    pd.DataFrame(
        {
            "team_id": [2, 2],
            "team_pace_szn": [101.0, 101.0],
            "team_off_rtg_szn": [114.0, 114.0],
            "team_def_rtg_szn": [106.0, 106.0],
        }
    ).to_parquet(day_b / "rates_training_base.parquet", index=False)

    # ------------------------------------------------------------------
    # Arrange: a mocked mini-slate (2 teams, 2 players per team).
    # ------------------------------------------------------------------
    live = pd.DataFrame(
        {
            "game_id": [100, 100, 100, 100],
            "player_id": [101, 102, 201, 202],
            "team_id": [1, 1, 2, 2],
            "opponent_team_id": [2, 2, 1, 1],
        }
    )

    warnings: list[str] = []
    with_ctx = builder._attach_team_context(live, warnings=warnings)

    # Backfilled context should be sourced per-team across multiple prior partitions.
    assert with_ctx.loc[with_ctx["team_id"] == 1, "team_pace_szn"].iloc[0] == 97.0
    assert with_ctx.loc[with_ctx["team_id"] == 2, "team_pace_szn"].iloc[0] == 101.0
    assert with_ctx.loc[with_ctx["team_id"] == 1, "opp_pace_szn"].iloc[0] == 101.0
    assert with_ctx.loc[with_ctx["team_id"] == 2, "opp_pace_szn"].iloc[0] == 97.0

    # ------------------------------------------------------------------
    # Arrange: historical labels to compute team_minutes_dispersion_prior.
    # Each team has a prior game with boxscore minutes.
    # ------------------------------------------------------------------
    labels_source = pd.DataFrame(
        {
            "team_id": [1, 1, 2, 2],
            "game_id": [9001, 9001, 9002, 9002],
            "game_date": [
                pd.Timestamp("2025-12-14"),
                pd.Timestamp("2025-12-14"),
                pd.Timestamp("2025-12-14"),
                pd.Timestamp("2025-12-14"),
            ],
            "minutes": [30.0, 18.0, 28.0, 12.0],
        }
    )

    with_disp = builder._attach_team_minutes_dispersion_prior(
        with_ctx, labels_source=labels_source, warnings=warnings
    )

    # team1: std([30,18]) = 6.0 ; team2: std([28,12]) = 8.0
    assert with_disp.loc[with_disp["team_id"] == 1, "team_minutes_dispersion_prior"].iloc[0] == 6.0
    assert with_disp.loc[with_disp["team_id"] == 2, "team_minutes_dispersion_prior"].iloc[0] == 8.0

    # ------------------------------------------------------------------
    # Regression thresholds: avoid NaN deserts.
    # ------------------------------------------------------------------
    team_cols = ["team_pace_szn", "team_off_rtg_szn", "team_def_rtg_szn"]
    opp_cols = ["opp_pace_szn", "opp_def_rtg_szn"]
    assert _nan_rate(with_disp, team_cols) <= 0.05
    assert _nan_rate(with_disp, opp_cols) <= 0.05
    assert _nan_rate(with_disp, ["team_minutes_dispersion_prior"]) == 0.0

